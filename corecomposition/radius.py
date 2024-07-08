import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import configparser
import argparse

from astropy.table import Table
from dustmaps.edenhofer2023 import Edenhofer2023Query
import wdphoto

def measure_radius(catalog, params, args):
    # load in the necessary parameters
    source_ids = np.array(catalog['wd_source_id'])
    obs_mag = np.array([catalog[param] for param in params['column_names'].split(' ')]).T
    e_obs_mag = np.array([catalog[param] for param in params['column_uncertainties'].split(' ')]).T
    distances = np.array(catalog[params['distance']])
    bands = params['pyphot_bands'].split(' ')

    # apply Edenhofer dereddening
    if args.deredden:
        l = catalog['wd_l']
        b = catalog['wd_b']

        bsq = Edenhofer2023Query() # initiate Edenhofer query
        obs_mag = wdphoto.deredden(bsq, l, b, obs_mag, distances, bands) # perform query
        del bsq # delete this to save memory

    table = Table()
    table['source_id'] = source_ids
    table['distance'] = distances


    # create interpolators and photometric engines
    warwick = wdphoto.WarwickDAInterpolator(bands)
    co_hrich_model = wdphoto.LaPlataInterpolator(bands, massive_params = ('CO', 'Hrich'))
    one_hrich_model = wdphoto.LaPlataInterpolator(bands, massive_params = ('ONe', 'Hrich'))
    co_hdef_model = wdphoto.LaPlataInterpolator(bands, massive_params = ('CO', 'Hdef'))
    one_hdef_model = wdphoto.LaPlataInterpolator(bands, massive_params = ('ONe', 'Hdef'))

    interpolators = {'Warwick': (warwick, 8), 'CO_Hrich': (co_hrich_model, 9), 'ONe_Hrich': (one_hrich_model, 9),
                     'CO_Hdef': (co_hdef_model, 9), 'ONe_Hdef': (one_hdef_model, 9)}
    engines = {key : (wdphoto.PhotometryEngine(interpolators[key][0]), interpolators[key][1]) for key in interpolators.keys()}

    outs = np.nan*np.zeros((len(engines), len(table), 7))
    for i in tqdm(range(len(obs_mag))):
        for j, key in enumerate(engines.keys()):
            outs[j,i] = engines[key][0](obs_mag[i], e_obs_mag[i], distances[i], p0 = [10000, engines[key][1], 0.003])#, p0=[catalog['teff'][i], catalog['logg'][i], 0.003])

            if args.plot_radii:
                model = wdphoto.utils.plot(obs_mag[i], e_obs_mag[i], distances[i], outs[j,i,0], outs[j,i,2], engines[key][1], key)
                plt.legend()
                plt.title(f'Gaia DR3 {source_ids[i]} {key} Model')
                model.savefig(f'/mnt/d/arsen/research/proj/core-composition/figures/{key}/{source_ids[i]}.png')
                plt.close()

    for j, key in enumerate(engines.keys()):
        table[f'{key}_radius'] = outs[j,:,0]
        table[f'{key}_e_radius'] = outs[j,:,1]
        table[f'{key}_teff'] = outs[j,:,2]
        table[f'{key}_e_teff'] = outs[j,:,3]
        table[f'{key}_chi2'] = outs[j,:,-1]
        table[f'{key}_roe'] = table[f'{key}_radius'] / table[f'{key}_e_radius']
        table[f'{key}_failed'] = np.any([table[f'{key}_chi2'] > 5, table[f'{key}_roe'] < 5], axis=0)

    table['all_clear'] = ~np.any([table[f'{key}_failed'] for key in engines.keys()], axis=0)

    print('\nFit Report:')
    for key in engines.keys():
        print(f'{key} failed={sum(table[f'{key}_failed'])/len(table)*100:2.2f}%')
    print(f'Total failed={(1 - (sum(table['all_clear']) / len(table)))*100:2.2f}%')   

    if args.radius_path is not None:
        table.write(args.radius_path, overwrite=True)

    return table, engines.keys()

