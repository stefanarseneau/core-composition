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
    #for i in range(obs_mag.shape[1]):
    #    table[params['column_names'].split(' ')[i]] = obs_mag[:,i]
    #    table[params['column_uncertainties'].split(' ')[i]] = e_obs_mag[:,i]

    # create interpolators and photometric engines
    warwick = wdphoto.WarwickDAInterpolator(bands)
    co_hrich_model = wdphoto.LaPlataInterpolator(bands, massive_params = ('CO', 'Hrich'))
    one_hrich_model = wdphoto.LaPlataInterpolator(bands, massive_params = ('ONe', 'Hrich'))
    co_hdef_model = wdphoto.LaPlataInterpolator(bands, massive_params = ('CO', 'Hdef'))
    one_hdef_model = wdphoto.LaPlataInterpolator(bands, massive_params = ('ONe', 'Hdef'))

    WarwickEngine = wdphoto.PhotometryEngine(warwick)
    CO_Hrich_Engine = wdphoto.PhotometryEngine(co_hrich_model)
    ONe_Hrich_Engine = wdphoto.PhotometryEngine(one_hrich_model)
    CO_Hdef_Engine = wdphoto.PhotometryEngine(co_hdef_model)
    ONe_Hdef_Engine = wdphoto.PhotometryEngine(one_hdef_model)

    outs = np.nan*np.zeros((6, len(table), 7))
    for i in tqdm(range(len(obs_mag))):
        outs[0,i] = CO_Hrich_Engine(obs_mag[i], e_obs_mag[i], distances[i], p0 = [10000, 9, 0.003])#, p0=[catalog['teff'][i], catalog['logg'][i], 0.003])
        outs[1,i] = CO_Hdef_Engine(obs_mag[i], e_obs_mag[i], distances[i], p0 = [10000, 9, 0.003])
        outs[2,i] = ONe_Hrich_Engine(obs_mag[i], e_obs_mag[i], distances[i], p0 = [10000, 9, 0.003])#, p0=[catalog['teff'][i], catalog['logg'][i], 0.003])
        outs[3,i] = ONe_Hdef_Engine(obs_mag[i], e_obs_mag[i], distances[i], p0 = [10000, 9, 0.003])
        outs[4,i] = WarwickEngine(obs_mag[i], e_obs_mag[i], distances[i], p0 = [10000, 8, 0.003])#, p0=[catalog['teff'][i], catalog['logg'][i], 0.003])

        if args.plot_radii:
            f_co_hrich = wdphoto.utils.plot(obs_mag[i], e_obs_mag[i], distances[i], outs[0,i,0], outs[0,i,2], 9, CO_Hrich_Engine)
            plt.legend()
            plt.title(f'Gaia DR3 {source_ids[i]} CO Hrich Model')
            f_co_hrich.savefig(f'/mnt/d/arsen/research/proj/core-composition/figures/co_hrich/{source_ids[i]}.png')
            plt.close()

            f_co_hdef = wdphoto.utils.plot(obs_mag[i], e_obs_mag[i], distances[i], outs[1,i,0], outs[1,i,2], 9, CO_Hdef_Engine)
            plt.legend()
            plt.title(f'Gaia DR3 {source_ids[i]} CO Hdef Model')
            f_co_hdef.savefig(f'/mnt/d/arsen/research/proj/core-composition/figures/co_hdef/{source_ids[i]}.png')
            plt.close()

            f_one_hrich = wdphoto.utils.plot(obs_mag[i], e_obs_mag[i], distances[i], outs[2,i,0], outs[2,i,2], 8, ONe_Hrich_Engine)
            plt.legend()
            plt.title(f'Gaia DR3 {source_ids[i]} ONe Hrich Model')
            f_one_hrich.savefig(f'/mnt/d/arsen/research/proj/core-composition/figures/one_hrich/{source_ids[i]}.png')
            plt.close()

            f_one_hdef = wdphoto.utils.plot(obs_mag[i], e_obs_mag[i], distances[i], outs[3,i,0], outs[3,i,2], 8, ONe_Hdef_Engine)
            plt.legend()
            plt.title(f'Gaia DR3 {source_ids[i]} ONe Hdef Model')
            f_one_hdef.savefig(f'/mnt/d/arsen/research/proj/core-composition/figures/one_hdef/{source_ids[i]}.png')
            plt.close()

            f_warwick = wdphoto.utils.plot(obs_mag[i], e_obs_mag[i], distances[i], outs[4,i,0], outs[4,i,2], 8, WarwickEngine)
            plt.legend()
            plt.title(f'Gaia DR3 {source_ids[i]} Warwick Model')
            f_warwick.savefig(f'/mnt/d/arsen/research/proj/core-composition/figures/warwick/{source_ids[i]}.png')
            plt.close()


    table['CO_Hrich_radius'] = outs[0,:,0]
    table['CO_Hrich_e_radius'] = outs[0,:,1]
    table['CO_Hrich_teff'] = outs[0,:,2]
    table['CO_Hrich_e_teff'] = outs[0,:,3]
    table['CO_Hrich_chi2'] = outs[0,:,-1]
    table['CO_Hrich_roe'] = table['CO_Hrich_radius'] / table['CO_Hrich_e_radius']

    table['CO_Hdef_radius'] = outs[1,:,0]
    table['CO_Hdef_e_radius'] = outs[1,:,1]
    table['CO_Hdef_teff'] = outs[1,:,2]
    table['CO_Hdef_e_teff'] = outs[1,:,3]
    table['CO_Hdef_chi2'] = outs[1,:,-1]
    table['CO_Hdef_roe'] = table['CO_Hdef_radius']/table['CO_Hdef_e_radius']  

    table['ONe_Hrich_radius'] = outs[2,:,0]
    table['ONe_Hrich_e_radius'] = outs[2,:,1]
    table['ONe_Hrich_teff'] = outs[2,:,2]
    table['ONe_Hrich_e_teff'] = outs[2,:,3]
    table['ONe_Hrich_chi2'] = outs[2,:,-1]
    table['ONe_Hrich_roe'] = table['ONe_Hrich_radius'] / table['ONe_Hrich_e_radius']

    table['ONe_Hdef_radius'] = outs[3,:,0]
    table['ONe_Hdef_e_radius'] = outs[3,:,1]
    table['ONe_Hdef_teff'] = outs[3,:,2]
    table['ONe_Hdef_e_teff'] = outs[3,:,3]
    table['ONe_Hdef_chi2'] = outs[3,:,-1]
    table['ONe_Hdef_roe'] = table['ONe_Hdef_radius']/table['ONe_Hdef_e_radius']  

    table['Warwick_radius'] = outs[4,:,0]
    table['Warwick_e_radius'] = outs[4,:,1]
    table['Warwick_teff'] = outs[4,:,2]
    table['Warwick_e_teff'] = outs[4,:,3]
    table['Warwick_chi2'] = outs[4,:,-1]
    table['Warwick_roe'] = table['Warwick_radius']/table['Warwick_e_radius']

    table['CO_Hrich_failed'] = np.any([table['CO_Hrich_chi2'] > 5, table['CO_Hrich_roe'] < 5], axis=0)
    table['CO_Hdef_failed'] = np.any([table['CO_Hdef_chi2'] > 5, table['CO_Hdef_roe'] < 5], axis=0)
    table['ONe_Hrich_failed'] = np.any([table['ONe_Hrich_chi2'] > 5, table['ONe_Hrich_roe'] < 5], axis=0)
    table['ONe_Hdef_failed'] = np.any([table['ONe_Hdef_chi2'] > 5, table['ONe_Hdef_roe'] < 5], axis=0)
    table['Warwick_failed'] = np.any([table['Warwick_chi2'] > 5, table['Warwick_roe'] < 5], axis=0)
    table['all_clear'] = ~np.any([table['CO_Hrich_failed'], table['CO_Hdef_failed'], 
                           table['ONe_Hrich_failed'], table['ONe_Hdef_failed'], 
                           table['Warwick_failed']], axis=0)

    print('\nFit Report:')
    print(f'CO Hrich failed={sum(table['CO_Hrich_failed'])/len(table)*100:2.2f}%')
    print(f'CO Hdef failed={sum(table['CO_Hdef_failed'])/len(table)*100:2.2f}%')
    print(f'ONe Hrich failed={sum(table['ONe_Hrich_failed'])/len(table)*100:2.2f}%')
    print(f'ONe Hdef failed={sum(table['ONe_Hdef_failed'])/len(table)*100:2.2f}%')
    print(f'Warwick failed={sum(table['Warwick_failed'])/len(table)*100:2.2f}%')
    print(f'Total failed={(1 - (sum(table['all_clear']) / len(table)))*100:2.2f}%')   

    if args.radius_path is not None:
        table.write(args.radius_path, overwrite=True)

    return table

