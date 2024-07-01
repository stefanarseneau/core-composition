import numpy as np
from tqdm import tqdm
import configparser
import argparse

from astropy.table import Table
from dustmaps.edenhofer2023 import Edenhofer2023Query
import wdphoto

def measure_radius(catalog, params, args):
    # load in the necessary parameters
    print('Loading parameters...', end=' ')
    source_ids = np.array(catalog['wd_source_id'])
    obs_mag = np.array([catalog[param] for param in params['column_names'].split(' ')]).T
    e_obs_mag = np.array([catalog[param] for param in params['column_uncertainties'].split(' ')]).T
    distances = np.array(catalog[params['distance']])
    bands = params['pyphot_bands'].split(' ')
    print('Done!') 

    # apply Edenhofer dereddening
    if args.deredden:
        print('Applying dereddening Using Edenhofer+2023')
        l = catalog['wd_l']
        b = catalog['wd_b']

        bsq = Edenhofer2023Query() # initiate Edenhofer query
        obs_mag = wdphoto.deredden(bsq, l, b, obs_mag, distances, bands) # perform query
        del bsq # delete this to save memory

    print('Creating radius table...', end=' ')
    table = Table()
    table['source_id'] = source_ids
    table['distance'] = distances
    for i in range(obs_mag.shape[1]):
        table[params['column_names'].split(' ')[i]] = obs_mag[:,i]
        table[params['column_uncertainties'].split(' ')[i]] = e_obs_mag[:,i]
    print('Done!')

    # create interpolators and photometric engines
    print('Instantiating photometry interpolators and engines...', end=' ')
    co_model = wdphoto.LaPlataInterpolator(bands, massive_params = ('CO', params['hlayer']))
    one_model = wdphoto.LaPlataInterpolator(bands, massive_params = ('ONe', params['hlayer']))
    CO_Engine = wdphoto.PhotometryEngine(co_model)
    ONe_Engine = wdphoto.PhotometryEngine(one_model)
    print('Done!')

    print('Measuring radii')
    outs = np.nan*np.zeros((2, len(table), 7))
    for i in tqdm(range(len(obs_mag))):
        outs[0,i] = CO_Engine(obs_mag[i], e_obs_mag[i], distances[i], p0 = [10000, 9, 0.003])#, p0=[catalog['teff'][i], catalog['logg'][i], 0.003])
        outs[1,i] = ONe_Engine(obs_mag[i], e_obs_mag[i], distances[i], p0 = [10000, 9, 0.003])

    table['CO_radius'] = outs[0,:,0]
    table['CO_e_radius'] = outs[0,:,1]
    table['CO_teff'] = outs[0,:,2]
    table['CO_e_teff'] = outs[0,:,3]
    table['CO_chi2'] = outs[0,:,-1]
    table['CO_roe'] = table['CO_radius'] / table['CO_e_radius']

    table['ONe_radius'] = outs[1,:,0]
    table['ONe_e_radius'] = outs[1,:,1]
    table['ONe_teff'] = outs[1,:,2]
    table['ONe_e_teff'] = outs[1,:,3]
    table['ONe_chi2'] = outs[1,:,-1]
    table['ONe_roe'] = table['ONe_radius']/table['ONe_e_radius']  

    table['CO_failed'] = np.any([table['CO_chi2'] > 5, table['CO_roe'] < 5], axis=0)
    table['ONe_failed'] = np.any([table['ONe_chi2'] > 5, table['ONe_roe'] < 5], axis=0)
    total_failed = np.any([table['CO_failed'], table['ONe_failed']], axis=0)

    print('\nFit Report:')
    print(f'CO failed={sum(table['CO_failed'])/len(table)*100:2.2f}%')
    print(f'ONe failed={sum(table['ONe_failed'])/len(table)*100:2.2f}%')
    print(f'Total failed={total_failed.sum() / len(table)*100:2.2f}%\n')    

    try:
        table.write(params['savepath'], overwrite=True)
    except:
        print('Did not write table to file')

    return table


if __name__ == "__main__":
    # read in the config parser
    config = configparser.ConfigParser()
    config.read('config.ini')
    params = config['radius']
    
    # read in the arguments from the cli
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-path", default=None)
    parser.add_argument('--deredden', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    # read in the specified table and operate on it
    catalog = Table.read(args.catalog_path)
    measure_radius(catalog, params, args)

