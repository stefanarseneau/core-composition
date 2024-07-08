import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from astropy.table import Table, join
import configparser
import argparse

import sys
sys.path.append('../')
import WD_models
import spectromancer as sp
import corv

from wdphoto.utils import plot
from .build_catalog import build_catalog
from .radius import measure_radius


radius_sun = 6.957e8
mass_sun = 1.9884e30
newton_G = 6.674e-11
pc_to_m = 3.086775e16
speed_light = 299792458 #m/s

def one_model(radarray, teffarray, lowmass = 'f', midmass = 'f', highmass = 'f'):
    ONe_model = WD_models.load_model('ft', 'ft', 'o', atm_type = 'H', HR_bands = ['bp3-rp3', 'G3'])
    g_acc = (10**ONe_model['logg'])/100
    rsun = np.sqrt(ONe_model['mass_array'] * mass_sun * newton_G / g_acc) / radius_sun
    
    rsun_teff_to_m = WD_models.interp_xy_z_func(x = rsun, y = 10**ONe_model['logteff'],\
                                                z = ONe_model['mass_array'], interp_type = 'linear')
    
    mass = rsun_teff_to_m(radarray, teffarray) * mass_sun
    radius = radarray * radius_sun
    rv = newton_G * mass / (speed_light * radius)

    return rv*1e-3

def co_model(radarray, teffarray, lowmass = 'f', midmass = 'f', highmass = 'f'):
    CO_model = WD_models.load_model('ft', 'ft', 'ft', atm_type = 'H', HR_bands = ['bp3-rp3', 'G3'])
    g_acc = (10**CO_model['logg'])/100
    rsun = np.sqrt(CO_model['mass_array'] * mass_sun * newton_G / g_acc) / radius_sun
    
    rsun_teff_to_m = WD_models.interp_xy_z_func(x = rsun, y = 10**CO_model['logteff'],\
                                                z = CO_model['mass_array'], interp_type = 'linear')
    
    mass = rsun_teff_to_m(radarray, teffarray) * mass_sun
    radius = radarray * radius_sun
    rv = newton_G * mass / (speed_light * radius)

    return rv*1e-3

# read in the config parser
config = configparser.ConfigParser()
config.read('config.ini')

# read in the arguments from the cli
parser = argparse.ArgumentParser()
parser.add_argument("path", nargs='?', default=None)
parser.add_argument('config', nargs='?', default='config.ini')
parser.add_argument("catfile", nargs='?', default=None)
parser.add_argument('outfile', nargs='?', default=None)

parser.add_argument('--highmass-path', nargs='?', default=None)
parser.add_argument('--radius-path', nargs='?', default=None)
parser.add_argument('--rv-path', nargs='?', default=None)

parser.add_argument('--deredden', action='store_true', default=False)
parser.add_argument('--plot-radii', action='store_true', default=False)
parser.add_argument('-v', '--verbose', action='store_true')
args = parser.parse_args()

# either build or read in the catalog
print('Building Catalog\n==============================')
catalog_params = config['catalog'] 
catalog, highmass = build_catalog(catalog_params, args)

# measure radii
print('\nMeasuring Radii\n==============================')
radius_params = config['radius']
radii, engine_keys = measure_radius(highmass, radius_params, args)

targets = join(highmass, radii, keys_left='wd_source_id', keys_right='source_id')
# sort the brightest stars first
targets.sort(['wd_phot_g_mean_mag'])
targets.write(args.catfile, overwrite=True)

if args.outfile is not None:
    # measure rvs
    print('\nMeasuring WD RVs\n==============================')
    model = corv.models.make_warwick_da_model(names=['a','b','g','d'])
    observation = sp.Observation('observations/')
    observation.fit_rvs(model, save_column=True, verbose=args.verbose)

    #mask = np.all([observation.table['rv_spread'] < 1], axis=0)
    rv_table = observation.table#[mask]

    print(f'Measured {len(rv_table)} WD RVs')

    if args.rv_path is not None:
        rv_table.write(args.rv_path, overwrite=True)

    # join files
    outfile = join(targets, rv_table, keys='source_id')

    outfile['gravz'] = outfile['rv'] - outfile['ms_rv']
    outfile['e_gravz'] = (outfile['e_rv'] + outfile['ms_erv'])

    print(f'Joined {len(outfile)} WD+MS Targets.')
    outfile.write(args.outfile, overwrite=True)

    if args.verbose:
        rad_array = np.linspace(0.0045, 0.007, 100)
        vg_array_one = one_model(rad_array, 16278)
        vg_array_co = co_model(rad_array, 16278)

        plt.style.use('./stefan.mplstyle')

        plt.figure(figsize = (10,10))
        plt.plot(rad_array, vg_array_one, label='O/Ne Core', color = 'k')
        plt.plot(rad_array, vg_array_co, label='C/O Core', color = 'red')

        # Datapoints
        colors = ['blue', 'orange', 'green', 'red', 'black']
        for i, key in enumerate(engine_keys):
            #if not np.any([outfile[f'{key}_failed']], axis=0):
            plt.errorbar(outfile[f'{key}_radius'], outfile['gravz'], 
                    xerr = outfile[f'{key}_e_radius'], yerr = outfile['e_gravz'], 
                    fmt='o', label = f'{key}', color=colors[i], ecolor = 'black')

        plt.xlabel(r'Radius $[R_\odot]$')
        plt.ylabel(r'$v_g$ $[km/s]$')

        plt.legend(framealpha=0)
        plt.show()