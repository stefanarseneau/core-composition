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
from .elbadry import build_catalog
from .radius import measure_radius


radius_sun = 6.957e8
mass_sun = 1.9884e30
newton_G = 6.674e-11
pc_to_m = 3.086775e16
speed_light = 299792458 #m/s

def build(config, catalog):
    catalog_params = config['catalog'] 
    radius_params = config['radius']

    # either build or read in the catalog
    print('Building Catalog\n==============================')
    catalog, highmass = build_catalog(catalog_params, catalog)

    # measure radii
    print('\nMeasuring Radii\n==============================')
    radii, engine_keys = measure_radius(highmass, radius_params, False, False)

    targets = join(catalog, radii, keys_left='wd_source_id', keys_right='source_id')
    # sort the brightest stars first
    targets.sort(['wd_phot_g_mean_mag'])
    return catalog, targets, engine_keys

def analyze(targets, config, obsfile):
    # measure rvs
    print('Measuring WD RVs\n==============================')
    model = corv.models.make_warwick_da_model(names=['a','b','g','d'])
    observation = sp.Observation(obsfile)
    observation.fit_rvs(model, save_column=True)
    mask = np.all([observation.table['rv_spread'] < 1], axis=0)
    rv_table = observation.table[mask]

    print(f'Measured {len(rv_table)} WD RVs')

    # join files
    outfile = join(targets, rv_table, keys='source_id')
    outfile['gravz'] = outfile['rv'] - outfile['ms_rv']
    outfile['e_gravz'] = (outfile['e_rv'] + outfile['ms_erv'])

    print(f'Joined {len(outfile)} WD+MS Targets.')    
    return outfile, rv_table
        