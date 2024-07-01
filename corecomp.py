import matplotlib.pyplot as plt
import numpy as np
import configparser
import argparse

from astropy.table import Table
import corecomposition as cc

if __name__ == "__main__":
    # read in the config parser
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    # read in the arguments from the cli
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-path", nargs='?', default=None)
    parser.add_argument('--deredden', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    # either build or read in the catalog
    print('Building Catalog\n==============================')
    catalog_params = config['catalog'] 
    catalog, highmass = cc.build_catalog(catalog_params, args)

    # measure radii
    print('\nMeasuring Radii\n==============================')
    radius_params = config['radius']
    radii = cc.measure_radius(highmass, radius_params, args)