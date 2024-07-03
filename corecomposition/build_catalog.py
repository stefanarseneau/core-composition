import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import configparser
import argparse

import sys
sys.path.append('/mnt/d/arsen/research/proj/')
from astropy.table import Table, vstack, join
from astroquery.gaia import Gaia
import WD_models

plt.style.use('/mnt/d/arsen/research/proj/core-composition/stefan.mplstyle')

def wd_separator(bp_rp):
    # the line that El-Badry use to define WDs
    return 3.25*bp_rp + 9.625

# split the catalog with standard prefixes
def build_catalog(params, save_path):
    # read in Kareem El-Badry's catalog
    print('Reading El-Badry+21 Chance Alignment Catalog...')
    catalog = Table.read('https://zenodo.org/records/4435257/files/all_columns_catalog.fits.gz')
    catalog = catalog[catalog['binary_type'] == b'WDMS'] # select only the WD+MS wide binaries
    catalog = catalog[catalog['R_chance_align'] < float(params['R_chance_align'])] # filter out the chance alignments
    print('no. WD+MS in El-Badry+2021:', len(catalog))

    # compute the absolute magnitudes for the first target
    M1 = catalog['phot_g_mean_mag1'] + 5 * (np.log10(catalog['parallax1'] / 100))

    # save the wd object and the ms object
    wd_obj = ['1' if M1[i] > wd_separator(catalog['bp_rp1'][i]) else '2' for i in range(len(M1))]
    ms_obj = ['1' if wd == '2' else '2' for wd in wd_obj]

    # identify all the columns to be saved
    keys = [key[:-1] for key in catalog.keys() if key.endswith('1')] # select the keys with star-specific data
    consts = [key for key in catalog.keys() if ((not key.endswith('1')) and (not key.endswith('2')))] # select El-Badry+21's binary columns

    # create a new table object to save info into
    table = Table()

    # iterate over all the star-specific keys
    for key in keys: 
        wd_column = [] # create temporary holder for wd data
        ms_column = [] # create temporary holder for ms data
        for j in range(len(catalog)):
            # append each row's wd data into wd_column and ms data into ms_column
            wd_column.append(catalog[key + wd_obj[j]][j]) 
            ms_column.append(catalog[key + ms_obj[j]][j])
        # save this as a new column and move on to the next one
        table['wd_' + key] = wd_column
        table['ms_' + key] = ms_column

    # next, add all the non-row specific data into the table
    for key in consts:
        table[key] = catalog[key]

    # if a save path is specified, save the file
    if save_path is not None:
        table.write(save_path, overwrite=True)
    return table

def make_physical_photometry(catalog):
    # convert gaia flux into physical units (see documentation)
    catalog['wd_phot_g_mean_flux'] = catalog['wd_phot_g_mean_flux'] * 1.736011e-33 * 2.99792458e+21 / 6217.9**2
    catalog['wd_phot_bp_mean_flux'] = catalog['wd_phot_bp_mean_flux'] * 2.620707e-33 * 2.99792458e+21 / 5109.7**2
    catalog['wd_phot_rp_mean_flux'] = catalog['wd_phot_rp_mean_flux'] * 3.2988153e-33 * 2.99792458e+21 / 7769.1**2

    # then convert the errors
    catalog['wd_phot_g_mean_flux_error'] = catalog['wd_phot_g_mean_flux_error'] * 1.736011e-33 * 2.99792458e+21 / 6217.9**2
    catalog['wd_phot_bp_mean_flux_error'] = catalog['wd_phot_bp_mean_flux_error'] * 2.620707e-33 * 2.99792458e+21 / 5109.7**2
    catalog['wd_phot_rp_mean_flux_error'] = catalog['wd_phot_rp_mean_flux_error'] * 3.2988153e-33 * 2.99792458e+21 / 7769.1**2

    # finally convert the magnitude uncertainties
    catalog['wd_e_gmag'] = catalog['wd_phot_g_mean_flux_error'] / (1.09 * catalog['wd_phot_g_mean_flux'])
    catalog['wd_e_bpmag'] = catalog['wd_phot_bp_mean_flux_error'] / (1.09 * catalog['wd_phot_bp_mean_flux'])
    catalog['wd_e_rpmag'] = catalog['wd_phot_rp_mean_flux_error'] / (1.09 * catalog['wd_phot_rp_mean_flux'])
    return catalog

def get_bailerjones(catalog):
    stardats = []
    iters = (len(catalog)+2000) // 2000
    for i in tqdm(range(iters)):
            ADQL_CODE1 = """SELECT dist.source_id, dist.r_med_geo
            FROM gaiadr3.gaia_source as gaia
            JOIN external.gaiaedr3_distance as dist
            ON gaia.source_id = dist.source_id      
            WHERE gaia.source_id in {}""".format(tuple(catalog['ms_source_id'][2000*i:2000*i+2000]))
            stardats.append(Gaia.launch_job(ADQL_CODE1,dump_to_file=False).get_results())
    gaia_d1 = vstack(stardats)
    gaia_d1.rename_column('source_id', 'ms_source_id')
    catalog = join(catalog, gaia_d1, keys = 'ms_source_id')
    return catalog

def get_msrv(catalog, params):
    stardats = []
    iters = (len(catalog)+2000) // 2000
    for i in tqdm(range(iters)):
            ADQL_CODE1 = """SELECT source_id, radial_velocity, radial_velocity_error
            FROM gaiadr3.gaia_source
            WHERE radial_velocity_error < {}
            AND source_id in {}""".format(params['max_erv'], tuple(catalog['ms_source_id'][2000*i:2000*i+2000]))
            stardats.append(Gaia.launch_job(ADQL_CODE1,dump_to_file=False).get_results())
    gaia_d1 = vstack(stardats)
    gaia_d1.rename_columns(['SOURCE_ID', 'radial_velocity', 'radial_velocity_error'], ['ms_source_id', 'ms_rv', 'ms_erv'])
    catalog = join(catalog, gaia_d1, keys = 'ms_source_id')
    return catalog
            
def initial_mass_threshold(catalog, params):
    # define units that will be useful later
    radius_sun = 6.957e8
    mass_sun = 1.9884e30
    newton_G = 6.674e-11

    # create the ONe model from WD_models and interpolate from bp-rp and G
    model = WD_models.load_model('ft', 'ft', 'ft', atm_type = 'H', HR_bands = ['bp3-rp3', 'G3'])
    
    # interpolate the column to WD_mass using the ONe model
    catalog['wd_mass'] = model['HR_to_mass'](catalog['wd_bp_rp'], catalog['wd_m_g'])
    catalog = catalog[~np.isnan(catalog['wd_mass'])] # get rid of anything that can't interpolate

    # return all the targets with interpolated masses above the cutoff
    return catalog, catalog[catalog['wd_mass'] > float(params['cutoff_mass'])]

def build_catalog(params, args):
    # either build or read in the catalog
    if args.path == None:
        catalog = build_catalog(params, args.save_path)
    else:
        catalog = Table.read(args.path)

    # query bailer-jones distances
    catalog = get_bailerjones(catalog)
    catalog = get_msrv(catalog, params)

    # compute useful absolute magnitude columns
    catalog['ms_m_g'] = catalog['ms_phot_g_mean_mag'] + 5 * (np.log10(catalog['ms_parallax'] / 100))
    catalog['wd_m_g'] = catalog['wd_phot_g_mean_mag'] + 5 * (np.log10(catalog['wd_parallax'] / 100))
    print(f'Found {len(catalog):d} WD+MS Wide Binaries')

    catalog = make_physical_photometry(catalog)
    catalog, highmass = initial_mass_threshold(catalog, params)

    if args.verbose:
        # print the color-magnitude diagram
        plt.figure(figsize=(10,5))
        plt.scatter(catalog['ms_bp_rp'], catalog['ms_m_g'], label='Main Sequence', alpha = 0.5, s=5)
        plt.scatter(catalog['wd_bp_rp'], catalog['wd_m_g'], label='White Dwarf', alpha = 0.5, s=5)
        plt.ylabel(r'$M_G$')
        plt.xlabel(r'bp-rp')
        plt.title(r'CMD')
        plt.gca().invert_yaxis()
        plt.legend(framealpha = 0)
        plt.show()

        # print the histogram of the interpolated masses
        f = plt.figure(figsize = (10, 3))
        N, bins, patches = plt.hist(catalog['wd_mass'], bins = 50)
        for ii, val in enumerate(bins):
            if val >= float(params['cutoff_mass']):
                point = ii-1
                break
        for p in range(len(patches)):
            if p >= point:
                patches[p].set_facecolor('green')
        plt.axvline(x = float(params['cutoff_mass']), c='k', ls = '--', alpha = 0.5)
        plt.xlabel('Interpolated WD Mass')
        plt.show()

        # print the CMD of the catalog
        plt.figure(figsize=(10,5))
        plt.scatter(catalog['wd_bp_rp'], catalog['wd_m_g'], label='White Dwarf', alpha = 0.5, s=5, c='k')
        plt.scatter(highmass['wd_bp_rp'], highmass['wd_m_g'], label='Massive White Dwarf', alpha = 0.5, s=10, c='red')
        plt.ylabel(r'$M_G$')
        plt.xlabel(r'bp-rp')
        plt.title(r'CMD')
        plt.gca().invert_yaxis()
        plt.legend(framealpha = 0)
        plt.show()

    print(f'Found {len(highmass)} High-Mass WD+MS Binaries')

    if args.highmass_path is not None:
        highmass.write(args.highmass_path, overwrite=True)
    
    return catalog, highmass

    