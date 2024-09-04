from __future__ import print_function
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

import re
import pickle
import glob
import os

from scipy.interpolate import RegularGridInterpolator
from astropy.table import Table

import pyphot
import extinction

def plot(obs_mag, e_obs_mag, distance, radius, teff, logg, Engine):
    model_flux = 4 * np.pi * Engine.interpolator(teff, logg)
    #convert to SI units
    radius = radius * 6.957e8 # Rsun to meter
    distance = distance * 3.086775e16 # Parsec to meter
    model_flux = (radius / distance)**2 * model_flux

    lib = pyphot.get_library()
    model_wavl = [lib[band].lpivot.to('angstrom').value for band in Engine.bands]

    obs_flux, e_obs_flux = Engine.mag_to_flux(obs_mag,  e_obs_mag)

    f = plt.figure(figsize = (8,7))
    plt.scatter(model_wavl, model_flux, c = 'blue', label='Model Photometry')
    plt.errorbar(model_wavl, obs_flux, yerr=e_obs_flux, linestyle = 'none', marker = 'None', capsize = 5, color = 'k', label=f'Teff={teff:6.0f}\nlogg={logg:1.1f}')
    plt.xlim(2500,15000)
    plt.xlabel(r'Wavelength $[\AA]$')
    plt.ylabel(r'Flux $[erg/s/cm^2/\AA]$')

    return f


def deredden(bsq, coords, photo, bands):
    # perform the query
    bsq_res = bsq.query(coords).copy()
    if np.isnan(bsq_res):
        bsq_res = 0

    # Convert to actual units
    Ebv = bsq_res*0.901*0.98
    e_Ebv = Ebv*0.2

    # Parameters for correcting using Gaia
    Rv = 3.1
    A_v0 = Ebv*Rv

    # Fetch Gaia photometric band wavelengths and store in `gaia_phot_wavl`
    lib = pyphot.get_library()
    phot = [lib[band] for band in bands]
    phot_wavl = np.array([x.lpivot.to('angstrom').value for x in phot])

    # For each point, find extinction using the parameters we defined above
    ext_all = extinction.fitzpatrick99(phot_wavl, A_v0, Rv)
    photo_corrected = np.array(photo) - ext_all
    return photo_corrected

def correct_gband(bp, rp, astrometric_params_solved, phot_g_mean_mag):
    """
    Correct the G-band fluxes and magnitudes for the input list of Gaia EDR3 data.
    
    Parameters
    ----------
    
    bp_rp: float, numpy.ndarray
        The (BP-RP) colour listed in the Gaia EDR3 archive.
    astrometric_params_solved: int, numpy.ndarray
        The astrometric solution type listed in the Gaia EDR3 archive.
    phot_g_mean_mag: float, numpy.ndarray
        The G-band magnitude as listed in the Gaia EDR3 archive.
        
    Returns
    -------
    
    The corrected G-band magnitudes and fluxes. The corrections are only applied to
    sources with a 2-paramater or 6-parameter astrometric solution fainter than G=13, 
    for which a (BP-RP) colour is available.
    
    Example
    -------
    
    gmag_corr = correct_gband(bp_rp, astrometric_params_solved, phot_g_mean_mag)
    """
    bp_rp = bp - rp

    if np.isscalar(bp_rp) or np.isscalar(astrometric_params_solved) or np.isscalar(phot_g_mean_mag):
        bp_rp = np.float64(bp_rp)
        astrometric_params_solved = np.int64(astrometric_params_solved)
        phot_g_mean_mag = np.float64(phot_g_mean_mag)
    
    if not (bp_rp.shape == astrometric_params_solved.shape == phot_g_mean_mag.shape):
        raise ValueError('Function parameters must be of the same shape!')
    
    do_not_correct = np.isnan(bp_rp) | (phot_g_mean_mag<13) | (astrometric_params_solved == 31)
    bright_correct = np.logical_not(do_not_correct) & (phot_g_mean_mag>=13) & (phot_g_mean_mag<=16)
    faint_correct = np.logical_not(do_not_correct) & (phot_g_mean_mag>16)
    bp_rp_c = np.clip(bp_rp, 0.25, 3.0)
    
    correction_factor = np.ones_like(phot_g_mean_mag)
    correction_factor[faint_correct] = 1.00525 - 0.02323*bp_rp_c[faint_correct] + \
        0.01740*np.power(bp_rp_c[faint_correct],2) - 0.00253*np.power(bp_rp_c[faint_correct],3)
    correction_factor[bright_correct] = 1.00876 - 0.02540*bp_rp_c[bright_correct] + \
        0.01747*np.power(bp_rp_c[bright_correct],2) - 0.00277*np.power(bp_rp_c[bright_correct],3)
    
    gmag_corrected = phot_g_mean_mag - 2.5*np.log10(correction_factor)
    
    return gmag_corrected

# convert air wavelengths to vacuum
def air2vac(wv):
    _tl=1.e4/np.array(wv)
    return (np.array(wv)*(1.+6.4328e-5+2.94981e-2/\
                          (146.-_tl**2)+2.5540e-4/(41.-_tl**2)))

def refactor_array(fl):
    # re-format an array of strings into floats
    for jj, num in enumerate(fl):
        if 'E' not in num: # if there is no exponential
            if '+' in num:
                num = num.split('-')
                num = 'E'.join(num)
            elif ('-' in num) and (num[0] != '-'):
                num = num.split('-')
                num = 'E-'.join(num)
            elif ('-' in num) and (num[0] == '-'):
                num = num.split('-')
                num = 'E-'.join(num)
                num = num[1:]
            fl[jj] = num

    try:
        fl = np.array([float(val) for val in fl])
    except:
        fl = fl[1:]
        fl = np.array([float(val) for val in fl])
    return fl


# create the Warwick DA interpolator
def build_warwick_da(path = '/data/warwick_da', outpath = None, flux_unit = 'flam'):
    dirpath = os.path.dirname(os.path.realpath(__file__)) # identify the current directory
    files = glob.glob(dirpath + path + '/*') # get a list of the files corresponding to the Warwick DA spectra

    # read in the first file
    with open(files[0]) as f:
        lines = f.read().splitlines()
        
    for ii in range(len(lines)):        
        if 'Effective temperature' in lines[ii]: # iterate over lines until the first line with actual data is found
            # create an array of all the data up until that line to create the base wavelength grid
            base_wavl = np.array(re.split('\s+', ''.join(lines[1:ii])))[1:].astype(float)
            break

    # instantiate holder objects
    dat = []        
                
    for file in files:
        # read each file in
        with open(file) as f:
            lines = f.read().splitlines()
                
        prev_ii = 0 # at what index do the file's wavelengths end?
        first = True # is this the first pass of that file?
        for ii in range(len(lines)):        
            if 'Effective temperature' in lines[ii]:
                if first: # if not done already, generate an array of the file's wavelength coverage
                    wavl = np.array(re.split('\s+', ''.join(lines[1:ii])))[1:].astype(float)

                    # update variables
                    first = False
                    prev_ii = ii

                    # store the temperature and logg of this part of the file
                    teff = float(re.split('\s+', lines[prev_ii])[4])
                    logg = np.log10(float(re.split('\s+', lines[prev_ii])[7]))

                else: # if this is not the first time,
                    # store the temperature and logg of this part of the file
                    teff = float(re.split('\s+', lines[prev_ii])[4])
                    logg = np.log10(float(re.split('\s+', lines[prev_ii])[7]))

                    # create a list of all the string fluxes from the last break to this one
                    fl = re.split('\s+', ''.join(lines[prev_ii+1:ii]))
                    fl = refactor_array(fl) # refactor the array from strings to floats
                    fl = np.interp(base_wavl, wavl, fl) # interpolate flux onto a consistent wavelength grid
                        
                    dat.append([logg, teff, fl, base_wavl]) # append to a holder for later use                                             
                    prev_ii = ii # update to the new end of the flux data
                
    table = Table() # create a table to hold this in 
    table['logg'] = np.array(dat, dtype=object).T[0]
    table['teff'] = np.array(dat, dtype=object).T[1] 
    table['fl'] = np.array(dat, dtype=object).T[2]   
    table['wl'] = np.array(dat, dtype=object).T[3] # convert air wavelengths to vacuum
    
    if flux_unit == 'flam': # convert to flam if desired
        table['fl'] = (2.99792458e18*table['fl'] / table['wl']**2)
    
    # create a sorted list of teffs and loggs for use in the array
    teffs = sorted(list(set(table['teff'])))
    loggs = sorted(list(set(table['logg'])))
    
    # instantiate the interpolation object
    values = np.zeros((len(teffs), len(loggs), len(base_wavl)))
    
    for i in range(len(teffs)):
        for j in range(len(loggs)):
            try:
                # append the value of flux corresponding to the current (teff, logg)
                values[i,j] = table[np.all([table['teff'] == teffs[i], table['logg'] == loggs[j]], axis = 0)]['fl'][0]
            except:
                # if that isn't included, append zeros
                values[i,j] = np.zeros(len(base_wavl))
    
    #NICOLE BUG FIX
    high_logg_grid=values[:,4:]
    high_loggs=loggs[4:]

    low_logg_grid=values[16:33,:]
    low_loggs_teffs=teffs[16:33]

    model_spec = RegularGridInterpolator((teffs, high_loggs), high_logg_grid)
    model_spec_low_logg = RegularGridInterpolator((low_loggs_teffs, loggs), low_logg_grid)
    
    if outpath is not None:
        # open a file, where you ant to store the data
        interp_file = open(outpath + '/warwick_da.pkl', 'wb')
        
        # dump information to that file
        pickle.dump(model_spec, interp_file)
        np.save(outpath + '/warwick_da_wavl', base_wavl)
        
    return base_wavl, model_spec, model_spec_low_logg, table