import numpy as np
from Payne import utils as payne_utils
from astropy.table import Table
import corv
from tqdm import tqdm

base_wavl, model_spec, model_spec_low_logg, table = corv.utils.build_warwick_da()
build_spec = lambda best_est, distance : 4*np.pi*model_spec((best_est[0], 9)) * ((best_est[1] * 6.957e8) / (distance * 3.086775e16))**2

def doppler_shift(wavelength, flux, radial_velocity):
    # Speed of light in km/s
    c = 299792.458
    # Calculate the Doppler shift factor
    doppler_factor = np.sqrt((1 + radial_velocity / c) / (1 - radial_velocity / c))
    # Apply the Doppler shift to the wavelength
    shifted_wavelength = wavelength * doppler_factor
    return shifted_wavelength, flux

def simulate_spec(wl, params, distance, rv, snr, R = 3000):
    highres_flux = build_spec(params, distance)
    highres_flux = np.interp(wl, base_wavl, highres_flux)
    # smooth down to the correct resolution
    smoothed_flux = payne_utils.smoothing.smoothspec(wl, highres_flux, resolution=R, smoothtype="R")
    smoothed_flux[smoothed_flux < 0] = 0
    # calculate and add noise into the measurement
    noise = smoothed_flux / snr
    smoothed_flux += np.random.normal(loc=0, scale = noise)
    # add a doppler shift 
    wl, flux = doppler_shift(wl, smoothed_flux, rv)
    return wl, flux, 1/(noise**2)

def fetch_distributions():
    # Tremblay+2019
    dist_catalog = Table.read('https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/fits?J/MNRAS/482/5222/tablea1.dat')
    # Raddi+2022
    wd_reference = Table.read('https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/fits?J/A+A/658/A22/table5.dat.gz')

    teff_counts, teff_bins = np.histogram(10**wd_reference['logTeff'], bins = 250)
    teff_counts = teff_counts / sum(teff_counts)

    dist_counts, dist_bins = np.histogram(100 / dist_catalog['plx'], bins = 250)
    dist_counts = dist_counts / sum(dist_counts)

    radius_counts, radius_bins = np.histogram(wd_reference['Radius'], bins = 100)
    radius_counts = radius_counts / sum(radius_counts)

    return (teff_counts, teff_bins), (dist_counts, dist_bins), (radius_counts, radius_bins)

def simulate(wl, n_sims, snr_grid, R, teff, distance):
    snrs = []
    teffs = []
    dists = []
    radii = []
    rvs = []
    measured_rvs = []
    e_rvs = []

    for ii, snr in enumerate(tqdm(snr_grid)):
        for jj in range(n_sims):
            radius = np.random.uniform(low = 0.0045, high = 0.0060)
            rv = np.random.uniform(low = -50, high = 50)

            wl_fetched, fl, ivar = simulate_spec(wl, (teff, radius), distance, rv, snr, R = R)
            corvmodel = corv.models.make_balmer_model(nvoigt = 2, names = ['a','b'])
            measured_rv, e_rv, redchi, param_res = corv.fit.fit_corv(wl_fetched, fl, ivar, corvmodel)
            #corv.utils.lineplot(wl_fetched, fl, ivar, corvmodel, param_res.params, printparams = False, gap = 0.3, figsize = (6, 5))

            snrs.append(snr)
            teffs.append(teff)
            dists.append(distance)
            radii.append(radius)
            rvs.append(rv)
            measured_rvs.append(measured_rv)
            e_rvs.append(e_rv)

    parameters = Table()
    parameters['snr'] = snrs
    parameters['teff'] = teffs
    parameters['distance'] = dists
    parameters['radius'] = radii
    parameters['rv'] = rvs
    parameters['measured_rv'] = measured_rvs
    parameters['measured_e_rv'] = e_rvs

    return parameters