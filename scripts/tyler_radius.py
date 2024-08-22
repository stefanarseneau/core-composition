# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:50:23 2021

@author: Tyler
"""

import numpy
import sys
from astropy.table import Table, vstack, Column
from matplotlib import pyplot
from scipy.interpolate import interp1d, interp2d, griddata, LinearNDInterpolator
import matplotlib
import glob
from astropy.io.votable import parse
from scipy.integrate import simps
from scipy.optimize import curve_fit, differential_evolution
import emcee
import corner
import time
import warnings
# ===================================================================================
def correct_gband(bp_rp, astrometric_params_solved, phot_g_mean_mag, phot_g_mean_flux):
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
    phot_g_mean_flux: float, numpy.ndarray
        The G-band flux as listed in the Gaia EDR3 archive.
        
    Returns
    -------
    
    The corrected G-band magnitudes and fluxes. The corrections are only applied to
    sources with a 2-paramater or 6-parameter astrometric solution fainter than G=13, 
    for which a (BP-RP) colour is available.
    
    Example
    -------
    
    gmag_corr, gflux_corr = correct_gband(bp_rp, astrometric_params_solved, phot_g_mean_mag, phot_g_mean_flux)
    """

    if numpy.isscalar(bp_rp) or numpy.isscalar(astrometric_params_solved) or numpy.isscalar(phot_g_mean_mag) \
                    or numpy.isscalar(phot_g_mean_flux):
        bp_rp = numpy.float64(bp_rp)
        astrometric_params_solved = numpy.int64(astrometric_params_solved)
        phot_g_mean_mag = numpy.float64(phot_g_mean_mag)
        phot_g_mean_flux = numpy.float64(phot_g_mean_flux)
    
    if not (bp_rp.shape == astrometric_params_solved.shape == phot_g_mean_mag.shape == phot_g_mean_flux.shape):
        raise ValueError('Function parameters must be of the same shape!')
    
    do_not_correct = numpy.isnan(bp_rp) | (phot_g_mean_mag<13) | (astrometric_params_solved == 31)
    bright_correct = numpy.logical_not(do_not_correct) & (phot_g_mean_mag>=13) & (phot_g_mean_mag<=16)
    faint_correct = numpy.logical_not(do_not_correct) & (phot_g_mean_mag>16)
    bp_rp_c = numpy.clip(bp_rp, 0.25, 3.0)
    
    correction_factor = numpy.ones_like(phot_g_mean_mag)
    correction_factor[faint_correct] = 1.00525 - 0.02323*bp_rp_c[faint_correct] + \
        0.01740*numpy.power(bp_rp_c[faint_correct],2) - 0.00253*numpy.power(bp_rp_c[faint_correct],3)
    correction_factor[bright_correct] = 1.00876 - 0.02540*bp_rp_c[bright_correct] + \
        0.01747*numpy.power(bp_rp_c[bright_correct],2) - 0.00277*numpy.power(bp_rp_c[bright_correct],3)
    
    gmag_corrected = phot_g_mean_mag - 2.5*numpy.log10(correction_factor)
    gflux_corrected = phot_g_mean_flux * correction_factor
    
    return gmag_corrected, gflux_corrected
# ===================================================================================
def convolve_filters_models(models, filters, filter_freq, names, zeropoints, bs):
    new_models = []
    for i in range(len(models)):
        new_model = []
        model = models[i]
        for j in range(len(filters)):
            if names[j] == 'Gaia':
                A_Gaia = 0.7278
                ## in photons per s per A per m squared ##
                f = interp1d(model['col1'],model['col2']*1e-7*10000*model['col1']/1e10/(6.626e-34*3e8))
                filt_w = filters[j]['wavelength']
                trans = filters[j]['t']
                flux = f(filt_w)
                tot_flux = simps(trans*flux, filt_w) # in photons per s per m2
                tot_flux = tot_flux*A_Gaia*numpy.power(10, -0.4*(zeropoints[j]+56.10)) ## in Watts per m squared per Hz
                # print(tot_flux)
                new_model.append(tot_flux)

            ## Need to make sure it is ok to do this!!!! ##
            else:#if names[j] == 'SDSS' or names[j] == 'pstarr' or names[j] == 'SkyMap':
                mod_flux = model['col2'] ## erg per sec per cm^2 per Ang
                mod_wav = model['col1'] ## Ang
                mod_flux = mod_flux*3.34e4*numpy.square(mod_wav)  ## Jy
                mod_freq = 3e8/(mod_wav*1e-10)
                f = interp1d(mod_freq, mod_flux)
                filt_w = filters[j]['wavelength']
                trans = filters[j]['t']
                filt_freq = 3e8/(filt_w*1e-10)
                flux = f(filt_freq)
                tot_flux = simps(trans*flux, filt_freq)
                f0_array = 3631
                f0 = simps(trans*f0_array, filt_freq)
                mag_AB = -2.5*numpy.log10(tot_flux/f0)
                tot_flux =  3631e-26*numpy.power(10, mag_AB/-2.5) # W/m2/Hz
                # tot_flux = tot_flux/3.34e4/numpy.square(filt_w) # erg per sec per cm^2 per Ang
                # print(tot_flux)
                new_model.append(tot_flux)
                
            # if names[j] == '2MASS':
                
            
            # if names[j] == 'APASS':
                
        # pyplot.figure(figsize = (18,9))
        # pyplot.plot(3e8/filter_freq*1e9, filter_freq*new_model, "k.", markersize = 15)
        # pyplot.xlabel("Wavelength [nm]", fontsize = 24)#r'$\nu$ [Hz]', fontsize = 24)
        # pyplot.ylabel(r'$\mathrm{\lambda F_\lambda\ [W\ m^{-2}]}$', fontsize = 24)
        # pyplot.yscale('log')
        # pyplot.tick_params(which = "both", direction='inout', length=14, labelsize = 20)
        # print(new_model)
        # pyplot.savefig("SED_convolve_test.png")
        # pyplot.show()
        # sys.exit(20)
        new_models.append({"col1" : filter_freq, "col2" : new_model})    
    return new_models
# ===================================================================================
def make_MR_coolfunc(MR):
    ## interpolates Mass Radius Relation and returns the radius in meters
    teffs = numpy.array(MR['Teff'])
    loggs = numpy.array(MR['Log(g)'])
    points = []
    for i in range(len(teffs)):
        # points.append(numpy.array([teffs[i], loggs[i]]))
        points.append(numpy.array([numpy.log10(teffs[i]), loggs[i]]))
    R = numpy.array(MR['R'])/100
    points = numpy.array(points)
    f = LinearNDInterpolator(points, numpy.log10(R), rescale = True)
    g = LinearNDInterpolator(points, numpy.log10(MR['Age']), rescale = True)
    return f, g

# ===================================================================================
def make_flux_interp(teffs, loggs, model_fluxes):
    ## generates interpolator of WD atmosphere models in teff and logg space
    points = []
    model_fluxes = numpy.array(model_fluxes).T
    for i in range(len(teffs)):
        points.append(numpy.array([teffs[i], loggs[i]]))
    fs = []
    for i in range(len(model_fluxes)):
        f = LinearNDInterpolator(points, model_fluxes[i], rescale = True)
        fs.append(f)
        
    return numpy.array(fs)

# ===================================================================================
def plot_Chi2(freq, fluxes, errors, plx, plx_error, models, loggs, teffs, model_fluxes, flux_interp, MR, ra, dec, source_id, mcmc_samples = None, index = 'None'):
    
    ## This function plots the chi2 space for the SED fit. This takes awhile so only ##
    ## run if you need to diagnose issues with different fits. the points are the    ## 
    ## (teff, logg) grid points. They need to be in a special format that can be seen## 
    ## in main() part of code. MR is an array of mass and radius values generated    ##
    ## from a mass-radius relation.
    pyplot.clf()
    def interp_models(freqs, teff, logg):
        sol = []
        R = numpy.power(10, MR([numpy.log10(teff), logg]))[0]
        for i in numpy.where((fluxes > 0) & (errors > 0))[0]:
            f = flux_interp[i] 
            sol.append(f([teff, logg])[0]*R**2/(10.0*3.086e16)**2)
            
        return numpy.array(sol)

    uncertainty = numpy.sqrt(numpy.square(errors/fluxes) + 2*numpy.square(plx_error/plx))*fluxes*numpy.square((1000/plx)/10)  
    fluxes_10 = fluxes*numpy.square((1000/plx)/10)
    T = numpy.linspace(5000, 20000, 100)
    L = numpy.linspace(6.5,9.5,100)
    dof = len(fluxes_10)
    chis = []
    TS = []
    LS = []
    i = 0
    for t in T:
        # if i % int(len(T)*0.1) == 0:
        #     print("{} % done\n".format(i/len(T)*100))
        for l in L:
            TS.append(t)
            LS.append(l)
            flux_model = interp_models(freq, t, l)
            log_errors = numpy.abs(uncertainty/(fluxes_10*numpy.log(10)))
            chi2 = numpy.sum(numpy.square((numpy.array(flux_model) - fluxes_10)/uncertainty))
            # chi2 = numpy.sqrt(numpy.sum(numpy.square((numpy.log10(numpy.array(flux_model)) - numpy.log10(fluxes))/log_errors)))#errors)))
            chis.append(chi2/dof)
        i+=1
    
    cm = pyplot.cm.get_cmap('viridis')            
    sc = pyplot.scatter(LS, TS, c = numpy.log10(chis), marker = '.', cmap = cm, vmax = 2.5)
    if mcmc_samples is not None:
        teff_mcmc = mcmc_samples[:, 0]
        logg_mcmc = mcmc_samples[:, 1]
        pyplot.scatter(logg_mcmc, teff_mcmc, c = "r", marker = ".", s = 1)
    sc.cmap.set_under('grey')
    sc.cmap.set_over('grey')
    cb = pyplot.colorbar(sc)
    pyplot.suptitle("WD eDR3 {} RA: {} DEC: {}".format(source_id, ra, dec), fontsize = 20)
    pyplot.savefig("plots\\Chi2_space_{}.png".format(index), bbox_inches = 'tight')
    pyplot.cla()
    # pyplot.show()

    # sys.exit(20)      

# ===================================================================================
def fit_SED_absolute(freq, fluxes, uncertainty, plx, plx_error, loggs, teffs, model_fluxes, flux_interp, MR, ra, dec, source_id, IFMR = None, t_ms = None, initial_guess = None, cooling_func = None, index = None, AV_range = [0,2], R = None):
    ## Calculate best fit SED:                                                           ##
    ##   I think the inputs are clear here but will say flux_interp and MR are functions ##
    ##   generated from a LinearNDInterpolator which uses flux, mass and radius values   ##
    ##   from a WD model.                                                                ##
    
    # Define function used to interpolate the models #
    def interp_models(freqs, teff, logg):
        sol = []
        R = numpy.power(10, MR([numpy.log10(teff), logg]))[0]
        for i in range(len(fluxes)):
            f = flux_interp[i]
            sol.append(f([teff, logg])[0]*R**2/(10.0*3.086e16)**2)
            
        return numpy.array(sol)
     
    # Below here is using a gradient solver #
    # Teff_ave = (numpy.max(teffs) + numpy.min(teffs))/2.0
    # Logg_ave = (numpy.max(loggs) + numpy.min(loggs))/2.0
    # initial_guess = (Teff_ave, Logg_ave)
    # popt, pcov = curve_fit(interp_models, freq, fluxes*numpy.square(1000/(plx*10)), p0 = initial_guess, sigma = uncertainty, absolute_sigma = True, bounds = ([numpy.min(teffs), numpy.min(loggs)], [numpy.max(teffs), numpy.max(loggs)]), maxfev=5000)
    
    # print("Gradient Solver finds {} : {}".format(popt, pcov))
    # sys.exit(20)

    ## LET'S TRY MCMC ##
    # print("Starting MCMC code...\n")
    # Define log likelihood. This comes from a Gaussian #
    def log_likelihood(params):
        flux_model = interp_models(None, params[0], params[1])
        return numpy.sum(-0.5*numpy.square(((fluxes - flux_model)/uncertainty)) - numpy.log(numpy.sqrt(2*numpy.pi)*uncertainty))
    
    # Define priors. I use uniform priors here #
    def log_prior(params):
        teff, logg = params
        if numpy.min(teffs) <= teff <= numpy.max(teffs) and numpy.min(loggs) <= logg <= numpy.max(loggs):# and AV_range[0] <= AV <= AV_range[1]:
            return 0.0
        else:
            return -numpy.inf

    # Define total probability
    def log_prob(params):
        lp = log_prior(params)
        ll = log_likelihood(params)
        if not numpy.isfinite(lp) or not numpy.isfinite(ll):
            return -numpy.inf
        return lp + ll

    nsteps = 2500
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            init_pos = initial_guess + 1e-2*numpy.random.randn(50,2)*initial_guess # intialize postions of walkers
            nwalkers, ndim = init_pos.shape
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)#, moves = [emcee.moves.StretchMove(a=1.75)])
            sampler.run_mcmc(init_pos, nsteps, progress = False) # run x steps of mcmc
        except:
            init_pos = initial_guess + 1e-2*numpy.random.randn(50,2)*initial_guess # intialize postions of walkers
            nwalkers, ndim = init_pos.shape
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)#, moves = [emcee.moves.StretchMove(a=1.75)])
            sampler.run_mcmc(init_pos, nsteps, progress = False) # run x steps of mcmc
        
        
        auto_corr_time = numpy.max(sampler.get_autocorr_time(quiet = True)) # get amount of steps for burn-in so we know how many steps we should run
        # print("Auto-Correlation Time = {}\n".format(auto_corr_time))
        
        if numpy.isfinite(auto_corr_time):
            if nsteps <= int(52*auto_corr_time):
                sampler.run_mcmc(None, int(52*auto_corr_time) - nsteps, progress = False)


            flat_samples = sampler.get_chain(discard = int(3*auto_corr_time), thin = int(0.5*auto_corr_time), flat = True)
        else:
            flat_samples = sampler.get_chain(discard = 0, flat = True)
            
    flat_samples[:, 0] = numpy.log10(flat_samples[:, 0]) # change temperature to log temperature to use MR and cooling func
    cool_age = numpy.power(10, cooling_func(flat_samples[:,:2]))/1e9
    G = 6.6743e-11
    mass = 1/100*numpy.power(10,flat_samples[:, 1])/G*numpy.square(numpy.power(10,MR(flat_samples[:,:2])))/1.98847e30
    # init_mass = IFMR(mass)
    # init_mass[numpy.where(init_mass < 0.8)] = 0.8
    # mslife = t_ms(init_mass)
    # tot_age = mslife + cool_age
    flat_samples[:, 0] = numpy.power(10, flat_samples[:, 0]) # change back to normal temp for plots
    if numpy.max(mass) > 0:
        flat_samples = numpy.append(flat_samples.T, [mass], axis = 0).T
        labels = ["Teff [K]", "log(g) [cm2/sec]", "Mass [Msun]"]
    else:
        labels = ["Teff [K]", "log(g) [cm2/sec]"]
    # print(cool_age)
    if numpy.max(cool_age) > 0:
        flat_samples = numpy.append(flat_samples.T, [cool_age], axis = 0).T
        # flat_samples = numpy.append(flat_samples.T, [init_mass], axis = 0).T
        # flat_samples = numpy.append(flat_samples.T, [mslife], axis = 0).T
        # flat_samples = numpy.append(flat_samples.T, [tot_age], axis = 0).T
        # labels = ["Teff [K]", "log(g) [cm2/sec]", "Mass [Msun]", "t_cool [Gyr]", "Mi [Msun]", "tms [Gyr]", "tot_age [Gyr]"]
        labels = ["Teff [K]", "log(g) [cm2/sec]", "Mass [Msun]", "t_cool [Gyr]"]

    else:
        labels = labels

    best_est = numpy.zeros((len(labels)))
    unc = []
    for i in range(len(labels)):
        mcmc = numpy.percentile(flat_samples[:, i],[16,50,84])
        best_est[i] = mcmc[1]
        unc.append(numpy.diff(mcmc))
    unc = numpy.array(unc)
    # print("Teff = {} +- {} : Logg = {} +- {} : Mass = {} +- {} : Cool Age = {} +- {}".format(best_est[0], unc[0], best_est[1], unc[1], best_est[2], unc[2], best_est[3], unc[3]))
    # print("Acceptance fraction = {}".format(numpy.mean(sampler.acceptance_fraction)))
    # print(flat_samples)
    # print(numpy.min(flat_samples[:,0]), numpy.min(flat_samples[:,1]), numpy.min(flat_samples[:,2]))#, flat_samples[:,3])
    fig = corner.corner(flat_samples, labels = labels, truths = best_est, quantiles = [0.16, 0.5, 0.84], show_titles = True)
    pyplot.suptitle("WD eDR3 {} RA: {} DEC: {}".format(source_id, ra, dec), fontsize = 18, y = 1.05)
    pyplot.savefig("plots\\corner_WDeDR3_{}.pdf".format(source_id), bbox_inches = 'tight')
    pyplot.close()
    pyplot.cla()
    # pyplot.show()
    # fig.show()
    # sys.exit(20)
    # return flat_samples
    
    # Calculate Chi2 and model values to return
    # teff_fit = popt[0]
    # logg_fit = popt[1]
    teff_fit = [best_est[0], unc[0][0], unc[0][1]]
    logg_fit = [best_est[1], unc[1][0], unc[1][1]]
    if numpy.max(mass) > 0:
        mass_fit = [best_est[2], unc[2][0], unc[2][1]]
    else:
        mass_fit = [numpy.nan, numpy.nan,numpy.nan]
    if numpy.max(cool_age) > 0:
        cool_age_fit = [best_est[3], unc[3][0], unc[3][1]]
        # mi_fit = [best_est[4], unc[4][0], unc[4][1]]
        # msl_fit = [best_est[5], unc[5][0], unc[5][1]]
        # tot_age_fit = [best_est[6], unc[6][0], unc[6][1]]
    else:
        cool_age_fit = [numpy.nan, numpy.nan, numpy.nan]
        # mi_fit =[numpy.nan, numpy.nan, numpy.nan]
        # msl_fit = [numpy.nan, numpy.nan, numpy.nan]
        # tot_age_fit = [numpy.nan, numpy.nan, numpy.nan]
    flux_model = interp_models(freq, teff_fit[0], logg_fit[0])
    model = {'col1' : freq, 'col2' : flux_model}

    # teff_fit = popt[0]#(popt[0], numpy.sqrt(pcov[0,0]))
    # logg_fit = popt[1]#(popt[1], numpy.sqrt(pcov[1,1]))
    # dof = len(flux_model) - 2    
    chi2 = numpy.sum(numpy.square((numpy.array(flux_model) - fluxes)/uncertainty))
    # end_time = time.time()
    # print("This one took {} seconds!\n".format(end_time - start_time))
    # return teff_fit, logg_fit, mass_fit, cool_age_fit, mi_fit, msl_fit, tot_age_fit, model, chi2, flat_samples
    return teff_fit, logg_fit, mass_fit, cool_age_fit, model, chi2, flat_samples
     
# ===================================================================================
def plot_SED(wavelengths, fluxes, fmts, colors, labels, ra, dec, source_id, errors = 0, teff = "?", logg = "?", sptype = "?", model = None, plx = 1, teff_fit = 0, logg_fit = 0, ratio = 1, num = 1, chi2 = "?", full_model = None, Gaia_only = False, AV = 0, R = None):
    # R = find8_R(logg_fit)
    fig, axes = pyplot.subplots(figsize = (18,9))
    # axes.set_xlim(300,1000)
    uplims = numpy.full(len(errors), False)
    uplims[numpy.where(errors == 0)] = True
    for i in range(len(wavelengths)):
        axes.errorbar(3e8/wavelengths[i]*1e9, wavelengths[i]*(fluxes[i]), yerr = wavelengths[i]*errors[i], fmt = fmts[i], color = colors[i], markersize = 10, label = labels[i], elinewidth = 3, capsize = 5)
    if model != None:
        axes.plot(3e8/numpy.array(model['col1'])*1e9, model['col1']*numpy.array(model['col2'])*ratio, "k*", markersize = 20,label = "Model w/ Teff = {:.0f}K : logg = {:.2f} : Chi2 = {:.2f}".format(teff_fit, logg_fit, chi2))
    
    # old_range = axes.get_ylim()
    
    if full_model != None:
        axes.plot(numpy.array(full_model['col1'])/10, numpy.array(full_model['col1'])*numpy.array(full_model['col2'])*ratio*1e-7*1e4, "k-")
    
    axes.tick_params(which = "both", direction='inout', length=16, labelsize = 20)
    # axes.text(x = 1.05, y = 0.5,s = "Teff = {:.0f}K\nlogg = {:.2f}\nSptype = {}".format(teff, logg, sptype), transform = axes.transAxes, fontsize = 24)
    axes.set_yscale('log')
#    axes.set_xscale('log')
    axes.set_xlabel("Wavelength [nm]", fontsize = 24)#r'$\nu$ [Hz]', fontsize = 24)
    axes.set_ylabel(r'$\mathrm{\lambda F_\lambda\ [W\ m^{-2}]}$', fontsize = 24)
    axes.legend(fontsize = 20)
    # axes.set_ylim(old_range)
    pyplot.suptitle("WD eDR3 {} RA: {} DEC: {}".format(source_id, ra, dec), fontsize = 20)

    if Gaia_only:
        fig.savefig("plots\\SED_GaiaOnly_WDeDR3_{}.png".format(source_id), bbox_inches = 'tight')    
    else:
        fig.savefig("plots\\SED_WDeDR3{}.png".format(source_id), bbox_inches = 'tight')
    
    pyplot.close()
    pyplot.cla()
    # pyplot.show()
    
# ===================================================================================
def convertSDSS_to_flux(mag, b, zeropoint = 3631.0, offset = 0):
    mag[numpy.where(mag == 0)] = numpy.ma.masked
    mag = mag + offset
    flux = zeropoint*1e-26*2*b*numpy.sinh(-mag*numpy.log(10)/2.5 - numpy.log(b))
    flux[numpy.where(numpy.isnan(mag))] = numpy.ma.masked
    return flux
    
# ===================================================================================
def convertSDSS_error_to_flux(error, mag, b, zeropoint = 3631, offset = 0):
    mag = mag+offset
    error = zeropoint*1e-26*error*1.84207*b*numpy.cosh(numpy.log(b) + 0.921034*mag)
    return error
# ===================================================================================
def save_progress(table, filename, i):
    print("Saving progress...\n")
    table.write(filename + "_sedcheck" + "_{}".format(i) + ".csv", overwrite = True)
    f = open("sed_progress.txt", 'w')
    f.write("{}".format(i))
    f.close()
# ===================================================================================
def main():
    
    # Read in Binaries #
    filename = "WDs_in_WDMS_xmatch_phot_wtd_pars"
    binaries = Table.read(filename + ".csv")
    
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    binaries = binaries[start:end]
    Gaia_only = False
    Plx = binaries['wtd_par']
    e_Plx = binaries['e_wtd_par']
    AV = binaries['ext']#binaries['meanAV']
    min_AV = binaries['ext_min']#binaries['minAV']
    max_AV = binaries['ext_max']#binaries['maxAV']
    R_table = Table.read("BV_extinction_SF.txt", format = 'ascii')
    R = numpy.array([0.835, 1.139,0.65, float(R_table['R_lambda'][numpy.where(R_table['Band'] == 'SDSS u')])/3.1, 
                      float(R_table['R_lambda'][numpy.where(R_table['Band'] == 'SDSS g')])/3.1, 
                      float(R_table['R_lambda'][numpy.where(R_table['Band'] == 'SDSS r')])/3.1, 
                      float(R_table['R_lambda'][numpy.where(R_table['Band'] == 'SDSS i')])/3.1, 
                      float(R_table['R_lambda'][numpy.where(R_table['Band'] == 'SDSS z')])/3.1,
                      float(R_table['R_lambda'][numpy.where(R_table['Band'] == 'PanSTARRS g')])/3.1, 
                      float(R_table['R_lambda'][numpy.where(R_table['Band'] == 'PanSTARRS r')])/3.1, 
                      float(R_table['R_lambda'][numpy.where(R_table['Band'] == 'PanSTARRS i')])/3.1, 
                      float(R_table['R_lambda'][numpy.where(R_table['Band'] == 'PanSTARRS z')])/3.1,
                      float(R_table['R_lambda'][numpy.where(R_table['Band'] == 'PanSTARRS y')])/3.1,
                      float(R_table['R_lambda'][numpy.where(R_table['Band'] == 'SkyMapper u')])/3.1, 
                      float(R_table['R_lambda'][numpy.where(R_table['Band'] == 'SkyMapper v')])/3.1, 
                      float(R_table['R_lambda'][numpy.where(R_table['Band'] == 'SkyMapper g')])/3.1, 
                      float(R_table['R_lambda'][numpy.where(R_table['Band'] == 'SkyMapper r')])/3.1, 
                      float(R_table['R_lambda'][numpy.where(R_table['Band'] == 'SkyMapper i')])/3.1,
                      float(R_table['R_lambda'][numpy.where(R_table['Band'] == 'SkyMapper z')])/3.1])    
    
    A_error_percent = 0.05
    e_A_range_lower = AV - min_AV
    e_A_range_upper = max_AV - AV
    e_A = A_error_percent*AV
    for i in range(len(e_A)):
        if e_A[i] < e_A_range_lower[i] or e_A_range_upper[i]:
            e_A[i] = numpy.max([e_A_range_lower[i], e_A_range_upper[i]])
        else:
            continue
        
    zeropoints = []

    ## Gaia eDR3 Bandpasses - zeropoints from Gaia Collab ##
    freq_G = 3e8/(583.63/1e9)
    freq_rp = 3e8/(758.88/1e9)   
    freq_bp = 3e8/(502.09/1e9)
    G_0 = 25.8010446445   
    e_G0 = 0.0027590522
    RP_0 = 25.1039837393   
    e_RP0 = 0.0015800349
    BP_0 = 25.3539555559   
    e_BP0 = 0.0023065687
    G_0_vega = 25.6873668671   
    BP_0_vega = 25.3385422158   
    RP_0_vega = 24.7478955012   

    zeropoints.append(G_0)
    zeropoints.append(BP_0)
    zeropoints.append(RP_0)
    gmag_corrected, gflux_corrected = correct_gband(binaries['bp_rp'], binaries['astrometric_params_solved'], binaries['phot_g_mean_mag'], binaries['phot_g_mean_flux'])
    ## These Fluxes are in W m^-2 Hz^-1 ##
    FG = numpy.power(10, ((gmag_corrected - G_0_vega)/-2.5))
    FBP = numpy.power(10, (binaries['phot_bp_mean_mag'] - BP_0_vega)/-2.5)
    FRP = numpy.power(10, (binaries['phot_rp_mean_mag'] - RP_0_vega)/-2.5)
            
    G = FG*numpy.power(10, -0.4*(G_0 + 56.10))
    BP = FBP*numpy.power(10, -0.4*(BP_0 + 56.10))
    RP = FRP*numpy.power(10, -0.4*(RP_0 + 56.10))

    # G = gflux_corrected*numpy.power(10, -0.4*(G_0 + 56.10))
    # BP = binaries['phot_bp_mean_flux']*numpy.power(10, -0.4*(BP_0 + 56.10))
    # RP = binaries['phot_rp_mean_flux']*numpy.power(10, -0.4*(RP_0 + 56.10))
    
    e_G = numpy.sqrt(numpy.square(binaries['phot_g_mean_flux_error']/gflux_corrected) + numpy.square(numpy.log(10)*e_G0*0.4))*G
    e_BP = numpy.sqrt(numpy.square(binaries['phot_bp_mean_flux_error']/binaries['phot_bp_mean_flux']) + numpy.square(numpy.log(10)*e_BP0*0.4))*BP
    e_RP = numpy.sqrt(numpy.square(binaries['phot_rp_mean_flux_error']/binaries['phot_rp_mean_flux']) + numpy.square(numpy.log(10)*e_RP0*0.4))*RP
    
    e_G[numpy.where((e_G/G < 0.03) & (e_G/G > 0))] = 0.03*G[numpy.where((e_G/G < 0.03) & (e_G/G > 0))]
    e_G[numpy.where((e_RP/RP < 0.03) & (e_RP/RP > 0))] = 0.03*G[numpy.where((e_RP/RP < 0.03) & (e_RP/RP > 0))]
    e_G[numpy.where((e_BP/BP < 0.03) & (e_RP/RP > 0))] = 0.03*G[numpy.where((e_BP/BP < 0.03) & (e_BP/BP > 0))]

    # ## SDSS Bandpasses ##
    # # Make sure magnitudes won't have negative flux!! #
    binaries['u'][numpy.where(numpy.array(binaries['u']) >= 24.63)] = 0
    binaries['g'][numpy.where(numpy.array(binaries['g']) >= 25.11)] = 0
    binaries['r'][numpy.where(numpy.array(binaries['r']) >= 24.80)] = 0
    binaries['i'][numpy.where(numpy.array(binaries['i']) >= 24.36)] = 0
    binaries['z'][numpy.where(numpy.array(binaries['z']) >= (22.83 - 0.2))] = 0
    
    binaries['u'][numpy.where(numpy.array(binaries['angular_distance_']) > 2)] = 0
    binaries['g'][numpy.where(numpy.array(binaries['angular_distance_']) > 2)] = 0
    binaries['r'][numpy.where(numpy.array(binaries['angular_distance_']) > 2)] = 0
    binaries['i'][numpy.where(numpy.array(binaries['angular_distance_']) > 2)] = 0
    binaries['z'][numpy.where(numpy.array(binaries['angular_distance_']) > 2)] = 0

    # Check flags for EDGE (bit 2)
    binaries['u'][numpy.where((numpy.array(binaries['flags_u'], dtype = numpy.int64) & numpy.power(2,2)) != 0)] = 0
    binaries['g'][numpy.where((numpy.array(binaries['flags_g'], dtype = numpy.int64) & numpy.power(2,2)) != 0)] = 0
    binaries['r'][numpy.where((numpy.array(binaries['flags_r'], dtype = numpy.int64) & numpy.power(2,2)) != 0)] = 0
    binaries['i'][numpy.where((numpy.array(binaries['flags_i'], dtype = numpy.int64) & numpy.power(2,2)) != 0)] = 0
    binaries['z'][numpy.where((numpy.array(binaries['flags_z'], dtype = numpy.int64) & numpy.power(2,2)) != 0)] = 0

    # Check flags for PEAKCENTER (bit 5)
    binaries['u'][numpy.where((numpy.array(binaries['flags_u'], dtype = numpy.int64) & numpy.power(2,5)) != 0)] = 0
    binaries['g'][numpy.where((numpy.array(binaries['flags_g'], dtype = numpy.int64) & numpy.power(2,5)) != 0)] = 0
    binaries['r'][numpy.where((numpy.array(binaries['flags_r'], dtype = numpy.int64) & numpy.power(2,5)) != 0)] = 0
    binaries['i'][numpy.where((numpy.array(binaries['flags_i'], dtype = numpy.int64) & numpy.power(2,5)) != 0)] = 0
    binaries['z'][numpy.where((numpy.array(binaries['flags_z'], dtype = numpy.int64) & numpy.power(2,5)) != 0)] = 0
    
    # Check flags for SATUR (bit 18)
    binaries['u'][numpy.where((numpy.array(binaries['flags_u'], dtype = numpy.int64) & numpy.power(2,18)) != 0)] = 0
    binaries['g'][numpy.where((numpy.array(binaries['flags_g'], dtype = numpy.int64) & numpy.power(2,18)) != 0)] = 0
    binaries['r'][numpy.where((numpy.array(binaries['flags_r'], dtype = numpy.int64) & numpy.power(2,18)) != 0)] = 0
    binaries['i'][numpy.where((numpy.array(binaries['flags_i'], dtype = numpy.int64) & numpy.power(2,18)) != 0)] = 0
    binaries['z'][numpy.where((numpy.array(binaries['flags_z'], dtype = numpy.int64) & numpy.power(2,18)) != 0)] = 0

    # Check flags for NOTCHECKED (bit 19)
    binaries['u'][numpy.where((numpy.array(binaries['flags_u'], dtype = numpy.int64) & numpy.power(2,19)) != 0)] = 0
    binaries['g'][numpy.where((numpy.array(binaries['flags_g'], dtype = numpy.int64) & numpy.power(2,19)) != 0)] = 0
    binaries['r'][numpy.where((numpy.array(binaries['flags_r'], dtype = numpy.int64) & numpy.power(2,19)) != 0)] = 0
    binaries['i'][numpy.where((numpy.array(binaries['flags_i'], dtype = numpy.int64) & numpy.power(2,19)) != 0)] = 0
    binaries['z'][numpy.where((numpy.array(binaries['flags_z'], dtype = numpy.int64) & numpy.power(2,19)) != 0)] = 0

    # # Make 0 values NaNs to make sure  to not get zeromag fluxes #
    binaries['u'][numpy.where(numpy.array(binaries['u']) == 0)] = numpy.ma.masked
    binaries['g'][numpy.where(numpy.array(binaries['g']) == 0)] = numpy.ma.masked
    binaries['r'][numpy.where(numpy.array(binaries['r']) == 0)] = numpy.ma.masked
    binaries['i'][numpy.where(numpy.array(binaries['i']) == 0)] = numpy.ma.masked
    binaries['z'][numpy.where(numpy.array(binaries['z']) == 0)] = numpy.ma.masked
    
    # Set minimum error value of magnitudes
    binaries['err_u'][numpy.where((numpy.array(binaries['err_u']) < 0.03) & (numpy.array(binaries['err_u']) > 0.0))] = 0.03
    binaries['err_g'][numpy.where((numpy.array(binaries['err_g']) < 0.03) & (numpy.array(binaries['err_g']) > 0.0))] = 0.03
    binaries['err_r'][numpy.where((numpy.array(binaries['err_r']) < 0.03) & (numpy.array(binaries['err_r']) > 0.0))] = 0.03
    binaries['err_i'][numpy.where((numpy.array(binaries['err_i']) < 0.03) & (numpy.array(binaries['err_i']) > 0.0))] = 0.03
    binaries['err_z'][numpy.where((numpy.array(binaries['err_z']) < 0.03) & (numpy.array(binaries['err_z']) > 0.0))] = 0.03

    u = convertSDSS_to_flux(binaries['u'], 1.4e-10, offset = -0.04)#, 1568.5)
    g = convertSDSS_to_flux(binaries['g'], 0.9e-10)#, 3965.9)
    r = convertSDSS_to_flux(binaries['r'], 1.2e-10)#, 3162.0)
    i = convertSDSS_to_flux(binaries['i'], 1.8e-10)#, 2602.0)
    z = convertSDSS_to_flux(binaries['z'], 7.4e-10, offset = 0.02)#, 2244.7)

    # e_u = 0.02*u + convertSDSS_error_to_flux(binaries['err_u'], binaries['u'], 1.4e-10, offset = -0.04)#, 1568.5)
    # e_g = 0.01*g + convertSDSS_error_to_flux(binaries['err_g'], binaries['g'], 0.9e-10)#, 3965.9)
    # e_r = 0.01*r + convertSDSS_error_to_flux(binaries['err_r'], binaries['r'], 1.2e-10)#, 3162.0)
    # e_i = 0.01*i + convertSDSS_error_to_flux(binaries['err_i'], binaries['i'], 1.8e-10)#, 2602.0)
    # e_z = 0.02*z + convertSDSS_error_to_flux(binaries['err_z'], binaries['z'], 7.4e-10, offset = 0.02)#, 2244.7)
    e_u = convertSDSS_error_to_flux(binaries['err_u'], binaries['u'], 1.4e-10, offset = -0.04)#, 1568.5)
    e_g = convertSDSS_error_to_flux(binaries['err_g'], binaries['g'], 0.9e-10)#, 3965.9)
    e_r = convertSDSS_error_to_flux(binaries['err_r'], binaries['r'], 1.2e-10)#, 3162.0)
    e_i = convertSDSS_error_to_flux(binaries['err_i'], binaries['i'], 1.8e-10)#, 2602.0)
    e_z = convertSDSS_error_to_flux(binaries['err_z'], binaries['z'], 7.4e-10, offset = 0.02)#, 2244.7)
    
    freq_u = 3e8/(359.49/1e9)
    freq_g = 3e8/(464.04/1e9)
    freq_r = 3e8/(612.23/1e9)
    freq_i = 3e8/(743.95/1e9)
    freq_z = 3e8/(889.71/1e9)
    zeropoints.append(3631e-26)
    zeropoints.append(3631e-26)
    zeropoints.append(3631e-26)
    zeropoints.append(3631e-26)
    zeropoints.append(3631e-26)
    # # bs = [1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10]
    
    
    # ## PanSTARR Fluxes ##
    freq_g_pstarr = 3e8/(481.09/1e9)
    freq_r_pstarr = 3e8/(615.64/1e9)
    freq_i_pstarr = 3e8/(750.37/1e9)  
    freq_z_pstarr = 3e8/(866.86/1e9)
    freq_y_pstarr = 3e8/(961.35/1e9)
    g_pstarr_0 = 3631e-26#3893e-26
    r_pstarr_0 = 3631e-26#3135.5e-26
    i_pstarr_0 = 3631e-26#2577e-26
    z_pstarr_0 = 3631e-26#2273e-26
    y_pstarr_0 = 3631e-26#2203.7e-26
    
    # # Check for flags related to rank detections

    binaries['g_mean_psf_mag'][numpy.where(numpy.array(binaries['angular_distance_pstarr']) > 2)] = 0
    binaries['r_mean_psf_mag'][numpy.where(numpy.array(binaries['angular_distance_pstarr']) > 2)] = 0
    binaries['i_mean_psf_mag'][numpy.where(numpy.array(binaries['angular_distance_pstarr']) > 2)] = 0
    binaries['z_mean_psf_mag'][numpy.where(numpy.array(binaries['angular_distance_pstarr']) > 2)] = 0
    binaries['y_mean_psf_mag'][numpy.where(numpy.array(binaries['angular_distance_pstarr']) > 2)] = 0
    
    binaries['g_mean_psf_mag'][numpy.where((numpy.array(binaries['g_flags_'], dtype = numpy.int64) & numpy.power(2,10)) != 0)] = 0
    binaries['r_mean_psf_mag'][numpy.where((numpy.array(binaries['r_flags_'], dtype = numpy.int64) & numpy.power(2,10)) != 0)] = 0
    binaries['i_mean_psf_mag'][numpy.where((numpy.array(binaries['i_flags_'], dtype = numpy.int64) & numpy.power(2,10)) != 0)] = 0
    binaries['z_mean_psf_mag'][numpy.where((numpy.array(binaries['z_flags_'], dtype = numpy.int64) & numpy.power(2,10)) != 0)] = 0
    binaries['y_mean_psf_mag'][numpy.where((numpy.array(binaries['y_flags'], dtype = numpy.int64) & numpy.power(2,10)) != 0)] = 0
 
    binaries['g_mean_psf_mag'][numpy.where((numpy.array(binaries['g_flags_'], dtype = numpy.int64) & numpy.power(2,11)) != 0)] = 0
    binaries['r_mean_psf_mag'][numpy.where((numpy.array(binaries['r_flags_'], dtype = numpy.int64) & numpy.power(2,11)) != 0)] = 0
    binaries['i_mean_psf_mag'][numpy.where((numpy.array(binaries['i_flags_'], dtype = numpy.int64) & numpy.power(2,11)) != 0)] = 0
    binaries['z_mean_psf_mag'][numpy.where((numpy.array(binaries['z_flags_'], dtype = numpy.int64) & numpy.power(2,11)) != 0)] = 0
    binaries['y_mean_psf_mag'][numpy.where((numpy.array(binaries['y_flags'], dtype = numpy.int64) & numpy.power(2,11)) != 0)] = 0
     
    binaries['g_mean_psf_mag'][numpy.where((numpy.array(binaries['g_flags_'], dtype = numpy.int64) & numpy.power(2,12)) != 0)] = 0
    binaries['r_mean_psf_mag'][numpy.where((numpy.array(binaries['r_flags_'], dtype = numpy.int64) & numpy.power(2,12)) != 0)] = 0
    binaries['i_mean_psf_mag'][numpy.where((numpy.array(binaries['i_flags_'], dtype = numpy.int64) & numpy.power(2,12)) != 0)] = 0
    binaries['z_mean_psf_mag'][numpy.where((numpy.array(binaries['z_flags_'], dtype = numpy.int64) & numpy.power(2,12)) != 0)] = 0
    binaries['y_mean_psf_mag'][numpy.where((numpy.array(binaries['y_flags'], dtype = numpy.int64) & numpy.power(2,12)) != 0)] = 0
    
    binaries['g_mean_psf_mag'][numpy.where(numpy.array(binaries['g_mean_psf_mag']) == 0)] = numpy.ma.masked
    binaries['r_mean_psf_mag'][numpy.where(numpy.array(binaries['r_mean_psf_mag']) == 0)] = numpy.ma.masked
    binaries['i_mean_psf_mag'][numpy.where(numpy.array(binaries['i_mean_psf_mag']) == 0)] = numpy.ma.masked
    binaries['z_mean_psf_mag'][numpy.where(numpy.array(binaries['z_mean_psf_mag']) == 0)] = numpy.ma.masked
    binaries['y_mean_psf_mag'][numpy.where(numpy.array(binaries['y_mean_psf_mag']) == 0)] = numpy.ma.masked
    
    # Set minimum error value of magnitudes
    binaries['g_mean_psf_mag_error'][numpy.where((numpy.array(binaries['g_mean_psf_mag_error']) < 0.03) & (numpy.array(binaries['g_mean_psf_mag_error']) > 0.0))] = 0.03
    binaries['r_mean_psf_mag_error'][numpy.where((numpy.array(binaries['r_mean_psf_mag_error']) < 0.03) & (numpy.array(binaries['r_mean_psf_mag_error']) > 0.0))] = 0.03
    binaries['i_mean_psf_mag_error'][numpy.where((numpy.array(binaries['i_mean_psf_mag_error']) < 0.03) & (numpy.array(binaries['i_mean_psf_mag_error']) > 0.0))] = 0.03
    binaries['z_mean_psf_mag_error'][numpy.where((numpy.array(binaries['z_mean_psf_mag_error']) < 0.03) & (numpy.array(binaries['z_mean_psf_mag_error']) > 0.0))] = 0.03
    binaries['y_mean_psf_mag_error'][numpy.where((numpy.array(binaries['y_mean_psf_mag_error']) < 0.03) & (numpy.array(binaries['y_mean_psf_mag_error']) > 0.0))] = 0.03
    
    g_pstarr = g_pstarr_0*numpy.power(10, numpy.array(binaries['g_mean_psf_mag'])/-2.5)
    r_pstarr = r_pstarr_0*numpy.power(10, numpy.array(binaries['r_mean_psf_mag'])/-2.5)
    i_pstarr = i_pstarr_0*numpy.power(10, numpy.array(binaries['i_mean_psf_mag'])/-2.5)
    z_pstarr = z_pstarr_0*numpy.power(10, numpy.array(binaries['z_mean_psf_mag'])/-2.5)
    y_pstarr = y_pstarr_0*numpy.power(10, numpy.array(binaries['y_mean_psf_mag'])/-2.5)

        
    e_g_pstarr = binaries['g_mean_psf_mag_error']*g_pstarr
    e_r_pstarr = binaries['r_mean_psf_mag_error']*r_pstarr
    e_i_pstarr = binaries['i_mean_psf_mag_error']*i_pstarr
    e_z_pstarr = binaries['z_mean_psf_mag_error']*z_pstarr
    e_y_pstarr = binaries['y_mean_psf_mag_error']*y_pstarr

    zeropoints.append(3631e-26)
    zeropoints.append(3631e-26)
    zeropoints.append(3631e-26)
    zeropoints.append(3631e-26)
    zeropoints.append(3631e-26)

    ## SkyMapper Fluxes ##
    freq_u_smap = 3e8/(349.82/1e9)
    freq_v_smap = 3e8/(387.09/1e9)
    freq_g_smap = 3e8/(496.85/1e9)
    freq_r_smap = 3e8/(604.01/1e9)
    freq_i_smap = 3e8/(692.91/1e9)  
    freq_z_smap = 3e8/(909.15/1e9)
    u_smap_0 = 3631e-26
    v_smap_0 = 3631e-26
    g_smap_0 = 3631e-26
    r_smap_0 = 3631e-26
    i_smap_0 = 3631e-26
    z_smap_0 = 3631e-26

    # Check for flags
    binaries['u_psf'][numpy.where(numpy.array(binaries['angular_distance']) > 2)] = 0
    binaries['v_psf'][numpy.where(numpy.array(binaries['angular_distance']) > 2)] = 0
    binaries['g_psf'][numpy.where(numpy.array(binaries['angular_distance']) > 2)] = 0
    binaries['r_psf'][numpy.where(numpy.array(binaries['angular_distance']) > 2)] = 0
    binaries['i_psf'][numpy.where(numpy.array(binaries['angular_distance']) > 2)] = 0
    binaries['z_psf'][numpy.where(numpy.array(binaries['angular_distance']) > 2)] = 0
    
    binaries['u_psf'][numpy.where(numpy.array(binaries['u_nimaflags']) > 0)] = 0
    binaries['v_psf'][numpy.where(numpy.array(binaries['v_nimaflags']) > 0)] = 0
    binaries['g_psf'][numpy.where(numpy.array(binaries['g_nimaflags']) > 0)] = 0
    binaries['r_psf'][numpy.where(numpy.array(binaries['r_nimaflags']) > 0)] = 0
    binaries['i_psf'][numpy.where(numpy.array(binaries['i_nimaflags']) > 0)] = 0
    binaries['z_psf'][numpy.where(numpy.array(binaries['z_nimaflags']) > 0)] = 0

    binaries['u_psf'][numpy.where(numpy.array(binaries['u_flags']) > 0)] = 0
    binaries['v_psf'][numpy.where(numpy.array(binaries['v_flags']) > 0)] = 0
    binaries['g_psf'][numpy.where(numpy.array(binaries['g_flags_smap']) > 0)] = 0
    binaries['r_psf'][numpy.where(numpy.array(binaries['r_flags_smap']) > 0)] = 0
    binaries['i_psf'][numpy.where(numpy.array(binaries['i_flags_smap']) > 0)] = 0
    binaries['z_psf'][numpy.where(numpy.array(binaries['z_flags_smap']) > 0)] = 0
    
    # Mask zero magnitude values
    binaries['u_psf'][numpy.where(numpy.array(binaries['u_psf']) == 0)] = numpy.ma.masked
    binaries['v_psf'][numpy.where(numpy.array(binaries['v_psf']) == 0)] = numpy.ma.masked
    binaries['g_psf'][numpy.where(numpy.array(binaries['g_psf']) == 0)] = numpy.ma.masked
    binaries['r_psf'][numpy.where(numpy.array(binaries['r_psf']) == 0)] = numpy.ma.masked
    binaries['i_psf'][numpy.where(numpy.array(binaries['i_psf']) == 0)] = numpy.ma.masked
    binaries['z_psf'][numpy.where(numpy.array(binaries['z_psf']) == 0)] = numpy.ma.masked
    
    # Set minimum error value of magnitudes
    binaries['e_u_psf'][numpy.where((numpy.array(binaries['e_u_psf']) < 0.03) & (numpy.array(binaries['e_u_psf']) > 0.0))] = 0.03
    binaries['e_v_psf'][numpy.where((numpy.array(binaries['e_v_psf']) < 0.03) & (numpy.array(binaries['e_v_psf']) > 0.0))] = 0.03
    binaries['e_g_psf'][numpy.where((numpy.array(binaries['e_g_psf']) < 0.03) & (numpy.array(binaries['e_g_psf']) > 0.0))] = 0.03
    binaries['e_r_psf'][numpy.where((numpy.array(binaries['e_r_psf']) < 0.03) & (numpy.array(binaries['e_r_psf']) > 0.0))] = 0.03
    binaries['e_i_psf'][numpy.where((numpy.array(binaries['e_i_psf']) < 0.03) & (numpy.array(binaries['e_i_psf']) > 0.0))] = 0.03
    binaries['e_z_psf'][numpy.where((numpy.array(binaries['e_z_psf']) < 0.03) & (numpy.array(binaries['e_z_psf']) > 0.0))] = 0.03
       
    u_smap = u_smap_0*numpy.power(10, numpy.array(binaries['u_psf'])/-2.5)
    v_smap = v_smap_0*numpy.power(10, numpy.array(binaries['v_psf'])/-2.5)
    g_smap = g_smap_0*numpy.power(10, numpy.array(binaries['g_psf'])/-2.5)
    r_smap = r_smap_0*numpy.power(10, numpy.array(binaries['r_psf'])/-2.5)
    i_smap = i_smap_0*numpy.power(10, numpy.array(binaries['i_psf'])/-2.5)
    z_smap = z_smap_0*numpy.power(10, numpy.array(binaries['z_psf'])/-2.5)

    e_u_smap = binaries['e_u_psf']*u_smap
    e_v_smap = binaries['e_v_psf']*v_smap    
    e_g_smap = binaries['e_g_psf']*g_smap
    e_r_smap = binaries['e_r_psf']*r_smap
    e_i_smap = binaries['e_i_psf']*i_smap
    e_z_smap = binaries['e_z_psf']*z_smap
    
    zeropoints.append(3631e-26)
    zeropoints.append(3631e-26)
    zeropoints.append(3631e-26)
    zeropoints.append(3631e-26)
    zeropoints.append(3631e-26)
    zeropoints.append(3631e-26)
    
    ## Combine everything into arrays to then fit data
    wavelengths = numpy.array([freq_G, freq_bp, freq_rp, freq_u, freq_g, freq_r, freq_i, freq_z, freq_g_pstarr, freq_r_pstarr, 
                               freq_i_pstarr, freq_z_pstarr, freq_y_pstarr, freq_u_smap, freq_v_smap, freq_g_smap, freq_r_smap, 
                               freq_i_smap, freq_z_smap])#, freq_J, freq_H, freq_K])
    fluxes = numpy.array([G, BP, RP, u, g, r, i, z, g_pstarr, r_pstarr, i_pstarr, z_pstarr, y_pstarr, u_smap, v_smap, g_smap, 
                          r_smap, i_smap, z_smap])#, J, H, K])
    mags = numpy.array([gmag_corrected, binaries['phot_bp_mean_mag'], binaries['phot_rp_mean_mag'], 
                        binaries['u'], binaries['g'], binaries['r'], binaries['i'], binaries['z'], 
                        binaries['g_mean_psf_mag'], binaries['r_mean_psf_mag'], binaries['i_mean_psf_mag'], binaries['z_mean_psf_mag'], binaries['y_mean_psf_mag'], 
                        binaries['u_psf'], binaries['v_psf'], binaries['g_psf'], binaries['r_psf'], binaries['i_psf'], binaries['z_psf']]) 
                        #binaries['j_m_tmass'], binaries['h_m_tmass'], binaries['ks_m_tmass']])
    errors = numpy.array([e_G, e_BP, e_RP, e_u, e_g, e_r, e_i, e_z, e_g_pstarr, e_r_pstarr, e_i_pstarr, e_z_pstarr, e_y_pstarr, 
                          e_u_smap, e_v_smap, e_g_smap, e_r_smap, e_i_smap, e_z_smap])
                         #, e_J, e_H, e_K])
    
    # below is for absolute fitting at 10 pc so include plx errors
    fluxes_10 = numpy.zeros(fluxes.shape)
    fluxes_10_dered = numpy.zeros(fluxes.shape)
    uncertainty = numpy.zeros(fluxes.shape)
    print(fluxes.shape)
    for i in range(len(fluxes)):
        fluxes_10[i] = fluxes[i]*numpy.square((1000/Plx)/10)
        fluxes_10_dered[i] = fluxes_10[i]*numpy.power(10,0.4*AV*R[i])
        uncertainty[i] = numpy.sqrt(numpy.square(errors[i]/fluxes[i]) + 2*numpy.square(e_Plx/Plx) 
                          + 0.046*e_A*R[i])*fluxes_10_dered[i]   
      
    # convolve models with filter profiles
    # models_convolve = convolve_filters_models(models, filters, wavelengths, names = ["Gaia", "Gaia", "Gaia", "SDSS", "SDSS", "SDSS", "SDSS", "SDSS", "pstarr", "pstarr", "pstarr", "pstarr", "pstarr", "SkyMap", "SkyMap", "SkyMap", "SkyMap", "SkyMap", "SkyMap", "2MASS", "2MASS", "2MASS", "APASS", "APASS", "APASS", "APASS", "APASS"], zeropoints = zeropoints, bs = bs)
    
    ## Make arrays for M-R relation using Montreal cooling models for M-R Relation
    ## Only using DA models for now. Can add DB after I get this working
    thickness = 'thick'
    H_files = glob.glob("C:\\Users\\Tyler\\Documents\\WD_Research\\Bergeron Cooling Models\\DA\\Sequences\\New_Sequences\\" + thickness + "\\*")
    cooling_grid_H = Table.read(H_files[0], format = 'ascii.fast_commented_header')
    for i in range(len(H_files) - 1):
        cooling_grid_H = vstack([cooling_grid_H, Table.read(H_files[i+1], format = 'ascii.fast_commented_header')])

    ## Make colormap and other plotting settings here
    colormap = matplotlib.cm.get_cmap(name = "RdBu")
    norm = matplotlib.colors.Normalize(vmin=numpy.min(wavelengths), vmax=numpy.max(wavelengths))
    colors = colormap(norm(wavelengths))
    fmts = numpy.array(["^", "^", "^", "s", "s", "s","s", "s", "o", "o", "o", "o", "o", "X", "X", "X", "X", "X", "X", "P", "P", "P"])
    colors = numpy.array(["red", "red", "red", "blue", "blue", "blue", "blue", "blue", "m", "m", "m", "m", "m", "c", "c", "c", "c", "c", "c", "brown", "brown", "brown"])
    labels = numpy.array(["Gaia", "", "", "SDSS", "", "", "", "", "PanSTARR", "", "", "", "", "SkyMap", "", "", "", "", "", "2MASS", "", "", "APASS", "", "", "", ""])
    
    # band_names = ["Gaia G", "Gaia BP", "Gaia RP", "SDSS u", "SDSS g", "SDSS r", "SDSS i", "SDSS z", "PanSTARR g", "PanSTARR r", "PanSTARR i", "PanSTARR z", "PanSTARR y", "SkyMap u", "SkyMap V", "SkyMap g", "SkyMap r", "SkyMap i", "SkyMap z", "2MASS J", "2MASS H", "2MASS Ks"]    
    
    # mwdd_matches = numpy.where(binaries['mwdd_teff_1'] > 0)[0]
    
    # model_fluxes = []
    # for i in range(len(models)):
    #     model_fs = []
    #     model_f = models_convolve[i]['col2']
    #     for j in range(len(model_f)):
    #         model_fs.append(model_f[j])
    #     model_fluxes.append(numpy.array(model_fs))
    # model_fluxes = numpy.array(model_fluxes)
    
    
    model_fluxes_table = Table.read("model_fluxes_3000K.csv")
    teffs = model_fluxes_table['Teff']
    loggs = model_fluxes_table['Logg']
    model_fluxes_dummy = model_fluxes_table["Gaia G", "Gaia BP", "Gaia RP", "SDSS u", "SDSS g", "SDSS r", "SDSS i", "SDSS z", "PanSTARR g", "PanSTARR r", "PanSTARR i", "PanSTARR z", "PanSTARR y", "SkyMap u", "SkyMap V", "SkyMap g", "SkyMap r", "SkyMap i", "SkyMap z", "2MASS J", "2MASS H", "2MASS Ks"]
    model_fluxes = []
    for i in range(len(model_fluxes_dummy)):
        model_fluxes.append(list(model_fluxes_dummy[i]))
    
    model_fluxes = numpy.array(model_fluxes)
    # print(model_fluxes)
    # model_table_save = Table(model_fluxes, names = band_names)
    # print(model_table_save)
    # model_table_save.add_columns([Column(teffs, name = 'Teff'), Column(loggs, 'Logg')])
    # model_table_save.write("model_fluxes_3000K.csv", overwrite = True)
    # print(model_table_save)
    # sys.exit(20)
    ## Generate interpolation grids under here ##
    flux_interp = make_flux_interp(teffs, loggs, model_fluxes)
    MR, cool_func = make_MR_coolfunc(cooling_grid_H)

    chis = []
    teffs_save = []
    e_teffs_upper = []
    e_teffs_lower = []
    loggs_save = []
    e_loggs_upper = []
    e_loggs_lower = []
    masses_save = []
    e_masses_upper = []
    e_masses_lower = []
    cool_ages_save = []
    e_cool_ages_upper = []
    e_cool_ages_lower = []
    # mis = []
    # e_mis_upper = []
    # e_mis_lower = []
    # tot_ages_save = []
    # e_tot_ages_upper = []
    # e_tot_ages_lower = []
    
    MESA_data = Table.read("C:\\Users\\Tyler\\Documents\\WD_Research\\MESA_IFMR\\MESA_IFMR_missing_one_point.csv")
    a=0.06367
    Mi_MESA = MESA_data["M_initial"]
    Mf_MESA = MESA_data["M_final"]
    IFMR = interp1d(Mf_MESA + a, Mi_MESA, kind = 'linear', fill_value = 'extrapolate', bounds_error = False)     
    
    init_m = numpy.load('init_mass_to_mslife\\mi.npy')
    msl = numpy.load('init_mass_to_mslife\\msl.npy')
    
    t_ms = interp1d(init_m, msl, fill_value = 'extrapolate', bounds_error = False)
    
    time_start = time.time()
    for i in range(len(binaries)): #mwdd_matches: #range(5):#len(binaries)):

        if i % 10 == 0:
            time_end = time.time()
            sys.stderr.write("Done with {}/{} -- It has taken {} minutes for last 10\n".format(i, len(binaries), (time_end - time_start)/60))
            time_start = time.time()        
        
        ##           RELATIVE FLUXES BELOW            ##
        ## ------------------------------------------ ##
        
        ## CHANGE NAME OF SAVEFILE IN plot_SED!!!!!!  ##
        
        # Use this code for not interpolating the models and just finding best fit grid point #
        # i_model, model, ratio, chi2 = fit_SED(wavelengths, fluxes[:,i], errors[:,i], Plx1[i], models_convolve, loggs, teffs, model_fluxes, points)
        # teff_fit, logg_fit, model, chi2 = fit_SED(wavelengths, fluxes[:,i], errors[:,i], Plx1[i], models_convolve, loggs, teffs, model_fluxes, points)

        # Code below does interpolate models #
        # teff_fit, logg_fit = fit_SED(wavelengths, fluxes[:,i], errors[:,i], Plx1[i], models_convolve, loggs, teffs, model_fluxes, points)
        # plot_SED(wavelengths, fluxes[:,i], fmts = fmts, labels = labels, errors = errors[:,i], logg = logg1[i], teff = Teff1[i], model = model, plx = Plx1[i], teff_fit = teffs[i_model], logg_fit = loggs[i_model], ratio = ratio, num = i, chi2 = chi2, full_model = models[i_model])
        
        ##           ABSOLUTE FLUXES BELOW            ##
        ## ------------------------------------------ ##
        
        if binaries['teff_H'][i] > 0 and binaries['logg_H'][i]:
            init_guess = (binaries['teff_H'][i], binaries['logg_H'][i])#, AV[i])
        else:
            init_guess = (10000, 8.0)#, AV[i])
            
        # AV_range = numpy.array([min_AV[i], max_AV[i]])
        if (gmag_corrected[i] + 5*numpy.log10(binaries['parallax'][i]/100.0) < 3.25*binaries['bp_rp'][i] + 9.625) or (not numpy.isfinite(AV[i])):# or (not numpy.isfinite(AV_range[0])) or (not numpy.isfinite(AV_range[1])):
            e_teff_lower_fit = numpy.ma.masked
            e_teff_upper_fit = numpy.ma.masked
            teff_fit = numpy.ma.masked
            
            e_logg_lower_fit = numpy.ma.masked
            e_logg_upper_fit = numpy.ma.masked
            logg_fit = numpy.ma.masked
            
            e_mass_upper_fit = numpy.ma.masked
            e_mass_lower_fit = numpy.ma.masked
            mass_fit = numpy.ma.masked
            
            e_cool_age_upper_fit = numpy.ma.masked
            e_cool_age_lower_fit = numpy.ma.masked
            cool_age_fit = numpy.ma.masked
            
            chi2 = numpy.ma.masked
            
        else:
            # Only use the Gaia Bandpasses #
            if Gaia_only:# or min_sep[i] < 2.0:            
                # teff_fit, logg_fit, mass_fit, cool_age_fit, mi_fit, msl_fit, tot_age_fit, model, chi2, sample = fit_SED_absolute(wavelengths[:3], fluxes[:3,i], errors[:3,i], Plx[i], e_Plx[i], loggs, teffs, model_fluxes, flux_interp, MR = MR, ra = binaries['ra'][i], dec = binaries['dec'][i], source_id = binaries['source_id'][i], initial_guess = init_guess, cooling_func = cool_func, IFMR = IFMR, t_ms = t_ms)
                teff_fit, logg_fit, mass_fit, cool_age_fit, model, chi2, sample = fit_SED_absolute(wavelengths[:3], fluxes_10_dered[:3,i], uncertainty[:3,i], Plx[i], e_Plx[i], loggs, teffs, model_fluxes, flux_interp, MR = MR, ra = binaries['ra'][i], dec = binaries['dec'][i], source_id = binaries['source_id'][i], initial_guess = init_guess, cooling_func = cool_func, IFMR = IFMR, t_ms = t_ms, AV_range = None, R = R)
                
                e_teff_lower_fit = teff_fit[1]
                e_teff_upper_fit = teff_fit[2]
                teff_fit = teff_fit[0]
                
                e_logg_lower_fit = logg_fit[1]
                e_logg_upper_fit = logg_fit[2]
                logg_fit = logg_fit[0]
                
                e_mass_upper_fit = mass_fit[1]
                e_mass_lower_fit = mass_fit[2]
                mass_fit = mass_fit[0]
                
                e_cool_age_upper_fit = cool_age_fit[1]
                e_cool_age_lower_fit = cool_age_fit[2]
                cool_age_fit = cool_age_fit[0]

                # e_mi_upper_fit = mi_fit[1]
                # e_mi_lower_fit = mi_fit[2]
                # mi_fit = mi_fit[0]
                
                # e_tot_age_upper_fit = tot_age_fit[1]
                # e_tot_age_lower_fit = tot_age_fit[2]
                # tot_age_fit = tot_age_fit[0]
                
                # plot_Chi2(wavelengths[:3], fluxes[:3,i], errors[:3,i], Plx[i], e_Plx[i], models_convolve, loggs, teffs, model_fluxes, flux_interp, MR = MR, mcmc_samples = sample, index = i + file_start)
                plot_SED(wavelengths[:3], fluxes_10_dered[:3,i], fmts = fmts[:3], colors = colors[:3], labels = labels[:3], ra = binaries['ra'][i], dec = binaries['dec'][i], source_id = binaries['source_id'][i], errors = uncertainty[:3,i], model = model, plx = Plx[i], teff_fit = teff_fit, logg_fit = logg_fit, chi2 = chi2, AV = AV[i], R = R)
               
            # Use all available photometry #
            else:
                where_non_zero_mag = numpy.where(mags[:,i] > 0)[0]
                where_non_zero_mag_and_error = numpy.where((mags[:,i] > 0) & (errors[:,i] > 0))[0]
                non_zero_fluxes_and_errors = fluxes_10_dered[:,i][where_non_zero_mag_and_error]
                # teff_fit, logg_fit, mass_fit, cool_age_fit, mi_fit, msl_fit, tot_age_fit, model, chi2, sample = fit_SED_absolute(wavelengths[where_non_zero_mag_and_error], non_zero_fluxes_and_errors, errors[:,i][where_non_zero_mag_and_error], Plx[i], e_Plx[i], loggs, teffs, model_fluxes, flux_interp[where_non_zero_mag_and_error], MR = MR, ra = binaries['ra'][i], dec = binaries['dec'][i], source_id = binaries['source_id'][i], initial_guess = init_guess, cooling_func = cool_func, IFMR = IFMR, t_ms = t_ms)
                teff_fit, logg_fit, mass_fit, cool_age_fit, model, chi2, sample = fit_SED_absolute(wavelengths[where_non_zero_mag_and_error], non_zero_fluxes_and_errors, uncertainty[:,i][where_non_zero_mag_and_error], Plx[i], e_Plx[i], loggs, teffs, model_fluxes, flux_interp[where_non_zero_mag_and_error], MR = MR, ra = binaries['ra'][i], dec = binaries['dec'][i], source_id = binaries['source_id'][i], initial_guess = init_guess, cooling_func = cool_func, IFMR = IFMR, t_ms = t_ms, AV_range = None, R = R)
                
                e_teff_upper_fit = teff_fit[1]
                e_teff_lower_fit = teff_fit[2]
                teff_fit = teff_fit[0]
                
                e_logg_upper_fit = logg_fit[1]
                e_logg_lower_fit = logg_fit[2]
                logg_fit = logg_fit[0]
                
                e_mass_upper_fit = mass_fit[1]
                e_mass_lower_fit = mass_fit[2]
                mass_fit = mass_fit[0]                
                
                e_cool_age_upper_fit = cool_age_fit[1]
                e_cool_age_lower_fit = cool_age_fit[2]
                cool_age_fit = cool_age_fit[0]
                
                # e_mi_upper_fit = mi_fit[1]
                # e_mi_lower_fit = mi_fit[2]
                # mi_fit = mi_fit[0]
                
                # e_tot_age_upper_fit = tot_age_fit[1]
                # e_tot_age_lower_fit = tot_age_fit[2]
                # tot_age_fit = tot_age_fit[0]
                
                
                # plot_Chi2(wavelengths[where_non_zero_mag], non_zero_fluxes, errors[:,i][where_non_zero_mag], Plx[i], e_Plx[i], models_convolve, loggs, teffs, model_fluxes, flux_interp, MR = MR, mcmc_samples = sample, index = i + file_start)
                plot_SED(wavelengths[where_non_zero_mag], fluxes_10_dered[:, i][where_non_zero_mag], fmts = fmts[where_non_zero_mag], colors = colors[where_non_zero_mag], labels = labels[where_non_zero_mag], ra = binaries['ra'][i], dec = binaries['dec'][i], source_id = binaries['source_id'][i], errors = uncertainty[:, i][where_non_zero_mag], model = model, plx = Plx[i], teff_fit = teff_fit, logg_fit = logg_fit, chi2 = chi2, AV = AV[i], R = R)
    
        chis.append(chi2)
        teffs_save.append(teff_fit)
        e_teffs_upper.append(e_teff_upper_fit)
        e_teffs_lower.append(e_teff_lower_fit)
        loggs_save.append(logg_fit)
        e_loggs_upper.append(e_logg_upper_fit)
        e_loggs_lower.append(e_logg_lower_fit)
        masses_save.append(mass_fit)
        e_masses_upper.append(e_mass_upper_fit)
        e_masses_lower.append(e_mass_lower_fit)
        cool_ages_save.append(cool_age_fit)
        e_cool_ages_upper.append(e_cool_age_upper_fit)
        e_cool_ages_lower.append(e_cool_age_lower_fit)
        # mis.append(mi_fit)
        # e_mis_upper.append(e_mi_upper_fit)
        # e_mis_lower.append(e_mi_lower_fit)
        # tot_ages_save.append(tot_age_fit)
        # e_tot_ages_upper.append(e_tot_age_upper_fit)
        # e_tot_ages_lower.append(e_tot_age_lower_fit)
    
    
    binaries = Table.read(filename + ".csv")
    binaries = binaries[start:end]
    binaries.add_column(teffs_save, name = "TeffH")
    binaries.add_column(e_teffs_lower, name = "e_TeffH_lower")
    binaries.add_column(e_teffs_upper, name = "e_TeffH_upper")
    binaries.add_column(loggs_save, name = "LoggH")
    binaries.add_column(e_loggs_lower, name = "e_LoggH_lower")
    binaries.add_column(e_loggs_upper, name = "e_LoggH_upper")
    binaries.add_column(masses_save, name = "MassH")
    binaries.add_column(e_masses_lower, name = "e_MassH_lower")
    binaries.add_column(e_masses_upper, name = "e_MassH_upper")
    binaries.add_column(cool_ages_save, name = "cool_ageH")
    binaries.add_column(e_cool_ages_lower, name = "e_cool_ageH_lower")
    binaries.add_column(e_cool_ages_upper, name = "e_cool_ageH_upper")
    # binaries.add_column(mis, name = "MiH")
    # binaries.add_column(e_mis_lower, name = "e_MiH_lower")
    # binaries.add_column(e_mis_upper, name = "e_MiH_upper")
    # binaries.add_column(tot_ages_save, name = "tot_ageH")
    # binaries.add_column(e_tot_ages_lower, name = "e_tot_ageH_lower")
    # binaries.add_column(e_tot_ages_upper, name = "e_tot_ageH_upper")
    binaries.add_column(chis, name = "Chi2H")
    binaries.write("tables\\" + filename + "_fit_seds_fluxes_{}_{}.csv".format(start,end), overwrite = True)
    # pyplot.show()    
    
# ===================================================================================
if __name__ == '__main__':
    main()