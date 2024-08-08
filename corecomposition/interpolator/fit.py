import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import lmfit
import pyphot
import emcee
import warnings


from . import interpolator

#physical constants in SI units
speed_light = 299792458 #m/s
radius_sun = 6.957e8 #m
mass_sun = 1.9884e30 #kg
newton_G = 6.674e-11 #N m^2/kg^2
pc_to_m = 3.086775e16

class CoarseEngine:
    def __init__(self, interpolator, assume_mrr = False, vary_logg = False):
        self.interpolator = interpolator
        self.bands = interpolator.bands
        self.teff_lims = interpolator.teff_lims
        self.logg_lims = interpolator.logg_lims

        self.assume_mrr = assume_mrr
        self.vary_logg = vary_logg

        lib = pyphot.get_library()
        self.filters = [lib[band] for band in self.bands]

    def mag_to_flux(self, mag, e_mag = None):
        """
        convert from magntiudes on the AB system to flux for a particular filter (Gaia magnitudes are on the Vega system)
        
        Args:
            mag (array[N]):    magnitude of the observation
            e_mag (array[N]):  magnitude uncertainty
        Returns:
            flux in flam if e_mag is not specified, a tuple containing flux and flux uncertainty if it is
        """
        if e_mag is not None:
            flux = [10**(-0.4*(mag[i] + filter.Vega_zero_mag)) for i, filter in enumerate(self.filters)]
            e_flux = [np.abs((np.log(10)*(-0.4)*10**(-0.4 * (mag[i] + filter.Vega_zero_mag)) * e_mag[i])) for i, filter in enumerate(self.filters)]
            return flux, e_flux
        else:
            flux = [10**(-0.4*(mag[i] + filter.Vega_zero_mag)) for i, filter in enumerate(self.filters)]
            return flux

    def get_model_flux(self, params):
        #get model photometric flux for a WD with a given radius, located a given distance away        
        fl= 4 * np.pi * self.interpolator(params['teff'], params['logg']) # flux in physical units

        #convert to SI units
        if self.assume_mrr:
            radius = self.interpolator.radius_interp(params['teff'], params['logg']) * radius_sun
        else:            
            radius = params['radius'] * radius_sun # Rsun to meter
        distance = params['distance'] * pc_to_m # Parsec to meter

        return (radius / distance)**2 * fl # scale down flux by distance

    def residual(self, params, obs_flux = None, e_obs_flux = None):
        # calculate the chi2 between the model and the fit
        model_flux = self.get_model_flux(params)
        chisquare = ((model_flux - obs_flux) / e_obs_flux)**2

        return chisquare

    def __call__(self, obs_mag, e_obs_mag, distance, p0 = [], method = 'leastsq', **fit_kws):    
        obs_flux, e_obs_flux = self.mag_to_flux(obs_mag,  e_obs_mag) # convert magnitudes to fluxes

        # if an initial guess is not specified, set it to the mean of the parameter range
        if len(p0) == 0:
            p0 = [np.average(self.teff_lims), np.average(self.logg_lims), 0.001]
        
        # initialize the parameter object
        params = lmfit.Parameters()
        params.add('teff', value = p0[0], min = self.teff_lims[0], max = self.teff_lims[1], vary = True)
        params.add('distance', value = distance, vary = False)

        if self.assume_mrr:
            # if we're assuming an MRR, we only need to vary logg
            params.add('logg', value = p0[1], min = self.logg_lims[0], max = self.logg_lims[1], vary = True)
        else:
            # if not assume_mrr, initialize a specific radius variable
            params.add('logg', value = p0[1], min = self.logg_lims[0], max = self.logg_lims[1], vary = self.vary_logg)
            params.add('radius', value = p0[2], min = 0.000001, max = 0.1, vary = True)

        # run the fit with the defined parameters
        result = lmfit.minimize(self.residual, params, kws = dict(obs_flux = obs_flux, e_obs_flux = e_obs_flux), method = method, **fit_kws)

        # save the variables that are the same for all versions
        teff = result.params['teff'].value
        e_teff = result.params['teff'].stderr
        logg = result.params['logg'].value
        e_logg = result.params['logg'].stderr

        if self.assume_mrr:
            radius = self.interpolator.radius_interp(teff, logg)
            e_radius = None
        else:
            radius = result.params['radius'].value
            e_radius = result.params['radius'].stderr            

        return radius, e_radius, teff, e_teff, logg, e_logg, result

class MCMCEngine():
    def __init__(self, interpolator):
        self.interpolator = interpolator
        self.bands = interpolator.bands
        self.teff_lims = interpolator.teff_lims
        self.logg_lims = interpolator.logg_lims

        lib = pyphot.get_library()
        self.filters = [lib[band] for band in self.bands]

    def mag_to_flux(self, mag, e_mag = None):
        """
        convert from magntiudes on the AB system to flux for a particular filter (Gaia magnitudes are on the Vega system)
        
        Args:
            mag (array[N]):    magnitude of the observation
            e_mag (array[N]):  magnitude uncertainty
        Returns:
            flux in flam if e_mag is not specified, a tuple containing flux and flux uncertainty if it is
        """
        if e_mag is not None:
            flux = np.array([10**(-0.4*(mag[i] + filter.Vega_zero_mag)) for i, filter in enumerate(self.filters)])
            e_flux = np.array([np.abs((np.log(10)*(-0.4)*10**(-0.4 * (mag[i] + filter.Vega_zero_mag)) * e_mag[i])) for i, filter in enumerate(self.filters)])
            return flux, e_flux
        else:
            flux = np.array([10**(-0.4*(mag[i] + filter.Vega_zero_mag)) for i, filter in enumerate(self.filters)])
            return flux

    def get_model_flux(self, params):
        #get model photometric flux for a WD with a given radius, located a given distance away        
        fl= 4 * np.pi * self.interpolator(params[0], 9) # flux in physical units
        #convert to SI units
        radius = params[1] * radius_sun # Rsun to meter
        distance = self.distance * pc_to_m # Parsec to meter
        return (radius / distance)**2 * fl # scale down flux by distance

    def log_prob(self, params):
        """
        Compute log probs for a set of parameters
        Inputs:
            params      -   lmfit parameters containing teff, logg, radius
        Outputs:
            log probability of the set of parameters
        """
        def log_likelihood(params):
            flux_model =  self.get_model_flux(params) # compute model fluxes
            return np.sum(-0.5*np.square(((self.fluxes - flux_model) / self.e_fluxes)) - np.log(np.sqrt(2*np.pi) * self.e_fluxes)) # convert to log likelihood
        
        def log_prior(params):
            teff, radius = params[0], params[1] # fetch the teff and logg values
            # ensure that teff and logg are within the bounds of possiblity
            #if (self.teff_lims[0] <= teff <= self.teff_lims[1]) and (self.logg_lims[0] <= logg <= self.logg_lims[1]) and (0 < radius < 1):
            if (self.teff_lims[0] <= teff <= self.teff_lims[1]) and (0 < radius < 1):
                return 0.0
            else:
                return -np.inf
            
        lp = log_prior(params)
        ll = log_likelihood(params)
        if not np.isfinite(lp) or not np.isfinite(ll):
            return -np.inf
        return lp + ll
    
    def run_mcmc(self, mags, e_mags, distance, initial_guess):
        """
        Main MCMC run function using emcee
        Inputs:
            fluxes      -   observed photometric fluxes [np array]
            e_fluxes    -   uncertainties in photometric fluxes [np array]
            initial_guess - [initial_teff, initial_logg, initial_radius]
        """
        self.distance = distance
        self.fluxes, self.e_fluxes = self.mag_to_flux(mags, e_mags)

        # first, run 2500 steps of MCMC to understand how much we actually need to run
        nsteps = 2500
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                init_pos = initial_guess + 1e-2*np.random.randn(50,2)*initial_guess # intialize postions of walkers
                nwalkers, ndim = init_pos.shape
                sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_prob)#, moves = [emcee.moves.StretchMove(a=1.75)])
                sampler.run_mcmc(init_pos, nsteps, progress = False) # run x steps of mcmc
            except:
                init_pos = initial_guess + 1e-2*np.random.randn(50,2)*initial_guess # intialize postions of walkers
                nwalkers, ndim = init_pos.shape
                sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_prob)#, moves = [emcee.moves.StretchMove(a=1.75)])
                sampler.run_mcmc(init_pos, nsteps, progress = False) # run x steps of mcmc
            
            auto_corr_time = np.max(sampler.get_autocorr_time(quiet = True)) # get amount of steps for burn-in so we know how many steps we should run
            print("Auto-Correlation Time = {}, additional steps = {}".format(auto_corr_time, int(52*auto_corr_time) - nsteps))
            
            if np.isfinite(auto_corr_time):
                if nsteps <= int(52*auto_corr_time):
                    sampler.run_mcmc(None, int(52*auto_corr_time) - nsteps, progress = True)

                flat_samples = sampler.get_chain(discard = int(3*auto_corr_time), thin = int(0.5*auto_corr_time), flat = True)
            else:
                flat_samples = sampler.get_chain(discard = 0, flat = True)

        self.distance, self.fluxes, self.e_fluxes = None, None, None
        return flat_samples
