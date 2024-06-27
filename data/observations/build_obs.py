import numpy as np
import glob
from astropy.io import fits
import corv

class Spectrum:
    def __init__(self, wave, flux, ivar, mask = None):
        self.wave = wave
        self.flux = flux
        self.ivar = ivar
        self.mask = mask

    def calculate_rv(self, model, **kwargs):
        # compute radial velocity
        rv, e_rv, redchi, param_res = corv.fit.fit_rv(self.wave, self.flux, self.ivar, model, **kwargs)

        # save the variables into the object
        self.rv = rv
        self.e_rv = e_rv
        self.rv_redchi = redchi

    def fit_rv(self, model):
        rv, e_rv, redchi, param_res = corv.fit.fit_corv(self.wave, self.flux, self.ivar, model)

        self.rv = rv # radial velocity in km/s
        self.e_rv = e_rv # radial velocity uncertainty
        self.rv_redchi = redchi # reduced chi2 of the fit
        self.rv_params = param_res # parameters of the fit

    def spectrum_coadd(self, other):
        # interpolate the other spectrum onto a consistent wavelength grid
        flux_rebase = np.interp(self.wave, other.flux, other.ivar)
        ivar_rebase = np.interp(self.wave, other.wave, other.ivar)

        # create holder objects for fluxes and ivars
        fls = np.array([self.flux, flux_rebase])
        ivars = np.array([self.ivar, ivar_rebase])

        # perform the coadd
        mask = (ivars == 0) # create a boolean mask of ivar = 0
        ivars[mask] = 1 # set those points to 1 to avoid divide by zero
        variances = ivars**-1 # compute variances of both spectra
        variance = np.sum(variances, axis = 0) / len(variances)**2 # central limit theorem
        ivar = variance**-1 # put this back into ivar
        flux = np.median(fls, axis=0) # compute the median of each point
        smask = (mask).all(axis=0) # reduce the dimensionality of the mask
        ivar[smask] = 1e-6 # reset the points to something near zero
        
        # return a new object that is the co-added spectra
        return Spectrum(self.wave, flux, ivar, mask)

    # the + operation performs a coadd of two spectra
    def __add__(self, other):
        return self.spectrum_coadd(other)

    def __radd__(self, other):
        return self + other

    
class target:
    def __init__(self, name):
        """
        assume that the file structure for n coadds is is something like:
        name/
            name_coadd_1
            ...
            name_coadd_n
        """
        self.name = name


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    model = corv.models.make_balmer_model(nvoigt=2)

    # read in the observation data
    spec_file1 = fits.open('./j1657m5438/j1657m5438_coadd_1.fits') 
    spec_file2 = fits.open('./j1657m5438/j1657m5438_coadd_2.fits') 

    cl = spec_file1[1].data['wave'] > 0
    wave1 = spec_file1[1].data['wave'][cl]
    flux1 = spec_file1[1].data['flux'][cl]
    ivar1 = spec_file1[1].data['ivar'][cl]
    mask1 = spec_file1[1].data['mask'][cl]

    cl = spec_file2[1].data['wave'] > 0
    wave2 = spec_file2[1].data['wave'][cl]
    flux2 = spec_file2[1].data['flux'][cl]
    ivar2 = spec_file2[1].data['ivar'][cl]
    mask2 = spec_file2[1].data['mask'][cl]

    spec1 = Spectrum(wave1, flux1, ivar1, mask1)
    spec2 = Spectrum(wave2, flux2, ivar2, mask2)

    plt.plot(spec1.wave, spec1.flux)
    plt.ylim(0,10)
    plt.show()

    spec3 = spec1 + spec2
    spec3.fit_rv(model)
    print(spec3.rv, spec3.e_rv)

    plt.plot(spec3.wave, spec3.flux)
    plt.ylim(0,10)
    plt.show()