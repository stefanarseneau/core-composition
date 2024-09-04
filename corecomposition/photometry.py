import pyvo as vo
from astropy.table import Table, join, vstack

import numpy as np
from tqdm import tqdm
import pickle

import pyphot
from .interpolator import MCMCEngine, WarwickDAInterpolator, LaPlataUltramassive, LaPlataBase
from .interpolator.utils import deredden

def gaia_to_ab(photo, gaia_flux, e_gaia_flux):
    mags = photo
    e_mags = np.array([gaia_flux[i] / (1.09 * e_gaia_flux[i]) for i in range(len(gaia_flux))])
    return mags, e_mags

def panstarrs_to_vega(photo, e_photo):
    # AB zero fluxes in W m^-2 Hz^-1 (10^26 Jy)
    pstarr_0 = [3631e-26]*5
    pivot = [4849.117, 6201.186, 7534.956, 8674.179, 9627.741] # grizy
    bands = ['PS1_g', 'PS1_r', 'PS1_i', 'PS1_z', 'PS1_y']
    
    lib = pyphot.get_library()
    filters = [lib[band] for band in bands]
    
    # https://iopscience.iop.org/article/10.1088/0004-637X/750/2/99/pdf
    # convert reported mags to fluxes in 1e-3 erg s^-1 cm^-2 Hz^-1 = W m^2 Hz^-1
    calibrated_flux = [(np.power(10, (photo[i] + 48.6)/-2.5) * 1e-3) 
                            if (photo[i] != -999) else -999 for i in range(len(pstarr_0))]
    e_calibrated_flux = [calibrated_flux[i] * e_photo[i] 
                            if photo[i] != -999 else -999 for i in range(len(pstarr_0))]

    # converts from W m^-2 Hz^-1 to flam using pyphot's pivots
    flam = [(2.99792458e21 * calibrated_flux[i] / pivot[i]**2) 
                if (photo[i] != -999) else -999 for i in range(len(calibrated_flux))]
    e_flam = [(2.99792458e21 * e_calibrated_flux[i] / pivot[i]**2) 
                if (photo[i] != -999) else -999 for i in range(len(e_calibrated_flux))]  

    # converts to mag
    mags = np.array([(-2.5 * np.log10(flam[i]) - filters[i].Vega_zero_mag) 
                if (photo[i] != -999) else -999 for i in range(len(flam))])
    e_mags = np.array([(e_flam[i] /(1.09 * flam[i])) 
                if (photo[i] != -999) else -999 for i in range(len(flam))])
    return mags, e_mags 


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

def des_to_ps1(bands, e_bands = None):
    # bands should be formatted as grizy
    # https://des.ncsa.illinois.edu/releases/dr2/dr2-docs/dr2-transformations
    conversion = lambda bands: np.array([bands[0] - 0.026 * (bands[0] - bands[2]) - 0.020,
                                        bands[1] + 0.139 * (bands[1] - bands[2]) + 0.014,
                                        bands[2] + 0.153 * (bands[1] - bands[2]) - 0.011,
                                        bands[3] + 0.112 * (bands[1] - bands[2]) + 0.013,
                                        bands[4] + 0.031 * (bands[1] - bands[2]) - 0.034])
    e_conversion = lambda bands: np.array([e_bands[0] + 0.026 * (e_bands[0] + e_bands[2]),
                                        e_bands[1] + 0.139 * (e_bands[1] + e_bands[2]),
                                        e_bands[2] + 0.153 * (e_bands[1] + e_bands[2]),
                                        e_bands[3] + 0.112 * (e_bands[1] + e_bands[2]),
                                        e_bands[4] + 0.031 * (e_bands[1] + e_bands[2])])
    if e_bands is not None:
        return conversion(bands), e_conversion(e_bands)
    else:
        return conversion(bands)


def des_photo(source_ids, convert = False):
    noir_tap_service = vo.dal.TAPService(" https://datalab.noirlab.edu/tap")
    des_query = """
        SELECT
        gs.id2 as source_id, des.mag_auto_g, des.mag_auto_r, des.mag_auto_i, des.mag_auto_z, des.mag_auto_y, 
        des.magerr_auto_g, des.magerr_auto_r, des.magerr_auto_i, des.magerr_auto_z, des.magerr_auto_y
        FROM des_dr2.x1p5__main__gaia_dr3__gaia_source as gs
        LEFT JOIN des_dr2.mag as des ON des.coadd_object_id = gs.id1
        WHERE gs.id2 in {}
        """.format(tuple(source_ids))
    des_result = noir_tap_service.search(des_query).to_table()

    if convert:
        photo = np.array([des_result['mag_auto_g'], des_result['mag_auto_r'], des_result['mag_auto_i'], 
                            des_result['mag_auto_z'], des_result['mag_auto_y']])
        e_photo = np.array([des_result['magerr_auto_g'], des_result['magerr_auto_r'], des_result['magerr_auto_i'], 
                            des_result['magerr_auto_z'], des_result['magerr_auto_y']])
        panstarrs, e_panstarrs = des_to_ps1(photo, e_photo)

        des_result['PS1_g'] = panstarrs[0]
        des_result['PS1_r'] = panstarrs[1]
        des_result['PS1_i'] = panstarrs[2]
        des_result['PS1_z'] = panstarrs[3]
        des_result['PS1_y'] = panstarrs[4]

        des_result['e_PS1_g'] = e_panstarrs[0]
        des_result['e_PS1_r'] = e_panstarrs[1]
        des_result['e_PS1_i'] = e_panstarrs[2]
        des_result['e_PS1_z'] = e_panstarrs[3]
        des_result['e_PS1_y'] = e_panstarrs[4]

        des_result.remove_columns(['mag_auto_g', 'mag_auto_r', 'mag_auto_i', 'mag_auto_z', 'mag_auto_y',
                                    'magerr_auto_g', 'magerr_auto_r', 'magerr_auto_i', 'magerr_auto_z', 'magerr_auto_y'])

    return des_result


def panstarrs_photo(source_ids):
    # first use the Gaia table to xmatch against panstarrs
    gaia_tap_service = vo.dal.TAPService("https://gea.esac.esa.int/tap-server/tap")
    gaia_query = f"""
        SELECT gs.source_id, gs.original_ext_source_id
        FROM gaiadr3.panstarrs1_best_neighbour as gs
        WHERE gs.source_id in {tuple(source_ids)}
        """
    gaia_result = gaia_tap_service.search(gaia_query).to_table()

    # then use the panstarrs ids to get the panstarrs data
    panstarrs_tap_service = vo.dal.TAPService('http://vao.stsci.edu/PS1DR2/tapservice.aspx')
    ps1_query = """select *
                    from dbo.MeanObjectView
                    where objID in {}""".format(tuple(gaia_result['original_ext_source_id']))
    panstarrs_photo = panstarrs_tap_service.search(ps1_query).to_table()
    panstarrs_photo = join(gaia_result, panstarrs_photo, keys_left='original_ext_source_id', keys_right='objID')
    panstarrs_photo = panstarrs_photo[['source_id', 'qualityFlag', 'gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag',
                                        'gMeanPSFMagErr', 'rMeanPSFMagErr', 'iMeanPSFMagErr', 'zMeanPSFMagErr', 'yMeanPSFMagErr']]

    # bitmasks for checking the quality of the source
    mask0 = 0x00000001 # the source is extended in PS
    mask1 = 0x00000004 # good-quality measurement in PS
    mask2 = 0x00000016 # good-quality object in the stack 
    mask3 = 0x00000020 # ghe primary stack measurements are the best
    mask = np.all([~(panstarrs_photo['qualityFlag'] & mask0), panstarrs_photo['qualityFlag'] & mask1,
                    panstarrs_photo['qualityFlag'] & mask2, panstarrs_photo['qualityFlag'] & mask3], axis=0)

    # change the column names to pyphot standards
    panstarrs_photo.rename_columns(['gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag',
                                    'gMeanPSFMagErr', 'rMeanPSFMagErr', 'iMeanPSFMagErr', 'zMeanPSFMagErr', 'yMeanPSFMagErr'],
                                    ['PS1_g', 'PS1_r', 'PS1_i', 'PS1_z', 'PS1_y', 'e_PS1_g', 'e_PS1_r', 'e_PS1_i', 'e_PS1_z', 'e_PS1_y'])
    return panstarrs_photo[mask]

def fetch_photometry(source_ids):
    try:
        des_photometry = des_photo(source_ids, convert = True)
        des_photometry['source'] = 'des'
        missing_ids = [id for id in source_ids if id not in des_photometry['source_id']]
    except:
        des_photometry = None
        missing_ids = source_ids

    panstarrs_photometry = panstarrs_photo(missing_ids)
    panstarrs_photometry['source'] = 'ps1'

    if des_photometry is not None:
        photometry = vstack([des_photometry, panstarrs_photometry])
    else:
        photometry = panstarrs_photometry.copy()

    ### Convert PanSTARRS photo to Vega
    for row in photometry:
        photo = np.array([row['PS1_g'], row['PS1_r'], row['PS1_i'], row['PS1_z'], row['PS1_y']])
        e_photo = np.array([row['e_PS1_g'], row['e_PS1_r'], row['e_PS1_i'], row['e_PS1_z'], row['e_PS1_y']])
        vega, e_vega = panstarrs_to_vega(photo, e_photo)

        row['PS1_g'] = vega[0]
        row['PS1_r'] = vega[1]
        row['PS1_i'] = vega[2]
        row['PS1_z'] = vega[3]
        row['PS1_y'] = vega[4]

        row['e_PS1_g'] = e_vega[0]
        row['e_PS1_r'] = e_vega[1]
        row['e_PS1_i'] = e_vega[2]
        row['e_PS1_z'] = e_vega[3]
        row['e_PS1_y'] = e_vega[4]

    return photometry

class Photometry:
    def __init__(self, source_ids, geometry, astrometric_params_solved, gaia_photo, e_gaia_photo, initial_guesses, bsq = None):
        panstarrs = fetch_photometry(source_ids)

        self.source_ids = source_ids
        self.geometry = geometry
        self.astrometric_params_solved = astrometric_params_solved
        self.initial_guess = initial_guesses

        self.bands = []
        self.photometry = []
        self.e_photometry = []
        
        for ii in tqdm(range(len(self.source_ids))):
            ix = np.where(panstarrs['source_id'] == self.source_ids[ii])[0]
            band = np.array(['Gaia_G', 'Gaia_BP', 'Gaia_RP'])
            photo = gaia_photo[ii]
            e_photo = e_gaia_photo[ii]

            # correct for g-band magnitude
            photo[0] = correct_gband(photo[1], photo[2], self.astrometric_params_solved[ii], photo[0])
        
            # add the extra photometry
            band = np.append(band, np.array(['PS1_g', 'PS1_r', 'PS1_i', 'PS1_z', 'PS1_y']))
            photo = np.append(photo, np.array([panstarrs['PS1_g'][ix], 
                                        panstarrs['PS1_r'][ix], 
                                        panstarrs['PS1_i'][ix], 
                                        panstarrs['PS1_z'][ix], 
                                        panstarrs['PS1_y'][ix]]))
            e_photo = np.append(e_photo, np.array([panstarrs['e_PS1_g'][ix], 
                                        panstarrs['e_PS1_r'][ix], 
                                        panstarrs['e_PS1_i'][ix], 
                                        panstarrs['e_PS1_z'][ix], 
                                        panstarrs['e_PS1_y'][ix]]))

            # check valid photometry
            invalid = np.where(photo < -50)
            band = np.delete(band, invalid)
            photo = np.delete(photo, invalid)
            e_photo = np.delete(e_photo, invalid)

            # optionally deredden
            if bsq is not None:
                photo = deredden(bsq, self.geometry[ii], photo, band)

            # create a list
            self.bands.append(band)
            self.photometry.append(photo)
            self.e_photometry.append(e_photo)            

    def run_mcmc(self, interpolator, **kwargs):
        chains = {}
        for ii in tqdm(range(len(self.source_ids))):
            interp = interpolator(self.bands[ii], **kwargs)
            engine = MCMCEngine(interp)

            chain = engine.run_mcmc(self.photometry[ii], self.e_photometry[ii], self.geometry[ii].distance.value, self.initial_guess[ii])
            chains[self.source_ids[ii]] = chain
        return chains

    def write(self, path):
        photo = {}
        for ii, id in enumerate(self.source_ids):
            photo[id] = [self.geometry[ii], self.astrometric_params_solved[ii], self.bands[ii], self.photo[ii], self.e_photo[ii], self.initial_guesses[ii]]

        with open(path, 'wb') as f:
            pickle.dump(self.photo, f)




