import pyvo as vo
from astropy.table import Table, join, vstack

import numpy as np
import pickle

from .interpolator import MCMCEngine, WarwickDAInterpolator, LaPlataUltramassive, LaPlataBase


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
    gaia_query = """
        SELECT
        gs.source_id, gs.original_ext_source_id
        FROM gaiadr3.panstarrs1_best_neighbour as gs
        WHERE gs.source_id in {}
        """.format(tuple(source_ids))
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

    return photometry

class Photometry:
    def __init__(self, source_ids, distances, astrometric_params_solved, gaia_photo, e_gaia_photo, initial_guesses):
        panstarrs = fetch_photometry(source_ids)

        self.photo = {}
        for ii, id in enumerate(source_ids):
            ix = np.where(panstarrs['source_id'] == id)[0]
            bands = np.array(['Gaia_G', 'Gaia_BP', 'Gaia_RP'])
            photo = np.array(gaia_photo[ii])
            e_photo = np.array(e_gaia_photo[ii])
        
            bands = np.append(bands, np.array(['PS1_g', 'PS1_r', 'PS1_i', 'PS1_z', 'PS1_y']))
            photo = np.append(photo, np.array([panstarrs['PS1_g'][ix], panstarrs['PS1_r'][ix], panstarrs['PS1_i'][ix], panstarrs['PS1_z'][ix], panstarrs['PS1_y'][ix]]))
            e_photo = np.append(e_photo, np.array([panstarrs['e_PS1_g'][ix], panstarrs['e_PS1_r'][ix], panstarrs['e_PS1_i'][ix], panstarrs['e_PS1_z'][ix], panstarrs['e_PS1_y'][ix]]))

            self.photo[id] = [distances[ii], astrometric_params_solved[ii], bands, photo, e_photo, initial_guesses[ii]]

        self.check_valid()

    def check_valid(self):
        for key in self.photo.keys():
            # find and delete invalid PanSTARRS photometry
            invalid = np.where(self.photo[key][1] == -999)
            for i in range(len(self.photo[key])):
                self.photo[key][i] = np.delete(self.photo[key][i], invalid)

    def run_mcmc(self, interpolator, **kwargs):
        chains = {}
        for id in self.photo.keys():
            interp = interpolator(self.bands, **kwargs)
            engine = MCMCEngine(interp)

            chain = engine.run_mcmc(self.photo, self.e_photo, self.distance, initial_params)
            chains[id] = chain
        return chains

    def write(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.photo, f)




