from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table, vstack

from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
from scipy.interpolate import griddata, interp1d


def build_interpolator(table):
    def interpolate_2d(x, y, z, method):
        if method == 'linear':
            interpolator = LinearNDInterpolator
        elif method == 'cubic':
            interpolator = CloughTocher2DInterpolator
        return interpolator((x, y), z, rescale=True)
        #return interp2d(x, y, z, kind=method)

    def interp(x, y, z, interp_type_atm='linear'):
            grid_z      = griddata(np.array((x, y)).T, z, (grid_x, grid_y), method=interp_type_atm)
            z_func      = interpolate_2d(x, y, z, interp_type_atm)
            return grid_z, z_func

    logteff_logg_grid=(3500, 80000, 1000, 8.77, 9.3, 0.01)
    grid_x, grid_y = np.mgrid[logteff_logg_grid[0]:logteff_logg_grid[1]:logteff_logg_grid[2],
                                logteff_logg_grid[3]:logteff_logg_grid[4]:logteff_logg_grid[5]]

    mask = table['teff'] > logteff_logg_grid[0]
    table = table[mask]

    grid_g, g_func = interp(table['teff'], table['logg'], table['Gaia_G'])
    grid_bp, bp_func = interp(table['teff'], table['logg'], table['Gaia_BP'])
    grid_rp, rp_func = interp(table['teff'], table['logg'], table['Gaia_RP'])

    photometry = lambda teff, logg: np.array([g_func(teff, logg), bp_func(teff, logg), rp_func(teff, logg)])
    key = {'Gaia_G' : 0, 'Gaia_BP' : 1, 'Gaia_RP' : 2}
    return photometry, key

def build_table_over_mass(core, layer, z, basepath = './'):
    masses = ['110', '116', '123', '129'] # solar masses (e.g. 1.10 Msun)
    table = Table()

    for mass in masses:
        # read in the table from the correct file
        path = f'{basepath}/{core}_{mass}_{layer}_{z}.dat'
        temp_table = Table.read(path, format='ascii')

        # put the columns into a pyphot-readable standard
        temp_table.rename_columns(['Teff', 'logg(CGS)'], ['teff', 'logg'])
        temp_table.remove_columns(['g_1', 'r_1', 'i_1', 'z_1', 'y', 'U', 'B',\
                            'V', 'R', 'I', 'J', 'H', 'K', 'FUV', 'NUV'])
        temp_table.rename_columns(['G3', 'Bp3', 'Rp3', 'u', 'g', 'r', 'i', 'z' ],
                                ['Gaia_G_mag', 'Gaia_BP_mag', 'Gaia_RP_mag', 'SDSS_u_mag', 'SDSS_g_mag', 'SDSS_r_mag', 'SDSS_i_mag', 'SDSS_z_mag'])
        
        # now convert from absolute magnitude to surface flux
        temp_table['Gaia_G'] = (4*np.pi)**-1 * 10**(-0.4*(temp_table['Gaia_G_mag'] + 21.50763)) * ((10*3.086775e16) / (6.957e8*temp_table['R/R_sun']))**2
        temp_table['Gaia_BP'] = (4*np.pi)**-1 * 10**(-0.4*(temp_table['Gaia_BP_mag'] + 20.97231)) * ((10*3.086775e16) / (6.957e8*temp_table['R/R_sun']))**2
        temp_table['Gaia_RP'] = (4*np.pi)**-1 * 10**(-0.4*(temp_table['Gaia_RP_mag'] + 22.26331)) * ((10*3.086775e16) / (6.957e8*temp_table['R/R_sun']))**2
        
        # stack the tables
        table = vstack([table, temp_table])
    return table


class Tests:
    def __init__(self, basepath):
        self.basepath = basepath
        self.cores = ['CO', 'ONe'] # La Plata core models
        self.Hlayers = ['Hrich', 'Hdef'] # available H layer thicknesses
        self.zs = ['0_001', '0_02', '0_04', '0_06'] # metallicities

        self.teffs = np.arange(4000, 73000, 500)
        self.loggs = np.arange(8.85, 9.35, 0.01)

        self.reset_grid()

    def reset_grid(self):
        self.grid = np.zeros((len(self.teffs), len(self.loggs), 3))

    def test_metallicity(self):
        """
        Holding all other variables constant, understand the effect of different metallicities
        """

        print('TEST METALLICITY VARIATION')

        # ONe have no z variation, so only test CO
        core = self.cores[0]

        for layer in self.Hlayers:
            print(f'CORE: {core} | H-LAYER: {layer} ========================')
            grids = {}
            
            for z in self.zs:
                table = build_table_over_mass(core, layer, z, basepath = self.basepath)
                photometry, key = build_interpolator(table)

                for i in range(len(self.teffs)):
                    for j in range(len(self.loggs)):
                        self.grid[i,j] = photometry(self.teffs[i], self.loggs[j])

                grids[z] = (photometry, self.grid)
                self.reset_grid()
            self.print_test(grids)

    def test_hlayers(self):
        """
        Holding all other variables constant, understand the effect of different H layers
        """

        print('TEST HLAYER VARIATION')

        for core in self.cores:
            for z in self.zs:
                if (core == 'ONe') and (z != '0_02'):
                    continue

                print(f'CORE: {core} | Z: {z} ========================')
                grids = {}
                
                for layer in self.Hlayers:
                    table = build_table_over_mass(core, layer, z, basepath = self.basepath)
                    photometry, key = build_interpolator(table)

                    for i in range(len(self.teffs)):
                        for j in range(len(self.loggs)):
                            self.grid[i,j] = photometry(self.teffs[i], self.loggs[j])

                    grids[layer] = (photometry, self.grid)
                    self.reset_grid()
                self.print_test(grids)


    def test_cores(self):
        """
        Holding all other variables constant, understand the effect of different H layers
        """

        print('TEST CORE VARIATION')

        z = self.zs[1]

        for layer in self.Hlayers:

            print(f'HLAYER: {layer} | Z: {z} ========================')
            grids = {}
            
            for core in self.cores:
                table = build_table_over_mass(core, layer, z, basepath = self.basepath)
                photometry, key = build_interpolator(table)

                for i in range(len(self.teffs)):
                    for j in range(len(self.loggs)):
                        self.grid[i,j] = photometry(self.teffs[i], self.loggs[j])

                grids[core] = (photometry, self.grid)
                self.reset_grid()
            self.print_test(grids)

    def print_test(self, grids):
        for key1 in grids.keys():
            for key2 in grids.keys():
                if key1 != key2:
                    print(f'    {key1} and {key2}:')
                    mean = np.nanmean(np.abs(grids[key1][1] - grids[key2][1]) / np.nanmean(grids[key1][1]))*100
                    median = np.nanmedian(np.abs(grids[key1][1] - grids[key2][1]) / np.nanmedian(grids[key1][1]))*100
                    print(f'    mean absolute difference: {mean:2.2f}%')
                    print(f'    median absolute difference: {median:2.2f}%\n')

    def run_tests(self, metallicity, layers, core_comps):
        if metallicity:
            self.test_metallicity()
        if layers:
            self.test_hlayers()
        if core_comps:
            self.test_cores()
            
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-metallicities", action="store_true")
    parser.add_argument("--test-hlayers", action="store_true")
    parser.add_argument("--test-cores", action="store_true")
    parser.add_argument("path")

    args = parser.parse_args()

    tester = Tests(args.path)
    tester.run_tests(args.test_metallicities, args.test_hlayers, args.test_cores)