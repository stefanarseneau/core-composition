from .interpolator import MassiveLaPlataInterpolator
import numpy as np
import argparse

class LaPlataTests:
    def __init__(self, basepath = './data/laplata'):
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
                interp = MassiveLaPlataInterpolator(core, layer, z, bands = ['Gaia_G', 'Gaia_BP', 'Gaia_RP'])

                for i in range(len(self.teffs)):
                    for j in range(len(self.loggs)):
                        self.grid[i,j] = interp(self.teffs[i], self.loggs[j])

                grids[z] = (interp.interp, self.grid)
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
                    interp = MassiveLaPlataInterpolator(core, layer, z, bands = ['Gaia_G', 'Gaia_BP', 'Gaia_RP'])

                    for i in range(len(self.teffs)):
                        for j in range(len(self.loggs)):
                            self.grid[i,j] = interp(self.teffs[i], self.loggs[j])

                    grids[layer] = (interp.interp, self.grid)
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
                interp = MassiveLaPlataInterpolator(core, layer, z, bands = ['Gaia_G', 'Gaia_BP', 'Gaia_RP'])

                for i in range(len(self.teffs)):
                    for j in range(len(self.loggs)):
                        self.grid[i,j] = interp(self.teffs[i], self.loggs[j])

                grids[core] = (interp.interp, self.grid)
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

    tester = LaPlataTests(args.path)
    tester.run_tests(args.test_metallicities, args.test_hlayers, args.test_cores)