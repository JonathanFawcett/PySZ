from context import *

class TestCosmology(unittest.TestCase):
    def setUp(self, plot_results=False):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.plot_results = plot_results

    def test_cosmology(self):
        # Constants from https://ned.ipac.caltech.edu/level5/Sept05/Carlstrom/Carlstrom4.html
        cos_model = cosmology.LambdaCDM(
            H0 = 42 << (u.km*u.s**-1*u.Mpc**-1), # Hubble constant at z=0, km/sec/Mpc. Arbitrary for this test
            Om0 = 0.3, # Omega matter at z=0, fraction of critical density
            Ode0 = 0.7 # Omega dark energy at z=0, fraction of critical density
        )

        # Load data from Carlstrom
        data = pd.read_csv(os.path.join(self.data_dir, 'CarlstromComovingVolume.csv'))
        z = data['z'].values
        comov_vol_carlstrom = data['dV'].values/cos_model.h**3*u.Mpc**3

        # Calculate and verify differential comoving volume
        comov_vol_calculated = cos_model.differential_comoving_volume(z)
        self.assertTrue(verifyOoM(comov_vol_calculated, comov_vol_carlstrom, 0.2),\
                        'Differential comoving volume was not within 0.2 orders of magnitude of Carlstrom')
        
        # Plot results
        if self.plot_results:
            plt.figure()
            plt.plot(z, comov_vol_calculated.to(u.Mpc**3)*cos_model.h**3, 'b.', label='Calculated')
            plt.plot(z, comov_vol_carlstrom*cos_model.h**3, 'k:', label='Carlstrom Data')
            plt.xlabel('Redshift [-]')
            plt.ylabel(r'Differential Comoving Volume [$h^{-3}Mpc^{3}/sr$]')
            plt.yscale('log')
            plt.xlim(left = 0, right = 3)
            plt.ylim(bottom = 10**8)
            plt.title('Differential Comoving Volume, LCDM\n h = ' + str(cos_model.h) + ', $\Omega_M$ = ' + str(cos_model.Om0) + ', $\Omega_\Lambda$ = ' + str(cos_model.Ode0))
            plt.legend()
            plt.show()

if __name__ == '__main__':
    # Run with plots
    test = TestCosmology()
    test.setUp(plot_results=True)
    test.test_cosmology()