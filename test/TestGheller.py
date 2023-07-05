from context import *

class TestGheller(unittest.TestCase):
    def setUp(self, plot_results=False):
        self.files = ['Gheller00.csv', 'Gheller05.csv', 'Gheller10.csv']
        self.z = [0, 0.5, 1.0]
        
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.plot_results = plot_results

    def test_filament_length(self):
        filament_model = structures.FilamentModelGheller()

        # Load data from Gheller, Figure 4, right panel, 100 Mpc
        for i in range(len(self.files)):
            data = pd.read_csv(os.path.join(self.data_dir, self.files[i]))
            z = self.z[i]
            M = data['M'].values*u.M_sun
            L_gheller = data['L'].values*u.Mpc

            # Calculate and verify filament length
            L_calculated = filament_model.getFilamentLength(z, M)
            self.assertTrue(verifyOoM(L_calculated, L_gheller, 0.1),\
                            'Failed order-of-magnitude verification on ' + self.files[i])
            
            if self.plot_results:
                plt.figure()
                plt.plot(M, L_calculated, 'b-', label='Calculated')
                plt.plot(M, L_gheller, 'k:', label='Gheller Data')
                plt.xlabel(r'M/$M_{\odot}$')
                plt.ylabel(r'Filament Length [Mpc]')
                plt.title('Filament Length Scaling\n100Mpc Box, z = ' + str(z))
                plt.xscale('log')
                plt.yscale('log')
                plt.legend()

        # Show all plots when finished
        plt.show()

    def test_base_temperature(self):
        filament_model = structures.FilamentModelGheller()
        L = 15*u.Mpc

        # Calculate mass of 15 Mpc filament at z=0
        base_M = (L/u.Mpc/(10**filament_model.beta_L[0]))**(1/filament_model.alpha_L[0])*u.M_sun

        # Calculate base temperature using M-T fit from Gheller
        base_T = filament_model.logScaleByMass(filament_model.alpha_T[0], filament_model.beta_T[0], base_M)*u.K

        # Verify scaling is properly implemented by asserting scale factor is within 1e-6 at base mass and temperature
        self.assertTrue(abs(filament_model.getTemperatureScaling(0, base_M)\
                            - filament_model.getTemperatureScaling(0, base_M, base_T = base_T)) < 1e-6,\
                                'Base temperature is not within error')
        
if __name__ == '__main__':
    # Run with plots
    test = TestGheller()
    test.setUp(plot_results=True)
    test.test_filament_length()
    test.test_base_temperature()