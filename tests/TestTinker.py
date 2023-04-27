from context import *

class TestTinker(unittest.TestCase):
    def setUp(self, cos_model=Planck18, plot_results=False):
        self.cos_model = cos_model

        self.files = ['Tinker200.csv', 'Tinker800.csv', 'Tinker3200.csv']
        self.deltas = [200, 800, 3200]
        
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.plot_results = plot_results

    def test_sigma_fit(self):
        # Test implemented sigma fit from Tinker (linear fit from Figure 5) and Lopez-Honorez
        sigma_fit_tinker = distributions.SigmaFitTinker()
        sigma_fit_LH = distributions.SigmaFitLopezHonorez(self.cos_model)

        z = 0
        logM = np.array([10, 11, 12, 13, 14, 15, 16], dtype='float')
        logSigmaTinker = np.array([-0.64, -0.52, -0.38, -0.21, -0.01, 0.24, 0.55])

        h = self.cos_model.h
        M = 10**(logM)*h*u.M_sun
        sigmaTinker = 1/(10**logSigmaTinker)

        self.assertTrue(verifyOoM(sigma_fit_tinker.getSigma(z, M), sigmaTinker, 0.2),\
                        'SigmaFitTinker failed order-of-magnitude check')
        self.assertTrue(verifyOoM(sigma_fit_LH.getSigma(z, M), sigmaTinker, 0.1),\
                        'SigmaFitLopezHonorez failed order-of-magnitude check')

        if self.plot_results:
            plt.figure()
            plt.plot(M/h, sigmaTinker, 'b.', label='Tinker Values')
            plt.plot(M/h, sigma_fit_tinker.getSigma(z, M), 'b:', label='Linear Tinker Fit')
            plt.plot(M/h, sigma_fit_LH.getSigma(z, M), 'k:', label='Lopez-Honorez Fit')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$M/(h^{-1}M_{\odot})$')
            plt.ylabel('$\sigma$')
            plt.legend()
            plt.title('Sigma Fits')
            plt.show()

    def test_distribution(self):
        sigma_fit = distributions.SigmaFitLopezHonorez(self.cos_model)

        for i in range(len(self.files)):
            z = 0
            h = self.cos_model.h
            mean_density = self.cos_model.critical_density(z)*self.cos_model.Om(z)

            # Load data from Tinker Figure 5
            data = pd.read_csv(os.path.join(self.data_dir, self.files[i]))
            delta = self.deltas[i]
            M = data['M'].values*h*u.M_sun
            dndM_tinker = data['dndM'].values*mean_density/M**2

            # Calculate and verify distribution
            distribution = distributions.MassDistributionTinker(self.cos_model, sigma_fit, delta)
            dndM_calculated = distribution.getMassDistribution(z, M)

            self.assertTrue(verifyOoM(dndM_calculated, dndM_tinker),\
                            'Failed order-of-magnitude verification on ' + self.files[i])
            
            if self.plot_results:
                plt.figure()
                plt.plot(M/h, dndM_calculated*(M**2)/mean_density, 'b.', label='Calculated')
                plt.plot(M/h, dndM_tinker*(M**2)/mean_density, 'k:', label='Tinker Data')
                plt.xlabel(r'$M/(h^{-1}M_{\odot})$')
                plt.ylabel(r'M$^2\rho_m$ dn/dM')
                plt.xscale('log')
                plt.yscale('log')
                plt.ylim(10**-4, 10**-1)
                plt.title('Tinker Mass Distribution\n$\Delta$ = '+str(delta))
                plt.legend()
                # Results similar to Tinker Fig. 5

        # Show all plots when finished
        plt.show()

if __name__=='__main__':
    # Run with plots
    test = TestTinker()
    test.setUp(plot_results=True)
    test.test_sigma_fit()
    test.test_distribution()