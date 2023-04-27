from context import *

class TestCautun(unittest.TestCase):
    def setUp(self, cos_model=Planck18, plot_results=False):
        self.cos_model = cos_model
        self.distribution = distributions.MassDistributionCautun(cos_model)
        
        self.files = ['Cautun00.csv', 'Cautun05.csv', 'Cautun10.csv', 'Cautun15.csv', 'Cautun20.csv']
        self.fit_z = [0, 0.5, 1, 1.5, 2]
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')

        self.plot_results = plot_results

    def test_distribution(self):
        h = self.cos_model.h

        # Load data from Cautun Figure 54
        for i in range(len(self.files)):
            data = pd.read_csv(os.path.join(self.data_dir, self.files[i]))
            z = self.fit_z[i]
            M = data['M'].values*u.M_sun # in units of h**-1
            dndM_cautun = data['dndM'].values*u.Gpc**-3 # in units of h**-3

            # Note mass distribution is corrected by M*ln(10) because Cautun plots dn/dlog10(M)
            dndM_calculated = self.distribution.getMassDistribution(z, M/h)*(M/h*np.log(10)/h**3)

            self.assertTrue(verifyOoM(\
                dndM_calculated, dndM_cautun),\
                        'Failed order-of-magnitude verification on ' + self.files[i]) 
            
            if self.plot_results:
                plt.figure()
                plt.plot(M.to(u.M_sun), dndM_calculated.to(u.Gpc**-3), 'b-', label='Calculated')
                plt.plot(M.to(u.M_sun), dndM_cautun.to(u.Gpc**-3), 'k:', label="Cautun Data")
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel('Filament Mass $M_f$ $[h^{-1}M_\odot]$')
                plt.ylabel('$dn/dlog_{10}M_f$  $[h^{3}Gpc^{-3}]$')
                plt.legend()
                plt.title('z = ' + str(self.fit_z[i]))
            
        # Show all plots when finished
        plt.show()

if __name__ == '__main__':
    # Run with plots
    test = TestCautun()
    test.setUp(plot_results=True)
    test.test_distribution()