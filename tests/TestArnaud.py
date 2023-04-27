from context import *

class TestArnaud(unittest.TestCase):
    def setUp(self, cos_model=cosmology.LambdaCDM(H0=70*u.km/u.s/u.Mpc, Om0=0.3, Ode0=0.7), plot_results=False):
        self.cos_model = cos_model
        self.cluster_model = structures.ClusterModelArnaud(cos_model)
        
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.plot_results = plot_results

    def test_universal_profile(self):
        ##* Test Universal Pressure Profile
        # Load data from Arnaud Figure 8, green line of best fit
        data = pd.read_csv(os.path.join(self.data_dir, 'ArnaudUniversalProfile.csv'))
        x = data['x'].values
        P_arnaud = data['P'].values
        P_calculated = self.cluster_model.universalProfile(x)
        self.assertTrue(verifyOoM(P_arnaud, P_calculated, 0.1),\
                        'Universal pressure profile was not within 0.1 orders of magnitude of Arnaud')

        if self.plot_results:
            plt.figure()
            plt.plot(x, P_calculated, 'b.', label='Calculated')
            plt.plot(x, P_arnaud, 'k:', label='Arnaud Data')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Radius (R$_{500}$)')
            plt.ylabel('P/P$_{500}$')
            plt.title('Arnaud Universal Pressure Profile')
            plt.legend()
            plt.show()
            # Results similar to Arnaud Figure 8
    
    def test_cluster_profile(self):
        ##* Test cluster profile of RXC J0003.8+0203
        # Values from Table 1 of Pratt, Arnaud, Piffaretti, et. al, 2010
        z = 0.0924
        h_70 = self.cos_model.H0/(70*u.km/(u.s*u.Mpc))
        M = 2.11e14*u.M_sun/h_70
        R_500_calculated = self.cluster_model.getClusterRadius(z, M).to(u.Mpc)
        R_500_arnaud = 0.879*u.Mpc/h_70 # from Arnaud Table C.1, first row
        self.assertTrue((R_500_calculated - R_500_arnaud) < 0.001*u.Mpc,\
                        'Calculated R_500 was not within rounding error of Arnaud')
        
        # Load data from Arnaud Appendix C, Figure 1, top-left graph
        arnaud_data = pd.read_csv(os.path.join(self.data_dir, 'ArnaudClusterProfile.csv'))
        R = arnaud_data['R'].values*u.kpc/h_70
        P_arnaud = arnaud_data['P'].values*(h_70**(1/2))*u.keV/u.cm**3

        # Verify calculated pressure
        P_calculated = self.cluster_model.getElectronPressure(z, M, R/R_500_calculated, \
                                    P_0_h70 = 3.93*h_70**(3/2), c_500 = 1.33, alpha = 1.41, gamma = 0.567).to(u.keV/u.cm**3)
        self.assertTrue(verifyOoM(P_arnaud, P_calculated, 0.1), 'Cluster pressure was not within 0.1 orders of magnitude of Arnaud')
        
        if self.plot_results:
            plt.figure()
            plt.plot(R, P_calculated/(h_70**(1/2)), 'b.', label="Calculated")
            plt.plot(R, P_arnaud/(h_70**(1/2)), 'k:', label="Arnaud Data")
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Radius [h$_{70}$$^{-1}$ kpc]')
            plt.ylabel('P [h$_{70}$$^{1/2}$ keV cm$^{-3}$]')
            plt.title('RXC J0003.8+0203 Pressure Profile')
            plt.legend()
            plt.show()

if __name__ == '__main__':
    # Run with plots
    test = TestArnaud()
    test.setUp(plot_results=True)
    test.test_universal_profile()
    test.test_cluster_profile()