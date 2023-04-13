from pysz.calculators import *
from pysz.integrators import *
from pysz.structure_models import *
from pysz.mass_distributions import *
from astropy.cosmology import Planck18
import matplotlib.pyplot as plt
import csv
import multiprocessing as mp
import pandas as pd

##* Test universal pressure profile
cos_model = Planck18
CMA = ClusterModelArnaud(cos_model)
x = np.logspace(-2, 0.6, 1000)
universal_profile = CMA.universalProfile(x)
plt.figure()
plt.plot(x, universal_profile)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Radius (R$_{500}$)')
plt.ylabel('P/P$_{500}$')
plt.title('Arnaud Universal Pressure Profile')
# Results similar to Arnaud Figure 8

##* Test cluster profile of RXC J0003.8+0203
# Values from Table 1 of Pratt, Arnaud, Piffaretti, et. al, 2010
z = 0.0924
h_70 = cos_model.H(z)/(70*u.km/(u.s*u.Mpc))
M = 2.11e14*u.M_sun/h_70
R_500 = CMA.getClusterRadius(z, M).to(u.Mpc)
print(R_500) # similar to R_500 provided by Arnaud Table C.1, 0.879 Mpc

# Load data from Arnaud
arnaud_data = pd.read_csv('./test_data/ArnaudClusterProfile.csv')

# Plot pressure
R = (np.logspace(1.5, 2.9, 20)*u.kpc)/h_70
P = CMA.getElectronPressure(z, M, R/R_500, \
                            P_0_h70 = 3.93*h_70**(3/2), c_500 = 1.33, alpha = 1.41, gamma = 0.567).to(u.keV/u.cm**3)
plt.figure()
plt.plot(R*h_70, P/(h_70**(1/2)), 'b.', label="Calculated")
plt.plot(arnaud_data['R'], arnaud_data['P'], 'k:', label="Arnaud Data")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Radius [h$_{70}$$^{-1}$ kpc]')
plt.ylabel('P [h$_{70}$$^{1/2}$ keV cm$^{-3}$]')
plt.title('RXC J0003.8+0203 Pressure Profile')
plt.legend()
# TODO should be similar to Arnaud Appendix C, Figure 1, top-left graph

# Show plots
plt.show()