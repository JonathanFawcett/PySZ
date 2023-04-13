from pysz.calculators import *
from pysz.integrators import *
from pysz.structure_models import *
from pysz.mass_distributions import *
from astropy.cosmology import Planck18
import matplotlib.pyplot as plt
import csv
import multiprocessing as mp
import scipy.optimize

cos_model = Planck18
SFLH = SigmaFitLopezHonorez(cos_model)
SFT = SigmaFitTinker()
MDT = MassDistributionTinker(cos_model, SFLH)

##* Plot mass vs. sigma
logM = np.array([10, 11, 12, 13, 14, 15, 16], dtype='float')
logSigmaTinker = np.array([-0.64, -0.52, -0.38, -0.21, -0.01, 0.24, 0.55])
z = 0
h = cos_model.H(z)/(100*u.km/u.s/u.Mpc)
M = 10**(logM)*h*u.M_sun
sigmaTinker = 1/(10**logSigmaTinker)

plt.figure()
plt.plot(M/h, sigmaTinker, 'b.', label='Tinker Values')
plt.plot(M/h, SFT.getSigma(z, M), 'b:', label='Linear Tinker Fit')
plt.plot(M/h, SFLH.getSigma(z, M), 'k:', label='Lopez-Honorez Fit')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M/(h^{-1}M_{\odot})$')
plt.ylabel('$\sigma$')
plt.legend()
plt.title('Tinker Sigma Fit')

##* Plot mass distribution
deltas = [200, 800, 3200]
z = 0
h = cos_model.H(z)/(100*u.km/u.s/u.Mpc)
mean_density = cos_model.critical_density(z)*cos_model.Om(z)
M = np.logspace(10, 16, 1000)*h*u.M_sun

plt.figure()
for delta in deltas:
    MDT = MassDistributionTinker(cos_model, SFLH, delta=delta)
    mass_dist = MDT.getMassDistribution(z, M)
    plt.plot(M/h, mass_dist*(M**2)/mean_density, label=str(delta))

plt.xlabel(r'$M/(h^{-1}M_{\odot})$')
plt.ylabel(r'M$^2\rho_m$ dn/dM')
plt.xscale('log')
plt.yscale('log')
plt.ylim(10**-4, 10**-1)
plt.title('Tinker Mass Distribution')
plt.legend()
# Results similar to Tinker Fig. 5

# Show plots
plt.show()