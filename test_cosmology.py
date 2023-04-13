from pysz.calculators import *
from pysz.integrators import *
from pysz.structure_models import *
from pysz.mass_distributions import *
from astropy.cosmology import Planck18
import matplotlib.pyplot as plt
import csv
import multiprocessing as mp

##* Create expansion model from astropy
# Constants from https://ned.ipac.caltech.edu/level5/Sept05/Carlstrom/Carlstrom4.html
LCDM = cosmology.LambdaCDM(
    H0 = 42 << (u.km*u.s**-1*u.Mpc**-1), # Hubble constant at z=0, km/sec/Mpc. Arbitrary for this test
    Om0 = 0.3, # Omega matter at z=0, fraction of critical density
    Ode0 = 0.7 # Omega dark energy at z=0, fraction of critical density
)

# Plot comoving volume from 0 to 3
z = np.linspace(0, 3, 1000)
comov_vol = LCDM.differential_comoving_volume(z)

plt.figure()
plt.plot(z, comov_vol * (LCDM.H0 / 100)**3)
plt.xlabel('Redshift [-]')
plt.ylabel(r'Differential Comoving Volume [$h^{-3}Mpc^{3}/sr$]')
plt.yscale('log')
plt.xlim(left = 0, right = 3)
plt.ylim(bottom = 10**8)
plt.title('Comoving Distance, LCDM\n H0 = ' + str(LCDM.H0) + ', Om0 = ' + str(LCDM.Om0) + ', Ode = ' + str(LCDM.Ode0))
# Result equivalent to https://ned.ipac.caltech.edu/level5/Sept05/Carlstrom/Carlstrom4.html

# Show plot
plt.show()