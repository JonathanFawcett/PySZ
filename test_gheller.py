from pysz.calculators import *
from pysz.integrators import *
from pysz.structure_models import *
from pysz.mass_distributions import *
from astropy.cosmology import Planck18
import matplotlib.pyplot as plt
import csv
import multiprocessing as mp

##* Test filament model
FMG = FilamentModelGheller()
filament_z = [0, 0.5, 1.0]
M = np.logspace(10, 15, 1000)*u.M_sun
plt.figure()

for z in filament_z:
    fil_length = FMG.getFilamentLength(z, M).to(u.Mpc)
    plt.plot(M, fil_length, label="z = " + str(z))

plt.xlabel(r'M/$M_{\odot}$')
plt.ylabel(r'Filament Length [Mpc]')
plt.xscale('log')
plt.yscale('log')
plt.title('Filament Length Scaling\n100 Mpc Box')
plt.legend()
# Result similar to 1607.01406 Figure 4, right panel, 100Mpc

# Get base temperature (for scaling), assuming L = 15 Mpc @ z = 0
L = 15
base_M = (L/(10**FMG.beta_L[0]))**(1/FMG.alpha_L[0])*u.M_sun
base_T = FMG.logScaleByMass(FMG.alpha_T[0], FMG.beta_T[0], base_M)*u.K
print("Base Mass: {M}".format(M = base_M))
print("Base Temperature: {T}".format(T = base_T))
print("Temperature Scale @ Base Mass: {s}".format(s = FMG.getTemperatureScaling(0, base_M)))

# Show plot
plt.show()