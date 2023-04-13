from pysz.calculators import *
from pysz.integrators import *
from pysz.structure_models import *
from pysz.mass_distributions import *
from astropy.cosmology import Planck18
import matplotlib.pyplot as plt
import csv
import multiprocessing as mp
import pandas as pd
import scipy.optimize
from verify_func import *

# ##* Fit Cautun data to parabola
# def parabola(x, a, b, c):
#     return a*x**2 + b*x + c

files = ['Cautun00.csv', 'Cautun05.csv', 'Cautun10.csv', 'Cautun15.csv', 'Cautun20.csv']
fit_z = [0, 0.5, 1, 1.5, 2]

# TODO: include this as test?
# for file in files:
#     data = pd.read_csv(file)
#     params, _ = scipy.optimize.curve_fit(parabola, np.log10(data['M']), np.log10(data['dndM']))
#     print(params)
#     plt.plot(data['M'], data['dndM'], 'b.')
#     plt.plot(data['M'], 10**parabola(np.log10(data['M']), params[0], params[1], params[2]), 'k:')
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.title(file)
#     plt.show()

##* Test mass distribution
cos_model = Planck18
h = cos_model.h
MDC = MassDistributionCautun(cos_model)
M = np.logspace(11, 16, 1000)*u.M_sun/h

# Load data from Cautun Figure 54
for i in range(len(files)):
    data = pd.read_csv('./test_data/' + files[i])

    plt.figure()
    plt.plot(M*h, (MDC.getMassDistribution(fit_z[i], M)*(M*np.log(10))/h**3).to(u.Gpc**-3), 'b-', label="Calculated") #TODO: can I do this?
    plt.plot(data['M'], data['dndM'], 'k:', label="Cautun Data")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Filament Mass $M_f$ $[h^{-1}M_\odot]$')
    plt.ylabel('$dn/dlog_{10}M_f$  $[h^{3}Gpc^{-3}]$')
    plt.legend()
    plt.title('z = ' + str(fit_z[i]))

    assert verify_order_of_magnitude(MDC.getMassDistribution(fit_z[i], data['M'].values*u.M_sun/h), data['dndM'].values*u.Gpc**-3), 'Failed order-of-magnitued verification on ' + files[i]

# Show plots
plt.show()