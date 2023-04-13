from pysz.calculators import *
from pysz.integrators import *
from pysz.structure_models import *
from pysz.mass_distributions import *
from astropy.cosmology import Planck18
import matplotlib.pyplot as plt
import csv
import multiprocessing as mp
from sympy.physics.wigner import wigner_3j
import pandas as pd

def getWigner3j(l1: u.Quantity, l2: u.Quantity, l3: u.Quantity):
    w3j = np.empty(len(l1))
    for i in range(len(l1)):
        w3j[i] = wigner_3j(l1[i]*u.rad, l2[i]*u.rad, l3[i]*u.rad, 0, 0, 0)
    return w3j

if __name__=='__main__':
    pool = mp.Pool(mp.cpu_count()) 

    ##* Test cluster calculator!
    cos_model = Planck18
    integrator = IntegratorTrapezoid()
    sigma_model = SigmaFitLopezHonorez(cos_model)
    mass_model = MassDistributionTinker(cos_model, sigma_model, delta=500/cos_model.Om(0))
    cluster_model = ClusterModelArnaud(cos_model)
    
    SZ_calc = GeneralSZCalculator(integrator, cos_model, cluster_model)
    mass_calc = MassCalculator(mass_model, SZ_calc, integrator, M_min=1e10*u.M_sun, M_max=1e16*u.M_sun)
    los_calc = LineOfSightCalculator(cos_model, mass_calc, integrator)

    # Angular Power Spectrum
    num_l = 100
    l = u.Quantity(np.logspace(1, 3, num_l, dtype='int'), u.rad**-1, dtype=None)

    ang_power_spec = los_calc.getLOSIntegral(l, pool=pool)*1e12*l*(l+1)/(2*np.pi*u.rad)

    # Load data from Planck 2015 Fig. 15
    planck_ang = pd.read_csv('./test_data/PlanckAngular.csv')

    plt.figure()
    plt.plot(l, ang_power_spec, 'b.', label="Calculated")
    plt.plot(planck_ang['l'], planck_ang['power'], 'k:', label="Planck 2015 Data")
    plt.xlabel('Multipole $\ell$')
    plt.ylabel('$10^{12} \ell(\ell+1) C_\ell/2\pi$')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(l.min(), l.max())

    # Equilateral bispectrum
    num_l = 11
    l = u.Quantity(np.array([np.linspace(200, 600, num_l, dtype='int')]*3).T, u.rad**-1, dtype=None)
    equil_bispec = los_calc.getLOSIntegral(l, pool=pool)*np.sqrt((2*l[:,0]+1)*(2*l[:,1]+1)*(2*l[:,2]+1)/(4*np.pi))*getWigner3j(l[:,0], l[:,1], l[:,2])

    # Load data from Planck 2015 Fig. 14a
    planck_equil = pd.read_csv('./test_data/PlanckEquil.csv')

    plt.figure()
    plt.plot(l[:,0], np.abs(equil_bispec), 'b.', label="Calculated")
    plt.plot(planck_equil['l'], planck_equil['power'], 'k:', label="Planck 2015 Data")
    plt.xlabel('Multipole $\ell$')
    plt.ylabel('abs($b(\ell,\ell,\ell)$)')
    plt.yscale('log')
    plt.title('Equilateral Bispectrum')
    plt.xlim(l[:,0].min(), l[:,0].max())
    plt.legend()

    # Flattened bispectrum
    num_l = 11
    l = u.Quantity(np.array([np.linspace(200, 600, num_l, dtype='int')]*3).T, u.rad**-1, dtype=None)
    l[:,2] = l[:,2]*2
    flat_bispec = los_calc.getLOSIntegral(l, pool=pool)*np.sqrt((2*l[:,0]+1)*(2*l[:,1]+1)*(2*l[:,2]+1)/(4*np.pi))*getWigner3j(l[:,0], l[:,1], l[:,2])

    # Load data from Planck 2015 Fig. 14c
    planck_flat = pd.read_csv('./test_data/PlanckFlat.csv')

    plt.figure()
    plt.plot(l[:,0], np.abs(flat_bispec), 'b.', label="Calculated")
    plt.plot(planck_flat['l'], planck_flat['power'], 'k:', label="Planck 2015 Data")
    plt.xlabel('Multipole $\ell$')
    plt.ylabel('abs($b(\ell,\ell,2\ell)$)')
    plt.yscale('log')
    plt.title('Flattened Bispectrum')
    plt.xlim(l[:,0].min(), l[:,0].max())
    plt.legend()

    # Close parallel pool
    # if __name__=='__main__':
    pool.close()

    # Show plots
    plt.show()