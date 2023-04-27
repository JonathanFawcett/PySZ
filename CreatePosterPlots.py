import pysz
from astropy import units as u
from astropy.cosmology import Planck18
import numpy as np
import multiprocessing as mp
from sympy.physics.wigner import wigner_3j
import matplotlib.pyplot as plt
import pandas as pd
import time

def getWigner3j(l1: u.Quantity, l2: u.Quantity, l3: u.Quantity):
    w3j = np.empty(len(l1))
    for i in range(len(l1)):
        w3j[i] = wigner_3j(l1[i]*u.rad, l2[i]*u.rad, l3[i]*u.rad, 0, 0, 0)
    return w3j

if __name__=='__main__':
  pool = mp.Pool(mp.cpu_count()) 
  start = time.perf_counter()

  ##* Filaments
  cos_model = Planck18
  integrator = pysz.integrators.IntegratorTrapezoid()
  mass_model_fil = pysz.distributions.MassDistributionCautun(cos_model)
  fil_model = pysz.structures.FilamentModelGheller()

  SZ_calc_fil = pysz.calculators.GaussianFilamentSZCalculator(integrator, cos_model, fil_model, num_pts=10)
  mass_calc_fil = pysz.calculators.MassCalculator(mass_model_fil, SZ_calc_fil, integrator, 1e11*u.M_sun, 1e16*u.M_sun, num_pts=500)
  los_calc_fil = pysz.calculators.LineOfSightCalculator(cos_model, mass_calc_fil, integrator)

  # Generate l arrays
  num_l_ang = 30
  num_l_bispec = 15

  l_ang = u.Quantity(np.logspace(1, 3, num_l_ang, dtype='int'), u.rad**-1, dtype=None)
  l_bispec_flat = u.Quantity(np.array([np.linspace(200, 600, num_l_bispec, dtype='int')]*3).T, u.rad**-1, dtype=None)
  l_bispec_flat[l_bispec_flat % 2 != 0] = l_bispec_flat[l_bispec_flat % 2 != 0] + 1 # turn odd values even
  l_bispec_flat[:,2] = l_bispec_flat[:,2]*2

  ##* Averaged over all orientations
  ang_power_spec_averaged = los_calc_fil.getLOSIntegral(l_ang, pool=pool)*1e12*l_ang*(l_ang+1)/(2*np.pi*u.rad)
  flat_bispec_averaged = los_calc_fil.getLOSIntegral(l_bispec_flat, pool=pool)*\
    np.sqrt((2*l_bispec_flat[:,0]+1)*(2*l_bispec_flat[:,1]+1)*(2*l_bispec_flat[:,2]+1)/(4*np.pi))*\
      getWigner3j(l_bispec_flat[:,0], l_bispec_flat[:,1], l_bispec_flat[:,2])

  ##* Filament length perpendicular to line of sight
  SZ_calc_fil.theta_min = 0*u.rad
  SZ_calc_fil.theta_max = 1e-5*u.rad
  SZ_calc_fil.phi_min = 0*u.rad
  SZ_calc_fil.phi_max = 1e-5*u.rad

  ang_power_spec_perp = los_calc_fil.getLOSIntegral(l_ang, pool=pool)*1e12*l_ang*(l_ang+1)/(2*np.pi*u.rad)
  flat_bispec_perp = los_calc_fil.getLOSIntegral(l_bispec_flat, pool=pool)*\
    np.sqrt((2*l_bispec_flat[:,0]+1)*(2*l_bispec_flat[:,1]+1)*(2*l_bispec_flat[:,2]+1)/(4*np.pi))*\
      getWigner3j(l_bispec_flat[:,0], l_bispec_flat[:,1], l_bispec_flat[:,2])

  ##* Filament length parallel to line of sight
  SZ_calc_fil.theta_min = (np.pi/2-1e-5)*u.rad
  SZ_calc_fil.theta_max = (np.pi/2+1e-5)*u.rad
  SZ_calc_fil.phi_min = (np.pi/2-1e-5)*u.rad
  SZ_calc_fil.phi_max = (np.pi/2+1e-5)*u.rad

  ang_power_spec_par = los_calc_fil.getLOSIntegral(l_ang, pool=pool)*1e12*l_ang*(l_ang+1)/(2*np.pi*u.rad)
  flat_bispec_par = los_calc_fil.getLOSIntegral(l_bispec_flat, pool=pool)*\
    np.sqrt((2*l_bispec_flat[:,0]+1)*(2*l_bispec_flat[:,1]+1)*(2*l_bispec_flat[:,2]+1)/(4*np.pi))*\
      getWigner3j(l_bispec_flat[:,0], l_bispec_flat[:,1], l_bispec_flat[:,2])
  
  ##* Clusters
  sigma_model = pysz.distributions.SigmaFitLopezHonorez(cos_model)
  mass_model_clus = pysz.distributions.MassDistributionTinker(cos_model, sigma_model, delta=500/cos_model.Om(0))
  cluster_model = pysz.structures.ClusterModelArnaud(cos_model)

  SZ_calc_clus = pysz.calculators.GeneralSZCalculator(integrator, cos_model, cluster_model)
  mass_calc_clus = pysz.calculators.MassCalculator(mass_model_clus, SZ_calc_clus, integrator, M_min=1e10*u.M_sun, M_max=1e16*u.M_sun)
  los_calc_clus = pysz.calculators.LineOfSightCalculator(cos_model, mass_calc_clus, integrator)

  # Angular Power Spectrum
  ang_power_spec = los_calc_clus.getLOSIntegral(l_ang, pool=pool)*1e12*l_ang*(l_ang+1)/(2*np.pi*u.rad)

  # Load data from Planck 2015 Fig. 15
  planck_ang = pd.read_csv(r'.\tests\data\PlanckAngular.csv')

  # Flattened bispectrum
  flat_bispec = los_calc_clus.getLOSIntegral(l_bispec_flat, pool=pool)*\
    np.sqrt((2*l_bispec_flat[:,0]+1)*(2*l_bispec_flat[:,1]+1)*(2*l_bispec_flat[:,2]+1)/(4*np.pi))*\
      getWigner3j(l_bispec_flat[:,0], l_bispec_flat[:,1], l_bispec_flat[:,2])

  # Load data from Planck 2015 Fig. 14c
  planck_flat = pd.read_csv(r'.\tests\data\PlanckFlat.csv')

  # Close parallel pool
  pool.close()

  ##* Angular power spectrum plot
  plt.figure()
  plt.plot(l_ang, ang_power_spec, '.', label="Clusters")
  plt.plot(l_ang, ang_power_spec_averaged, '.', label="Filaments Averaged")
  plt.plot(l_ang, ang_power_spec_perp, '.', label="Filaments Perpendicular")
  plt.plot(l_ang, ang_power_spec_par, '.', label="Filaments Parallel")
  plt.plot(planck_ang['l'], planck_ang['power'], 'k:', label="Planck 2015 Data")
  plt.xlabel('Multipole $\ell$')
  plt.ylabel('$10^{12} \ell(\ell+1) C_\ell/2\pi$')
  plt.xscale('log')
  plt.yscale('log')
  plt.xlim(l_ang.min(), l_ang.max())
  plt.legend()

  # Flattened bispectrum plot
  plt.figure()
  plt.plot(l_bispec_flat[:,0], np.abs(flat_bispec), '.', label="Clusters")
  plt.plot(l_bispec_flat[:,0], np.abs(flat_bispec_averaged), '.', label="Filaments Averaged")
  plt.plot(l_bispec_flat[:,0], np.abs(flat_bispec_perp), '.', label="Filaments Perpendicular")
  plt.plot(l_bispec_flat[:,0], np.abs(flat_bispec_par), '.', label="Filaments Parallel")
  plt.plot(planck_flat['l'], planck_flat['power'], 'k:', label="Planck 2015 Data")
  plt.xlabel('Multipole $\ell$')
  plt.ylabel('abs($b(\ell,\ell,2\ell)$)')
  plt.xscale('log')
  plt.yscale('log')
  plt.title('Flattened Bispectrum')
  plt.xlim(l_bispec_flat[:,0].min(), l_bispec_flat[:,0].max())
  plt.legend()

  end = time.perf_counter()
  print("Time to calculate and create plots = {t:.3f}s".format(t=end - start))

  # Show plots
  plt.show()