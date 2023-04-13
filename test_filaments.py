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

  ##* Test filament calculator!
  cos_model = Planck18
  integrator = IntegratorTrapezoid()
  mass_model = MassDistributionCautun(cos_model)
  fil_model = FilamentModelGheller()

  SZ_calc = GaussianFilamentSZCalculator(integrator, cos_model, fil_model, num_pts=10)
  mass_calc = MassCalculator(mass_model, SZ_calc, integrator, 1e11*u.M_sun, 1e16*u.M_sun, num_pts=500)
  los_calc = LineOfSightCalculator(cos_model, mass_calc, integrator)

  # Generate l arrays
  num_l_ang = 30
  num_l_bispec = 15

  l_ang = u.Quantity(np.logspace(1, 3, num_l_ang, dtype='int'), u.rad**-1, dtype=None)
  l_bispec_equil = u.Quantity(np.array([np.logspace(1, 3, num_l_bispec, dtype='int')]*3).T, u.rad**-1, dtype=None)
  l_bispec_equil[l_bispec_equil % 2 != 0] = l_bispec_equil[l_bispec_equil % 2 != 0] + 1 # turn odd values even
  l_bispec_flat = l_bispec_equil.copy()
  l_bispec_flat[:,2] = l_bispec_flat[:,2]*2

  ##* Averaged over all orientations
  ang_power_spec_averaged = los_calc.getLOSIntegral(l_ang, pool=pool)*1e12*l_ang*(l_ang+1)/(2*np.pi*u.rad)
  equil_bispec_averaged = los_calc.getLOSIntegral(l_bispec_equil, pool=pool)*\
    np.sqrt((2*l_bispec_equil[:,0]+1)*(2*l_bispec_equil[:,1]+1)*(2*l_bispec_equil[:,2]+1)/(4*np.pi))*\
      getWigner3j(l_bispec_equil[:,0], l_bispec_equil[:,1], l_bispec_equil[:,2])
  flat_bispec_averaged = los_calc.getLOSIntegral(l_bispec_flat, pool=pool)*\
    np.sqrt((2*l_bispec_flat[:,0]+1)*(2*l_bispec_flat[:,1]+1)*(2*l_bispec_flat[:,2]+1)/(4*np.pi))*\
      getWigner3j(l_bispec_flat[:,0], l_bispec_flat[:,1], l_bispec_flat[:,2])

  ##* Filament length perpendicular to line of sight
  SZ_calc.theta_min = 0*u.rad
  SZ_calc.theta_max = 1e-5*u.rad
  SZ_calc.phi_min = 0*u.rad
  SZ_calc.phi_max = 1e-5*u.rad

  ang_power_spec_perp = los_calc.getLOSIntegral(l_ang, pool=pool)*1e12*l_ang*(l_ang+1)/(2*np.pi*u.rad)
  equil_bispec_perp = los_calc.getLOSIntegral(l_bispec_equil, pool=pool)*\
    np.sqrt((2*l_bispec_equil[:,0]+1)*(2*l_bispec_equil[:,1]+1)*(2*l_bispec_equil[:,2]+1)/(4*np.pi))*\
      getWigner3j(l_bispec_equil[:,0], l_bispec_equil[:,1], l_bispec_equil[:,2])
  flat_bispec_perp = los_calc.getLOSIntegral(l_bispec_flat, pool=pool)*\
    np.sqrt((2*l_bispec_flat[:,0]+1)*(2*l_bispec_flat[:,1]+1)*(2*l_bispec_flat[:,2]+1)/(4*np.pi))*\
      getWigner3j(l_bispec_flat[:,0], l_bispec_flat[:,1], l_bispec_flat[:,2])

  ##* Filament length parallel to line of sight
  SZ_calc.theta_min = (np.pi/2-1e-5)*u.rad
  SZ_calc.theta_max = (np.pi/2+1e-5)*u.rad
  SZ_calc.phi_min = (np.pi/2-1e-5)*u.rad
  SZ_calc.phi_max = (np.pi/2+1e-5)*u.rad

  ang_power_spec_par = los_calc.getLOSIntegral(l_ang, pool=pool)*1e12*l_ang*(l_ang+1)/(2*np.pi*u.rad)
  equil_bispec_par = los_calc.getLOSIntegral(l_bispec_equil, pool=pool)*\
    np.sqrt((2*l_bispec_equil[:,0]+1)*(2*l_bispec_equil[:,1]+1)*(2*l_bispec_equil[:,2]+1)/(4*np.pi))*\
      getWigner3j(l_bispec_equil[:,0], l_bispec_equil[:,1], l_bispec_equil[:,2])
  flat_bispec_par = los_calc.getLOSIntegral(l_bispec_flat, pool=pool)*\
    np.sqrt((2*l_bispec_flat[:,0]+1)*(2*l_bispec_flat[:,1]+1)*(2*l_bispec_flat[:,2]+1)/(4*np.pi))*\
      getWigner3j(l_bispec_flat[:,0], l_bispec_flat[:,1], l_bispec_flat[:,2])

  # Close parallel pool
  pool.close()

  # Angular power spectrum plot
  plt.figure()
  plt.plot(l_ang, ang_power_spec_averaged, 'b.', label="Averaged")
  plt.plot(l_ang, ang_power_spec_perp, 'r.', label="Filament Perpendicular")
  plt.plot(l_ang, ang_power_spec_par, 'g.', label="Filament Parallel")
  plt.xlabel('Multipole $\ell$')
  plt.ylabel('$10^{12} \ell(\ell+1) C_\ell/2\pi$')
  plt.xscale('log')
  plt.yscale('log')
  plt.legend()

  # Equilateral bispectrum plot
  plt.figure()
  plt.plot(l_bispec_equil[equil_bispec_averaged>0,0], equil_bispec_averaged[equil_bispec_averaged>0], 'b.', label="Averaged >0")
  plt.plot(l_bispec_equil[equil_bispec_averaged<0,0], -equil_bispec_averaged[equil_bispec_averaged<0], 'bx', label="Averaged <0")
  plt.plot(l_bispec_equil[equil_bispec_perp>0,0], equil_bispec_perp[equil_bispec_perp>0], 'r.', label="Filament Perpendicular >0")
  plt.plot(l_bispec_equil[equil_bispec_perp<0,0], -equil_bispec_perp[equil_bispec_perp<0], 'rx', label="Filament Perpendicular <0")
  plt.plot(l_bispec_equil[equil_bispec_par>0,0], equil_bispec_par[equil_bispec_par>0], 'g.', label="Filament Parallel >0")
  plt.plot(l_bispec_equil[equil_bispec_par<0,0], -equil_bispec_par[equil_bispec_par<0], 'gx', label="Filament Parallel <0")
  plt.xlabel('Multipole $\ell$')
  plt.ylabel('abs($b(\ell,\ell,\ell)$)')
  plt.xscale('log')
  plt.yscale('log')
  plt.title('Equilateral Bispectrum')
  plt.legend()

  # Flattened bispectrum plot
  plt.figure()
  plt.plot(l_bispec_flat[flat_bispec_averaged>0,0], flat_bispec_averaged[flat_bispec_averaged>0], 'b.', label="Averaged >0")
  plt.plot(l_bispec_flat[flat_bispec_averaged<0,0], -flat_bispec_averaged[flat_bispec_averaged<0], 'bx', label="Averaged <0")
  plt.plot(l_bispec_flat[flat_bispec_perp>0,0], flat_bispec_perp[flat_bispec_perp>0], 'r.', label="Filament Perpendicular >0")
  plt.plot(l_bispec_flat[flat_bispec_perp<0,0], -flat_bispec_perp[flat_bispec_perp<0], 'rx', label="Filament Perpendicular <0")
  plt.plot(l_bispec_flat[flat_bispec_par>0,0], flat_bispec_par[flat_bispec_par>0], 'g.', label="Filament Parallel >0")
  plt.plot(l_bispec_flat[flat_bispec_par<0,0], -flat_bispec_par[flat_bispec_par<0], 'gx', label="Filament Parallel <0")
  plt.xlabel('Multipole $\ell$')
  plt.ylabel('abs($b(\ell,\ell,2\ell)$)')
  plt.xscale('log')
  plt.yscale('log')
  plt.title('Flattened Bispectrum')
  plt.legend()

  # Show plots
  plt.show()
  