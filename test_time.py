from pysz.calculators import *
from pysz.integrators import *
from pysz.structure_models import *
from pysz.mass_distributions import *
from astropy.cosmology import Planck18
import matplotlib.pyplot as plt
import csv
import multiprocessing as mp
import time

if __name__=='__main__':
    #* Setup cluster test for bispectrum, with equilateral (l1=l2=l3) and scalene bispectra
    pool = mp.Pool(mp.cpu_count()) 

    cos_model = Planck18
    integrator = IntegratorTrapezoid()
    sigma_model = SigmaFitLopezHonorez(cos_model)
    mass_model = MassDistributionTinker(cos_model, sigma_model, delta=500/cos_model.Om(0))
    cluster_model = ClusterModelArnaud(cos_model)
    SZ_calc = GeneralSZCalculator(integrator, cos_model, cluster_model)
    mass_calc = MassCalculator(mass_model, SZ_calc, integrator, M_min=1e10*u.M_sun, M_max=1e16*u.M_sun)
    los_calc = LineOfSightCalculator(cos_model, mass_calc, integrator)

    num_l = 20

    l_bispec_equil = u.Quantity(np.array([np.logspace(1, 3, num_l, dtype='int')]*3).T, u.rad**-1, dtype=None)
    l_bispec_alt = l_bispec_equil.copy()
    l_bispec_alt[:,1] = l_bispec_alt[:, 1]*2
    l_bispec_alt[:,2] = l_bispec_alt[:, 2]*3

    #* Test parallelized/not parallelized and equilateral/scalene bispectra
    start = time.perf_counter()
    los_calc.getLOSIntegral(l_bispec_equil, pool=None)
    end = time.perf_counter()
    print("Time for {l:d} equilateral bispectra single-threaded = {t:.3f}s".format(l=num_l, t=end - start))

    start = time.perf_counter()
    los_calc.getLOSIntegral(l_bispec_equil, pool=pool)
    end = time.perf_counter()
    print("Time for {l:d} equilateral bispectra with {cpu:d} threads = {t:.3f}s".format(l=num_l, cpu=mp.cpu_count(), t=end - start))

    start = time.perf_counter()
    los_calc.getLOSIntegral(l_bispec_alt, pool=pool)
    end = time.perf_counter()
    print("Time for {l:d} scalene bispectra with {cpu:d} threads = {t:.3f}s".format(l=num_l, cpu=mp.cpu_count(), t=end - start))

    pool.close()

    #* Test numba compiling
    num_z = 1000
    num_M = 1000

    cos_model = Planck18
    sigma_model = SigmaFitLopezHonorez(cos_model)

    z = np.expand_dims(np.geomspace(1, 3, num_z), (1,2,3))
    M = np.expand_dims(np.logspace(10, 15, num_M), (0,2,3))*u.M_sun

    start = time.perf_counter()
    sigma_model.getSigma(z, M)
    end = time.perf_counter()
    print("Time for 1 computation of {z:d}x{M:d} matrix with compilation = {t:.3f}s".format(z=num_z, M=num_M, t=end-start))

    num_runs = 10
    t = np.ones(num_runs)*np.Inf
    for i in range(num_runs):
        start = time.perf_counter()
        sigma_model.getSigma(z, M)
        end = time.perf_counter()
        t[i] = end-start
    print("Average time for {n} computation(s) after compilation = {t:.3f}s".format(n=num_runs, t=np.mean(t)))