from context import *

class TestPerformance(unittest.TestCase):
    def setUp(self, pool=None, verbose=False):
        self.pool = pool
        self.verbose = verbose

    def tearDown(self):
        if self.pool != None:
            self.pool.close()

    def test_parallelization(self, num_l=16):
        if self.pool==None:
            self.skipTest('Must have parallel pool available')

        #* Test clusters with equilateral (l1=l2=l3) bispectra
        cos_model = Planck18
        integrator = integrators.IntegratorTrapezoid()
        sigma_model = distributions.SigmaFitLopezHonorez(cos_model)
        mass_model = distributions.MassDistributionTinker(cos_model, sigma_model, delta=500/cos_model.Om(0))
        cluster_model = structures.ClusterModelArnaud(cos_model)
        SZ_calc = calculators.GeneralSZCalculator(integrator, cos_model, cluster_model)
        mass_calc = calculators.MassCalculator(mass_model, SZ_calc, integrator, M_min=1e10*u.M_sun, M_max=1e16*u.M_sun)
        los_calc = calculators.LineOfSightCalculator(cos_model, mass_calc, integrator)

        l_bispec_equil = u.Quantity(np.array([np.logspace(1, 3, num_l, dtype='int')]*3).T, u.rad**-1, dtype=None)

        start = time.perf_counter()
        los_calc.getLOSIntegral(l_bispec_equil, pool=None)
        t_single = time.perf_counter() - start

        start = time.perf_counter()
        los_calc.getLOSIntegral(l_bispec_equil, pool=self.pool)
        t_parallel = time.perf_counter() - start

        speed_up = (t_single - t_parallel)/t_single

        if self.verbose:
            print("Time for {l:d} equilateral bispectra single-threaded = {t:.3f}s".format(\
                l=num_l, t=t_single))
            print("Time for {l:d} equilateral bispectra with {cpu:d} threads = {t:.3f}s".format(\
                l=num_l, cpu=self.pool._processes, t=t_parallel))
            print("Total speed-up from parallelization: {speed:.1f}%\n".format(speed=speed_up*100))

        # Verify that performance is at least 20% better for parallel vs. single-threaded
        self.assertTrue(speed_up > 0.2,\
                        'Failed parallel performance test')
            
    def test_scalene(self, num_l=16):
        #* Test clusters with equilateral (l1=l2=l3) and scalene (l1 =/= l2 =/= l3) bispectra
        cos_model = Planck18
        integrator = integrators.IntegratorTrapezoid()
        sigma_model = distributions.SigmaFitLopezHonorez(cos_model)
        mass_model = distributions.MassDistributionTinker(cos_model, sigma_model, delta=500/cos_model.Om(0))
        cluster_model = structures.ClusterModelArnaud(cos_model)
        SZ_calc = calculators.GeneralSZCalculator(integrator, cos_model, cluster_model)
        mass_calc = calculators.MassCalculator(mass_model, SZ_calc, integrator, M_min=1e10*u.M_sun, M_max=1e16*u.M_sun)
        los_calc = calculators.LineOfSightCalculator(cos_model, mass_calc, integrator)

        l_bispec_equil = u.Quantity(np.array([np.logspace(1, 3, num_l, dtype='int')]*3).T, u.rad**-1, dtype=None)
        l_bispec_scalene = l_bispec_equil.copy()
        l_bispec_scalene[:,1] = l_bispec_scalene[:, 1]*2
        l_bispec_scalene[:,2] = l_bispec_scalene[:, 2]*3

        start = time.perf_counter()
        los_calc.getLOSIntegral(l_bispec_equil, pool=self.pool)
        t_equilateral = time.perf_counter() - start

        start = start = time.perf_counter()
        los_calc.getLOSIntegral(l_bispec_scalene, pool=self.pool)
        t_scalene = time.perf_counter() - start

        speed_up = (t_scalene - t_equilateral)/t_scalene

        if self.verbose:
            print("Time for {l:d} equilateral bispectra with {cpu:d} threads = {t:.3f}s".format(\
                l=num_l, cpu=self.pool._processes, t=t_equilateral))
            print("Time for {l:d} scalene bispectra with {cpu:d} threads = {t:.3f}s".format(\
                l=num_l, cpu=self.pool._processes, t=t_scalene))
            print("Total speed-up from duplicate l values: {speed:.1f}%\n".format(speed=speed_up*100))

        # Verify that performance is at least 50% better for equilateral vs. scalene
        self.assertTrue(speed_up > 0.5,\
                        'Failed equilateral/scalene performance test')
            
    def test_compiling(self, num_z=5000, num_M=5000, num_runs=5):
        #* Test numba compiling
        cos_model = Planck18
        sigma_model = distributions.SigmaFitLopezHonorez(cos_model)
        mass_distribution = distributions.MassDistributionTinker(cosmology_model=cos_model, sigma_model=sigma_model)

        z = np.expand_dims(np.geomspace(1, 3, num_z), (1,2,3))
        M = np.expand_dims(np.logspace(10, 15, num_M), (0,2,3))*u.M_sun
        sigma = sigma_model.getSigma(z, M)

        # First calculation will compile and then run
        start = time.perf_counter()
        mass_distribution.f(sigma.value, mass_distribution.A, mass_distribution.a, mass_distribution.b, mass_distribution.c)
        t_uncompiled = time.perf_counter() - start
        
        # Subsequent calculations are already compiled
        t = np.ones(num_runs)*np.Inf
        for i in range(num_runs):
            start = time.perf_counter()
            mass_distribution.f(sigma.value, mass_distribution.A, mass_distribution.a, mass_distribution.b, mass_distribution.c)
            end = time.perf_counter()
            t[i] = end-start
        t_compiled=np.mean(t)

        speed_up = (t_uncompiled - t_compiled)/t_uncompiled

        if self.verbose:
            print("Time for 1 computation of {z:d}x{M:d} matrix with compilation = {t:.3f}s".format(z=num_z, M=num_M, t=t_uncompiled))
            print("Average time of {n} computation(s) after compilation = {t:.3f}s".format(n=num_runs, t=t_compiled))
            print("Total speed-up from compiling: {speed:.1f}%\n".format(speed=speed_up*100))

        # Verify that performance is at least 20% better for compiled vs. uncompiled
        self.assertTrue(speed_up > 0.2,\
                        'Failed compiler performance test')

if __name__=='__main__':
    # Run with verbose outputs
    test = TestPerformance()
    test.setUp(pool=mp.Pool(), verbose=True)
    test.test_parallelization()
    test.test_scalene()
    test.test_compiling()