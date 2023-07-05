from context import *

class TestIntegrators(unittest.TestCase):
    def test_rectangle(self, integrator: integrators.Integrator, tol=1e-6):
        # Test integrator with a simple constant function
        x_min = u.Quantity(1, u.m)
        x_max = u.Quantity(5, u.m)
        func = lambda x: np.ones(np.shape(x)) * 17 * u.m # Returns an array of [17, 17, 17, ...] the same size as the input

        solution = func(0) * (x_max - x_min)
        result = integrator(func, x_min, x_max)

        self.assertTrue(solution.unit == result.unit,\
            'Incorrect units returned by ' + str(integrator.__class__) + \
                ', solution was ' + str(solution.unit) + ' but returned was ' + str(result.unit))
        error = abs((result - solution)/solution)
        self.assertTrue(error < tol,\
            'Failed rectangular integration by ' + str(integrator.__class__) + \
                ', tolerance was ' + str(tol) + ' but error was ' + str(error))
        
    def test_exponential(self, integrator: integrators.Integrator, tol=1e-6):
        # Test integrator with exp(-x**2)
        x_min = u.Quantity(0, u.dimensionless_unscaled)
        x_max = u.Quantity(100, u.dimensionless_unscaled)
        func = lambda x: np.exp(-x**2)*u.dimensionless_unscaled

        solution = np.sqrt(np.pi)/2 * u.dimensionless_unscaled
        result = integrator(func, x_min, x_max)

        self.assertTrue(solution.unit == result.unit,\
            'Incorrect units returned by ' + str(integrator.__class__) + \
                ', solution was ' + str(solution.unit) + ' but returned was ' + str(result.unit))
        error = abs((result - solution)/solution)
        self.assertTrue(error < tol,\
            'Failed exponential integration by ' + str(integrator.__class__) + \
                ', tolerance was ' + str(tol) + ' but error was ' + str(error))

if __name__ == '__main__':
    test = TestIntegrators()

    # Run with trapezoid integrator
    test.test_rectangle(integrators.IntegratorTrapezoid())
    test.test_exponential(integrators.IntegratorTrapezoid())

    # Run with SciPy integrator
    test.test_rectangle(integrators.IntegratorSciPy())
    test.test_exponential(integrators.IntegratorSciPy())