# This module includes the integrator classes that calculate the many integrals, optimized for speed or precision

from .imports import *

class Integrator(ABC):
    """Abstract class to integrate a function between 2 points
    """
    @abstractmethod
    def __call__(self, func, x_min, x_max):
        """Abstract class to integrate a function between 2 points

        Args:
            func (Callable): Callable function to integrate
            x_min (Any): Lower bound of integral
            x_max (Any): Upper bound of integral

        Returns:
            Any: Result of integral
        """
        pass

class IntegratorTrapezoid(Integrator):
    """Integrator wrapping numpy's trapz function
    """
    @staticmethod
    def __call__(func, x_min: u.Quantity, x_max: u.Quantity,
                 num_pts: int = 1000, dist='lin',
                 axis: int = 0, ndmin: int = 1):
        """Integrator wrapping numpy's trapz function

        Args:
            func (Callable): Callable function to integrate
            x_min (u.Quantity): Lower bound of integral
            x_max (u.Quantity): Upper bound of integral
            num_pts (int, optional): Number of points to evaluate. Defaults to 1000.
            dist (str, optional): Integral point distribution, can be 'log'arithmic or 'lin'ear. Defaults to 'lin'.
            axis (int, optional): Axis to integrate along. Defaults to 0.
            ndmin (int, optional): Number of dimensions to return in the array-like. Defaults to 1.

        Returns:
            u.Quantity: Array-like of results, integrated along the "axis" dimension and with "ndmin" final dimensions
        """

        if dist=='lin':
            x = np.linspace(x_min, x_max, num_pts)
        elif dist=='log':
            x = np.geomspace(x_min, x_max, num_pts)
        else:
            raise TypeError('Distribution should be "lin" or "log"')

        # Correct array dimensions if required
        i = axis
        while i > 0:
            x = x[np.newaxis, :]
            i -= 1

        while x.ndim < ndmin:
            x = np.expand_dims(x, axis=x.ndim)

        # Perform integration
        y = np.trapz(func(x), x, axis=axis)

        # Return with correct number of dimensions
        if y.ndim < x.ndim:
            return np.expand_dims(y, axis=axis)
        else:
            return y

class IntegratorSciPy(Integrator):
    """Integrator wrapping SciPy's integrate.quad function
    """
    @staticmethod
    def __call__(func, x_min: u.Quantity, x_max: u.Quantity):
        """Integrator wrapping SciPy's integrate.quad function

        Args:
            func (Callable): Callable function to integrate
            x_min (u.Quantity): Lower bound of integral
            x_max (u.Quantity): Upper bound of integral

        Returns:
            u.Quantity: Integral result
        """

        if x_min.unit != x_max.unit:
            raise TypeError('Units of lower and upper bounds must be equal')
        
        # Determine final return units (unit of function * unit of bounds)
        units = func(x_min).unit * x_min.unit

        # Create callable function for SciPy that does not use units
        func_no_units = lambda x: func(x).value

        # Return result with units
        return scipy.integrate.quad(func_no_units, x_min.value, x_max.value)[0] * units # do not return error