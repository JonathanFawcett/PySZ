from abc import ABC, abstractmethod
import numpy as np
import scipy
from astropy import cosmology
from astropy import constants
from astropy import units as u
from numba import njit, jit, prange

class Integrator(ABC):
    @abstractmethod
    def __call__(self, func, x_min, x_max):
        pass

class IntegratorTrapezoid(Integrator):
    @staticmethod
    def __call__(func, x_min: u.Quantity, x_max: u.Quantity,
                 num_pts: int = 1000, dist='lin',
                 axis: int = 0, ndmin: int = 1):
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
    @staticmethod
    def __call__(func, x_min: u.Quantity, x_max: u.Quantity):
        # TODO make this work with astropy units by stripping units, then integrating, then multiplying units
        return scipy.integrate.quad(func, x_min, x_max)[0] # do not return error