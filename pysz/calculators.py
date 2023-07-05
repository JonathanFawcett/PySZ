# This module includes the calculator classes that calculate y-tilde and perform the integration across mass and redshift

from .imports import *

from . import integrators
from . import structures
from . import distributions

class SZCalculator(ABC):
    """ Abstract class for the Sunyaev-Zeldovich calculator
    """
    sigma_T = constants.sigma_T
    m_e = constants.m_e
    c = constants.c

    @abstractmethod
    def getYTilde(self, z: u.Quantity, M: u.Quantity, l: u.Quantity):
        """Abstract method for calculating the Fourier transform of the y-parameter

        Args:
            z (u.Quantity): Array-like of redshift values
            M (u.Quantity): Array-like of structure masses
            l (u.Quantity): Array-like of angular scales

        Returns:
            u.Quantity: Array-like of calculated y-tilde values
        """
        pass

class GeneralSZCalculator(SZCalculator):
    """ General Sunyaev-Zeldovich calculator for clusters and other structures with radial electron pressure formulas
    """
    def __init__(self, integrator: integrators.Integrator,
                 cosmology_model: cosmology.FLRW,
                 cluster_model: structures.ClusterModel,
                 R_min = 1e-5, R_max = 5, num_pts: int = 100):
        """ General Sunyaev-Zeldovich calculator for clusters and other structures with radial electron pressure formulas

        Args:
            integrator (integrators.Integrator): Integration class to use for calculations
            cosmology_model (cosmology.FLRW): Astropy cosmology model to use
            cluster_model (structures.ClusterModel): Cluster model to use
            R_min (_type_, optional): Minimum fraction of the scale radius to evaluate. Defaults to 1e-5.
            R_max (int, optional): Maximum fraction of the scale radius to evaluate. Defaults to 5.
            num_pts (int, optional): Number of points to evaluate in integral. Defaults to 100.
        """
        self.integrator = integrator
        self.cosmology_model = cosmology_model
        self.structure_model = cluster_model

        self.R_min = R_min
        self.R_max = R_max
        self.num_pts = num_pts

        self.prefactor = 4*np.pi*self.sigma_T/(self.m_e*self.c**2)
    
    def getYTilde(self, z: u.Quantity, M: u.Quantity, l: u.Quantity):
        """Calculates the Fourier transform of the y-parameter by radial integration

        Args:
            z (u.Quantity): Array-like of redshift values
            M (u.Quantity): Array-like of structure masses
            l (u.Quantity): Array-like of angular scales

        Returns:
            u.Quantity: Array-like of calculated y-tilde values
        """
        scale_radius = self.structure_model.getClusterRadius(z, M)
        l_s = self.cosmology_model.angular_diameter_distance(z)/(u.rad*scale_radius)
        func = lambda x: (x**2)*self.structure_model.getElectronPressure(z, M, x)*np.sinc(l*x/(l_s*np.pi))

        return self.prefactor*(scale_radius/l_s**2)*self.integrator(func, self.R_min, self.R_max, num_pts=self.num_pts, axis=2,  ndmin=4)

class GaussianFilamentSZCalculator(SZCalculator):
    """Sunyaev-Zeldovich calculator for filaments, assuming a uniform double-Gaussian profile
    """

    def __init__(self, integrator: integrators.Integrator,
                 cosmology_model: cosmology.FLRW,
                 filament_model: structures.FilamentModel,
                 A_1: u.Quantity = 1.2e-5*u.keV*u.cm**-3, A_2: u.Quantity = 2.66e-5*u.keV*u.cm**-3, 
                 c_1=0.5186, c_2=0.1894, 
                 theta_min: u.Quantity = 0*u.rad, theta_max: u.Quantity = np.pi*u.rad, 
                 phi_min: u.Quantity = 0*u.rad, phi_max: u.Quantity = 2*np.pi*u.rad, 
                 num_pts: int = 20):
        """Sunyaev-Zeldovich calculator for filaments, assuming a uniform double-Gaussian profile

        Args:
            integrator (integrators.Integrator): Integration class to use for calculations
            cosmology_model (cosmology.FLRW): Astropy cosmology model to use
            filament_model (structures.FilamentModel): Filament model to use
            A_1 (u.Quantity, optional): Amplitude for the first Gaussian term. Defaults to 1.2e-5*u.keV*u.cm**-3.
            A_2 (u.Quantity, optional): Amplitude of the second Gaussian term. Defaults to 2.66e-5*u.keV*u.cm**-3.
            c_1 (float, optional): Exponent of the first Gaussian term. Defaults to 0.5186.
            c_2 (float, optional): Exponent of the second Gaussian term. Defaults to 0.1894.
            theta_min (u.Quantity, optional): Minimum rotation about the Y axis (see Figure 3). Defaults to 0*u.rad.
            theta_max (u.Quantity, optional): Maximum rotation about the Y axis (see Figure 3). Defaults to np.pi*u.rad.
            phi_min (u.Quantity, optional): Minimum rotation about the Z axis (see Figure 3). Defaults to 0*u.rad.
            phi_max (u.Quantity, optional): Maximum rotation about the Z axis (see Figure 3). Defaults to 2*np.pi*u.rad.
            num_pts (int, optional): Number of points to evaluate in integral. Defaults to 20.
        """
        self.A_1 = A_1
        self.A_2 = A_2
        self.c_1 = c_1
        self.c_2 = c_2
        self.integrator = integrator
        self.cosmology_model = cosmology_model
        self.filament_model = filament_model

        self.theta_min = theta_min
        self.theta_max = theta_max
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.num_pts = num_pts

        self.prefactor = 2*(1/(4*np.pi))*self.sigma_T/(self.m_e*self.c**2)
        self.factor_1 = A_1 * c_1**2
        self.factor_2 = A_2 * c_2**2

    def getYTilde(self, z: u.Quantity, M: u.Quantity, l: u.Quantity):    
        """Calculates the Fourier transform of the y-parameter using Equation 15

        Args:
            z (u.Quantity): Array-like of redshift values
            M (u.Quantity): Array-like of filament masses
            l (u.Quantity): Array-like of angular scales

        Returns:
            u.Quantity: Array-like of calculated y-tilde values
        """
        dA = self.cosmology_model.angular_diameter_distance(z)/u.rad
        L = self.filament_model.getFilamentLength(z, M)
        R_sq = self.filament_model.getFilamentRadius(z, M)**2
        k = l/dA
        temp_scale = self.filament_model.getTemperatureScaling(z, M)

        # Correct for integral limits
        correction_factor = 1/((self.phi_max - self.phi_min) * self.integrator(lambda x: np.sin(x), self.theta_min, self.theta_max))

        # Perform double integral
        inner_func = lambda theta, phi: np.sinc(self.k_z(k, theta, phi)*(L/2)/np.pi)*(L/2)*\
            (self.factor_1*np.exp(-self.k_r_sq(k, theta, phi)*R_sq*self.c_1**2/4) +\
                self.factor_2*np.exp(-self.k_r_sq(k, theta, phi)*R_sq*self.c_2**2/4)) *\
                    np.sin(theta)
        inner_integral = lambda phi: self.integrator(lambda theta: inner_func(theta, phi), self.theta_min, self.theta_max, num_pts=self.num_pts, axis=3, ndmin=4)
        outer_integral = self.integrator(inner_integral, self.phi_min, self.phi_max, num_pts=self.num_pts, axis=2,  ndmin=4)

        return self.prefactor * correction_factor * temp_scale * R_sq / dA**2 * outer_integral
    
    def k_z(self, k: u.Quantity, theta: u.Quantity, phi: u.Quantity):
        """Internal function to calculate k_z using Equation 14

        Args:
            k (u.Quantity): Linear scale
            theta (u.Quantity): Rotation about the Y axis (see Figure 3)
            phi (u.Quantity): Rotation about the Z axis (see Figure 3)

        Returns:
            u.Quantity: Z component of the linear scale
        """
        return k * np.sqrt((np.cos(phi)*np.sin(theta))**2 + np.cos(theta)**2)
    
    def k_r_sq(self, k: u.Quantity, theta: u.Quantity, phi: u.Quantity):
        """Internal function to calculate k_r**2 using Equation 16

        Args:
            k (u.Quantity): Linear scale
            theta (u.Quantity): Rotation about the Y axis (see Figure 3)
            phi (u.Quantity): Rotation about the Z axis (see Figure 3)

        Returns:
            u.Quantity: Radial scale squared
        """
        return k**2 * ((np.cos(phi)*np.cos(theta))**2 + np.sin(theta)**2 + np.cos(phi)**2)

class MassCalculator:
    """Calculator class for performing the mass integral
    """

    def __init__(self, mass_model: distributions.MassDistributionModel,
                 sz_calc: SZCalculator,
                 integrator: integrators.Integrator,
                 M_min: u.Quantity = 1e11*u.M_sun, M_max: u.Quantity = 1e16*u.M_sun,
                 num_pts: int = 1000, dist='log'):
        """Calculator class for performing the mass integral

        Args:
            mass_model (distributions.MassDistributionModel): Mass distribution model to use
            sz_calc (SZCalculator): Sunyaev-Zeldovich calculator to use
            integrator (integrators.Integrator): Integrator class to use for the calculations
            M_min (u.Quantity, optional): Minimum structure mass to integrate over. Defaults to 1e11*u.M_sun.
            M_max (u.Quantity, optional): Maximum structure mass. Defaults to 1e16*u.M_sun.
            num_pts (int, optional): Number of points to evaluate in integral. Defaults to 1000.
            dist (str, optional): Integral point distribution, can be 'log'arithmic or 'lin'ear. Defaults to 'log'.
        """
        self.mass_model = mass_model
        self.sz_calc = sz_calc
        self.integrator = integrator
        self.M_min = M_min
        self.M_max = M_max
        self.num_pts = num_pts
        self.dist = dist

    def getMassIntegral(self, l: u.Quantity, z: u.Quantity):
        """Performs the mass integral from Equation 1

        Args:
            l (u.Quantity): Array-like of angular scales
            z (u.Quantity): Array-like of redshift values

        Returns:
            u.Quantity: Array-like result of the mass integral at provided z values
        """

        # Filter out unique l values only
        unique_l, unique_counts = np.unique(l, return_counts=True)
        func = lambda M: self.mass_model.getMassDistribution(z, M)*\
            self.distribute_l(z, M, unique_l, np.array(unique_counts, dtype=object), self.sz_calc)

        return self.integrator(func, self.M_min, self.M_max, num_pts=self.num_pts, dist=self.dist, axis=1, ndmin=4)
    
    @staticmethod
    def distribute_l(z, M, unique_l, unique_counts, sz_calc: SZCalculator):
        """ Internal function for calculating the integral, parallelized by unique l values

        Args:
            z (u.Quantity): Array-like of redshift values
            M (u.Quantity): Array-like of structure masses
            unique_l (u.Quantity): Array-like of unique angular scales
            unique_counts (int): Array-like of exponent used for each unique angular scale
            sz_calc (SZCalculator): Sunyaev-Zeldovich calculator to use when calculating each y-parameter

        Returns:
            u.Quantity: Array-like of final integral
        """
        integ = 1
        for i in prange(len(unique_l)):
            integ *= sz_calc.getYTilde(z, M, unique_l[i])**unique_counts[i]
        return integ

class LineOfSightCalculator:
    """ Calculator class for performing the line-of-sight integral along redshift
    """
    def __init__(self, cosmology_model: cosmology.FLRW,
                 mass_calc: MassCalculator,
                 integrator: integrators.Integrator,
                 z_min: u.Quantity = 1e-6, z_max: u.Quantity = 5,
                 num_pts: int = 100, dist = 'log'):
        """Calculator class for performing the line-of-sight integral along redshift

        Args:
            cosmology_model (cosmology.FLRW): Astropy cosmology model to use
            mass_calc (MassCalculator): Mass calculator class to use
            integrator (integrators.Integrator): Integrator class to use for the calculation
            z_min (u.Quantity, optional): Minimum redshift value to evaluate. Defaults to 1e-6.
            z_max (u.Quantity, optional): Maximum redshift value to evaluate. Defaults to 5.
            num_pts (int, optional): Number of points to evaluate in the integral. Defaults to 100.
            dist (str, optional): Integral point distribution, can be 'log'arithmic or 'lin'ear. Defaults to 'log'.
        """
        self.cosmology_model = cosmology_model
        self.mass_calc = mass_calc
        self.integrator = integrator

        self.z_min = z_min
        self.z_max = z_max
        self.num_pts = num_pts
        self.dist = dist

    def getLOSIntegral(self, l: u.Quantity, pool=None):
        """ Perform the integral along the line-of-sight of the observer

        Args:
            l (u.Quantity): Array-like of angular scales, with rows corresponding to [l_1, l_2, ... l_i] for i-th order statistics
            pool (optional): Multiprocessing pool for parallel computing. Defaults to None.

        Returns:
            u.Quantity: Power spectrum, bispectrum, or higher order results, as in Equations 1 and2  
        """
        num_l = np.shape(l)[0]
        result = np.empty(num_l)

        # If one dimensional array of l, assume we want angular power spectrum
        if np.ndim(l) < 2:
            l = np.tile(np.expand_dims(l, 1), (1,2))

        if pool is not None:
            # Run each set of l values in parallel
            result = np.squeeze(np.array(pool.map(self.parallelFunc, l)))*u.sr
        else:
            # Loop through each l value, calculate sequentially
            for i in range(num_l):
                result[i] = self.parallelFunc(l[i, :])

        return result
    
    def parallelFunc(self, l: u.Quantity):
        """Internal function for parallelizing calculations across angular scales

        Args:
            l (u.Quantity): Scalar l value to calculate

        Returns:
            u.Quantity: Integral for a single l value
        """
        
        # Calculates the integral for a single l value
        integ_func = lambda z: self.mass_calc.getMassIntegral(l, z)*self.cosmology_model.differential_comoving_volume(z)
        return self.integrator(integ_func, self.z_min, self.z_max, num_pts=self.num_pts, dist=self.dist, axis=0, ndmin=4).to(u.sr)