from abc import ABC, abstractmethod
import numpy as np
import scipy
from astropy import cosmology
from astropy import constants
from astropy import units as u
from numba import njit, jit, prange

from .integrators import *
from .structure_models import *
from .mass_distributions import *

class SZCalculator(ABC):
    sigma_T = constants.sigma_T
    m_e = constants.m_e
    c = constants.c

    @abstractmethod
    def getYTilde(self, z: u.Quantity, M: u.Quantity, l: u.Quantity):
        pass

class GeneralSZCalculator(SZCalculator):
    def __init__(self, integrator: Integrator, cosmology_model: cosmology.FLRW, cluster_model: ClusterModel,
                 R_min = 1e-5, R_max = 5, num_pts: int = 100):
        self.integrator = integrator
        self.cosmology_model = cosmology_model
        self.structure_model = cluster_model

        self.R_min = R_min
        self.R_max = R_max
        self.num_pts = num_pts

        self.prefactor = 4*np.pi*self.sigma_T/(self.m_e*self.c**2)
    
    # @njit
    def getYTilde(self, z: u.Quantity, M: u.Quantity, l: u.Quantity):
        scale_radius = self.structure_model.getClusterRadius(z, M)
        l_s = self.cosmology_model.angular_diameter_distance(z)/(u.rad*scale_radius)
        func = lambda x: (x**2)*self.structure_model.getElectronPressure(z, M, x)*np.sinc(l*x/(l_s*np.pi))

        return self.prefactor*(scale_radius/l_s**2)*self.integrator(func, self.R_min, self.R_max, num_pts=self.num_pts, axis=2,  ndmin=4)

class GaussianFilamentSZCalculator(SZCalculator):

    def __init__(self, integrator: Integrator, cosmology_model: cosmology.FLRW, filament_model: FilamentModel,
                 A_1: u.Quantity = 1.2e-5*u.keV*u.cm**-3, A_2: u.Quantity = 2.66e-5*u.keV*u.cm**-3, 
                 c_1=0.5186, c_2=0.1894, 
                 theta_min: u.Quantity = 0*u.rad, theta_max: u.Quantity = np.pi*u.rad, 
                 phi_min: u.Quantity = 0*u.rad, phi_max: u.Quantity = 2*np.pi*u.rad, 
                 num_pts: int = 20):
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

    # @njit
    def getYTilde(self, z: u.Quantity, M: u.Quantity, l: u.Quantity):    
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
        return k * np.sqrt((np.cos(phi)*np.sin(theta))**2 + np.cos(theta)**2)
    
    def k_r_sq(self, k: u.Quantity, theta: u.Quantity, phi: u.Quantity):
        return k**2 * ((np.cos(phi)*np.cos(theta))**2 + np.sin(theta)**2 + np.cos(phi)**2)

class MassCalculator:
    def __init__(self, mass_model: MassDistributionModel, sz_calc: SZCalculator, integrator: Integrator,
                 M_min: u.Quantity = 1e11*u.M_sun, M_max: u.Quantity = 1e16*u.M_sun,
                 num_pts: int = 1000, dist='log'):
        self.mass_model = mass_model
        self.sz_calc = sz_calc
        self.integrator = integrator
        self.M_min = M_min
        self.M_max = M_max
        self.num_pts = num_pts
        self.dist = dist

    def getMassIntegral(self, l: u.Quantity, z: u.Quantity):
        # Filter out unique l values only
        unique_l, unique_counts = np.unique(l, return_counts=True)
        func = lambda M: self.mass_model.getMassDistribution(z, M)*\
            self.distribute_l(z, M, unique_l, np.array(unique_counts, dtype=object), self.sz_calc)

        return self.integrator(func, self.M_min, self.M_max, num_pts=self.num_pts, dist=self.dist, axis=1, ndmin=4)
    
    #TODO re-evaluate
    @staticmethod
    # @jit(parallel=True, forceobj=True)
    def distribute_l(z, M, unique_l, unique_counts, sz_calc: SZCalculator):
        integ = 1
        for i in prange(len(unique_l)):
            integ *= sz_calc.getYTilde(z, M, unique_l[i])**unique_counts[i]
        return integ

class LineOfSightCalculator:
    def __init__(self, cosmology_model: cosmology.FLRW, mass_calc: MassCalculator, integrator: Integrator,
                 z_min: u.Quantity = 1e-6, z_max: u.Quantity = 5,
                 num_pts: int = 100, dist = 'log'):
        self.cosmology_model = cosmology_model
        self.mass_calc = mass_calc
        self.integrator = integrator

        self.z_min = z_min
        self.z_max = z_max
        self.num_pts = num_pts
        self.dist = dist

    def getLOSIntegral(self, l: u.Quantity, pool=None):
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
        # Calculates the integral for a single l value
        integ_func = lambda z: self.mass_calc.getMassIntegral(l, z)*self.cosmology_model.differential_comoving_volume(z)
        return self.integrator(integ_func, self.z_min, self.z_max, num_pts=self.num_pts, dist=self.dist, axis=0, ndmin=4).to(u.sr)