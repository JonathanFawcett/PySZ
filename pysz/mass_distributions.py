from abc import ABC, abstractmethod
import numpy as np
import scipy
from astropy import cosmology
from astropy import constants
from astropy import units as u
from numba import njit, jit, prange

class MassDistributionModel(ABC):
    def getMassDistribution(self, z, M):
        pass

class MassDistributionCautun(MassDistributionModel):
    z_fit = [0.0, 0.5, 1.0, 1.5, 2.0]
    a_fit = [-0.366977, -0.4116775, -0.46416863, -0.50786612, -0.54501788]
    b_fit = [9.33717549, 10.36896198, 11.59627844, 12.60073412, 13.44676056]
    c_fit = [-53.43898052, -59.14115095, -66.09902676, -71.71366663, -76.38153011]

    def __init__(self, cosmology_model: cosmology.FLRW):
        self.cosmology_model = cosmology_model
        
    # @njit
    def getMassDistribution(self, z: u.Quantity, M: u.Quantity):
        if (np.log10(M/u.M_sun) < 11).any():
            raise ValueError('Cautun fit only valid for M >= 10^11 solar massses')

        # Interpolate to get fit values
        a = np.interp(z, self.z_fit, self.a_fit)
        b = np.interp(z, self.z_fit, self.b_fit)
        c = np.interp(z, self.z_fit, self.c_fit)

        h = self.cosmology_model.h
        logM = np.log10(M*h/u.M_sun)

        # Calculate according to parabolic fit
        dndlogM = 10**(a*logM**2 + b*logM + c)*h**3*u.Gpc**-3

        return dndlogM/(M*np.log(10))

class SigmaModel(ABC):
    @abstractmethod
    def getSigma(self, z: u.Quantity, M: u.Quantity):
        pass

    @abstractmethod
    def getSigmaDeriv(self, M: u.Quantity):
        # Return dln(1/sigma)/dM
        pass

class SigmaFitLopezHonorez(SigmaModel):
    def __init__(self, cosmology_model: cosmology.FLRW):
        self.cosmology_model = cosmology_model
        self.D0 = self.D(0)

    # @njit
    def getSigma(self, z: u.Quantity, M: u.Quantity):
        # From 1305.5094
        return self.D(z)/self.D0*np.exp(-1*(0.2506*(M/u.M_sun)**(0.07536)-2.6*(M/u.M_sun)**(0.001745)))
    
    # @njit
    def getSigmaDeriv(self, M: u.Quantity):
        # Returns dln(1/sigma)/dM
        # From 1305.5094
        return 0.2506*0.07536*((M/u.M_sun)**0.07536)/M-2.6*0.001745*((M/u.M_sun)**0.001745)/M
    
    # @njit
    def D(self, z: u.Quantity):
        # Correction factor for redshift
        # From 1305.5094
        Om = self.cosmology_model.Om(z)
        Ode = self.cosmology_model.Ode(z)

        return 5/2 * 1/(1+z) * Om/(Om**(4/7) - Ode + (1 + Om/2)*(1 + Ode/70))
    
class SigmaFitTinker(SigmaModel):
    # From linear fit of Tinker Fig. 5 axes
    fit_slope = -0.44900409313699874
    fit_intercept = 6.156125716494604

    # @njit
    def getSigma(self, z: u.Quantity, M: u.Quantity):
        # From linear fit of Tinker Fig. 5 axes
        return np.exp(self.fit_slope*np.log10(M/u.M_sun)+self.fit_intercept)
    
    # @njit
    def getSigmaDeriv(self, M: u.Quantity):
        # From linear fit of Tinker Fig. 5 axes
        return -self.fit_slope/(M/u.M_sun*np.log(10)) # dln(1/sigma)/dM
    
class MassDistributionTinker(MassDistributionModel):
    # TODO: add auto-convert for Delta, since Tinker uses rho_m instead of rho_crit

    def __init__(self, cosmology_model: cosmology.FLRW, sigma_model: SigmaModel, delta=500):
        self.cosmology_model = cosmology_model
        self.sigma_model = sigma_model
        self.delta = delta

        # Fit parameters from Tinker Eq. B1-B4
        log_delta = np.log10(delta)
        if delta < 1600:
            self.A = 0.1*log_delta - 0.05
        else:
            self.A = 0.26

        self.a = 1.43 + (np.max([log_delta - 2.3, 0]))**1.5
        self.b = 1.0 + (np.max([log_delta - 1.6, 0]))**-1.5
        self.c = 1.2 + (np.max([log_delta - 2.35, 0]))**1.6

    # @njit
    def getMassDistribution(self, z: u.Quantity, M: u.Quantity):
        # From Tinker, Eq. 2
        sigma = self.sigma_model.getSigma(z, M)
        mean_density = self.cosmology_model.critical_density(z)*self.cosmology_model.Om(z) # From Tinker Eq. 1
        # return self.f(sigma) * mean_density/(M*constants.M_sun) * self.sigma_model.getSigmaDeriv(M)
        return self.f(sigma) * mean_density/M * self.sigma_model.getSigmaDeriv(M)

    def f(self, sigma):
        # From Tinker, Eq. 3
        return self.A*((sigma/self.b)**(-self.a)+1)*np.exp(-self.c/sigma**2)