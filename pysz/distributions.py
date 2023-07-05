# This module includes the structure distribution classes that calculate the number density, derived from literature

from .imports import *

class MassDistributionModel(ABC):
    """Abstract mass distribution class
    """
    def getMassDistribution(self, z, M):
        """Abstract function to calculate the number density of structures per unit mass

        Args:
            z (u.Quantity): Array-like of redshift values
            M (u.Quantity): Array-like of masses

        Returns:
            u.Quantity: Array-like of number densities per unit mass
        """
        pass

class MassDistributionCautun(MassDistributionModel):
    """Mass distribution of filaments based on a 2nd order fit of data from [20] (Cautun et. al.)
    """

    # Least-squared values of a 2nd order polynomial fit of data from Figure 31 of [20] (Cautun et. al.)
    z_fit = [0.0, 0.5, 1.0, 1.5, 2.0]
    a_fit = [-0.366977, -0.4116775, -0.46416863, -0.50786612, -0.54501788]
    b_fit = [9.33717549, 10.36896198, 11.59627844, 12.60073412, 13.44676056]
    c_fit = [-53.43898052, -59.14115095, -66.09902676, -71.71366663, -76.38153011]

    def __init__(self, cosmology_model: cosmology.FLRW):
        """Mass distribution of filaments based on a 2nd order fit of data from [20] (Cautun et. al.)
        Args:
            cosmology_model (cosmology.FLRW): Cosmology model to use when determining h
        """
        self.cosmology_model = cosmology_model
        
    def getMassDistribution(self, z: u.Quantity, M: u.Quantity):
        """Calculates number density of filaments per unit mass, interpolating data for z=0 to z=2 from [20] (Cautun et. al.)

        Args:
            z (u.Quantity): Array-like of redshift values
            M (u.Quantity): Array-like of masses

        Returns:
            u.Quantity: Array-like of number densities per unit mass
        """

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
    """Abstract class for calculating sigma values, which represent the smoothed RMS variance of the linear density field
    """
    @abstractmethod
    def getSigma(self, z: u.Quantity, M: u.Quantity):
        """Abstract method to calculate sigma values

        Args:
            z (u.Quantity): Array-like of redshifts
            M (u.Quantity): Array-like of masses

        Returns:
            u.Quantity: Array-like of sigma values, representing the smoothed RMS variance of the linear density field
        """
        pass

    @abstractmethod
    def getSigmaDeriv(self, M: u.Quantity):
        """Abstract method to calculates dln(1/sigma)/dM

        Args:
            M (u.Quantity): Array-like of masses

        Returns:
            u.Quantity: Array-like of derivatives of the natural logarithm of sigma inverse
        """
        pass

class SigmaFitLopezHonorez(SigmaModel):
    """Class to calculate sigma values based on a fit from [23] (Lopez-Honorez et. al.)
    """
    def __init__(self, cosmology_model: cosmology.FLRW):
        """Class to calculate sigma values based on a fit from [23] (Lopez-Honorez et. al.)

        Args:
            cosmology_model (cosmology.FLRW): Cosmology model to use when determining density factors
        """
        self.cosmology_model = cosmology_model
        self.D0 = self.D(0)

    def getSigma(self, z: u.Quantity, M: u.Quantity):
        """Calculates sigma values based on a fit from [23] (Lopez-Honorez et. al.)

        Args:
            z (u.Quantity): Array-like of redshifts
            M (u.Quantity): Array-like of masses

        Returns:
            u.Quantity: Array-like of sigma values, representing the smoothed RMS variance of the linear density field
        """

        # From Equations B14 & B15 of [23] (Lopez-Honorez et. al.)
        return self.D(z)/self.D0*np.exp(-1*(0.2506*(M/u.M_sun)**(0.07536)-2.6*(M/u.M_sun)**(0.001745)))
    
    def getSigmaDeriv(self, M: u.Quantity):
        """Calculates dln(1/sigma)/dM based on a fit from [23] (Lopez-Honorez et. al.)

        Args:
            M (u.Quantity): Array-like of masses

        Returns:
            u.Quantity: Array-like of derivatives of the natural logarithm of sigma inverse
        """

        # Returns dln(1/sigma)/dM
        # From Equation B15 of [23] (Lopez-Honorez et. al.)
        return 0.2506*0.07536*((M/u.M_sun)**0.07536)/M-2.6*0.001745*((M/u.M_sun)**0.001745)/M
    
    def D(self, z: u.Quantity):
        """Calculates the redshift correction factor from [23] (Lopez-Honorez et. al.)

        Args:
            z (u.Quantity): Array-like of redshift values

        Returns:
            u.Quantity: Array-like of redshift correction factors, D(z)
        """

        Om = self.cosmology_model.Om(z)
        Ode = self.cosmology_model.Ode(z)

        # From Equation B17 of [23] (Lopez-Honorez et. al.)
        return 5/2 * 1/(1+z) * Om/(Om**(4/7) - Ode + (1 + Om/2)*(1 + Ode/70))
    
class SigmaFitTinker(SigmaModel):
    """Class to calculate sigma values based on a linear fit from [22] (Tinker et. al.)
    """

    # Least-squares linear fit of the axes of Figure 5 from [22] (Tinker et. al.)
    fit_slope = -0.44900409313699874
    fit_intercept = 6.156125716494604

    def getSigma(self, z: u.Quantity, M: u.Quantity):
        """Calculates sigma based on a linear fit of the axes of Figure 5 from [22] (Tinker et. al.)

        Args:
            z (u.Quantity): Array-like of redshift values
            M (u.Quantity): Array-like of masses

        Returns:
            u.Quantity: Array-like of sigma values
        """
        # From linear fit of Tinker Fig. 5 axes
        return np.exp(self.fit_slope*np.log10(M/u.M_sun)+self.fit_intercept)
    
    def getSigmaDeriv(self, M: u.Quantity):
        """Calculates dln(1/sigma)/dM based on a linear fit of the axes of Figure 5 from [22] (Tinker et. al.)

        Args:
            M (u.Quantity): Array-like of masses

        Returns:
            u.Quantity: Array-like of derivatives of the natural logarithm of sigma inverse
        """

        # From linear fit of Tinker Fig. 5 axes
        return -self.fit_slope/(M/u.M_sun*np.log(10)) # dln(1/sigma)/dM
    
class MassDistributionTinker(MassDistributionModel):
    """Mass distribution of clusters based on data from [22] (Tinker et. al.)
    """

    def __init__(self, cosmology_model: cosmology.FLRW, sigma_model: SigmaModel, delta: float=1600):
        """Mass distribution of clusters based on a fitting function from [22] (Tinker et. al.)

        Args:
            cosmology_model (cosmology.FLRW): Cosmology model to use when determining densities
            sigma_model (SigmaModel): Model to use when calculating sigma
            delta (float, optional): Overdensity ratio of cluster compared to the mean density (NOT critical density) of the universe. Defaults to 1600.
        """
        self.cosmology_model = cosmology_model
        self.sigma_model = sigma_model
        self.delta = delta

        # Fit parameters from Equations B1 to B4 of [22] (Tinker et. al.)
        log_delta = np.log10(delta)
        if delta < 1600:
            self.A = 0.1*log_delta - 0.05
        else:
            self.A = 0.26

        self.a = 1.43 + (np.max([log_delta - 2.3, 0]))**1.5
        self.b = 1.0 + (np.max([log_delta - 1.6, 0]))**-1.5
        self.c = 1.2 + (np.max([log_delta - 2.35, 0]))**1.6

    def getMassDistribution(self, z: u.Quantity, M: u.Quantity):
        """Calculates the number density of clusters per unit mass, based on a fitting function from [22] (Tinker et. al.)

        Args:
            z (u.Quantity): Array-like of redshift values
            M (u.Quantity): Array-like of masses

        Returns:
            u.Quantity: Array-like of number densities per unit mass, calculated using Equation 2 of [22] (Tinker et. al.)
        """
        
        sigma = self.sigma_model.getSigma(z, M)
        mean_density = self.cosmology_model.critical_density(z)*self.cosmology_model.Om(z) # From Tinker Eq. 1

        # From Eq. 2 of Tinker
        return self.f(sigma.value, self.A, self.a, self.b, self.c) * mean_density/M * self.sigma_model.getSigmaDeriv(M)
    
    @staticmethod
    @njit
    def f(sigma:float, A:float, a:float, b:float, c:float):
        """Compiled internal function to compute f(sigma) from [22] (Tinker et. al.)

        Args:
            sigma (float): Array-like of sigma values, representing the smoothed RMS variance of the linear density field
            A (float): Fit parameter from Equation B1
            a (float): Fit parameter from Equation B2
            b (float): Fit parameter from Equation B3
            c (float): Fit parameter from Equation B4

        Returns:
            u.Quantity: Array-like of f(sigma), calculated using Equation 3 of [22] (Tinker et. al.)
        """
        # From Eq. 3 of Tinker
        return A*((sigma/b)**(-a)+1)*np.exp(-c/sigma**2)