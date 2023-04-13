from abc import ABC, abstractmethod
import numpy as np
import scipy
from astropy import cosmology
from astropy import constants
from astropy import units as u
from numba import njit, jit, prange

class StructureModel(ABC):
    def logScaleByMass(self, alpha, beta, M):
        return (10**beta)*(M/u.M_sun)**alpha

class FilamentModel(StructureModel):
    @abstractmethod
    def getFilamentRadius(self, z: u.Quantity, M: u.Quantity):
        pass

    @abstractmethod
    def getFilamentLength(self, z: u.Quantity, M: u.Quantity):
        pass

class ClusterModel(StructureModel):
    @abstractmethod
    def getClusterRadius(self, z: u.Quantity, M: u.Quantity):
        pass

    @abstractmethod
    def getElectronPressure(self, z: u.Quantity, M: u.Quantity, x):
        pass

class FilamentModelGheller(FilamentModel):
    # Using BOX100 results (more accurate?) from Tables 2 and 3 of 1607.01406
    z_data = [0.0, 0.5, 1.0]
    alpha_V = [0.824, 0.850, 0.864]
    beta_V = [-9.525, -9.712, -9.990]
    alpha_L = [0.381, 0.427, 0.440]
    beta_L = [-3.795, -4.210, -4.390]
    alpha_T = [0.431, 0.375, 0.375]
    beta_T = [0.377, 1.035, 1.193]

    # @njit
    def getFilamentRadius(self, z: u.Quantity, M: u.Quantity):
        return np.sqrt(self.getFilamentVolume(z, M) / (np.pi * self.getFilamentLength(z, M)))

    # @njit
    def getFilamentVolume(self, z: u.Quantity, M: u.Quantity):
        alpha = np.interp(z, self.z_data, self.alpha_V)
        beta = np.interp(z, self.z_data, self.beta_V)
        return self.logScaleByMass(alpha, beta, M)*(u.Mpc**3)

    # @njit
    def getFilamentLength(self, z: u.Quantity, M: u.Quantity):
        alpha = np.interp(z, self.z_data, self.alpha_L)
        beta = np.interp(z, self.z_data, self.beta_L)
        return self.logScaleByMass(alpha, beta, M)*(u.Mpc)
    
    # @njit
    def getTemperatureScaling(self, z: u.Quantity, M: u.Quantity,
                              base_T: u.Quantity = 1001072.3*u.K):
        alpha = np.interp(z, self.z_data, self.alpha_T)
        beta = np.interp(z, self.z_data, self.beta_T)
        return self.logScaleByMass(alpha, beta, M)*u.K/base_T

class ClusterModelArnaud(ClusterModel):
    def __init__(self, cosmology_model: cosmology.FLRW):
        self.cosmology_model = cosmology_model

    # @njit
    def getClusterRadius(self, z: u.Quantity, M: u.Quantity):
        # Use R_500 per Arnaud, assume constant density sphere per Arnaud Footnote 1
        delta = 500
        return (M/(4*np.pi/3*delta*self.cosmology_model.critical_density(z)))**(1/3)
    
    # @njit
    def getElectronPressure(self, z: u.Quantity, M: u.Quantity, x,
                            P_0_h70 = 8.403, c_500 = 1.177, 
                            gamma = 0.3081, alpha = 1.0510, beta = 5.4905):
        # Default fit values for universalProfile from Arnaud Eq. 12
        alpha_P = 0.12 # From Arnaud Eq. 7
        alpha_P_prime = 0.10 - (alpha_P + 0.10)*((x/0.5)**3)/(1+(x/0.5)**3) # From Arnaud Eq. 9
        h_70 = self.cosmology_model.H(z)/(70*u.km/(u.s*u.Mpc))
        P_0 = P_0_h70*h_70**(-3/2)

        # Arnaud Eq. 13
        return 1.65e-3 * h_70**(8/3)*((M/u.M_sun)/(3e14/h_70))**(2/3 + alpha_P + alpha_P_prime)*(self.universalProfile(x, P_0=P_0, c_500=c_500, gamma=gamma, alpha=alpha, beta=beta))*(h_70**2)*(u.keV*u.cm**-3)

    # @njit
    def universalProfile(self, x,
                         P_0 = 8.403, c_500 = 1.177, 
                         gamma = 0.3081, alpha = 1.0510, beta = 5.4905):
        # Equation from Arnaud Eq. 11, default fit values from Eq. 12
        return P_0/((c_500*x)**gamma * (1+(c_500*x)**alpha)**((beta-gamma)/alpha))