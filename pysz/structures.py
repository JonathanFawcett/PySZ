# This module includes the models of clusters and filaments derived from literature

from .imports import *

class StructureModel(ABC):
    """Abstract structure class, for use with the SZ calculators
    """
    @staticmethod
    def logScaleByMass(alpha, beta, M):
        """Static method to scale logarithmically by mass

        Args:
            alpha (float): Array-like of exponents to use
            beta (float): Array-like of normalizations to use
            M (u.Quantity): Mass to be scaled

        Returns:
            u.Quantity: Scaled quantity, where Y = beta*(M/M_sun)**alpha, from [6] (Gheller et. al.)
        """
        return (10**beta)*(M/u.M_sun)**alpha

class FilamentModel(StructureModel):
    """Abstract filament model, assuming a constant profile along its length
    """
    @abstractmethod
    def getFilamentRadius(self, z: u.Quantity, M: u.Quantity):
        """Abstract method to obtain the radius of a filament at a certain redshift and mass

        Args:
            z (u.Quantity): Array-like of redshift values
            M (u.Quantity): Array-like of mass values

        Returns:
            u.Quantity: Array-like of filament radii
        """
        pass

    @abstractmethod
    def getFilamentLength(self, z: u.Quantity, M: u.Quantity):
        """Abstract method to obtain the length of a filament at a certain redshift and mass

        Args:
            z (u.Quantity): Array-like of redshift values
            M (u.Quantity): Array-like of mass values

        Returns:
            u.Quantity: Array-like of filament lengths
        """
        pass

class ClusterModel(StructureModel):
    """Abstract cluster model, assuming a radial distribution
    """
    @abstractmethod
    def getClusterRadius(self, z: u.Quantity, M: u.Quantity):
        """Abstract method to obtain the radius of a cluster at a certain redshift and mass

        Args:
            z (u.Quantity): Array-like of redshift values
            M (u.Quantity): Array-like of mass values

        Returns:
            u.Quantity: Array-like of cluster radii
        """
        pass

    @abstractmethod
    def getElectronPressure(self, z: u.Quantity, M: u.Quantity, x):
        """Abstract method to obtain the electron pressure of a cluster at a specific point along its radius

        Args:
            z (u.Quantity): Array-like of redshift values
            M (u.Quantity): Array-like of mass values
            x (float): Reduced radial distance from the center of the cluster, x = r/R_cluster

        Returns:
            u.Quantity: Array-like of electron pressures
        """
        pass

class FilamentModelGheller(FilamentModel):
    """Filament model based on data from the BOX100 results in Tables 2 and 3 of [6] (Gheller et. al.)
    """
    z_data = [0.0, 0.5, 1.0]
    alpha_V = [0.824, 0.850, 0.864]
    beta_V = [-9.525, -9.712, -9.990]
    alpha_L = [0.381, 0.427, 0.440]
    beta_L = [-3.795, -4.210, -4.390]
    alpha_T = [0.431, 0.375, 0.375]
    beta_T = [0.377, 1.035, 1.193]

    def getFilamentRadius(self, z: u.Quantity, M: u.Quantity):
        """Calculates the characteristic radius of a filament assuming a constant density cylinder, as in [6] (Gheller et. al.)

        Args:
            z (u.Quantity): Array-like of redshift values
            M (u.Quantity): Array-like of masses

        Returns:
            u.Quantity: Array-like of filament radii
        """
        return np.sqrt(self.getFilamentVolume(z, M) / (np.pi * self.getFilamentLength(z, M)))

    def getFilamentVolume(self, z: u.Quantity, M: u.Quantity):
        """Calculates the volume of a filament using BOX100 data from [6] (Gheller et. al.)

        Args:
            z (u.Quantity): Array-like of redshift values
            M (u.Quantity): Array-like of masses

        Returns:
            u.Quantity: Array-like of filament volumes
        """
        alpha = np.interp(z, self.z_data, self.alpha_V)
        beta = np.interp(z, self.z_data, self.beta_V)
        return self.logScaleByMass(alpha, beta, M)*(u.Mpc**3)

    def getFilamentLength(self, z: u.Quantity, M: u.Quantity):
        """Calculates the length of a filament using BOX100 data from [6] (Gheller et. al.)

        Args:
            z (u.Quantity): Array-like of redshift values
            M (u.Quantity): Array-like of masses

        Returns:
            u.Quantity: Array-like of filament lengths
        """
        alpha = np.interp(z, self.z_data, self.alpha_L)
        beta = np.interp(z, self.z_data, self.beta_L)
        return self.logScaleByMass(alpha, beta, M)*(u.Mpc)
    
    def getTemperatureScaling(self, z: u.Quantity, M: u.Quantity,
                              base_T: u.Quantity = 1001072.3*u.K):
        """Calculates the correction factor for the electron pressure due to temperature scaling, using BOX100 data from [6] (Gheller et. al.)

        Args:
            z (u.Quantity): Array-like of redshift values
            M (u.Quantity): Array-like of masses
            base_T (u.Quantity, optional): Temperature of a characteristic filament. Defaults to 1001072.3*u.K, assuming a 15 Mpc characteristic filament at z=0, as calculated in TestGheller/test_base_temperature.

        Returns:
            u.Quantity: Array-like of temperature correction factors
        """
        alpha = np.interp(z, self.z_data, self.alpha_T)
        beta = np.interp(z, self.z_data, self.beta_T)
        return self.logScaleByMass(alpha, beta, M)*u.K/base_T

class ClusterModelArnaud(ClusterModel):
    """Cluster model based on REXCESS data in [21] (Arnaud et. al.)
    """
    def __init__(self, cosmology_model: cosmology.FLRW):
        """Cluster model based on REXCESS data in [21] (Arnaud et. al.)

        Args:
            cosmology_model (cosmology.FLRW): Cosmology model to use to determine the critical density
        """
        self.cosmology_model = cosmology_model

    def getClusterRadius(self, z: u.Quantity, M: u.Quantity):
        """Calculates the R_500 radius of a cluster, using data from [21] (Arnaud et. al.)

        Args:
            z (u.Quantity): Array-like of redshift values
            M (u.Quantity): Array-like of masses

        Returns:
            u.Quantity: Array-like of cluster radii, assuming a constant density sphere per Footnote 1 in [21] (Arnaud et. al.)
        """

        # Use R_500 per Arnaud, assume constant density sphere per Arnaud Footnote 1
        delta = 500
        return (M/(4*np.pi/3*delta*self.cosmology_model.critical_density(z)))**(1/3)
    
    def getElectronPressure(self, z: u.Quantity, M: u.Quantity, x,
                            P_0_h70 = 8.403, c_500 = 1.177, 
                            gamma = 0.3081, alpha = 1.0510, beta = 5.4905):
        """Calculates the electron pressure at a specific radial distance, using data from [21] (Arnaud et. al.)

        Args:
            z (u.Quantity): Array-like of redshift values
            M (u.Quantity): Array-like of masses
            x (float): Reduced radial distance from the center of the cluster, x = r/R_500
            P_0_h70 (float, optional): Central electron pressure corrected by h_70. Defaults to 8.403 per Equation 12 from [21] (Arnaud et. al.).
            c_500 (float, optional): Correction factor to the cluster radius. Defaults to 1.177 per Equation 12 from [21] (Arnaud et. al.).
            gamma (float, optional): Central slope of the cluster profile for x << 1/c_500. Defaults to 0.3081 per Equation 12 from [21] (Arnaud et. al.).
            alpha (float, optional): Intermediate slope of the cluster profile for x ~ 1/c_500. Defaults to 1.0510 per Equation 12 from [21] (Arnaud et. al.).
            beta (float, optional): Outer slope of the cluster profile for x >> 1/c_500. Defaults to 5.4905 per Equation 12 from [21] (Arnaud et. al.).

        Returns:
            u.Quantity: Array-like of electron pressures, calculated using Equation 13 in [21] (Arnaud et. al.) 
        """

        alpha_P = 0.12 # From Arnaud Eq. 7
        alpha_P_prime = 0.10 - (alpha_P + 0.10)*((x/0.5)**3)/(1+(x/0.5)**3) # From Arnaud Eq. 9
        h_70 = self.cosmology_model.H(z)/(70*u.km/(u.s*u.Mpc))
        P_0 = P_0_h70*h_70**(-3/2)

        # Arnaud Eq. 13
        return 1.65e-3 * h_70**(8/3)*((M/u.M_sun)/(3e14/h_70))**(2/3 + alpha_P + alpha_P_prime)*(self.universalProfile(x, P_0=P_0, c_500=c_500, gamma=gamma, alpha=alpha, beta=beta))*(h_70**2)*(u.keV*u.cm**-3)

    def universalProfile(self, x,
                         P_0 = 8.403, c_500 = 1.177, 
                         gamma = 0.3081, alpha = 1.0510, beta = 5.4905):
        """Internal function to calculate the universal profile from [21] (Arnaud et. al.)

        Args:
            x (float): Reduced radial distance from the center of the cluster, x = r/R_500
            P_0 (float, optional): Central electron pressure. Defaults to 8.403 per Equation 12 from [21] (Arnaud et. al.).
            c_500 (float, optional): Correction factor to the cluster radius. Defaults to 1.177 per Equation 12 from [21] (Arnaud et. al.).
            gamma (float, optional): Central slope of the cluster profile for x << 1/c_500. Defaults to 0.3081 per Equation 12 from [21] (Arnaud et. al.).
            alpha (float, optional): Intermediate slope of the cluster profile for x ~ 1/c_500. Defaults to 1.0510 per Equation 12 from [21] (Arnaud et. al.).
            beta (float, optional): Outer slope of the cluster profile for x >> 1/c_500. Defaults to 5.4905 per Equation 12 from [21] (Arnaud et. al.).

        Returns:
            u.Quantity: Array-like of universal profile values, calculated using Equation 11 from [21] (Arnaud et. al.)
        """
        # Arnaud Eq. 11
        return P_0/((c_500*x)**gamma * (1+(c_500*x)**alpha)**((beta-gamma)/alpha))