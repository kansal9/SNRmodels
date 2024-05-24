
import numpy as np
import snr_gui as gui
from scipy.integrate import quad, nquad
import functools
from helpers import *

#####################################################################################################################
class SNREmissivity:
    """Calculate and store emissivity data for a specific instance of a SuperNovaRemnant.

    Attributes:
        data (dict): dictionary of input parameters for emissivity model
        nrg_to_edt (float): conversion factor to get unitless energy divided by shock temperature from energy
        plots (dict): dictionary of OutputPlot instances shown on emissivity window
        cnst (dict): constants used in emissivity calculations
        root (str): ID of emissivity window, used to access widgets
    """
######################################################################################################################
    def __init__(self, snr, root):
        """Create instance of SNREmissivity class. Note that caching is used for functions when possible to speed up
        calculations. In some cases, scalar versions of vector functions are created so that a cache can be used (see
        scalar_temperature and scalar_density).

        Args:
            snr (SuperNovaRemnant): SuperNovaRemnant instance, used to pass input parameters to emissivity model
            root (str): ID of emissivity window, used to access widgets
        """
        self.widgets = {'model': None, 'c_tau': None}
        keys = ("abundance", "ej_abundance", "mu_H", "mu_e", "mu_I", "mu_H_ej", "mu_e_ej", "mu_I_ej", "Z_sq", "Z_sq_ej",
                "n_0")
        self.data = {key: snr.data[key] for key in keys}
        # r_c is the radius of the contact discontinuity, em_point lists points of difficulty in EM integral
        self.data.update({"r_c": 0, "em_point": None, "radius": snr.calc["r"] * PC_TO_KM * 100000,
                          "T_s": snr.calc["T"], "r_min": 0})
        self.nrg_to_edt = KEV_TO_ERG / BOLTZMANN / self.data["T_s"]
        # Graphs defined in snr.py
        self.plots = {}
        self.cnst = {"intensity": (2 * (4 * self.data["n_0"]) ** 2 * self.data["mu_H"] ** 2 / self.data["mu_e"] /
                                   self.data["mu_I"] * self.data["radius"] / self.data["T_s"] ** 0.5)}
        self.cnst["spectrum"] = 8 * np.pi ** 2 * self.data["radius"] ** 2 * self.cnst["intensity"]
        self.cnst["luminosity"] = self.data["T_s"] * BOLTZMANN / PLANCK * self.cnst["spectrum"]
        self.root = root
        
        if((snr.data["t"] <= snr.calc["t_st"]) or (snr.data["s"] == 2)):
            self.data["model"] = "chev"
            if snr.data["s"] == 0:
                if snr.data["n"] >= 6:
                    self.data["r_c"] = 1 / S0_CHEV_RMAX[snr.data["n"]]    #where 1.181 is is the max value for r on each file
                    self.data["r_min"] = S0_CHEV_RMIN[snr.data["n"]] * self.data["r_c"] #where 0.935 is is the min value for r on each file
                    # r_difficult gives radius values at which pressure/temperature change dramatically
                    self.data["r_difficult"] = [self.data["r_c"]]
                    self.data["em_point"] = [self.data["r_difficult"], None]
            else:  #s=2 case
                self.data["r_c"] = 1 / S2_CHEV_RMAX[snr.data["n"]]
                self.data["r_min"] = S2_CHEV_RMIN[snr.data["n"]] * self.data["r_c"]
                self.data["r_difficult"] = [0.9975 * self.data["r_c"], 1.02 * self.data["r_c"]]
                self.data["em_point"] = [[self.data["r_difficult"][0]], [self.data["r_difficult"][1]]]
            # ej_correction used to account for different composition of reverse shock (ejecta vs. ISM abundances)
            self.data["ej_correction"] = (self.data["Z_sq_ej"] / self.data["Z_sq"] * self.data["mu_I"] /
                                          self.data["mu_I_ej"] * self.data["mu_e"] / self.data["mu_e_ej"])
            # Get radius, density, and pressure profiles
            lines = np.loadtxt("data/Chev_s{0:.0f}n{1}.txt".format(snr.data["s"], snr.data["n"]))
            radius = lines[:, 0] * self.data["r_c"]
            density = lines[:, 1]
            pressure = lines[:, 2]
            mu_norm = np.ones_like(radius)
            mu_norm[np.where(radius < self.data["r_c"])] = (1 / self.data["mu_e"] + 1 / self.data["mu_I"]) / (
                1 / self.data["mu_e_ej"] + 1 / self.data["mu_I_ej"])
            temperature = pressure / density * mu_norm
            # Interpolate density and temperature profiles to create density and temperature functions
            self.vector_density = lambda x: np.interp(x, radius, density)
            self.vector_temperature = lambda x: np.interp(x, radius, temperature)
            self.scalar_temperature = functools.lru_cache(maxsize=None)(self.vector_temperature)
            self.scalar_density = functools.lru_cache(maxsize=None)(self.vector_density)
            # Overwrite default class methods
            self._s_lim = self._chev_s_lim
            self._s_point = self._chev_s_point
            self._opt_dict = self._chev_opt_dict
            
        elif (snr.data["model"] == "cism"):
            self.data["model"] = "cism"
            self.data["c_tau"] = snr.data["c_tau"]
            self.data["em_point"] = []
            # Get temperature and density profiles
            self.vector_temperature = self._file_interpTemp("WLsoln")
            self.vector_density = self._file_interpDens("WLsoln")
            self.scalar_temperature = functools.lru_cache(maxsize=None)(self.vector_temperature)
            self.scalar_density = functools.lru_cache(maxsize=None)(self.vector_density)
        elif ((snr.data["model"] == "cism") and (snr.data["c_tau"] == -1)):
            self.data["model"] = "cism"
            self.data["c_tau"] = snr.data["c_tau"]
            self.data["em_point"] = []
            self.vector_temperature = self._file_interpTemp("WLsoln", self.data["T_s"])
            self.scalar_temperature = self._sedov_scalar_temp
            self.vector_density = self._file_interpDens("WLsoln")
            self.scalar_density = functools.lru_cache(maxsize=None)(self.vector_density)
        else:
            self.data["model"] = "sedov"
            self.vector_temperature = self._sedov_vector_temp
            self.scalar_temperature = self._sedov_scalar_temp
            self.vector_density = self._sedov_vector_density
            self.scalar_density = self._sedov_scalar_density
        # Cache instance methods
        self._jnu_scaled = functools.lru_cache(maxsize=None)(self._jnu_scaled)
        self._intensity_integrand = functools.lru_cache(maxsize=None)(self._intensity_integrand)
        self._luminosity_integrand = functools.lru_cache(maxsize=None)(self._luminosity_integrand)

########################################################################################################################
    def _file_interpTemp(self, prefix, multiplier=1):
        """Read CISM data file and linearly interpolate temperature and density data.

        Args:
            prefix (str): file name excluding the C/tau value and extension
            multiplier (float): constant multiplied by all y values found

        Returns:
            function: interpolating function for data in specified file
        """

        lines = np.genfromtxt("data/{}{}_xfgh.txt".format(prefix, self.data["c_tau"]), delimiter=" ")
        lines[:, 1] = multiplier * lines[:, 1] / lines[:, 2]
        return lambda x: np.interp(x, lines[:, 0], lines[:, 1])
    
########################################################################################################################
    def _file_interpDens(self, prefix, multiplier=1):
        """Read CISM data file and linearly interpolate temperature and density data.

        Args:
            prefix (str): file name excluding the C/tau value and extension
            multiplier (float): constant multiplied by all y values found

        Returns:
            function: interpolating function for data in specified file
        """

        lines = np.genfromtxt("data/{}{}_xfgh.txt".format(prefix, self.data["c_tau"]), delimiter=" ")
        lines[:, 2] = multiplier * lines[:, 2]
        return lambda x: np.interp(x, lines[:, 0], lines[:, 2])
    
########################################################################################################################
    def update_output(self):
        """Update all plots and output values."""

        self.data.update(gui.InputParam.get_values(self.root))
        self.update_plot("Lnu", (self.data["emin"], self.data["emax"]))
        self.update_plot("Inu", (0, 1))
        self.update_plot("temp", (self.data["r_min"], 1))
        self.update_plot("density", (self.data["r_min"], 1))
        em = self.emission_measure()
        lum = self.total_luminosity()
        if self.data["model"] == "chev":
            # Show forward and reverse shock values separately in addition to total values
            output = {"lum": lum[0], "lum_f": lum[1], "lum_r": lum[2], "em": em[0], "em_f": em[1], "em_r": em[2],
                      "Tem": em[3], "Tem_f": em[4], "Tem_r": em[5]}
        else:
            output = {"lum": lum, "em": em[0], "Tem": em[1]}
            
        #if (self.widgets["model"].get_value() == "cism" and self.widgets["c_tau"].get_value() == 0):
        if (self.widgets["model"] == "cism" and self.widgets["c_tau"] == 0):
            gui.OutputValue.update(output, self.root, 1)
        else:
            gui.OutputValue.update(output, self.root, 0)
        

#######################################################################################################################
    def update_specific_intensity(self):
        """Update specific intensity plot."""

        self.data.update(gui.InputParam.get_values(self.root))
        self.update_plot("Inu", (0, 1))

#######################################################################################################################
    def update_luminosity_spectrum(self):
        """Update luminosity plot and total luminosity value(s)."""
        self.data.update(gui.InputParam.get_values(self.root))
        self.update_plot("Lnu", (self.data["emin"], self.data["emax"]))
        lum = self.total_luminosity()
        if self.data["model"] == "chev":
            # Show forward and reverse shock values separately
            output = {"lum": lum[0], "lum_f": lum[1], "lum_r": lum[2]}
        else:
            output = {"lum": lum}
        #if (self.widgets["model"].get_value() == "cism" and self.widgets["c_tau"].get_value() == 0):
        if (self.widgets["model"] == "cism" and self.widgets["c_tau"] == 0):
            gui.OutputValue.update(output, self.root, 1)
        else:
            gui.OutputValue.update(output, self.root, 0)

#######################################################################################################################
    def update_plot(self, key, limits):
        """Get data and redraw plot within given x-axis limits.

        Args:
            key (str): key used to identify which plot to update (Inu, Lnu, temp, or density)
            limits (tuple): x-axis limits for plot
        """

        plot = self.plots[key]
        plot.clear_plot()
        x_data = np.linspace(*limits, 150)
        # Get new data using function associated with plot
        y_data = plot.properties["function"](x_data)
        plot.add_data(x_data, y_data, color=plot.properties["color"])
        plot.display_plot(limits=limits)

#######################################################################################################################
    @classmethod
    @functools.lru_cache(maxsize=None)
    def _sedov_scalar_temp(cls, x):
        """Temperature profile of Sedov phase SNR.

        Args:
            x (float): normalized radius of SNR (r/r_total)

        Returns:
            float: normalized temperature at specified radius (T/T_shock)
        """

        if x < 0.4:
            return 0.4 ** -4.32
        else:
            return x ** -4.32

#######################################################################################################################
    @classmethod
    def _sedov_vector_temp(cls, x):
        """Temperature profile of Sedov phase SNR as a vector function.

        Args:
            x (np.ndarray): normalized radii of SNR (r/r_total)

        Returns:
            np.ndarray: normalized temperatures at specified radii (T/T_shock)
        """

        lower_length = x[np.where(x < 0.4)].size
        upper = x[np.where(x >= 0.4)]
        return np.concatenate([np.full(lower_length, 0.4 ** -4.32), upper ** -4.32])

#######################################################################################################################
    @classmethod
    @functools.lru_cache(maxsize=None)
    def _sedov_scalar_density(cls, x):
        """Density profile of Sedov phase SNR.

        Args:
            x (float): normalized radius of SNR (r/r_total)

        Returns:
            float: normalized density at specified radius (rho/rho_shock)
        """

        if x < 0.5:
            return 0.31 / cls._sedov_scalar_temp(x)
        else:
            return (0.31 + 2.774 * (x - 0.5) ** 3 + 94.2548 * (x - 0.5) ** 8.1748) / cls._sedov_scalar_temp(x)

#######################################################################################################################
    @classmethod
    def _sedov_vector_density(cls, x):
        """Density profile of Sedov phase SNR as a vector function.

        Args:
            x (float): normalized radii of SNR (r/r_total)

        Returns:
            float: normalized densities at specified radii (rho/rho_shock)
        """

        lower_length = x[np.where(x < 0.5)].size
        upper = x[np.where(x >= 0.5)]
        return np.concatenate([np.full(lower_length, 0.31), 0.31 + 2.774 * (upper - 0.5) ** 3 + 94.2548 * (
            upper - 0.5) ** 8.1748]) / cls._sedov_vector_temp(x)

#######################################################################################################################
    def _chev_s_lim(self, b, *args):
        """Get limits of integral over s (where s = (r^2 - b^2)^(1/2)) for ED phase emissivity model. Note *args is used
        since function is called by nquad, which also provides edt as a parameter.

        Args:
            b (float): normalized impact parameter

        Returns:
            tuple: lower and upper limits for integral over s
        """

        return 0 if b >= self.data["r_min"] else (self.data["r_min"] ** 2 - b ** 2) ** 0.5, (1 - b ** 2) ** 0.5

#########################################################################################################################
    @staticmethod
    def _s_lim(b, *args):
        """Get limits of integral over s (where s = (r^2 - b^2)^(1/2)). Note *args is used since function is called
        by nquad, which also provides edt as a parameter.

        Args:
            b (float): normalized impact parameter

        Returns:
            tuple: lower and upper limits for integral over s
        """

        return (0, (1 - b ** 2) ** 0.5)

##########################################################################################################################
    def _chev_s_point(self, b):
        """Get points of difficulty in integral over s.

        Args:
            b (float): normalized impact parameter

        Returns:
            list: points that could cause difficulty in the integral over s OR None if no such points exist
        """

        points = []
        for radius in self.data["r_difficult"]:
            if b < radius:
                points.append((radius ** 2 - b ** 2) ** 0.5)
        if b < self.data["r_c"]:
            points.append((self.data["r_c"] ** 2 - b ** 2) ** 0.5)
        if len(points) == 0:
            return None
        else:
            return points

##########################################################################################################################
    @staticmethod
    def _s_point(b):
        """Get points of difficulty in integral over s. This is the default method used as a placeholder until a more
        specific method is introduced for a given model, so no points are returned.

            Args:
                b (float): normalized impact parameter

            Returns:
                list: points that could cause difficulty in the integral over s OR None if no such points exist
        """

        return None

###########################################################################################################################
    def emission_measure(self):
        """Get emission measure and emission weighted temperature for the current emissivity model. (Commented lines can
        print off values used in density_finder.py)

        Returns:
            tuple: emission measure, emission weighted temperature (note that three of each value are given for the ED
                   phase model in the order total, forward shock, and reverse shock)
        """

        integrand = lambda x: self.scalar_density(x) ** 2 * 4 * np.pi * x ** 2
        temp_integrand = lambda x: integrand(x) * self.scalar_temperature(x)
        if self.data["model"] == "chev":
            em_dimless = (quad(integrand, self.data["r_min"], self.data["r_c"], epsabs=1e-5,
                          points=self.data["em_point"][0])[0],
                          quad(integrand, self.data["r_c"], 1, epsabs=1e-5, points=self.data["em_point"][1])[0])
            em_f = (16 * self.data["n_0"] ** 2 * self.data["radius"] ** 3 * self.data["mu_H"] * em_dimless[1] /
                    self.data["mu_e"])
            em_r = (16 * self.data["n_0"] ** 2 * self.data["radius"] ** 3 * self.data["mu_H"] ** 2 * em_dimless[0] /
                    self.data["mu_H_ej"] / self.data["mu_e_ej"])
            em_temp = (quad(temp_integrand, self.data["r_min"], self.data["r_c"], epsabs=1e-5,
                            points=self.data["em_point"][0])[0] * self.data["T_s"],
                       quad(temp_integrand, self.data["r_c"], 1, epsabs=1e-5,
                            points=self.data["em_point"][1])[0] * self.data["T_s"])
            #em_temp =  (quad(temp_integrand, self.data["r_min"], self.data["r_c"], epsabs=1e-5, points=self.data["em_point"][0], limit=100), quad(temp_integrand, self.data["r_c"], 1, epsabs=1e-5, points=self.data["em_point"][1], limit=100))
            em_temp_tot = (em_temp[0] + em_temp[1]) / (em_dimless[0] + em_dimless[1])
            return em_f + em_r, em_f, em_r, em_temp_tot, em_temp[1] / em_dimless[1], em_temp[0] / em_dimless[0]
        else:
            em_dimless = quad(integrand, self.data["r_min"], 1, epsabs=1e-5, points=self.data["em_point"])[0]
            return (em_dimless * 16 * self.data["n_0"] ** 2 * self.data["mu_H"] / self.data["mu_e"] *
                    self.data["radius"] ** 3,
                    quad(temp_integrand, self.data["r_min"], 1, epsabs=1e-5, points=self.data["em_point"])[0] /
                    em_dimless * self.data["T_s"])

############################################################################################################################
    def _jnu_scaled(self, x, edt):
        """Get emission coefficient for thermal bremsstrahlung.

        Args:
            x (float): normalized radius (r/r_total)
            edt (float): energy (h \nu) divided by shock temperature (unitless)
            temp is in units of shock temperature

        Returns:
            float: emission coefficient at given x and edt
        """

        temp = self.scalar_temperature(x)
        val = ((np.log10(edt / temp) + 1.5) / 2.5)
        gaunt = (3.158 - 2.524 * val + 0.4049 * val ** 2 + 0.6135 * val ** 3 + 0.6289 * val ** 4 + 0.3294 * val ** 5 -
                 0.1715 * val ** 6 - 0.3687 * val ** 7 - 0.07592 * val ** 8 + 0.1602 * val ** 9 + 0.08377 * val ** 10)
        return 5.4e-39 * self.data["Z_sq"] / temp ** 0.5 * gaunt * np.exp(-edt / temp)

############################################################################################################################
    @staticmethod
    @functools.lru_cache(maxsize=None)
    def norm_radius(b, s):
        """Get normalized radius from impact parameter and s value.

        Args:
            b (float): normalized impact parameter
            s (float): defined as (r^2-b^2)^(1/2)

        Returns:
            float: normalized radius value
        """
        return (s ** 2 + b ** 2) ** 0.5

############################################################################################################################
    def specific_intensity(self, b):
        """Get specific intensity at a given impact parameter and energy.

        Args:
            b (float): normalized impact parameter

        Returns:
            float: specific intensity at impact parameter b (and energy self.data["energy"])
        """
        edt = self.data["energy"] * self.nrg_to_edt
        integral = np.fromiter((quad(self._intensity_integrand, *self._s_lim(b_val), args=(b_val, edt),
                                     points=self._s_point(b_val))[0] for b_val in b), np.float64)
        return self.cnst["intensity"] * integral

############################################################################################################################
    def _intensity_integrand(self, s, b, edt):
        """Integrand used in specific intensity integral.

        Args:
            s (float): defined as (r^2-b^2)^(1/2)
            b (float): normalized impact parameter
            edt (float): energy divided by shock temperature (unitless)

        Returns:
            float: value of integrand for given parameters
        """

        radius = self.norm_radius(b, s)
        # Multiplier needed to account for different composition of ejecta (for reverse shock)
        multiplier = self.data["ej_correction"] if radius < self.data["r_c"] else 1
        return self._jnu_scaled(radius, edt) * self.scalar_density(radius) ** 2 * multiplier

############################################################################################################################
    def _chev_opt_dict(self, b, *args):
        """Get option dictionary for luminosity integral over s in ED phase model.

        Args:
            b (float): normalized impact parameter
            *args: unused, provided since nquad provides edt as an additional parameter

        Returns:
            dict: options for luminosity integral over s
        """

        points = self._chev_s_point(b)
        if points is None:
            return {}
        else:
            return {"points": points}

#############################################################################################################################
    @staticmethod
    def _opt_dict(b, *args):
        """Get option dictionary for luminosity integral over s. Used as a placeholder until a new function is defined
        for a specific case.

        Args:
            b (float): normalized impact parameter
            *args: unused, provided since nquad provides edt as an additional parameter

        Returns:
            dict: options for luminosity integral over s (empty since function is a placeholder)
        """

        return {}

#############################################################################################################################
    def luminosity_spectrum(self, energy):
        """Get luminosity at a given energy for luminosity spectrum.

        Args:
            energy (float): photon energy in keV

        Returns:
            float: luminosity at given energy
        """

        edt_array = energy * self.nrg_to_edt
        integral = np.fromiter((nquad(self._luminosity_integrand, [self._s_lim, [0, 1]],
                                      args=(edt,), opts=[self._opt_dict, {}])[0] for edt in edt_array), np.float64)
        return integral * self.cnst["spectrum"]

#############################################################################################################################
    def total_luminosity(self):
        """Get total luminosity over a given energy range.

        Returns:
            tuple: total luminosity, luminosity of forward shock, luminosity of reverse shock (for ED phase model)
            float: total luminosity (for other models)
        """

        edt_min = self.data["emin"] * self.nrg_to_edt
        edt_max = self.data["emax"] * self.nrg_to_edt
        if self.data["model"] == "chev":
            lum_r = nquad(lambda s, b, edt: self._luminosity_integrand(s, b, edt) * self._reverse(s, b),
                          [self._s_lim, [0, 1], [edt_min, edt_max]],
                          opts=[self._opt_dict,{},{"points": [0.001, 0.1, 1, 10, 100]}])[0] * self.cnst["luminosity"]
            lum_f = nquad(lambda s, b, edt: self._luminosity_integrand(s, b, edt) * self._forward(s, b),
                          [self._s_lim, [0, 1], [edt_min, edt_max]],
                          opts=[self._opt_dict,{},{"points": [0.001, 0.1, 1, 10, 100]}])[0] * self.cnst["luminosity"]
            return lum_r + lum_f, lum_f, lum_r
        else:
            integral = nquad(self._luminosity_integrand, [self._s_lim, [0, 1], [edt_min, edt_max]],
                             opts=[self._opt_dict,{},{"points": [0.001, 0.1, 1, 10, 100]}])[0]
            return integral * self.cnst["luminosity"]

##############################################################################################################################
    def _luminosity_integrand(self, s, b, edt):
        """Integrand of luminosity integral in total_luminosity and luminosity_spectrum.

        Args:
            s (float): defined as (r^2-b^2)^(1/2)
            b (float): normalized impact parameter
            edt (float): energy divided by shock temperature (unitless)

        Returns:
            float: integrand value for given parameters
        """

        radius = self.norm_radius(b, s)
        multiplier = self.data["ej_correction"] if radius < self.data["r_c"] else 1
        return b * self._jnu_scaled(radius, edt) * self.scalar_density(radius) ** 2 * multiplier

#############################################################################################################################
    def _forward(self, s, b):
        """Isolate forward shock values.

        Args:
            s (float): defined as (r^2-b^2)^2
            b (float): normalized impact parameter

        Returns:
             int: 1 if position is part of forward shock, 0 otherwise
        """

        return 1 if self.norm_radius(b, s) > self.data["r_c"] else 0

############################################################################################################################
    def _reverse(self, s, b):
        """Isolate reverse shock values.

        Args:
            s (float): defined as (r^2-b^2)^2
            b (float): normalized impact parameter

        Returns:
             int: 1 if position is part of reverse shock, 0 otherwise
        """

        return 1 if self.norm_radius(b, s) <= self.data["r_c"] else 0

##############################################################################################################################

