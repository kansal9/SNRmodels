"""SNR Project Calculation Module.

Classes used to represent a general SNR for main window and specific case of SNR for emissivity window.

Authors: Denis Leahy, Bryson Lawton, Jacqueline Williams
Version: April 1st, 2019
"""

import numpy as np
import snr_gui as gui
from scipy.optimize import brentq, newton
from scipy.integrate import quad, cumtrapz
from helpers import *
##################################
# Input Data in the following units:
#
#  Age: SNR.data["t"] - (years)
#  Energy: self.data["e_51"] - (x10^51 ergs)
#  ISM Temperature: self.data["temp_ism"] - (K)
#  Ejected Mass: self.data["m_ej"] - (Msun)
#  Electron to Ion Temperature Ratio: self.data["T_ratio"] - (no units)
#  Stellar Wind Mass Loss: self.data["m_w"] - (Msun/yr)
#  Wind Speed: self.data["v_w"] - (km/s)
#


##############################################################################################################################
##############################################################################################################################
class SuperNovaRemnant:
    """Calculate and store data for a supernova remnant.

    Attributes:
        root (str): ID of GUI main window, used to access widgets
        widgets (dict): widgets used for input
        buttons (dict): emissivity and abundance buttons in program window
        data (dict): values from input window
        calc (dict): values calculated from input values
        graph (snr_gui.TimePlot): plot used to show radius and velocity as functions of time
        cnst (dict): named tuple of constants used for supernova remnant calculations, used only for s=0
        radius_functions (dict): functions used to calculate radius as a function of time
        velocity_functions (dict): functions used to calculate velocity as a function of time or radius
        time_functions (dict): functions used to calculate time as a function of radius
    """
###########################################################################################################################
    def __init__(self, root):
        """Initialize SuperNovaRemnant instance.

        Args:
            root (str): ID of parent window, used to access widgets only from parent
        """

        self.root = root
        self.widgets = gui.InputParam.instances[self.root]
        self.buttons = {}
        self.data = {}
        self.calc = {}
        # Graph is defined in snr.py module
        self.graph = None
        # Constants are defined when an n value is specified
        self.cnst = None
        self.radius_functions = {}
        self.velocity_functions = {}
        self.time_functions = {}
        self._init_functions()

############################################################################################################################
    def update_output(self):
        """Recalculate and update data, plot, and output values using input values from main window."""
        global MU_H, MU_I, MU_e, MU_t, MU_ratej, MU_tej, MU_I2, MU_e2, MU_Iej, MU_eej
        self.data.update(gui.InputParam.get_values(self.root))
        
        ab_sum = sum(10 ** (self.data["abundance"][key] - self.data["abundance"]["H"]) for key in A_VALS)
        self.data["mu_H"] = sum(A_VALS[key] * 10 ** (self.data["abundance"][key] - self.data["abundance"]["H"]) for key in A_VALS)
        self.data["mu_e"] = self.data["mu_H"] / sum(Z_VALS[key] * 10 ** (self.data["abundance"][key] - self.data["abundance"]["H"]) for key in Z_VALS)
        self.data["mu_I"] = self.data["mu_H"] / ab_sum
        self.data["mu_t"] = 1/((1/self.data["mu_I"])+(1/self.data["mu_e"]))
        self.data["Z_sq"] = sum(Z_VALS[key] ** 2 * 10 ** (self.data["abundance"][key] - self.data["abundance"]["H"]) for key in A_VALS) / ab_sum
                
        ab_sum_ej = sum(10 ** (self.data["ej_abundance"][key] - self.data["ej_abundance"]["H"]) for key in A_VALS)
        self.data["mu_H_ej"] = sum(A_VALS[key] * 10 ** (self.data["ej_abundance"][key] - self.data["ej_abundance"]["H"]) for key in A_VALS)
        self.data["mu_e_ej"] = self.data["mu_H_ej"] / sum(Z_VALS[key] * 10 ** (self.data["ej_abundance"][key] - self.data["ej_abundance"]["H"]) for key in Z_VALS)
        self.data["mu_I_ej"] = self.data["mu_H_ej"] / ab_sum_ej
        self.data["mu_t_ej"] = 1/((1/self.data["mu_I_ej"])+(1/self.data["mu_e_ej"]))
        self.data["Z_sq_ej"] = sum(Z_VALS[key] ** 2 * 10 ** (self.data["ej_abundance"][key] - self.data["ej_abundance"]["H"]) for key in A_VALS) / ab_sum_ej
        
        self.data["mu_ratio"] = (self.data["mu_H"]*self.data["mu_e"])/(self.data["mu_H_ej"]*self.data["mu_e_ej"])
        
        MU_H, MU_I, MU_e, MU_t, MU_ratej, MU_tej, MU_I2, MU_e2 = self.data["mu_H"], self.data["mu_I"], self.data["mu_e"], self.data["mu_t"], self.data["mu_ratio"], self.data["mu_t_ej"], self.data["mu_I_ej"], self.data["mu_e_ej"]
        MU_Iej, MU_eej = MU_I2, MU_e2

        # n must be converted to an integer since it is used as a key in the function dictionaries
        self.data["n"] = round(self.data["n"])
        if str(self.widgets["T_ratio"].input.cget("state")) == "readonly" or type(self.data["T_ratio"]) == str:
            # Calculate Te/Ti ratio
            temp_est = (3 * 10 ** 6 * (self.data["t"] / 10 ** 4) ** -1.2 * (self.data["e_51"] / 0.75 / self.data["n_0"]) **
                        0.4)
            func = 5 / 3 * self.data["n_0"] / 81 / temp_est ** 1.5 * self.data["t"] * YR_TO_SEC * np.log(
                1.2 * 10 ** 5 * 0.1 * temp_est * (temp_est / 4 / self.data["n_0"]) ** 0.5)
            ratio = 1 - 0.97 * np.exp(-func ** 0.4 * (1 + 0.3 * func ** 0.6))
            self.widgets["T_ratio"].value_var.set("{:.2g}".format(ratio))
            self.data["T_ratio"] = ratio
        # Calculate phase transition times and all values needed for future radius and velocity calculations.
        if self.data["s"] == 0:
            if "t_pds" in self.calc:
                pds_old = round(self.calc["t_pds"])
            else:
                pds_old = -1

            self.cnst = VALUE_DICT_S0[self.data["n"]]

            self.calc["v_ej"] = (100 * self.data["e_51"] / self.data["m_ej"]) ** 0.5
            self.calc["c_0"] = ((5 * BOLTZMANN * self.data["temp_ism"] / 3 / M_H / self.data["mu_H"]) ** 0.5 /
                                100000)
            self.calc["c_net"] = (self.calc["c_0"] ** 2 + self.data["sigma_v"] ** 2) ** 0.5
            #Characteristic Radius Calculation
            self.calc["r_ch"] = ((self.data["m_ej"] * SOLAR_MASS_TO_GRAM) ** (1 / 3) /
                                 (self.data["n_0"] * self.data["mu_H"] * M_H) ** (1 / 3) / PC_TO_KM / 10 ** 5)
            #Characteristic Time Calculation
            self.calc["t_ch"] = ((self.data["m_ej"] * SOLAR_MASS_TO_GRAM) ** (5 / 6) / (self.data["e_51"] * 10 ** 51) ** 0.5 /
                                 (self.data["n_0"] * self.data["mu_H"] * M_H) ** (1 / 3) / YR_TO_SEC)
            #Characteristic Velocity Calculation
            self.calc["v_ch"] = self.calc["r_ch"] / self.calc["t_ch"] * PCyr_TO_KMs
            #Sedov-Taylor Time Calculation
            self.calc["t_st"] = self.cnst.t_st * self.calc["t_ch"]
            #Pressure Driven Snowplow Time Calculation
            self.calc["t_pds"] = (13300 * self.data["e_51"] ** (3 / 14) * self.data["n_0"] ** (-4 / 7) *
                                  self.data["zeta_m"] ** (-5 / 14))
            #MCS Time Calculation
            if (self.data["model"] == "cism"):
                self.calc["t_mcs"] = ((14.63 * CISM_EM_WEIGHTED["beta"][self.data["c_tau"]] * (self.data["mu_H"] * M_H) ** (3 / 5)
                / (self.data["zeta_m"] * CISM_EM_WEIGHTED["alpha"][self.data["c_tau"]]) ** (2 / 3) / BOLTZMANN) ** (15 / 28)
                * (K_DICT[self.data["c_tau"]] * self.data["e_51"] * 10 ** 51 / 4 / np.pi) ** (3 / 14)
                * self.data["n_0"] ** (-4 / 7) / YR_TO_SEC)
            else:
                self.calc["t_mcs"] = self.calc["t_pds"] * min(61 * self.calc["v_ej"] ** 3 / self.data["zeta_m"] ** (
                    9 / 14) / self.data["n_0"] ** (3 / 7) / self.data["e_51"] ** (3 / 14), 476 / (
                    self.data["zeta_m"] * PHI_C) ** (9 / 14))
            #Reversal Time Calculation
            self.calc["t_rev"] = newton(self.radius_functions[self.data["n"], "late"], 3 * self.calc["t_ch"])
            #Core Time Calculation
            self.calc["t_c"] = ((2 / 5) ** 5 * self.data["e_51"] * 10 ** 51 * XI_0 / (
                self.calc["c_0"] * 100000) ** 5 / (self.data["n_0"] * self.data["mu_H"] * M_H)) ** (
                1 / 3) / YR_TO_SEC
                
            if self.data["model"] == "fel":
                self.calc["gamma_0"] = 5.0/3.0
                self.calc["epsi"] = (4 * (self.calc["gamma_0"] - self.data["gamma_1"])) / ((self.calc["gamma_0"] - 1.0)*((self.data["gamma_1"] + 1)**2))

                # Set default FEL start time to t_PDS if already set to previous t_PDS
            if self.data["t_fel"] == round(pds_old):
                self.widgets["t_fel"].value_var.set(round(self.calc["t_pds"]))
                self.widgets["t_fel"].previous = round(self.calc["t_pds"])
                self.data["t_fel"] = self.calc["t_pds"]
            #Merger Time Calculation
            self.calc["t_mrg"] = self.merger_time(self.data["n"])
            if self.calc["t_mrg"]["ED"] < self.calc["t_st"] and (self.data["model"] == "cism"):
                # Change back to standard model if CISM phase doesn't occur
                self.widgets["model"].value_var.set("standard")
                self.update_output()
                return
            phases = self.get_phases()
            self.calc["t_mrg_final"] = round(self.calc["t_mrg"][phases[-1]])
            
            # Values to be displayed as output, note r and v are added later and rr and vr are updated later if t<t_rev
            output_data = {
                "epsi": "",
                "rr": "N/A",
                "vr": "N/A",
                "Tr": "N/A",
                "Core": "N/A",
                "Rev": "N/A",
                "t-ST": "N/A",
                "t-CISM": self.calc["t_st"],
                "t-PDS": self.calc["t_pds"],
                "t-MCS": self.calc["t_mcs"],
                "t-HLD": self.calc["t_c"] * 0.1,
                "t-FEL": self.data["t_fel"],
                "t-MRG": self.calc["t_mrg_final"]
            }
            if self.data["model"] != "sedtay":
                output_data["Core"] = "RS Reaches Core:  " + str(round(self.cnst.t_rchg * self.calc["t_ch"],1)) + " yr"
                output_data["Rev"] = "RS Reaches Center:  " + str(round(self.calc["t_rev"],1)) + " yr"
                output_data["t-ST"] = self.calc["t_st"]
            else:
                output_data["Core"] = "RS Reaches Core:  N/A"
                output_data["Rev"] = "RS Reaches Center:  N/A"
                output_data["t-ST"] = "N/A"
            del output_data["epsi"]
            if self.widgets["model"].get_value() == "fel":
                output_data["epsi"] = "Fractional energy loss \u03B5: " + str(round(self.calc["epsi"],6))
            else:
                output_data["epsi"] = ""
             
            # Calculate cx to emission measure and temperature graphs
            #if (self.data["model"] == "standard" and self.data["n"] > 5):
                #self.calc["cx"] = ((((27*self.cnst.l_ed**(self.data["n"]-2))/(4*np.pi*self.data["n"]*(self.data["n"]-3)*self.cnst.phi_ed))
                                #*(((10*(self.data["n"]-5))/(3*(self.data["n"]-3)))**((self.data["n"]-3)/2)))**(1/self.data["n"]))
                
            # Check if HLD model is valid with current conditions and change state of radio button accordingly
            if (0.1 * self.calc["t_c"] <= self.calc["t_pds"] and
                    str(self.widgets["model"].input["hld"].cget("state")) == "disabled"):
                self.widgets["model"].input["hld"].config(state="normal")
            elif 0.1 * self.calc["t_c"] > self.calc["t_pds"] and (
                    str(self.widgets["model"].input["hld"].cget("state")) == "normal"):
                self.widgets["model"].input["hld"].config(state="disabled")
                if self.widgets["model"].get_value() == "hld":
                    self.widgets["model"].value_var.set("standard")
                    self.data["model"] = "standard"
                    self.update_output()
                    return
        else: #s=2 case
            output_data = {
                "epsi": "",
                "rr": "N/A",
                "vr": "N/A",
                "Tr": "N/A",
                "Core": "",
                "Rev": "",
                "t-s2": "This model only includes the\nejecta-dominated phase."   # Set transition time output value to display message explaining lack of transition times
            }
            self.cnst = VALUE_DICT_S2[self.data["n"]]
            # Change units of m_w and v_w to fit with those used in Truelove and McKee
            if (self.data["n"] > 5):
                mdot_GRAMsec = self.data["m_w"]*SOLAR_MASS_TO_GRAM/YR_TO_SEC
                mej_GRAM = self.data["m_ej"]*SOLAR_MASS_TO_GRAM
                self.calc["q"] = (mdot_GRAMsec/10**5)/(4*np.pi*self.data["v_w"]) #
                self.calc["gcn"] = ((1-(3/self.data["n"]))*mej_GRAM) / (((4*np.pi)/3) * ((10/3)*((self.data["n"]-5)/(self.data["n"]-3))*((self.data["e_51"] * 10**51)/(mej_GRAM)))**((3-self.data["n"])/2))
                self.calc["RCn"] = ((self.cnst.a2*self.calc["gcn"])/(self.calc["q"]))**(1/(self.data["n"]-2)) #cm
                self.calc["v0"] = (((self.data["e_51"]*10**51*10*(self.data["n"]-5))/(mej_GRAM*3*(self.data["n"]-3)))**0.5)/(10**5) #km/s
            # Change units of m_w and v_w to fit with those used in Truelove and McKee
            self.data["m_w"] /= 1e-5  #units [(Msun*10^5)/yr]
            self.data["v_w"] /= 10    #units [(km*10^2)/s]
            self.calc["r_ch"] = 12.9 * self.data["m_ej"] / self.data["m_w"] * (self.data["v_w"]) #pc
            self.calc["t_ch"] = (1770 * self.data["e_51"] ** -0.5 * self.data["m_ej"] ** 1.5 / self.data["m_w"] * self.data["v_w"]) #yrs
            self.calc["v_ch"] = self.calc["r_ch"] / self.calc["t_ch"] * PCyr_TO_KMs  #km/s

            # Note despite the key "t_mrg", this is simply the model ending time since only the ED phase is used
            self.calc["t_mrg"] = {"s2": self.calc["t_ch"]}
            phases = self.get_phases()
            self.calc["t_mrg_final"] = round(self.calc["t_mrg"][phases[-1]])
        # Get correct merge time to set spinbox maximum values
        if self.data["model"] == "fel" and self.data["t_fel"] > self.calc["t_mrg_final"]:
            t_fel = round(self.calc["t_st"])
            self.widgets["t_fel"].value_var.set(t_fel)
            self.data["t_fel"] = t_fel
            self.widgets["t_fel"].previous = t_fel
            output_data["t-MRG-FEL"] = self.merger_time(self.data["n"])["FEL"]
            phases = self.get_phases()
        if self.data["t"] >= self.calc["t_mrg_final"]:
            self.data["t"] = self.calc["t_mrg_final"]
            self.widgets["t"].value_var.set(self.calc["t_mrg_final"])
        
        output_data.update(self.get_specific_data())
        
        self.calc["r"] = output_data["r"]
        self.calc["T"] = output_data["T"]  #This T is electron shock temperature
        self.widgets["xmin"].input.config(to=self.calc["t_mrg_final"])
        self.widgets["xmax"].input.config(to=self.calc["t_mrg_final"])
        
        self.update_plot(phases)
        if (self.widgets["model"].get_value() == "cism" and self.widgets["c_tau"].get_value() == 0):
            gui.OutputValue.update(output_data, self.root, 1, phases)
        else:
            gui.OutputValue.update(output_data, self.root, 0, phases)
        if self.emissivity_model_exists():
            self.buttons["em"].config(state="normal")
        else:
            self.buttons["em"].config(state="disabled")
            
###########################################################################################################################
    def emissivity_model_exists(self):
        """Check if an emissivity model is available for the current set of input parameters.

        Returns:
            bool: True if the model exists, False otherwise
        """

        time = self.data["t"]
        if (self.data["n"] in (6,7,8,9,10,11,12,13,14)):
            if (((self.data["model"] in ("standard")) and (self.data["s"] == 2)) or 
                ((self.data["model"] in ("standard")) and (self.data["s"] == 0) and (time < (self.cnst.t_rchg * self.calc["t_ch"]))) or 
                ((self.data["model"] in ("cism")) and (self.data["s"] == 0) and (self.calc["t_rev"] < time < min(self.calc["t_mcs"], self.calc["t_mrg_final"])))):
                return True
        elif (self.data["n"] in (0,1,2,4)):
            if (((self.data["model"] in ("cism")) and (self.data["s"] == 0) and (self.calc["t_rev"] < time < min(self.calc["t_mcs"], self.calc["t_mrg_final"]))) or
                ((self.data["model"] in ("sedtay")) and (self.data["s"] == 0) and (time < min(self.calc["t_pds"], self.calc["t_mrg_final"])))):
                return True
        else:
            return False
        
##########################################################################################################################
    def update_plot(self, phases):
        """Add necessary lines to output plot and redraw with the correct limits.

        Args:
            phases (list): list of SNR phases for current model
        """

        plot_data = self.get_plot_data(phases)
        # Clear previous lines
        self.graph.clear_plot()
        # Add data for forward and reverse shocks - radius vs. time or velocity vs. time depending on plot_type
        direction = "forward"
        self.graph.add_data(plot_data[direction]["t"], plot_data[direction][self.data["plot_type"]],
                            color="r", label="Blast-Wave Shock")
        if self.data["model"] != "sedtay":
            direction = "reverse"
            if ((self.data["n"] >= 6) and (self.data["s"] == 0) and (self.data["model"] == "standard") and (self.data["plot_type"] == "eMeas" or self.data["plot_type"] == "temper")):
                self.graph.add_data(plot_data["forward"]["t"], plot_data[direction][self.data["plot_type"]],
                            color="b", label="Reverse Shock")
            else:
                self.graph.add_data(plot_data[direction]["t"], plot_data[direction][self.data["plot_type"]],
                            color="b", label="Reverse Shock")
            
        # Add vertical lines to show phase transitions
        if self.data["model"] != "sedtay": 
            if "ST" in phases or "CISM" in phases:
                self.graph.graph.axvline(x=self.calc["t_st"], ls="dashed", c="black", label=r"$t_{\mathrm{ST}}$")
            if "PDS" in phases:
                self.graph.graph.axvline(x=self.calc["t_pds"], ls="-.", c="black", label=r"$t_{\mathrm{PDS}}$")
            if "MCS" in phases:
                self.graph.graph.axvline(x=self.calc["t_mcs"], ls="dotted", c="black", label=r"$t_{\mathrm{MCS}}$")
            if "FEL" in phases:
                self.graph.graph.axvline(x=self.data["t_fel"], ls="dotted", lw=3, c="black", label=r"$t_{\mathrm{FEL}}$")
            if "HLD" in phases:
                self.graph.graph.axvline(x=self.calc["t_c"]*0.1, ls="dotted", lw=3, c="black", label=r"$t_{\mathrm{HLD}}$")
        # Update plot display
        self.graph.display_plot(self.get_limits())

############################################################################################################################
    def get_limits(self):
        """Get limits for x-axis on output plot.

        Returns:
            list: minimum and maximum values for x-axis on radius/velocity vs. time plot
        """

        # Account for FEL and HLD models ending other phases earlier than expected
        if self.data["model"] == "fel":
            alt_upper = self.data["t_fel"]
        elif self.data["model"] == "hld":
            alt_upper = self.calc["t_c"] * 0.1
        else:
            # Ensure that alt_upper will never be the minimum value in expressions below
            if self.data["s"] != 2 and self.calc["t_mrg"]["ED"] < self.calc["t_st"]:
                alt_upper = self.calc["t_mrg"]["ED"]
            else:
                alt_upper = float("inf")
        if "Current" in self.data["range"]:
            # Detect current phase and switch range to that type
            phase = self.get_phase(self.data["t"])
            if phase in ("ED", "ST") and "ED-ST" in self.widgets["range"].input.cget("values"):
                phase = "ED-ST"
            elif phase in ("ED", "CISM") and "ED-CISM" in self.widgets["range"].input.cget("values"):
                phase = "ED-CISM"
            self.data["range"] = phase
            # Change s2 to ED for display purposes - s2 represents the ED phase for the case s=2
            if phase == "s2":
                phase = "ED"
            self.widgets["range"].value_var.set("Current ({})".format(phase))
        if self.data["range"] in ("ED-ST", "ED", "ST"):
            limits = (0, min(self.calc["t_pds"], self.calc["t_mrg"]["ST"], alt_upper))
        elif self.data["range"] == "PDS":
            limits = (self.calc["t_pds"], min(self.calc["t_mcs"], self.calc["t_mrg"]["PDS"], alt_upper))
        elif self.data["range"] == "MCS":
            limits = (self.calc["t_mcs"], self.calc["t_mrg"]["MCS"])
        elif self.data["range"] == "Custom":
            limits = (self.data["xmin"], self.data["xmax"])
        elif self.data["range"] == "FEL":
            limits = (self.data["t_fel"], self.calc["t_mrg"]["FEL"])
        elif self.data["range"] == "HLD":
            limits = (self.calc["t_c"] * 0.1, self.calc["t_mrg"]["HLD"])
        elif self.data["range"] == "s2":
            limits = (0, self.calc["t_ch"])
        elif self.data["range"] in ("CISM", "ED-CISM"):
            limits = (0, min(self.calc["t_mrg"]["CISM"], self.calc["t_mcs"], alt_upper))
        else:
            # Corresponds to reverse shock lifetime
            limits = (0, min(self.calc["t_rev"], self.calc["t_mrg_final"]))
        if self.data["range"] != "Custom":
            # Set spinbox limit values to match new x-axis limits
            self.widgets["xmin"].value_var.set(int(round(limits[0])))
            self.widgets["xmax"].value_var.set(int(round(limits[1])))
            self.data["xmin"] = limits[0]
            self.data["xmax"] = limits[1]
        else:
            # Ensure range does not include values greater than the model end time
            maximum = self.widgets["xmax"].input.cget("to")
            if limits[0] > maximum and limits[1] > maximum:
                # Prevents code below from making both upper and lower limits have the same value
                limits = (0, maximum)
                self.widgets["xmin"].value_var.set(round(limits[0]))
                self.widgets["xmax"].value_var.set(round(limits[1]))
                self.data["xmin"] = limits[0]
                self.data["xmax"] = limits[1]
            elif limits[1] > maximum:
                limits = (limits[0], maximum)
                self.widgets["xmax"].value_var.set(round(limits[1]))
                self.data["xmax"] = limits[1]
            elif limits[0] > maximum:
                limits = (maximum, limits[0])
                self.widgets["xmin"].value_var.set(round(limits[0]))
                self.data["xmin"] = limits[0]
        return limits

############################################################################################################################
    def get_phase(self, time):
        """Get current phase of SNR model.

        Args:
            time (float): age of SNR

        Returns:
            str: abbreviation representing phase of model
        """

        if self.data["s"] == 2:
            phase = "s2"
        elif self.data["model"] == "fel" and time > self.data["t_fel"]:
            phase = "FEL"
        elif self.data["model"] == "hld" and time > self.calc["t_c"] * 0.1:
            phase = "HLD"
        elif (self.data["model"] == "cism") and time > self.calc["t_st"]:
            if time > self.calc["t_mcs"]:
                phase = "MCS"
            else:
                phase = "CISM"
        elif (time < self.calc["t_st"] < self.calc["t_pds"]) or (time < self.calc["t_pds"] < self.calc["t_st"]):
            phase = "ED"
        elif time < self.calc["t_pds"]:
            phase = "ST"
        elif time < self.calc["t_mcs"]:
            phase = "PDS"
        else:
            phase = "MCS"
        return phase

###########################################################################################################################
    def get_plot_data(self, phases):
        """Get of forward and reverse shock data used to plot radius vs. time or velocity vs. time or emission measure vs. time or temperature vs. time

        Args:
            phases (list): list of SNR phases for current model

        Returns:
            dict: dictionary of np.ndarrays with shock radius, velocity, emission measure, temperature and time used to create output plots
        """

        plot_data = {}
        forward_time = {}
        # Account for different reverse shock phases between s=0 and s=2
        if self.data["s"] == 0:
            rev_phases = ("early", "late")
        else:
            rev_phases = "s2r",
        all_phases = {"forward": phases, "reverse": rev_phases}
        for direction, phase_list in all_phases.items():
            plot_data[direction] = {"t": [], "r": [], "v": [], "eMeas": [], "temper": []}
            for phase in phase_list:
                t, r, v = self.get_data(phase)
                plot_data[direction]["t"] = np.concatenate([plot_data[direction]["t"], t])
                plot_data[direction]["r"] = np.concatenate([plot_data[direction]["r"], r])
                plot_data[direction]["v"] = np.concatenate([plot_data[direction]["v"], v])
            if (direction == "forward"):
                forward_time = plot_data[direction]["t"]
            if ((self.data["n"] >= 6) and (self.data["s"] == 0) and (self.data["model"] == "standard") and (self.data["plot_type"] == "eMeas" or self.data["plot_type"] == "temper")):
                if (self.data["plot_type"] == "eMeas"):
                    if (direction == "forward"):
                        eMeasf = self.get_EMandTempData(forward_time, "eMeasf")
                        plot_data[direction]["eMeas"] = np.concatenate([plot_data[direction]["eMeas"], eMeasf])
                    elif (direction == "reverse"):
                        eMeasr = self.get_EMandTempData(forward_time, "eMeasr")
                        plot_data[direction]["eMeas"] = np.concatenate([plot_data[direction]["eMeas"], eMeasr])
                else:
                    if (direction == "forward"):
                        temperf = self.get_EMandTempData(forward_time, "temperf")
                        plot_data[direction]["temper"] = np.concatenate([plot_data[direction]["temper"], temperf])
                    elif (direction == "reverse"):
                        temperr = self.get_EMandTempData(forward_time, "temperr")
                        plot_data[direction]["temper"] = np.concatenate([plot_data[direction]["temper"], temperr])

        return plot_data


###########################################################################################################################
    def get_specific_data(self):
        """Get output values (radius, velocity, and temperature of both shocks) at a specific time.

        Returns:
            dict: dictionary of output values
        """

        output = {}
        time = self.data["t"]
        phase = self.get_phase(time)
        t, output["r"], output["v"] = self.get_data(phase, time)
        if self.data["model"] != "sedtay":
            if self.data["s"] == 2:
                t, output["rr"], output["vr"] = self.get_data("s2r", time)
            elif time < self.cnst.t_rchg * self.calc["t_ch"]:
                t, output["rr"], output["vr"] = self.get_data("early", time)
            elif time < self.calc["t_rev"]:
                t, output["rr"], output["vr"] = self.get_data("late", time)
        output["T"] = (self.data["T_ratio"]*(3 / 16 * self.data["mu_I"] * M_H / BOLTZMANN * (output["v"] * 100000) ** 2))  #T_e_shock
        if "vr" in output:
            output["Tr"] = (self.data["T_ratio"]*(3 / 16 * self.data["mu_I"] * M_H / BOLTZMANN * (output["vr"] * 100000) ** 2)) #T_e_shock
        return output

#############################################################################################################################
    def get_data(self, phase, t=None):
        """Returns dictionary of time, radius, and velocity data for a forward or reverse shock.

        Args:
            phase (str): phase to get data for
            t (float, optional): time, only specified for finding values at a specific time rather than a list of times

        Returns:
            np.ndarray/float: time value(s)
            np.ndarray/float: radius value(s)
            np.ndarray/float: velocity value(s)
            Note that float values are returned if t is specified, np.ndarray objects are returned otherwise.
        """
        output_values = self.velocity(phase, self.radius_time(phase, t))
        return output_values["t"], output_values["r"], output_values["v"]

#############################################################################################################################
    def get_EMandTempData(self, timeArray, output = "tempAndEMeas"):
        """Returns dictionary of emission measure and temperature for a forward or reverse shock.

        Args:
            timeArray: time array for all time values

        Returns:
            np.ndarray/float: forward emission measure values
            np.ndarray/float: forward temperature values
            np.ndarray/float: reverse emission measure values
            np.ndarray/float: reverse temperature values    
        """
        if(output == "eMeasf"):
            output_values = self.GetEMandTempArrays(timeArray, "eMeasf")
            return output_values["eMeasf"]
        elif(output == "temperf"):
            output_values = self.GetEMandTempArrays(timeArray, "temperf")
            return  output_values["Tempf"]
        elif(output == "eMeasr"):
            output_values = self.GetEMandTempArrays(timeArray, "eMeasr")
            return output_values["eMeasr"]
        elif(output == "temperr"):
            output_values = self.GetEMandTempArrays(timeArray, "temperr")
            return output_values["Tempr"]
        else:
            output_values = self.GetEMandTempArrays(timeArray)
            return output_values["eMeasf"], output_values["Tempf"], output_values["eMeasr"], output_values["Tempr"]

##########################################################################################################################
    def GetEMandTempArrays(self, timeArray, output = "tempAndEMeas"):
        """Returns dictionary of output for emission measure and temperature values.

        Args:
           timeArray: time array for all time values

        Returns:
            output: dictionary including time, emission measure & temperature data arrays
        """

        # Determine whether the input needs to be radius or time depending on if t(r) or r(t) is used
        if isinstance(timeArray, np.ndarray):
            # Ensure that original input arrays are not altered in the velocity functions
            timeArray = np.array(timeArray)
        output1_array = {}
        output2_array = {}
        output3_array = {}
        output4_array = {}
        input_arr = timeArray
        
        output1_func = self.emissionMeasure_functions[self.data["n"], "eMeasf"]
        output2_func = self.temperature_functions[self.data["n"], "Tempf"]
        output3_func = self.emissionMeasure_functions[self.data["n"], "eMeasr"]
        output4_func = self.temperature_functions[self.data["n"], "Tempr"]
        if (output == "eMeasf"):
            for t in timeArray:
                output1_array = np.append(output1_array, output1_func(t))
            output1_array = np.delete(output1_array, 0)
            output1_array = np.array(output1_array, dtype='float64')
        elif (output == "temperf"):
            for t in timeArray:
                output2_array = np.append(output2_array, output2_func(t))
            output2_array = np.delete(output2_array, 0)
            output2_array = np.array(output2_array, dtype='float64')
        elif (output == "eMeasr"):
            for t in timeArray:
                output3_array = np.append(output3_array, output3_func(t))
            output3_array = np.delete(output3_array, 0)
            output3_array = np.array(output3_array, dtype='float64')
        elif (output == "temperr"):
            for t in timeArray:
                output4_array = np.append(output4_array, output4_func(t))
            output4_array = np.delete(output4_array, 0)
            output4_array = np.array(output4_array, dtype='float64')
        else:
            for t in timeArray:
                output1_array = np.append(output1_array, output1_func(t))
                output2_array = np.append(output2_array, output2_func(t))
                output3_array = np.append(output3_array, output3_func(t))
                output4_array = np.append(output4_array, output4_func(t)) 
                #Fix this
            output1_array = np.delete(output1_array, 0)
            output2_array = np.delete(output2_array, 0)
            output3_array = np.delete(output3_array, 0)
            output4_array = np.delete(output4_array, 0)
            
            output1_array = np.array(output1_array, dtype='float64')
            output2_array = np.array(output2_array, dtype='float64')
            output3_array = np.array(output3_array, dtype='float64')
            output4_array = np.array(output4_array, dtype='float64')

        # Generate output
        output = {
            "t": input_arr,
            "eMeasf": output1_array,
            "Tempf": output2_array,
            "eMeasr": output3_array,
            "Tempr": output4_array
        }
        
        return output

#########################################################################################################################
    def _EMTemp_solutions(self, n, key):
        """Returns specific type of function used to calculate values for emission measure and temperature

        Args:
            n (int): the n value for the current case
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified 
        """

        def EM_forward(t):
            t /= self.calc["t_ch"]
            return self.get_plot_dEMF(n,t)

        def EM_reverse(t):
            t /= self.calc["t_ch"]
            return self.get_plot_dEMR(n,t)

        def temp_forward(t):
            t /= self.calc["t_ch"]
            return self.get_plot_dTF(n,t)

        def temp_reverse(t):
            t /= self.calc["t_ch"]
            return self.get_plot_dTR(n,t)

        function_dict = {
            "emf": EM_forward,
            "emr": EM_reverse,
            "Tempf": temp_forward,
            "Tempr": temp_reverse
        }
        return function_dict[key]
  
     
############################################################################################################################
    def time_array(self, phase):
        """Returns an array of times appropriate for a given phase.

        Args:
            phase (str): phase to get data for

        Returns:
            np.ndarray: array of time values used to create plots for given phase
        """

        # Account for phases that may end early due to start of FEL or HLD phases
        if self.data["model"] == "fel":
            alt_upper = self.data["t_fel"]
        elif self.data["model"] == "hld":
            alt_upper = self.calc["t_c"] * 0.1
        else:
            # Ensure alt_upper won't be found as a minimum in the expressions below
            alt_upper = float("inf")
        if phase in ("s2", "s2r"):
            t_array = np.concatenate([np.linspace(1, 100, 1000), np.linspace(100, self.calc["t_ch"], 50000)])
        elif phase == "ED":
            t_array = np.linspace(1, min(self.calc["t_st"], alt_upper, self.calc["t_pds"], self.calc["t_mrg"]["ED"]),
                                  1000)
        elif phase == "ST":
            t_array = np.linspace(self.calc["t_st"], min(self.calc["t_pds"], self.calc["t_mrg"]["ST"], alt_upper), 1000)
        elif phase == "PDS":
            t_array = np.linspace(self.calc["t_pds"], min(self.calc["t_mcs"], self.calc["t_mrg"]["PDS"], alt_upper),
                                  10000)
        elif phase == "MCS":
            t_array = np.linspace(self.calc["t_mcs"], self.calc["t_mrg"]["MCS"], 10000)
        elif phase == "FEL":
            t_array = np.linspace(self.data["t_fel"], self.calc["t_mrg"]["FEL"], 50000)
        elif phase == "HLD":
            t_array = np.linspace(self.calc["t_c"] * 0.1, self.calc["t_mrg"]["HLD"],
                                  int(round(self.calc["t_mrg"]["HLD"]/10)))
        elif phase == "CISM":
            t_array = np.linspace(self.calc["t_st"], min(self.calc["t_mrg"]["CISM"], self.calc["t_mcs"]), 50000)
        elif phase == "early":
            t_array = np.linspace(1, self.cnst.t_rchg * self.calc["t_ch"], 1000)
        elif phase == "late":
            t_array = np.linspace(self.cnst.t_rchg * self.calc["t_ch"], self.calc["t_rev"], 1000)
        else:
            t_array = []
        return t_array

##########################################################################################################################
    def radius_time(self, phase, t=None):
        """Returns a dictionary with time and radius data.

        Args:
            phase (str): phase to get data for
            t (float, optional): time, only specified for finding values at a specific time rather than a list of times

        Returns:
            dict: dictionary with radius and time values
        """

        if t is None:
            # Generate arrays of data for plotting purposes

            # Determine input type and function needed to compute results
            if (self.data["n"] in (1,2, 4) and (phase == "ED" or phase == "early")) or (                              #Fix this
                    self.data["n"] < 3 and phase in ("s2", "s2r")):
                # Set input as radius and use a time function since these cases provided t(r) rather than r(t)
                if phase == "ED" and self.data["s"] == 0:
                    r_chg = self.cnst.r_st * self.calc["r_ch"]
                    num = 100
                elif phase == "s2":
                    r_chg = 1 / 1.5 ** 2 / (3 - self.data["n"]) * self.calc["r_ch"] - 1
                    num = 100000
                elif phase == "s2r":
                    r_chg = 1 / 1.5 ** 2 / (3 - self.data["n"]) / 1.19 * self.calc["r_ch"] - 1
                    num = 100000
                else:
                    r_chg = self.cnst.r_rchg * self.calc["r_ch"]
                    num = 100
                # Create array of radius values
                input_arr = np.linspace(0.01, r_chg, num)
                input_key = "r"
                output_key = "t"
                output_func = self.time_functions[self.data["n"], phase]
            else:
                # Use time as the input and a function to determine radius these cases provided r(t)
                input_arr = self.time_array(phase)
                input_key = "t"
                output_key = "r"
                output_func = self.radius_functions[self.data["n"], phase]
            # Generate output
            output = {
                input_key: input_arr,
                output_key: output_func(np.array(input_arr))
            }
        else:
            # Find output values only at a specific time
            output = {
                "r": self.radius_functions[self.data["n"], phase](t),
                "t": t
            }
        return output

##########################################################################################################################
    def velocity(self, phase, output):
        """Returns dictionary of output with velocity added.

        Args:
            phase (str): phase to get data for
            output (dict): partially completed output dictionary, output of radius_time

        Returns:
            dict: dictionary including radius, time, and velocity (radius and time from output argument dictionary)
        """

        # Determine whether the input needs to be radius or time depending on if t(r) or r(t) is used
        if (self.data["n"] in (1, 2, 4) and ((phase in ("ED", "early")) or (self.data["s"] == 2))):
            input_ = output["r"]
        else:
            input_ = output["t"]
        if isinstance(input_, np.ndarray):
            # Ensure that original input arrays are not altered in the velocity functions
            input_ = np.array(input_)
        output["v"] = self.velocity_functions[self.data["n"], phase](input_)
        return output

#########################################################################################################################
    def _s0n0_solution(self, key, phase):
        """Returns specific type of function used to calculate values for the s=0, n=0 case.

        Args:
            key (str): determines which function is returned, see function_dict for possible values
            phase (str): evolutionary phase of SNR to get data for

        Returns:
            function: a function to calculate the parameter specified for s=0, n=0 case
        """

        def radius_b(t):
            t /= self.calc["t_ch"]
            if phase == "ED":
                return 2.01 * t * (1 + 1.72 * t ** 1.5) ** (-2 / 3) * self.calc["r_ch"]
            else:
                return (1.42 * t - 0.254) ** 0.4 * self.calc["r_ch"]

        def radius_r(t):
            t /= self.calc["t_ch"]
            if phase == "early":
                return 1.83 * t * (1 + 3.26 * t ** 1.5) ** (-2 / 3) * self.calc["r_ch"]
            else:
                return t * (0.779 - 0.106 * t - 0.533 * np.log(t)) * self.calc["r_ch"]

        def velocity_b(t):
            t /= self.calc["t_ch"]
            if phase == "ED":
                return 2.01 * (1 + 1.72 * t ** 1.5) ** (-5 / 3) * self.calc["v_ch"]
            else:
                return 0.569 * (1.42 * t - 0.254) ** -0.6 * self.calc["v_ch"]

        def velocity_r(t):
            t /= self.calc["t_ch"]
            if phase == "early":
                return 5.94 * t ** 1.5 * (1 + 3.26 * t ** 1.5) ** (-5 / 3) * self.calc["v_ch"]
            else:
                return (0.533 + 0.106 * t) * self.calc["v_ch"]

        function_dict = {
            "r": radius_b,
            "rr": radius_r,
            "v": velocity_b,
            "vr": velocity_r
        }
        return function_dict[key]
  
################################################################################################################
    def _s2nlt3_solution(self, n, key):
        """Returns a specific type of function used for the s=2, n<3 case.

        Args:
            n (int): n value for SNR
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified for s=2, n<3 case
        """

        def time_b(r):
            r /= self.calc["r_ch"]
            return (0.594 * ((3 - n) / (5 - n)) ** 0.5 * r * (1 - 1.5 * (3 - n) ** 0.5 * r ** 0.5) ** (-2 / (3 - n)) *
                    self.calc["t_ch"])

        def time_r(r):
            l_ed = 1.19
            return time_b(r * l_ed)

        def velocity_b(r):
            r /= self.calc["r_ch"]
            return 1.68 * ((5 - n) / (3 - n)) ** 0.5 * (1 - 1.5 * (3 - n) ** 0.5 * r ** 0.5) / (1 + 1.5 * (n - 2) / (
                3 - n) ** 0.5 * r ** 0.5) * self.calc["v_ch"]

        def velocity_r(r):
            r /= self.calc["r_ch"]
            return (2.31 * (5 - n) ** 0.5 / (3 - n) * r ** 0.5 * (1 - 1.63 * (3 - n) ** 0.5 * r ** 0.5) / (1 + 1.63 * (n - 2) / (3 - n) ** 0.5 * r ** 0.5) * self.calc["v_ch"])

        function_dict = {
            "t": time_b,
            "tr": time_r,
            "v": velocity_b,
            "vr": velocity_r
        }
        return function_dict[key]
    
########################################################################################################################
    def _s2ngt5_solution(self, n, key):
        """Returns a specific type of function used for the s=2, n>5 cases.

        Args:
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified for s=2, n>5 cases
        """
        #Units cm/s
        def radius_b(t):
            return (self.calc["RCn"]/(PC_TO_KM*10**5))*self.cnst.bbm * (t*YR_TO_SEC) ** ((n-3)/(n-2))  

        def radius_r(t):
            return (self.calc["RCn"]/(PC_TO_KM*10**5))*self.cnst.bbm * (t*YR_TO_SEC) ** ((n-3)/(n-2))/self.cnst.l_ed

        def velocity_b(t):
            return ((n-3)/(n-2)) *((self.calc["RCn"]/(10**5))*self.cnst.bbm) * (t*YR_TO_SEC) ** ((-1)/(n-2)) 

        def velocity_r(t):
            vf2 = ((n-3)/(n-2)) *((self.calc["RCn"]/(10**5))*self.cnst.bbm) * (t*YR_TO_SEC) ** ((-1)/(n-2))
            rr2 = (self.calc["RCn"]/(10**5))*self.cnst.bbm * (t*YR_TO_SEC) ** ((n-3)/(n-2))/self.cnst.l_ed
            return (rr2/(t*YR_TO_SEC)) - (vf2/self.cnst.l_ed)
        function_dict = {
            "r": radius_b,
            "rr": radius_r,
            "v": velocity_b,
            "vr": velocity_r
        }
        return function_dict[key]
    
#########################################################################################################################
    def _ed_solution(self, n, key):
        """General ED solution as defined in Truelove and McKee.

        Args:
            n (int): n value for SNR
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified using the ED solution
        """

        def time_b(r):
            r /= self.calc["r_ch"]
            return ((self.cnst.alpha / 2) ** 0.5 * r / self.cnst.l_ed * (1 - (3 - n) / 3 * (
                self.cnst.phi_eff / self.cnst.l_ed / self.cnst.f_n) ** 0.5 * r ** 1.5) ** (
                -2 / (3 - n)) * self.calc["t_ch"])

        def time_r(r):
            r /= self.calc["r_ch"]
            return (self.cnst.alpha / 2) ** 0.5 * r * (1 - (3 - n) / 3 * (
                self.cnst.phi_ed / self.cnst.l_ed / self.cnst.f_n) ** 0.5 * (r * self.cnst.l_ed) ** 1.5) ** ( -2 / (
                3 - n)) * self.calc["t_ch"]

        def velocity_b(r):
            r /= self.calc["r_ch"]
            return ((2 / self.cnst.alpha) ** 0.5 * self.cnst.l_ed * ((1 - (3 - n) / 3 * (
                self.cnst.phi_eff / self.cnst.l_ed / self.cnst.f_n) ** 0.5 * r ** 1.5) ** ((5 - n) / (3 - n))) / (
                1 + n / 3 * (self.cnst.phi_eff / self.cnst.l_ed / self.cnst.f_n)** 0.5 * r ** 1.5) * self.calc["v_ch"])

        def velocity_r(r):
            r /= self.calc["r_ch"]
            return ((2 * self.cnst.phi_ed / self.cnst.alpha / self.cnst.f_n) ** 0.5 * self.cnst.l_ed * r ** 1.5 * (
                (1 - (3 - n) / 3 * (self.cnst.phi_ed / self.cnst.f_n) ** 0.5 * self.cnst.l_ed * r ** 1.5) ** (
                2 / (3 - n))) / (1 + n / 3 * (self.cnst.phi_ed / self.cnst.f_n) ** 0.5 * self.cnst.l_ed * r ** 1.5) *
                    self.calc["v_ch"])

        function_dict = {
            "t": time_b,
            "tr": time_r,
            "v": velocity_b,
            "vr": velocity_r
        }
        return function_dict[key]
    
########################################################################################################################
    def _st_solution(self, key):
        """Offset ST solution for forward shock and constant acceleration solution for reverse shock.

        Args:
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified using the ST solution
        """

        def radius_b(t):
            t /= self.calc["t_ch"]
            return ((self.cnst.r_st ** 2.5 + XI_0 ** 0.5 * (t - self.cnst.t_st)) ** 0.4 *
                    self.calc["r_ch"])

        def velocity_b(t):
            t /= self.calc["t_ch"]
            return (2 / 5 * XI_0 ** 0.5 * (self.cnst.r_st ** 2.5 + XI_0 ** 0.5 * (t - self.cnst.t_st)) **
                    -0.6 * self.calc["v_ch"])

        def radius_r(t):
            t /= self.calc["t_ch"]
            return (t * (self.cnst.r_rchg / self.cnst.t_rchg - self.cnst.a_rchg * (t - self.cnst.t_rchg) - (
                self.cnst.v_rchg - self.cnst.a_rchg * self.cnst.t_rchg) * np.log(t / self.cnst.t_rchg)) *
                    self.calc["r_ch"])

        def velocity_r(t):
            t /= self.calc["t_ch"]
            return (self.cnst.v_rchg + self.cnst.a_rchg * (t - self.cnst.t_rchg)) * self.calc["v_ch"]

        function_dict = {
            "r": radius_b,
            "rr": radius_r,
            "v": velocity_b,
            "vr": velocity_r
        }
        return function_dict[key]

####################################################################################################################
    def _cn_solution(self, n, key):
        """CN solution from Truelove and Mckee.

        Args:
            n (int): n value for SNR
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified using the CN solution
        """

        def radius_b(t):
            t /= self.calc["t_ch"]
            return (27 * self.cnst.l_ed ** (n - 2) / 4 / np.pi / n / (n - 3) / self.cnst.phi_ed * (10 / 3 * (n - 5) / (
                n - 3)) ** ((n - 3) / 2)) ** (1 / n) * t ** ((n - 3) / n) * self.calc["r_ch"]

        def radius_r(t):
            return radius_b(t) / self.cnst.l_ed

        def velocity_b(t):
            t /= self.calc["t_ch"]
            return ((n - 3) / n * (27 * self.cnst.l_ed ** (n - 2) / 4 / np.pi / n / (n - 3) / self.cnst.phi_ed * (
                10 / 3 * (n - 5) / (n - 3)) ** ((n - 3) / 2)) ** (1 / n) * t ** (-3 / n) * self.calc["v_ch"])

        def velocity_r(t):
            t /= self.calc["t_ch"]
            return (3 / n / self.cnst.l_ed * (27 * self.cnst.l_ed ** (n - 2) / 4 / np.pi / n / (
                n - 3) / self.cnst.phi_ed * (10 / 3 * (n - 5) / (n - 3)) ** ((n - 3) / 2)) ** (1 / n) * t ** (
                -3 / n) * self.calc["v_ch"])

        function_dict = {
            "r": radius_b,
            "rr": radius_r,
            "v": velocity_b,
            "vr": velocity_r
        }
        return function_dict[key]

########################################################################################################################
    def _pds_solution(self, n, key):
        """PDS solution adapted from Cioffi et al.

        Args:
            n (int): n value for SNR
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified for PDS phase
        """

        def radius(t):
            if self.calc["t_pds"] < self.calc["t_st"]:
                previous = "ED"
            else:
                previous = "ST"
            r_end = self.radius_functions[n, previous](self.calc["t_pds"])
            if isinstance(t, np.ndarray):
                r = cumtrapz(velocity(np.array(t))/PCyr_TO_KMs, t, initial=0) + r_end
            else:
                r = r_end + quad(lambda t: velocity(t)/PCyr_TO_KMs, self.calc["t_pds"], t)[0]
            return r

        def velocity(t):
            t /= self.calc["t_pds"]
            if isinstance(t, float):
                if t < 1.1:
                    return lin_velocity(t)
                else:
                    return reg_velocity(t)
            else:
                return np.concatenate([lin_velocity(t[np.where(t < 1.1)]), reg_velocity(t[np.where(t >= 1.1)])])

        def reg_velocity(t):
            v_pds = 413 * self.data["n_0"] ** (1 / 7) * self.data["zeta_m"] ** (3 / 14) * self.data["e_51"] ** (1 / 14)
            return v_pds * (4 * t / 3 - 1 / 3) ** -0.7

        def lin_velocity(t):
            """Linear velocity function to join previous phase to v_pds (reg_velocity)."""

            if self.calc["t_pds"] < self.calc["t_st"]:
                previous = "ED"
            else:
                previous = "ST"
            v_end = self.velocity_functions[n, previous](self.calc["t_pds"])
            return (reg_velocity(1.1) - v_end) / 0.1 * (t - 1) + v_end

        function_dict = {
            "r": radius,
            "v": velocity
        }
        return function_dict[key]

######################################################################################################################
    def _mcs_solution(self, n, key):
        """MCS solution adapted from Cioffi et al.

        Args:
            n (int): n value for SNR
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified for MCS phase
        """

        def radius(t):
            if (self.data["model"] == "cism"):
                previous = "CISM"
            else:
                previous = "PDS"
            r_end = self.radius_functions[n, previous](self.calc["t_mcs"])
            if isinstance(t, np.ndarray):
                r = cumtrapz(velocity(np.array(t))/PCyr_TO_KMs, t, initial=0) + r_end
            else:
                r = r_end + quad(lambda t: velocity(t)/PCyr_TO_KMs, self.calc["t_mcs"], t)[0]
            return r

        def velocity(t):
            if (self.data["model"] == "cism"):
                v_mcs = self.velocity_functions[n, "CISM"](self.calc["t_mcs"])
                r_mcs = self.radius_functions[n, "CISM"](self.calc["t_mcs"])
                return v_mcs * (1 + 4 * v_mcs / r_mcs * (t - self.calc["t_mcs"]) / PCyr_TO_KMs) ** (-3 / 4)
            else:
                t /= self.calc["t_pds"]
                t_ratio = self.calc["t_mcs"] / self.calc["t_pds"]
                if isinstance(t, float):
                    if t < t_ratio * 1.1:
                        return lin_velocity(t)
                    else:
                        return reg_velocity(t)
                else:
                    return np.concatenate([lin_velocity(t[np.where(t < t_ratio * 1.1)]),
                                           reg_velocity(t[np.where(t >= t_ratio * 1.1)])])

        def reg_velocity(t):
            r_pds = 14 * self.data["e_51"] ** (2 / 7) / self.data["n_0"] ** (3 / 7) / self.data["zeta_m"] ** (1 / 7)
            r_mcs = (4.66 * t * (1 - 0.939 * t ** -0.17 + 0.153 / t)) ** 0.25 * r_pds
            t_mcs = self.calc["t_mcs"] / self.calc["t_pds"]
            return PCyr_TO_KMs * r_pds / 4 * (4.66 / self.calc["t_pds"] * (1 - 0.779 * t_mcs ** -0.17)) * (
                4.66 * (t - t_mcs) * (1 - 0.779 * t_mcs ** -0.17) + (r_mcs / r_pds) ** 4) ** -0.75

        def lin_velocity(t):
            """Linear velocity function to join previous phase to v_mcs (reg_velocity)."""

            v_end = self.velocity_functions[n, "PDS"](self.calc["t_mcs"])
            t_ratio = self.calc["t_mcs"] / self.calc["t_pds"]
            return (reg_velocity(t_ratio * 1.1) - v_end) / 0.1 / t_ratio * (t - t_ratio) + v_end

        function_dict = {
            "r": radius,
            "v": velocity
        }
        return function_dict[key]

##########################################################################################################################
    def _fel_solution(self, n, key):
        """Fractional energy loss solution from Liang and Keilty.

        Args:
            n (int): n value for SNR
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified for FEL model
        """

        def radius(t):
            phase = self.get_phase(self.data["t_fel"])
            r_0 = self.radius_functions[n, phase](self.data["t_fel"])
            v_0 = self.velocity_functions[n, phase](self.data["t_fel"]) / PCyr_TO_KMs
            alpha_1 = (2 - self.calc["gamma_0"] + ((2 - self.calc["gamma_0"]) ** 2 + 4 * (self.data["gamma_1"] - 1)) **
                       0.5) / 4 #LK Equation 8
            n1 = 1 / (4 - 3 * alpha_1) #LK Equation Under 5
            return (r_0 ** (1 / n1) + (4 - 3 * alpha_1) * v_0 * (t - self.data["t_fel"]) / r_0 ** (3 * (
                alpha_1 - 1))) ** n1

        def velocity(t):
            phase = self.get_phase(self.data["t_fel"])
            r_0 = self.radius_functions[n, phase](self.data["t_fel"])
            v_0 = self.velocity_functions[n, phase](self.data["t_fel"])
            alpha_1 = (2 - self.calc["gamma_0"] + ((2 - self.calc["gamma_0"]) ** 2 + 4 * (self.data["gamma_1"] - 1)) **
                       0.5) / 4 #LK Equation 8
            return v_0 * (radius(t) / r_0) ** (3 * (alpha_1 - 1))

        function_dict = {
            "r": radius,
            "v": velocity
        }
        return function_dict[key]

#########################################################################################################################
    def _hld_solution(self, n, key):
        """High temperature solution adapted from Tang and Wang.

        Args:
            n (int): n value for SNR
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified for HLD model
        """

        def velocity(t):
            t_0 = 0.1 * self.calc["t_c"]
            phase = self.get_phase(t_0)
            correction = ((self.velocity_functions[n, phase](t_0) / self.calc["c_0"]) ** (5 / 3) - 1) / 10
            return self.calc["c_0"] * (correction * self.calc["t_c"] / t + 1) ** (3 / 5)

        def radius(t):
            t_0 = 0.1 * self.calc["t_c"]
            phase = self.get_phase(t_0)
            r_0 = self.radius_functions[n, phase](t_0)
            if isinstance(t, np.ndarray):
                r = cumtrapz(velocity(t)/PCyr_TO_KMs, t, initial=0) + r_0
            else:
                correction = ((self.velocity_functions[n, phase](t_0) / self.calc["c_0"]) ** (5 / 3) - 1) / 10
                r = r_0 + self.calc["c_0"] * self.calc["t_c"] / PCyr_TO_KMs * quad(lambda t_dl: (
                    correction / t_dl + 1) ** (3 / 5), 0.1, t / self.calc["t_c"])[0]
            return r

        function_dict = {
            "r": radius,
            "v": velocity
        }
        return function_dict[key]
##########################################################################################################################
    def _cism_solution(self, n, key):
        """Cloudy ISM solution from White and Long.

        Args:
            n (int): n value for SNR
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified for CISM model
        """

        def radius(t):
            r_end = self.radius_functions[n, "ED"](self.calc["t_st"])
            if isinstance(t, np.ndarray):
                r = cumtrapz(velocity(np.array(t))/PCyr_TO_KMs, t, initial=0) + r_end
            else:
                r = r_end + quad(lambda t: velocity(t)/PCyr_TO_KMs, self.calc["t_st"], t)[0]
            return r

        def velocity(t):
            transition_end = self.calc["t_st"] * 1.1
            if isinstance(t, np.ndarray):
                return np.concatenate([lin_velocity(t[np.where(t < transition_end)]),
                                       reg_velocity(t[np.where(t >= transition_end)])])
            else:
                if t < transition_end:
                    return lin_velocity(t)
                else:
                    return reg_velocity(t)

        def reg_velocity(t):
            k_cism = K_DICT[self.data["c_tau"]]
            rho_0 = self.data["n_0"] * M_H * self.data["mu_H"]
            gamma = 5 / 3
            r_cism = (25 * (gamma + 1) * k_cism * self.data["e_51"] * 10 ** 51 / 16 / np.pi / rho_0) ** 0.2 * (
                t * YR_TO_SEC) ** 0.4 #r_s in White and Long
            return ((gamma + 1) * k_cism * self.data["e_51"] * 10 ** 51 / 4 / np.pi / rho_0 / r_cism ** 3) ** 0.5 / (
                10 ** 5) #V_s in White and Long

        def lin_velocity(t):
            """Linear velocity function to join ED phase to v_cism (reg_velocity)."""

            if n in (1,2,4):                                                                                          
                v_end = self.velocity_functions[n, "ED"](self.radius_functions[n, "ED"](self.calc["t_st"]))
            else:
                v_end = self.velocity_functions[n, "ED"](self.calc["t_st"])
            return (reg_velocity(1.1 * self.calc["t_st"]) - v_end) / 0.1 / self.calc["t_st"] * (t - self.calc["t_st"]) + v_end

        function_dict = {
            "r": radius,
            "v": velocity
        }
        return function_dict[key]
    
##########################################################################################################################
    def _sedtay_solution(self, n, key):
        """Sedov-Taylor solution, which is the C/Tau=0 solution of the Cloudy ISM model 

        Args:
            n (int): n value for SNR
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified for SedovTaylor model
        """

        def radius(t):
            rho_0 = self.data["n_0"] * M_H * self.data["mu_H"]
            e_0 = self.data["e_51"] * (10 ** 51)
            r = (((XI_0*e_0)/rho_0)**0.2)*(t**0.4)
            return r

        def velocity(t):
            rho_0 = self.data["n_0"] * M_H * self.data["mu_H"]
            e_0 = self.data["e_51"] * (10 ** 51)
            r = 0.4*(((XI_0*e_0)/rho_0)**0.2)*(t**-0.6)
            return r
                
        function_dict = {
            "r": radius,
            "v": velocity
        }
        return function_dict[key]


##########################################################################################################################
    def merger_time(self, n):
        """Gets merger times used to determine which phases occur for the s=0 case.

        Args:
            n (int): n value for SNR

        Returns:
            dict: dictionary of merger times for different phases of SNR evolution
        """

        t_mrg = {}
        if n == 0:
            t_mrg["ED"] = newton(lambda t: self.velocity_functions[n, "ED"](t) - self.calc["c_net"] * BETA, self.calc["t_st"])
        elif n in (1, 2, 4):                                                                     
            try:
                t_mrg["ED"] = newton(lambda t: self.velocity_functions[n, "ED"](
                    self.radius_functions[n, "ED"](t)) - self.calc["c_net"] * BETA, self.calc["t_st"])
            except ValueError:
                t_mrg["ED"] = np.inf
        else:
            t_mrg["ED"] = (self.calc["c_net"] * BETA / self.calc["v_ch"] * n / (n - 3) / (
                27 * self.cnst.l_ed ** (n - 2) / 4 / np.pi / n / (n - 3) / self.cnst.phi_ed * (10 * (n - 5) / 3 / (
                    n - 3)) ** ((n - 3) / 2)) ** (1 / n)) ** (-n / 3) * self.calc["t_ch"]
        if n == 0:
            t_mrg["ST"] = self.calc["t_ch"] / 1.42 * ((0.569 * self.calc["v_ch"] / BETA / self.calc["c_net"]) **
                                                      (5 / 3) + 0.254)
        else:
            t_mrg["ST"] = self.calc["t_ch"] * (((2 * XI_0 ** 0.5 * self.calc["v_ch"] / 5 / BETA / self.calc[
                "c_net"]) ** (5 / 3) - self.cnst.r_st ** 2.5) / XI_0 ** 0.5 + self.cnst.t_st)
        t_mrg["PDS"] = newton(lambda t: self.velocity_functions[n, "PDS"](t) - self.calc["c_net"] * BETA,
                              self.calc["t_pds"] * 2)
        try:
            t_mrg["MCS"] = newton(lambda t: self.velocity_functions[n, "MCS"](t) - self.calc["c_net"] * BETA,
                                  self.calc["t_mcs"])
        except RuntimeError:
            t_mrg["MCS"] = 0
        if self.data["model"] == "fel":
            phase = self.get_phase(self.data["t_fel"])
            r_0 = self.radius_functions[n, phase](self.data["t_fel"]) * PC_TO_KM
            v_0 = self.velocity_functions[n, phase](self.data["t_fel"])
            alpha_1 = (2 - self.calc["gamma_0"] + ((2 - self.calc["gamma_0"]) ** 2 + 4 * (self.data["gamma_1"] - 1)) **
                       0.5) / 4
            t_mrg["FEL"] = (((v_0 / BETA / self.calc["c_net"]) ** (-(4 - 3 * alpha_1) / 3 / (alpha_1 - 1)) - 1) *
                           r_0 / (v_0 * YR_TO_SEC) / (4 - 3 * alpha_1) + self.data["t_fel"])
        else:
            t_mrg["FEL"] = "N/A"
        t_mrg["HLD"] = self.data["t_hld"]
        if (self.data["model"] == "cism"):
            t_mrg["CISM"] = (K_DICT[self.data["c_tau"]] * self.data["e_51"] * 10 ** 51 / 4 / np.pi / self.data["mu_H"] /
                           M_H / self.data["n_0"]) ** (1 / 3) * (4 / 5 * (4 / 45) ** 0.2) ** (5 / 6) / \
                          (self.calc["c_net"] * 10**5 * 2) ** (5 / 3) / YR_TO_SEC
            if t_mrg["CISM"] < self.calc["t_st"] * 1.1:
                t_mrg["CISM"] = newton(lambda t: self.velocity_functions[n, "CISM"](t) - self.calc["c_net"] * BETA, self.calc["t_st"])
        else:
            t_mrg["CISM"] = "N/A"
        return t_mrg

#######################################################################################################################
    def get_phases(self):
        """Get phases that occur for the current input parameters and update the plot dropdown options accordingly.

        Returns:
            list: list of phases that occur for given input parameters
        """

        dropdown_values = ["Current", "Reverse Shock Lifetime", "ED-ST", "PDS", "MCS", "FEL", "HLD"]
        if self.data["s"] == 2:
            phases = "s2",
            dropdown_values = ["Current"]
        elif self.data["model"] == "fel":
            dropdown_values.remove("HLD")
            if self.data["t_fel"] == round(self.calc["t_st"]):
                dropdown_values.remove("PDS")
                dropdown_values.remove("MCS")
                phases = ("ED", "FEL")
                dropdown_values[dropdown_values.index("ED-ST")] = "ED"
            elif self.data["t_fel"] <= min(self.calc["t_pds"], self.calc["t_mrg"]["ST"]):
                dropdown_values.remove("PDS")
                dropdown_values.remove("MCS")
                phases = ("ED", "ST", "FEL")
            elif self.data["t_fel"] < min(self.calc["t_mcs"], self.calc["t_mrg"]["PDS"]):
                dropdown_values.remove("MCS")
                phases = ("ED", "ST", "PDS", "FEL")
            else:
                phases = ("ED", "ST", "PDS", "MCS", "FEL")
        elif self.data["model"] == "hld":
            dropdown_values.remove("FEL")
            dropdown_values.remove("PDS")
            dropdown_values.remove("MCS")
            if self.calc["t_c"] * 0.1 < self.calc["t_st"]:
                phases = ("ED", "HLD")
                dropdown_values[dropdown_values.index("ED-ST")] = "ED"
            else:
                phases = ("ED", "ST", "HLD")
        elif (self.data["model"] == "cism"):
            dropdown_values.remove("FEL")
            dropdown_values.remove("PDS")
            dropdown_values.remove("HLD")
            dropdown_values[dropdown_values.index("ED-ST")] = "ED-CISM"
            if self.calc["t_mcs"] < self.calc["t_mrg"]["CISM"]:
                phases = ("ED", "CISM", "MCS")
            else:
                dropdown_values.remove("MCS")
                phases = ("ED", "CISM")
        else:
            dropdown_values.remove("HLD")
            dropdown_values.remove("FEL")
            if self.calc["t_pds"] > self.calc["t_mrg"]["ST"]:
                dropdown_values.remove("PDS")
                dropdown_values.remove("MCS")
                phases = ("ED","ST")
            elif self.calc["t_mcs"] > self.calc["t_mrg"]["PDS"]:
                dropdown_values.remove("MCS")
                phases = ("ED", "ST", "PDS")
            else:
                phases = ("ED", "ST", "PDS", "MCS")
            if "ST" in phases and self.calc["t_st"] > self.calc["t_mrg"]["ED"]:
                dropdown_values[dropdown_values.index("ED-ST")] = "ED"
                phases = tuple([phase for phase in phases if phase != "ST"])
                self.widgets["model"].input["cism"].config(state="disabled")
        if self.data["s"] != 2 and str(self.widgets["model"].input["cism"].cget("state")) == "disabled" and self.calc["t_st"] < self.calc["t_mrg"]["ED"]:
            self.widgets["model"].input["cism"].config(state="normal")
        old = self.widgets["range"].get_value()
        self.widgets["range"].input.config(values=tuple(dropdown_values))
        
        # Set dropdown value to current if the old value is no longer an option
        if old not in dropdown_values and old != "Custom":
            self.widgets["range"].value_var.set("Current")
            self.data["range"] = "Current"
        return phases

##########################################################################################################################
    def _init_functions(self):
        """Create the function dictionaries used to calculate time, radius, and velocity. Note this function only needs
        to be called once.
        """

        self.radius_functions = {}
        self.velocity_functions = {}
        self.time_functions = {}
        self.emissionMeasure_functions = {}
        self.temperature_functions = {}

        for n in VALUE_DICT_S0:
            self.radius_functions[n, "PDS"] = self._pds_solution(n, "r")
            self.velocity_functions[n, "PDS"] = self._pds_solution(n, "v")
            self.radius_functions[n, "MCS"] = self._mcs_solution(n, "r")
            self.velocity_functions[n, "MCS"] = self._mcs_solution(n, "v")
            self.radius_functions[n, "FEL"] = self._fel_solution(n, "r")
            self.velocity_functions[n, "FEL"] = self._fel_solution(n, "v")
            self.radius_functions[n, "HLD"] = self._hld_solution(n, "r")
            self.velocity_functions[n, "HLD"] = self._hld_solution(n, "v")
            self.radius_functions[n, "CISM"] = self._cism_solution(n, "r")
            self.velocity_functions[n, "CISM"] = self._cism_solution(n, "v")
            
            self.emissionMeasure_functions[n, "eMeasf"] = self._EMTemp_solutions(n, "emf")
            self.emissionMeasure_functions[n, "eMeasr"] = self._EMTemp_solutions(n, "emr")
            self.temperature_functions[n, "Tempf"] = self._EMTemp_solutions(n, "Tempf")
            self.temperature_functions[n, "Tempr"] = self._EMTemp_solutions(n, "Tempr")
            
            if n == 0:
                self.radius_functions[n, "sedTay"] = self._sedtay_solution(n, "r")
                self.velocity_functions[n, "sedTay"] = self._sedtay_solution(n, "v")
                
            if n != 0:
                self.radius_functions[n, "ST"] = self._st_solution("r")
                self.radius_functions[n, "late"] = self._st_solution("rr")
                self.velocity_functions[n, "ST"] = self._st_solution("v")
                self.velocity_functions[n, "late"] = self._st_solution("vr")
                if n not in (1, 2, 4):                                                                     
                    self.radius_functions[n, "ED"] = self._cn_solution(n, "r")
                    self.radius_functions[n, "early"] = self._cn_solution(n, "rr")
                    self.velocity_functions[n, "ED"] = self._cn_solution(n, "v")
                    self.velocity_functions[n, "early"] = self._cn_solution(n, "vr")
                else:
                    self.time_functions[n, "ED"] = self._ed_solution(n, "t")
                    self.time_functions[n, "early"] = self._ed_solution(n, "tr")
                    self.radius_functions[n, "ED"] = lambda t, n=n: brentq(
                        lambda r, t: self.time_functions[n, "ED"](r) - t, 0, self.cnst.r_st * self.calc["r_ch"], t)
                    self.radius_functions[n, "early"] = lambda t, n=n: brentq(
                        lambda r, t: self.time_functions[n, "early"](r) - t, 0, self.cnst.r_rchg * self.calc["r_ch"], t)
                    self.velocity_functions[n, "ED"] = self._ed_solution(n, "v")
                    self.velocity_functions[n, "early"] = self._ed_solution(n, "vr")
            else:
                self.radius_functions[n, "ED"] = self._s0n0_solution("r", "ED")
                self.radius_functions[n, "ST"] = self._s0n0_solution("r", "ST")
                self.radius_functions[n, "early"] = self._s0n0_solution("rr", "early")
                self.radius_functions[n, "late"] = self._s0n0_solution("rr", "late")
                self.velocity_functions[n, "ED"] = self._s0n0_solution("v", "ED")
                self.velocity_functions[n, "ST"] = self._s0n0_solution("v", "ST")
                self.velocity_functions[n, "early"] = self._s0n0_solution("vr", "early")
                self.velocity_functions[n, "late"] = self._s0n0_solution("vr", "late")

        for n in (0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14):                                                                      
            if n >= 6:
                self.radius_functions[n, "s2"] = self._s2ngt5_solution(n, "r")
                self.radius_functions[n, "s2r"] = self._s2ngt5_solution(n, "rr")
                self.velocity_functions[n, "s2"] = self._s2ngt5_solution(n, "v")
                self.velocity_functions[n, "s2r"] = self._s2ngt5_solution(n, "vr")
            else:
                self.time_functions[n, "s2"] = self._s2nlt3_solution(n, "t")
                self.time_functions[n, "s2r"] = self._s2nlt3_solution(n, "tr")
                self.radius_functions[n, "s2"] = lambda t, n=n: brentq(
                    lambda r, t: self.time_functions[n, "s2"](r) - t, 0, 1 / 1.5 ** 2 / (
                        3 - self.data["n"]) * self.calc["r_ch"] - 1, t)
                self.radius_functions[n, "s2r"] = lambda t, n=n: brentq(
                    lambda r, t: self.time_functions[n, "s2r"](r) - t, 0, 1 / 1.5 ** 2 / (
                        3 - self.data["n"]) / 1.19 * self.calc["r_ch"] - 1, t)
                self.velocity_functions[n, "s2"] = self._s2nlt3_solution(n, "v")
                self.velocity_functions[n, "s2r"] = self._s2nlt3_solution(n, "vr")

#######################################################################################################################
    def get_plot_dEMF(self, n, t):
        """Get dEMF from n and t values

        Returns:
            dEMF: dEMF value from input parameters
        """
        self.dEMFcnsts =dEMFInv_DICT[n]
        self.calc["cx"] = ((((27*self.cnst.l_ed**(self.data["n"]-2))/(4*np.pi*self.data["n"]*(self.data["n"]-3)*self.cnst.phi_ed))
                                *(((10*(self.data["n"]-5))/(3*(self.data["n"]-3)))**((self.data["n"]-3)/2)))**(1/self.data["n"]))
        self.calc["RCRf"] = 0
        if (t < self.cnst.t_st):
            self.calc["R0STf"] = self.calc["r_ch"]*self.calc["cx"]*(t)**((self.data["n"]-3)/self.data["n"])
            self.calc["RCRf"] = self.calc["R0STf"]#*PC_TO_KM*10**5
        elif (t >= self.cnst.t_st):
            self.calc["RTMf"] = self.calc["r_ch"]*((self.cnst.r_st**2.5)+(((XI_0)**0.5)*(t-self.cnst.t_st)))**(0.4)
            self.calc["RCRf"] = self.calc["RTMf"]#*PC_TO_KM*10**5
        #self.calc["EM0"] = (16*self.data["n_0"]**2*(MU_H/MU_e)*self.calc["RCRf"]**3)
        self.calc["EM0"] = (0.4*0.473*self.calc["RCRf"]**3)
       
        if (t < self.dEMFcnsts.t1ef):
           dEMF = self.dEMFcnsts.def0
        
        elif ((self.dEMFcnsts.t1ef <= t) and (t < self.dEMFcnsts.t2ef)):
            dEMF = self.dEMFcnsts.def0 * ((t/self.dEMFcnsts.t1ef)**(self.dEMFcnsts.a1ef))
            
        elif ((self.dEMFcnsts.t2ef <= t) and (t < self.dEMFcnsts.t3ef)):
            dEMF = (self.dEMFcnsts.def0 * ((self.dEMFcnsts.t2ef/self.dEMFcnsts.t1ef)**(self.dEMFcnsts.a1ef)) 
                                        * ((t/self.dEMFcnsts.t2ef)**(self.dEMFcnsts.a2ef)))

        elif ((self.dEMFcnsts.t3ef <= t) and (t < self.dEMFcnsts.t4ef)):
            dEMF = (self.dEMFcnsts.def0 * ((self.dEMFcnsts.t2ef/self.dEMFcnsts.t1ef)**(self.dEMFcnsts.a1ef)) 
                                        * ((self.dEMFcnsts.t3ef/self.dEMFcnsts.t2ef)**(self.dEMFcnsts.a2ef)) 
                                        * ((t/self.dEMFcnsts.t3ef)**(self.dEMFcnsts.a3ef)))
            
        elif (self.dEMFcnsts.t4ef <= t):
            dEMF = (self.dEMFcnsts.def0 * ((self.dEMFcnsts.t2ef/self.dEMFcnsts.t1ef)**(self.dEMFcnsts.a1ef)) 
                                        * ((self.dEMFcnsts.t3ef/self.dEMFcnsts.t2ef)**(self.dEMFcnsts.a2ef)) 
                                        * ((self.dEMFcnsts.t4ef/self.dEMFcnsts.t3ef)**(self.dEMFcnsts.a3ef))
                                        * ((t/self.dEMFcnsts.t4ef)**(self.dEMFcnsts.a4ef)))
        return self.calc["EM0"]*dEMF*10**58
            

#######################################################################################################################
    def get_plot_dTF(self, n, t):
        """Get dTF from n and t values

        Returns:
            dTF: dTF value from input parameters
        """
        self.dTFcnsts = dTFInv_DICT[n]
        self.calc["cx"] = ((((27*self.cnst.l_ed**(self.data["n"]-2))/(4*np.pi*self.data["n"]*(self.data["n"]-3)*self.cnst.phi_ed))
                                *(((10*(self.data["n"]-5))/(3*(self.data["n"]-3)))**((self.data["n"]-3)/2)))**(1/self.data["n"]))
        self.calc["VCRf"] = 0
        if (t < self.cnst.t_st):
            self.calc["V0STf"] = self.calc["v_ch"]*10**5*((self.data["n"]-3)/self.data["n"])*self.calc["cx"]*(t)**(-3/self.data["n"])
            self.calc["VCRf"] = self.calc["V0STf"] 
        elif (t >= self.cnst.t_st):
            self.calc["VTMf"] = 0.4*((XI_0)**0.5)*self.calc["v_ch"]*10**5*((((self.cnst.r_st)**(2.5))+(((XI_0)**0.5)*(t-self.cnst.t_st)))**(-0.6))
            self.calc["VCRf"] = self.calc["VTMf"]
        
        self.calc["T0"] = (3/16)*MU_t*M_H/KEV_TO_ERG*self.calc["VCRf"]**2

        if (t < self.dTFcnsts.t1tf):
           dTF = self.dTFcnsts.dtf
        
        elif ((self.dTFcnsts.t1tf <= t) and (t < self.dTFcnsts.t2tf)):
            dTF = self.dTFcnsts.dtf * ((t/self.dTFcnsts.t1tf)**(self.dTFcnsts.a1tf))
            
        elif (self.dTFcnsts.t2tf <= t):
            dTF = (self.dTFcnsts.dtf * ((self.dTFcnsts.t2tf/self.dTFcnsts.t1tf)**(self.dTFcnsts.a1tf)) 
                                        * ((t/self.dTFcnsts.t2tf)**(self.dTFcnsts.a2tf)))
        return self.calc["T0"]*dTF     
            

#######################################################################################################################
    def get_plot_dEMR(self, n, t):
        """Get dEMR from n and t values

        Returns:
            dEMR: dEMR value from input parameters
        """
        self.dEMRcnsts = dEMRInv_DICT[n]
        self.calc["cx"] = ((((27*self.cnst.l_ed**(self.data["n"]-2))/(4*np.pi*self.data["n"]*(self.data["n"]-3)*self.cnst.phi_ed))
                                *(((10*(self.data["n"]-5))/(3*(self.data["n"]-3)))**((self.data["n"]-3)/2)))**(1/self.data["n"]))
        self.calc["RCRf"] = 0
        if (t < self.cnst.t_st):
            self.calc["R0STf"] = self.calc["r_ch"]*self.calc["cx"]*(t)**((self.data["n"]-3)/self.data["n"])
            self.calc["RCRf"] = self.calc["R0STf"]#*PC_TO_KM*10**5
        elif (t >= self.cnst.t_st):
            self.calc["RTMf"] = self.calc["r_ch"]*((self.cnst.r_st**2.5)+(((XI_0)**0.5)*(t-self.cnst.t_st)))**(0.4)
            self.calc["RCRf"] = self.calc["RTMf"]#*PC_TO_KM*10**5
            
        #self.calc["EM0"] = (16*self.data["n_0"]**2*(MU_H/MU_e)*self.calc["RCRf"]**3)
        self.calc["EM0"] = (0.4*0.473*self.calc["RCRf"]**3)
       
        if (t < self.dEMRcnsts.t1er):
           dEMR = self.dEMRcnsts.der * ((t/self.dEMRcnsts.t1er)**(-1.0*self.dEMRcnsts.a1er))
        
        elif ((self.dEMRcnsts.t1er <= t) and (t < self.dEMRcnsts.t2er)):
            dEMR = self.dEMRcnsts.der * ((t/self.dEMRcnsts.t1er)**(-1.0*self.dEMRcnsts.a2er))
            
        elif ((self.dEMRcnsts.t2er <= t) and (t < self.dEMRcnsts.t3er)):
            dEMR = (self.dEMRcnsts.der * ((self.dEMRcnsts.t2er/self.dEMRcnsts.t1er)**(-1.0*self.dEMRcnsts.a2er)) 
                                        * ((t/self.dEMRcnsts.t2er)**(-1.0*self.dEMRcnsts.a3er)))

        elif (self.dEMRcnsts.t3er <= t):
            dEMR = (self.dEMRcnsts.der * ((self.dEMRcnsts.t2er/self.dEMRcnsts.t1er)**(-1.0*self.dEMRcnsts.a2er)) 
                                        * ((self.dEMRcnsts.t3er/self.dEMRcnsts.t2er)**(-1.0*self.dEMRcnsts.a3er)) 
                                        * ((t/self.dEMRcnsts.t3er)**(-1.0*self.dEMRcnsts.a4er)))
        return self.calc["EM0"]*dEMR*10**58
            

#######################################################################################################################
    def get_plot_dTR(self, n, t):
        """Get dTR from n and t values

        Returns:
            dTR: dTR value from input parameters
        """

        self.dTRcnsts = dTRInv_DICT[n]
        self.calc["cx"] = ((((27*self.cnst.l_ed**(self.data["n"]-2))/(4*np.pi*self.data["n"]*(self.data["n"]-3)*self.cnst.phi_ed))
                                *(((10*(self.data["n"]-5))/(3*(self.data["n"]-3)))**((self.data["n"]-3)/2)))**(1/self.data["n"]))
        self.calc["VCRf"] = 0
        if (t < self.cnst.t_st):
            self.calc["V0STf"] = self.calc["v_ch"]*10**5*((self.data["n"]-3)/self.data["n"])*self.calc["cx"]*(t)**(-3/self.data["n"])
            self.calc["VCRf"] = self.calc["V0STf"]
        elif (t >= self.cnst.t_st):
            self.calc["VTMf"] = 0.4*((XI_0)**0.5)*self.calc["v_ch"]*10**5*((((self.cnst.r_st)**(2.5))+(((XI_0)**0.5)*(t-self.cnst.t_st)))**(-0.6))
            self.calc["VCRf"] = self.calc["VTMf"]
        
        self.calc["T0"] = (3/16)*MU_t*M_H/KEV_TO_ERG*self.calc["VCRf"]**2
       
        if (t < self.dTRcnsts.t1tr):
            dTR = self.dTRcnsts.dtr * ((t/self.dTRcnsts.t1tr)**(self.dTRcnsts.a1tr))
        
        elif ((self.dTRcnsts.t1tr <= t) and (t < self.dTRcnsts.t2tr)):
            dTR = self.dTRcnsts.dtr * ((t/self.dTRcnsts.t1tr)**(self.dTRcnsts.a2tr))
            
        elif ((self.dTRcnsts.t2tr <= t) and (t < self.dTRcnsts.t3tr)):
            dTR = (self.dTRcnsts.dtr * ((self.dTRcnsts.t2tr/self.dTRcnsts.t1tr)**(self.dTRcnsts.a2tr)) 
                                        * ((t/self.dTRcnsts.t2tr)**(self.dTRcnsts.a3tr)))

        elif (self.dTRcnsts.t3tr <= t):
            dTR = (self.dTRcnsts.dtr * ((self.dTRcnsts.t2tr/self.dTRcnsts.t1tr)**(self.dTRcnsts.a2tr)) 
                                        * ((self.dTRcnsts.t3tr/self.dTRcnsts.t2tr)**(self.dTRcnsts.a3tr)) 
                                        * ((t/self.dTRcnsts.t3tr)**(self.dTRcnsts.a4tr)))
        return self.calc["T0"]*dTR     
   
       
#####################################################################################################################
##############################################################################################################################
#=============================================================================================================================
##############################################################################################################################
class SuperNovaRemnantInverse:
    """Calculate and store data for supernova remnant inverse calculations.
    Attributes:
        root (str): ID of GUI main window, used to access widgets
        widgets (dict): widgets used for input
        buttons (dict): emissivity and abundance buttons in program window
        data (dict): values from input window
        calc (dict): values calculated from input values
        graph (snr_gui.TimePlot): plot used to show radius and velocity as functions of time
        cnst (dict): named tuple of constants used for supernova remnant calculations, used only for s=0
        radius_functions (dict): functions used to calculate radius as a function of time
        velocity_functions (dict): functions used to calculate velocity as a function of time or radius
        time_functions (dict): functions used to calculate time as a function of radius
    """
###########################################################################################################################
    def __init__(self, root):
        """Initialize SuperNovaRemnant instance.
        Args:
            root (str): ID of parent window, used to access widgets only from parent
        """

        self.root = root
        #self.widgetsInv = gui.InputParam.instances[self.root]
        self.buttons = {}
        self.data = {}
        self.calc = {}
        # Graph is defined in snr.py module
        self.graph = None
        # Constants are defined when an n value is specified
        self.cnst = None
        #self.radius_functions = {}
        #self.velocity_functions = {}
        #self.time_functions = {}
        #self._init_functions()

############################################################################################################################
    def update_output(self):

        self.data.update(gui.InputParam.get_values(self.root))

        # n must be converted to an integer since it is used as a key in the function dictionaries
        self.data["n_inv"] = round(self.data["n_inv"])
        # Calculate phase transition times and all values needed for future radius and velocity calculations.
        if self.data["s_inv"] == 0:
            self.cnst = VALUE_DICT_S0[self.data["n_inv"]]
            phases = self.get_phases()

            if (self.data["model_inv"] == "standard_forward"):
            # Values to be displayed as output, note r and v are added later and rr and vr are updated later if t<t_rev
                self.calculate_Standard_Forward_values()
                self.verify_forward_generalT_n6to14()
                output_data = {
                    "t_inv": "Age: " + str(round(self.calc["tcn_inv"],4)) + " yrs",
                    "E51_inv": "Energy: {:.4e} erg".format(self.calc["E51c_inv"]*10**51),
                    "n_0_inv": "ISM number density: " + str(round(self.calc["n0_inv"],4)) + " cm\u207B\u00B3",
                    "R_f_out": str(round(self.calc["Rs_ver"],4)) + " pc",
                    "t_f_out": str(round(self.calc["TE_ver"],4)) + " keV",
                    "EM_f_out": "{:.4e} cm\u207B\u00B3".format(self.calc["EM_ver"]*10**58),
                    "R_r_out": str(round(self.calc["Rrev_predict"],4)) + " pc",
                    "t_r_out": str(round(self.calc["Terav_predict"],4)) + " keV",
                    "EM_r_out": "{:.4e} cm\u207B\u00B3".format(self.calc["EMR_predict"]*10**58),
                    "Core": "RS Reaches Core: " + str(round(self.calc["tch_inv"]*self.cnst.t_rchg,4)) + " yrs",
                    "Rev": "RS Reaches Center: " + str(round(self.calc["tch_inv"]*self.cnst.t_rev,4)) + " yrs"
                }

            elif (self.data["model_inv"] == "standard_reverse"):
                self.calculate_Standard_Reverse_values()
                self.verify_reverse_generalT_n6to14()
                output_data = {
                    "t_inv": "Age: " + str(round(self.calc["tcn_inv"],4)) + " yrs",
                    "E51_inv": "Energy: {:.4e} erg".format(self.calc["E51c_inv"]*10**51),
                    "n_0_inv": "ISM number density: " + str(round(self.calc["n0_inv"],4)) + " cm\u207B\u00B3",
                    "R_f_out": str(round(self.calc["Rs_ver"],4)) + " pc",
                    "t_f_out": str(round(self.calc["TEf_predict"],4)) + " keV",
                    "EM_f_out": "{:.4e} cm\u207B\u00B3".format(self.calc["EMF_predict"]*10**58),
                    "R_r_out": str(round(self.calc["Rrev_predict"],4)) + " pc",
                    "t_r_out": str(round(self.calc["TE_ver"],4)) + " keV",
                    "EM_r_out": "{:.4e} cm\u207B\u00B3".format(self.calc["EM_ver"]*10**58),
                    "Core": "RS Reaches Core: " + str(round(self.calc["tch_inv"]*self.cnst.t_rchg,4)) + " yrs",
                    "Rev": "RS Reaches Center: " + str(round(self.calc["tch_inv"]*self.cnst.t_rev,4)) + " yrs"
                }
            elif (self.data["model_inv"] == "cloudy_forward"):
                self.cloudyCnst = VALUE_DICT_Cloudy[self.data["ctau_inv"]]
                self.calculate_Cloudy_Forward_values()
                self.verify_forward_cloudyT_n6to14()
                output_data = {
                    "t_inv": "Age: " + str(round(self.calc["t_inv"],4)) + " yrs",
                    "E51_inv": "Energy: {:.4e} erg".format(self.calc["E51_inv"]*10**51),
                    "n_0_inv": "ISM number density: " + str(round(self.calc["n0_inv"],4)) + " cm\u207B\u00B3",
                    "R_f_out": str(round(self.calc["Rs_ver"],4)) + " pc",
                    "t_f_out": str(round(self.calc["TE_ver"],4)) + " keV",
                    "EM_f_out": "{:.4e} cm\u207B\u00B3".format(self.calc["EM_ver"]*10**58),
                    "R_r_out": "N/A",
                    "t_r_out": "N/A",
                    "EM_r_out": "N/A",
                    "Core": "This model only has the \nself-similar phase.",
                    "Rev": "",
                    "t-s2": ""
                }

            else:
            # Values to be displayed as output, note r and v are added later and rr and vr are updated later if t<t_rev
                self.calculate_Sedov_Forward_values()
                self.verify_forward_sedovT_n6to14()
                output_data = {
                    "t_inv": "Age: " + str(round(self.calc["t_inv"],4)) + " yrs",
                    "E51_inv": "Energy: {:.4e} erg".format(self.calc["E51_inv"]*10**51),
                    "n_0_inv": "ISM number density: " + str(round(self.calc["n0_inv"],4)) + " cm\u207B\u00B3",
                    "R_f_out": str(round(self.calc["Rs_ver"],4)) + " pc",
                    "t_f_out": str(round(self.calc["TE_ver"],4)) + " keV",
                    "EM_f_out": "{:.4e} cm\u207B\u00B3".format(self.calc["EM_ver"]*10**58),
                    "R_r_out": "N/A",
                    "t_r_out": "N/A",
                    "EM_r_out": "N/A",
                    "Core": "This model only has the \nself-similar phase.",
                    "Rev": "",
                    "t-s2": ""
                }


        else: #s=2 case
            self.cnst = VALUE_DICT_S2[self.data["n_inv"]]
            phases = self.get_phases()
            if (self.data["model_inv"] == "standard_forward"):
                self.calculate_S2_Standard_Forward_values()
                self.verify_S2_forward_generalT_n6to14()
                output_data = {
                    "t_inv": "Age: " + str(round(self.calc["t_inv"],4)) + " yrs",
                    "E51_inv": "Energy: {:.4e} erg".format(self.calc["E51_inv"]*10**51),
                    "n_0_inv": "\u1E40/\u00284\u03C0V\u1D65\u1D65\u0029: {:.4e} g/cm".format(self.calc["q_inv"]),
                    "R_f_out": str(round(self.calc["Rs_ver"],4)) + " pc",
                    "t_f_out": str(round(self.calc["TE_ver"],4)) + " keV",
                    "EM_f_out": "{:.4e} cm\u207B\u00B3".format(self.calc["EM_ver"]*10**58),
                    "R_r_out": str(round(self.calc["Rrev_predict"],4)) + " pc",
                    "t_r_out": str(round(self.calc["Terav_predict"],4)) + " keV",
                    "EM_r_out": "{:.4e} cm\u207B\u00B3".format(self.calc["EMR_predict"]*10**58),
                    "Core": "",
                    "Rev": "",
                    "t-s2": "This model only includes the\nejecta-dominated phase \nfor t < tc\u2092\u1D63\u2091."   # Set transition time output value to display message explaining lack of transition times
                }
            elif (self.data["model_inv"] == "standard_reverse"):
                self.calculate_S2_Standard_Reverse_values()
                self.verify_S2_reverse_generalT_n6to14()
                output_data = {
                    "t_inv": "Age: " + str(round(self.calc["t_inv"],4)) + " yrs",
                    "E51_inv": "Energy: {:.4e} erg".format(self.calc["E51_inv"]*10**51),
                    "n_0_inv": "\u1E40/\u00284\u03C0V\u1D65\u1D65\u0029: {:.4e} g/cm".format(self.calc["q_inv"]),
                    "R_f_out": str(round(self.calc["Rs_ver"],4)) + " pc",
                    "t_f_out": str(round(self.calc["Terav_predict"],4)) + " keV",
                    "EM_f_out": "{:.4e} cm\u207B\u00B3".format(self.calc["EMF_predict"]*10**58),
                    "R_r_out": str(round(self.calc["Rrev_predict"],4)) + " pc",
                    "t_r_out": str(round(self.calc["TE_ver"],4)) + " keV",
                    "EM_r_out": "{:.4e} cm\u207B\u00B3".format(self.calc["EM_ver"]*10**58),
                    "Core": "",
                    "Rev": "",
                    "t-s2": "This model only includes the\nejecta-dominated phase \nfor t < tc\u2092\u1D63\u2091."   # Set transition time output value to display message explaining lack of transition times
                }
            else: #Standard Forward
                self.calculate_S2_Standard_Forward_values()
                self.verify_S2_forward_generalT_n6to14()
                output_data = {
                    "t_inv": "Age: " + str(round(self.calc["t_inv"],4)) + " yrs",
                    "E51_inv": "Energy: {:.4e} erg".format(self.calc["E51_inv"]*10**51),
                    "n_0_inv": "\u1E40/\u00284\u03C0V\u1D65\u1D65\u0029: {:.4e} g/cm".format(self.calc["q_inv"]),
                    "R_f_out": str(round(self.calc["Rs_ver"],4)) + " pc",
                    "t_f_out": str(round(self.calc["TE_ver"],4)) + " keV",
                    "EM_f_out": "{:.4e} cm\u207B\u00B3".format(self.calc["EM_ver"]*10**58),
                    "R_r_out": str(round(self.calc["Rrev_predict"],4)) + " pc",
                    "t_r_out": str(round(self.calc["Terav_predict"],4)) + " keV",
                    "EM_r_out": "{:.4e} cm\u207B\u00B3".format(self.calc["EMR_predict"]*10**58),
                    "Core": "",
                    "Rev": "",
                    "t-s2": "This model only includes the\nejecta-dominated phase \nfor t < tc\u2092\u1D63\u2091."   # Set transition time output value to display message explaining lack of transition times
                }
            # Change units of m_w and v_w to fit with those used in Truelove and McKee
       # output_data.update(self.get_specific_data())

        gui.OutputValue.update(output_data, self.root, 0, phases)

#######################################################################################################################
    def outputFile_createLine(self, inputValueDict):

        self.data.update(inputValueDict)

        # n must be converted to an integer since it is used as a key in the function dictionaries
        self.data["n_inv"] = round(self.data["n_inv"])
        # Calculate phase transition times and all values needed for future radius and velocity calculations.
        if self.data["s_inv"] == 0:
            self.cnst = VALUE_DICT_S0[self.data["n_inv"]]
            #phases = self.get_phases()

            if (self.data["model_inv"] == "standard_forward"):
            # Values to be displayed as output, note r and v are added later and rr and vr are updated later if t<t_rev
                self.calculate_Standard_Forward_values()
                self.verify_forward_generalT_n6to14()
                output_dataline = ("standard_forward," +
                    "{:.4e}".format(self.calc["tcn_inv"]) + "," +
                    "{:.4e}".format(self.calc["E51c_inv"]*10**51) + "," +
                    "{:.4e}".format(self.calc["n0_inv"]) + "," +
                    "{:.4e}".format(self.calc["Rs_ver"]) + "," +
                    "{:.4e}".format(self.calc["TE_ver"]) + "," +
                    "{:.4e}".format(self.calc["EM_ver"]*10**58) + "," +
                    "{:.4e}".format(self.calc["Rrev_predict"]) + "," +
                    "{:.4e}".format(self.calc["Terav_predict"]) + "," +
                    "{:.4e}".format(self.calc["EMR_predict"]*10**58) + "," +
                    "{:.4e}".format(self.calc["tch_inv"]*self.cnst.t_rchg) + "," +
                    "{:.4e}".format(self.calc["tch_inv"]*self.cnst.t_rev))

            elif (self.data["model_inv"] == "standard_reverse"):
                self.calculate_Standard_Reverse_values()
                self.verify_reverse_generalT_n6to14()
                output_dataline = ("standard_reverse," +
                    "{:.4e}".format(self.calc["tcn_inv"]) + "," +
                    "{:.4e}".format(self.calc["E51c_inv"]*10**51) + "," +
                    "{:.4e}".format(self.calc["n0_inv"]) + "," +
                    "{:.4e}".format(self.calc["Rs_ver"]) + "," +
                    "{:.4e}".format(self.calc["TEf_predict"]) + "," +
                    "{:.4e}".format(self.calc["EMF_predict"]*10**58) + "," +
                    "{:.4e}".format(self.calc["Rrev_predict"]) + "," +
                    "{:.4e}".format(self.calc["TE_ver"]) + "," +
                    "{:.4e}".format(self.calc["EM_ver"]*10**58) + "," +
                    "{:.4e}".format(self.calc["tch_inv"]*self.cnst.t_rchg) + "," +
                    "{:.4e}".format(self.calc["tch_inv"]*self.cnst.t_rev))

            elif (self.data["model_inv"] == "cloudy_forward"):
                self.cloudyCnst = VALUE_DICT_Cloudy[self.data["ctau_inv"]]
                self.calculate_Cloudy_Forward_values()
                self.verify_forward_cloudyT_n6to14()
                output_dataline = ("cloudy_forward," +
                    "{:.4e}".format(self.calc["t_inv"]) + "," +
                    "{:.4e}".format(self.calc["E51_inv"]*10**51) + "," +
                    "{:.4e}".format(self.calc["n0_inv"]) + "," +
                    "{:.4e}".format(self.calc["Rs_ver"]) + "," +
                    "{:.4e}".format(self.calc["TE_ver"]) + "," +
                    "{:.4e}".format(self.calc["EM_ver"]*10**58) + "," +
                    "N/A" + "," + "N/A" + "," + "N/A" + "," + "N/A" + "," + "N/A")

            else:
            # Values to be displayed as output, note r and v are added later and rr and vr are updated later if t<t_rev
                self.calculate_Sedov_Forward_values()
                self.verify_forward_sedovT_n6to14()
                output_dataline = ("sedov," +
                    "{:.4e}".format(self.calc["t_inv"]) + "," +
                    "{:.4e}".format(self.calc["E51_inv"]*10**51) + "," +
                    "{:.4e}".format(self.calc["n0_inv"]) + "," +
                    "{:.4e}".format(self.calc["Rs_ver"]) + "," +
                    "{:.4e}".format(self.calc["TE_ver"]) + "," +
                    "{:.4e}".format(self.calc["EM_ver"]*10**58) + "," +
                    "N/A" + "," + "N/A" + "," + "N/A" + "," + "N/A" + "," + "N/A")


        else: #s=2 case
            self.cnst = VALUE_DICT_S2[self.data["n_inv"]]
            #phases = self.get_phases()
            if (self.data["model_inv"] == "standard_forward"):
                self.calculate_S2_Standard_Forward_values()
                self.verify_S2_forward_generalT_n6to14()
                output_dataline = ("standard_forward," +
                    "{:.4e}".format(self.calc["t_inv"]) + "," +
                    "{:.4e}".format(self.calc["E51_inv"]*10**51) + "," +
                    "{:.4e}".format(self.calc["q_inv"]) + "," +
                    "{:.4e}".format(self.calc["Rs_ver"]) + "," +
                    "{:.4e}".format(self.calc["TE_ver"]) + "," +
                    "{:.4e}".format(self.calc["EM_ver"]*10**58) + "," +
                    "{:.4e}".format(self.calc["Rrev_predict"]) + "," +
                    "{:.4e}".format(self.calc["Terav_predict"]) + "," +
                    "{:.4e}".format(self.calc["EMR_predict"]*10**58) + "," +
                    "N/A" + "," + "N/A")

            elif (self.data["model_inv"] == "standard_reverse"):
                self.calculate_S2_Standard_Reverse_values()
                self.verify_S2_reverse_generalT_n6to14()
                output_dataline = ("standard_reverse," +
                    "{:.4e}".format(self.calc["t_inv"]) + "," +
                    "{:.4e}".format(self.calc["E51_inv"]*10**51) + "," +
                    "{:.4e}".format(self.calc["q_inv"]) + "," +
                    "{:.4e}".format(self.calc["Rs_ver"]) + "," +
                    "{:.4e}".format(self.calc["Terav_predict"]) + "," +
                    "{:.4e}".format(self.calc["EMF_predict"]*10**58) + "," +
                    "{:.4e}".format(self.calc["Rrev_predict"]) + "," +
                    "{:.4e}".format(self.calc["TE_ver"]) + "," +
                    "{:.4e}".format(self.calc["EM_ver"]*10**58) + "," +
                    "N/A" + "," + "N/A")

            else: #Standard Forward
                self.calculate_S2_Standard_Forward_values()
                self.verify_S2_forward_generalT_n6to14()
                output_dataline = ("standard_forward," +
                    "{:.4e}".format(self.calc["t_inv"]) + "," +
                    "{:.4e}".format(self.calc["E51_inv"]*10**51) + "," +
                    "{:.4e}".format(self.calc["q_inv"]) + "," +
                    "{:.4e}".format(self.calc["Rs_ver"]) + "," +
                    "{:.4e}".format(self.calc["TE_ver"]) + "," +
                    "{:.4e}".format(self.calc["EM_ver"]*10**58) + "," +
                    "{:.4e}".format(self.calc["Rrev_predict"]) + "," +
                    "{:.4e}".format(self.calc["Terav_predict"]) + "," +
                    "{:.4e}".format(self.calc["EMR_predict"]*10**58) + "," +
                    "N/A" + "," + "N/A")

        return output_dataline

#######################################################################################################################
    def get_phases(self):
        """Get phases that occur for the current input parameters and update the plot dropdown options accordingly.
        Returns:
            list: list of phases that occur for given input parameters
        """


        if self.data["s_inv"] == 2:
            phases = "s2"

        elif (self.data["model_inv"] == "cloudy_forward"):
            phases = ("t-MCS")

        else:
            phases = ("Core", "Rev", "t-PDS")

        return phases

#######################################################################################################################
    def calculate_Standard_Forward_values(self):

        self.calc["td_inv"] = (self.cnst.t_rev + self.cnst.t_rchg)/2
        self.calc["n0_inv"] = (((self.data["EM58_f_inv"]*10**58*MU_e)/(16*self.get_dEMF(self.data["n_inv"], self.calc["td_inv"])*MU_H*((self.data["R_f_inv"]*PC_TO_KM*10**5)**3)))**(0.5))
        self.calc["Te_inv"] = self.data["Te_f_inv"] * KEV_TO_ERG / BOLTZMANN
        self.calc["Tes_inv"] = self.calc["Te_inv"] / self.get_dTF(self.data["n_inv"], self.calc["td_inv"])
        self.calc["Vfs1_inv"] = (((16*BOLTZMANN*3*(self.calc["Tes_inv"]))/(3*MU_t*M_H))**0.5) #cm/s
        self.calc["E51c_inv"] = 1.0
        self.calc["tch_inv"] = (((self.data["m_eject_inv"] * SOLAR_MASS_TO_GRAM) ** (5.0 / 6.0)) / ((self.calc["E51c_inv"] * 10 ** 51) ** 0.5) /
                                ((self.calc["n0_inv"] * MU_H * M_H) ** (1.0 / 3.0)) )/ YR_TO_SEC
        self.calc["ty_inv"] = ((0.4*self.data["R_f_inv"]*PC_TO_KM*10**5/(self.calc["Vfs1_inv"]))
                                - (self.calc["tch_inv"]*YR_TO_SEC*(((self.cnst.r_st**2.5)/(XI_0**0.5))-self.cnst.t_st)))/YR_TO_SEC #yrs
        self.calc["E51_inv"] = ((self.calc["n0_inv"]*((self.data["R_f_inv"]/(0.3163))**5))/((self.calc["ty_inv"])**2))/0.27
        self.calculate_VCRf(self.calc["n0_inv"], self.calc["ty_inv"], self.calc["E51_inv"], self.data["n_inv"])
        self.calc["ts_inv"] = (3.0/16.0)*MU_t*M_H/BOLTZMANN*((self.calc["VCRf_inv"]*10**5)**2) #K

        self.calc["lnDelf_inv"] = np.log(1.2*(10**5)*0.5*(self.calc["ts_inv"]**1.5)/(2*(self.calc["n0_inv"]**0.5)))
        self.calc["flf_inv"] = (5*self.calc["lnDelf_inv"]*4*self.calc["n0_inv"]*self.calc["ty_inv"]*YR_TO_SEC)/(3*81*((self.calc["ts_inv"])**1.5)*4)
        self.calc["geif_inv"] = 1 - 0.97*np.exp(-1.0*((self.calc["flf_inv"])**0.4)*(1+0.3*self.calc["flf_inv"]**0.6))
        self.calc["TES_inv"] = self.calc["Te_inv"] / self.get_dTF(self.data["n_inv"], (self.calc["ty_inv"]/self.calc["tch_inv"])) #K
        self.calc["TFS_inv"] = MU_t*self.calc["TES_inv"]*((1/(MU_I*self.calc["geif_inv"]))+(1/MU_e)) #K
        self.calc["VFS_inv"] = ((16*BOLTZMANN*self.calc["TFS_inv"])/(3*MU_t*M_H))**0.5 #cm/s

        self.calculate_FCRf(self.calc["n0_inv"], self.calc["ty_inv"], self.calc["E51_inv"], self.data["n_inv"])

        for x in range (0,11):
            self.calculate_FCRf(self.calc["n0_inv"], self.calc["tcn_inv"], self.calc["E51c_inv"], self.data["n_inv"])


#######################################################################################################################
    def calculate_FCRf(self, n0, ty, E51, n):
        self.calc["tch_inv"] = ((self.data["m_eject_inv"] * SOLAR_MASS_TO_GRAM) ** (5.0 / 6.0) / ((E51) * 10 ** 51) ** 0.5 /
                (n0 * MU_H * M_H) ** (1.0 / 3.0) )/ YR_TO_SEC # in years

        self.calculate_VFSf(n0, ty, E51, n)

        if (ty >= self.calc["tch_inv"]*self.cnst.t_st):
            self.calc["tcn_inv"] = ((0.4*self.data["R_f_inv"]*PC_TO_KM*10**5/self.calc["VFS_inv"])-(self.calc["tch_inv"]*YR_TO_SEC*((self.cnst.r_st**2.5/((XI_0)**0.5))-self.cnst.t_st)))/ YR_TO_SEC # in years
        else:
            self.calc["tcn_inv"] = (((self.data["n_inv"]-3)*self.data["R_f_inv"]*PC_TO_KM*10**5)/((self.data["n_inv"])*self.calc["VFS_inv"]))/ YR_TO_SEC #in years
        self.calc["td_inv"] = (self.calc["tcn_inv"]) / self.calc["tch_inv"]
        self.calc["n0_inv"] = (((self.data["EM58_f_inv"]*10**58*MU_e)/(16*self.get_dEMF(self.data["n_inv"], self.calc["td_inv"])*MU_H*((self.data["R_f_inv"]*PC_TO_KM*10**5)**3)))**(0.5))

        self.calculate_VFSf(self.calc["n0_inv"], self.calc["tcn_inv"], E51, n)

        if (self.calc["td_inv"] >= self.cnst.t_st):
            self.calc["E51c_inv"] = (25/(4*XI_0))*(MU_H*M_H*self.calc["n0_inv"])*(self.calc["VFS_inv"]**2)*((self.data["R_f_inv"]*PC_TO_KM*10**5)**3)/(10**51)
        else:
            self.calc["E51c_inv"] = ((((self.data["m_eject_inv"] * SOLAR_MASS_TO_GRAM)**(5.0/3.0))*((MU_H*M_H*self.calc["n0_inv"])**(-2.0/3.0))/((10**51)*((self.calc["tcn_inv"]* YR_TO_SEC)**2)))*(((self.data["R_f_inv"]*PC_TO_KM*10**5)/(self.calc["cx_inv"]*self.calc["rch_inv"] * PC_TO_KM * 10 ** 5))**((2*n)/(n-3))))


#######################################################################################################################
    def calculate_VFSf(self, n0, ty, E51, n):
        self.calculate_geif(n0, ty, E51, n)
        self.calc["TES_inv"] = self.calc["Te_inv"] / self.get_dTF(n, (ty/self.calc["tch_inv"])) #K
        self.calc["TFS_inv"] = MU_t*self.calc["TES_inv"]*((1/(MU_I*self.calc["geif_inv"]))+(1/MU_e)) #K
        self.calc["VFS_inv"] = ((16*BOLTZMANN*self.calc["TFS_inv"])/(3*MU_t*M_H))**0.5 #cm/s

#######################################################################################################################
    def calculate_geif(self, n0, ty, E51, n):
        self.calculate_VCRf(n0, ty, E51, n)
        self.calc["ts_inv"] = (3.0/16.0)*MU_t*M_H/BOLTZMANN*((self.calc["VCRf_inv"]*10**5)**2) #K
        self.calc["lnDelf_inv"] = np.log(1.2*(10**5)*0.5*(self.calc["ts_inv"]**1.5)/(2*(n0**0.5)))
        self.calc["flf_inv"] = (5*self.calc["lnDelf_inv"]*4*n0*ty*YR_TO_SEC)/(3*81*((self.calc["ts_inv"])**1.5)*4)
        self.calc["geif_inv"] = 1 - 0.97*np.exp(-1.0*((self.calc["flf_inv"])**0.4)*(1+0.3*self.calc["flf_inv"]**0.6))

#######################################################################################################################
    def calculate_VCRf(self, n0, ty, E51, n):
        self.calc["tch_inv"] = ((self.data["m_eject_inv"] * SOLAR_MASS_TO_GRAM) ** (5.0 / 6.0) / ((E51) * 10 ** 51) ** 0.5 /
                (n0 * MU_H * M_H) ** (1.0 / 3.0) )/ YR_TO_SEC      # in years
        self.calc["rch_inv"] = ((self.data["m_eject_inv"] * SOLAR_MASS_TO_GRAM / (n0 * MU_H * M_H)) ** (1.0 / 3.0)) / PC_TO_KM / 10 ** 5 # in parsecs
        self.calc["cx_inv"] = ((((27*self.cnst.l_ed**(n-2))/(4*np.pi*n*(n-3)*self.cnst.phi_ed))
                                *(((10*(n-5))/(3*(n-3)))**((n-3)/2.0)))**(1.0/n))
        self.calc["vch_inv"] = self.calc["rch_inv"] / self.calc["tch_inv"] * PCyr_TO_KMs #km/s
        self.calc["v0STf_inv"] = self.calc["vch_inv"] * ((n - 3)/n) * self.calc["cx_inv"] * ((ty/self.calc["tch_inv"])**(-3/n))
        self.calc["vTMf_inv"] = 0.4*((XI_0)**0.5)*self.calc["vch_inv"]*((((self.cnst.r_st)**(2.5))+(((XI_0)**0.5)*((ty/self.calc["tch_inv"])-self.cnst.t_st)))**(-0.6))
        if((ty/self.calc["tch_inv"])<(self.cnst.t_st)):
            self.calc["VCRf_inv"] = self.calc["v0STf_inv"]
        elif((ty/self.calc["tch_inv"])>=(self.cnst.t_st)):
            self.calc["VCRf_inv"] = self.calc["vTMf_inv"] #km/s

#######################################################################################################################
    def calculate_VCRr(self, n0, ty, E51, n):
        self.calc["tch_inv"] = ((self.data["m_eject_inv"] * SOLAR_MASS_TO_GRAM) ** (5.0 / 6.0) / ((E51) * 10 ** 51) ** 0.5 /
                (n0 * MU_H * M_H) ** (1.0 / 3.0) )/ YR_TO_SEC      # in years
        self.calc["rch_inv"] = ((self.data["m_eject_inv"] * SOLAR_MASS_TO_GRAM / (n0 * MU_H * M_H)) ** (1.0 / 3.0)) / PC_TO_KM / 10 ** 5 # in parsecs
        self.calc["cx_inv"] = ((((27*self.cnst.l_ed**(n-2))/(4*np.pi*n*(n-3)*self.cnst.phi_ed))
                                *(((10*(n-5))/(3*(n-3)))**((n-3)/2.0)))**(1.0/n))
        self.calc["vch_inv"] = self.calc["rch_inv"] / self.calc["tch_inv"] * PCyr_TO_KMs #km/s
        self.calc["v0STf_inv"] = self.calc["vch_inv"] * ((n - 3)/n) * self.calc["cx_inv"] * ((ty/self.calc["tch_inv"])**(-3/n))
        self.calc["v0CR_inv"] = self.calc["v0STf_inv"]*(3/((n-3)*self.cnst.l_ed))
        self.calc["vTr_inv"] = self.calc["vch_inv"] * (self.cnst.v_rchg+self.cnst.a_rchg*((ty/self.calc["tch_inv"])-(self.cnst.t_rchg)))
        if((ty/self.calc["tch_inv"])<(self.cnst.t_rchg)):
            self.calc["VCRr_inv"] = self.calc["v0CR_inv"]
        elif((ty/self.calc["tch_inv"])>=(self.cnst.t_rchg)):
            self.calc["VCRr_inv"] = self.calc["vTr_inv"] #km

#######################################################################################################################
    def calculate_RCRf(self, n0, ty, E51, n):
        self.calc["tch_inv"] = ((self.data["m_eject_inv"] * SOLAR_MASS_TO_GRAM) ** (5.0 / 6.0) / ((E51) * 10 ** 51) ** 0.5 /
                (n0 * MU_H * M_H) ** (1.0 / 3.0) )/ YR_TO_SEC      # in years
        self.calc["rch_inv"] = ((self.data["m_eject_inv"] * SOLAR_MASS_TO_GRAM / (n0 * MU_H * M_H)) ** (1.0 / 3.0)) # in kms
        self.calc["cx_inv"] = ((((27*self.cnst.l_ed**(n-2))/(4*np.pi*n*(n-3)*self.cnst.phi_ed))
                                *(((10*(n-5))/(3*(n-3)))**((n-3)/2.0)))**(1.0/n))
        self.calc["r0STf_inv"] = self.calc["rch_inv"] * self.calc["cx_inv"] * ((ty/self.calc["tch_inv"])**((n-3)/n))
        self.calc["rTMf_inv"] = self.calc["rch_inv"]*((((self.cnst.r_st)**(2.5))+(((XI_0)**0.5)*((ty/self.calc["tch_inv"])-self.cnst.t_st)))**(0.4))
        if((ty/self.calc["tch_inv"])<(self.cnst.t_st)):
            self.calc["RCRf_inv"] = self.calc["r0STf_inv"]
        elif((ty/self.calc["tch_inv"])>=(self.cnst.t_st)):
            self.calc["RCRf_inv"] = self.calc["rTMf_inv"] #km

#######################################################################################################################
    def calculate_RCRr(self, n0, ty, E51, n):
        self.calc["tch_inv"] = ((self.data["m_eject_inv"] * SOLAR_MASS_TO_GRAM) ** (5.0 / 6.0) / ((E51) * 10 ** 51) ** 0.5 /
                (n0 * MU_H * M_H) ** (1.0 / 3.0) )/ YR_TO_SEC      # in years
        self.calc["rch_inv"] = ((self.data["m_eject_inv"] * SOLAR_MASS_TO_GRAM / (n0 * MU_H * M_H)) ** (1.0 / 3.0)) / PC_TO_KM / 10 ** 5 # in parsecs
        self.calc["cx_inv"] = ((((27*self.cnst.l_ed**(n-2))/(4*np.pi*n*(n-3)*self.cnst.phi_ed))
                                *(((10*(n-5))/(3*(n-3)))**((n-3)/2.0)))**(1.0/n))
        self.calc["vch_inv"] = self.calc["rch_inv"] / self.calc["tch_inv"] * PCyr_TO_KMs #km/s
        self.calc["r0STf_inv"] = self.calc["rch_inv"] * PC_TO_KM * self.calc["cx_inv"] * ((ty/self.calc["tch_inv"])**((n-3)/n)) #km
        self.calc["r0Cr_inv"] = self.calc["r0STf_inv"] / self.cnst.l_ed
        self.calc["rTr_inv"] = self.calc["vch_inv"] * ty * YR_TO_SEC * ((self.cnst.r_rchg/self.cnst.t_rchg)-(self.cnst.a_rchg*((ty/self.calc["tch_inv"])-(self.cnst.t_rchg)))-((self.cnst.v_rchg-self.cnst.a_rchg*self.cnst.t_rchg)*np.log(ty/(self.calc["tch_inv"]*self.cnst.t_rchg))))
        if((ty/self.calc["tch_inv"])<(self.cnst.t_rchg)):
            self.calc["RCRr_inv"] = self.calc["r0Cr_inv"]
        elif((ty/self.calc["tch_inv"])>=(self.cnst.t_rchg)):
            self.calc["RCRr_inv"] = self.calc["rTr_inv"] # in kms

#######################################################################################################################
    def verify_forward_generalT_n6to14(self):
        self.calc["td_inv"] = (self.calc["tcn_inv"]) / self.calc["tch_inv"]
        self.calculate_RCRf(self.calc["n0_inv"], self.calc["tcn_inv"], self.calc["E51c_inv"], self.data["n_inv"])
        self.calc["Rs_ver"] = self.calc["RCRf_inv"]
        self.calc["EM_ver"]=16*((self.calc["n0_inv"])**2)*(MU_H/MU_e)*self.get_dEMF(self.data["n_inv"],self.calc["td_inv"])*((self.calc["Rs_ver"])**3)

        self.calculate_VCRf(self.calc["n0_inv"], self.calc["tcn_inv"], self.calc["E51c_inv"], self.data["n_inv"])
        self.calc["ts_inv"] = (3.0/16.0)*MU_t*M_H/BOLTZMANN*((self.calc["VCRf_inv"]*10**5)**2) #K
        self.calc["lnDelf_inv"] = np.log(1.2*(10**5)*0.5*(self.calc["ts_inv"]**1.5)/(2*(self.calc["n0_inv"]**0.5)))
        self.calc["flf_inv"] = (5*self.calc["lnDelf_inv"]*4*self.calc["n0_inv"]*self.calc["tcn_inv"]*YR_TO_SEC)/(3*81*((self.calc["ts_inv"])**1.5)*4)
        self.calc["geif_inv"] = 1 - 0.97*np.exp(-1.0*((self.calc["flf_inv"])**0.4)*(1+0.3*self.calc["flf_inv"]**0.6))
        self.calc["TeS_ver"] = self.calc["ts_inv"]/MU_t/((1/(MU_I*self.calc["geif_inv"]))+(1/MU_e)) #K
        self.calc["TE_ver"] = self.calc["TeS_ver"]*self.get_dTF(self.data["n_inv"], self.calc["td_inv"])

        self.calc["EMR_predict"] = MU_ratej*16*((self.calc["n0_inv"])**2)*(MU_H/MU_e)*self.get_dEMR(self.data["n_inv"],self.calc["td_inv"])*((self.calc["Rs_ver"])**3)
        self.calc["Trav_predict"] = (MU_tej/MU_t)*self.calc["ts_inv"]*self.get_dTR(self.data["n_inv"], self.calc["td_inv"])
        self.calc["Terav_predict"] = self.calc["Trav_predict"]/MU_tej/((1/(MU_Iej*self.calc["geif_inv"]))+(1/MU_eej))
        self.calc["Rrev_predict"] = 0
        if((self.calc["tcn_inv"])<(self.calc["tch_inv"]*self.cnst.t_rev)):
            self.calculate_RCRr(self.calc["n0_inv"], self.calc["tcn_inv"], self.calc["E51c_inv"], self.data["n_inv"])
            self.calc["Rrev_predict"] = self.calc["RCRr_inv"]

        self.calc["Rs_ver"] = self.calc["Rs_ver"]/PC_TO_KM/10**5
        self.calc["EM_ver"] = self.calc["EM_ver"] /10**58
        self.calc["TE_ver"] = self.calc["TE_ver"]/KEV_TO_ERG*BOLTZMANN

        self.calc["Terav_predict"] = self.calc["Terav_predict"]/KEV_TO_ERG*BOLTZMANN
        self.calc["EMR_predict"] = self.calc["EMR_predict"]/10**58
        self.calc["Rrev_predict"] = self.calc["Rrev_predict"]/PC_TO_KM

#######################################################################################################################
#######################################################################################################################
    def calculate_Standard_Reverse_values(self):

        self.calc["td_inv"] = (self.cnst.t_rev + self.cnst.t_rchg)/2
        self.calc["n0_inv"] = (((self.data["EM58_f_inv"]*10**58*MU_e)/(MU_ratej*16*self.get_dEMR(self.data["n_inv"], self.calc["td_inv"])*MU_H*((self.data["R_f_inv"]*PC_TO_KM*10**5)**3)))**(0.5))
        self.calc["E51c_inv"] = 1.0
        self.calc["tch_inv"] = (((self.data["m_eject_inv"] * SOLAR_MASS_TO_GRAM) ** (5.0 / 6.0)) / ((self.calc["E51c_inv"] * 10 ** 51) ** 0.5) /
                                ((self.calc["n0_inv"] * MU_H * M_H) ** (1.0 / 3.0)) )/ YR_TO_SEC
        self.calc["Te_inv"] = self.data["Te_f_inv"] * KEV_TO_ERG / BOLTZMANN
        self.calculate_geif(self.calc["n0_inv"], self.calc["tch_inv"], self.calc["E51c_inv"], self.data["n_inv"])

        self.calc["Tr_inv"] = self.calc["Te_inv"]*MU_tej*((1/(MU_Iej*self.calc["geif_inv"]))+(1/MU_eej))
        self.calc["Tfs1_inv"] = (self.calc["Tr_inv"]*MU_t) / (self.get_dTR(self.data["n_inv"], self.calc["td_inv"])*MU_tej)
        self.calc["Vfs1_inv"] = (((16*BOLTZMANN*(self.calc["Tfs1_inv"]))/(3*MU_t*M_H))**0.5) #cm/s

        self.calc["ts_inv"] = ((0.4*self.data["R_f_inv"]*PC_TO_KM*10**5/(self.calc["Vfs1_inv"]))
                                - (self.calc["tch_inv"]*YR_TO_SEC*(((self.cnst.r_st**2.5)/(XI_0**0.5))-self.cnst.t_st)))/YR_TO_SEC #yrs
        self.calc["ty_inv"] = self.calc["ts_inv"]
        self.calc["E51_inv"] = ((self.calc["n0_inv"]*((self.data["R_f_inv"]/(0.3163))**5))/((self.calc["ty_inv"])**2))/0.27 ##WHATS UP WITH THE UNITS HERE?!?!

        self.calc["tch_inv"] = ((self.data["m_eject_inv"] * SOLAR_MASS_TO_GRAM) ** (5.0 / 6.0) / ((self.calc["E51_inv"]) * 10 ** 51) ** 0.5 /
                (self.calc["n0_inv"] * MU_H * M_H) ** (1.0 / 3.0) )/ YR_TO_SEC      # in years
        self.calculate_VFSr(self.calc["n0_inv"], self.calc["ty_inv"], self.calc["E51_inv"], self.data["n_inv"])
        self.calculate_FCRr(self.calc["n0_inv"], self.calc["ty_inv"], self.calc["E51_inv"], self.data["n_inv"])

        for x in range (0,39):
            self.calculate_FCRr(self.calc["n0_inv"], self.calc["tcn_inv"], self.calc["E51c_inv"], self.data["n_inv"])
#######################################################################################################################
    def calculate_VFSr(self, n0, ty, E51, n):
        self.calculate_geif(n0, ty, E51, n)
        self.calc["Tr_inv"] = self.calc["Te_inv"]*MU_tej*((1/(MU_Iej*self.calc["geif_inv"]))+(1/MU_eej))
        self.calc["TFS_inv"] = (self.calc["Tr_inv"]*MU_t) / (self.get_dTR(self.data["n_inv"], self.calc["td_inv"])*MU_tej)
        self.calc["VFS_inv"] = ((16*BOLTZMANN*self.calc["TFS_inv"])/(3*MU_t*M_H))**0.5 #cm/s

#######################################################################################################################
    def calculate_FCRr(self, n0, ty, E51, n):
        self.calc["tch_inv"] = ((self.data["m_eject_inv"] * SOLAR_MASS_TO_GRAM) ** (5.0 / 6.0) / ((E51) * 10 ** 51) ** 0.5 /
                (n0 * MU_H * M_H) ** (1.0 / 3.0) )/ YR_TO_SEC # in years
        #print(self.calc["tch_inv"])

        self.calculate_VFSr(n0, ty, E51, n)

        if (ty >= self.calc["tch_inv"]*self.cnst.t_st):
            self.calc["tcn_inv"] = ((0.4*self.data["R_f_inv"]*PC_TO_KM*10**5/self.calc["VFS_inv"])-(self.calc["tch_inv"]*YR_TO_SEC*((self.cnst.r_st**2.5/((XI_0)**0.5))-self.cnst.t_st)))/ YR_TO_SEC # in years
        else:
            self.calc["tcn_inv"] = (((self.data["n_inv"]-3)*self.data["R_f_inv"]*PC_TO_KM*10**5)/((self.data["n_inv"])*self.calc["VFS_inv"]))/ YR_TO_SEC #in years
        self.calc["td_inv"] = (self.calc["tcn_inv"]) / self.calc["tch_inv"]
        self.calc["n0_inv"] = (((self.data["EM58_f_inv"]*10**58*MU_e)/(MU_ratej*16*self.get_dEMR(self.data["n_inv"], self.calc["td_inv"])*MU_H*((self.data["R_f_inv"]*PC_TO_KM*10**5)**3)))**(0.5))

        self.calculate_VFSr(self.calc["n0_inv"], self.calc["tcn_inv"], E51, n)

        if (self.calc["td_inv"] >= self.cnst.t_st):
            self.calc["E51c_inv"] = (25/(4*XI_0))*(MU_H*M_H*self.calc["n0_inv"])*(self.calc["VFS_inv"]**2)*((self.data["R_f_inv"]*PC_TO_KM*10**5)**3)/(10**51)
        else:
            self.calc["E51c_inv"] = ((((self.data["m_eject_inv"] * SOLAR_MASS_TO_GRAM)**(5.0/3.0))*((MU_H*M_H*self.calc["n0_inv"])**(-2.0/3.0))/((10**51)*((self.calc["tcn_inv"]* YR_TO_SEC)**2)))*(((self.data["R_f_inv"]*PC_TO_KM*10**5)/(self.calc["cx_inv"]*self.calc["rch_inv"] * PC_TO_KM * 10 ** 5))**((2*n)/(n-3))))

#######################################################################################################################
    def verify_reverse_generalT_n6to14(self):
        self.calc["td_inv"] = (self.calc["tcn_inv"]) / self.calc["tch_inv"]
        self.calculate_RCRf(self.calc["n0_inv"], self.calc["tcn_inv"], self.calc["E51c_inv"], self.data["n_inv"])
        self.calc["Rs_ver"] = self.calc["RCRf_inv"]
        self.calc["EM_ver"]=MU_ratej*16*((self.calc["n0_inv"])**2)*(MU_H/MU_e)*self.get_dEMR(self.data["n_inv"],self.calc["td_inv"])*((self.calc["Rs_ver"])**3)

        self.calculate_VCRf(self.calc["n0_inv"], self.calc["tcn_inv"], self.calc["E51c_inv"], self.data["n_inv"])
        self.calc["ts_inv"] = (3.0/16.0)*MU_t*M_H/BOLTZMANN*((self.calc["VCRf_inv"]*10**5)**2) #K
        self.calc["lnDelf_inv"] = np.log(1.2*(10**5)*0.5*(self.calc["ts_inv"]**1.5)/(2*(self.calc["n0_inv"]**0.5)))
        self.calc["flf_inv"] = (5*self.calc["lnDelf_inv"]*4*self.calc["n0_inv"]*self.calc["tcn_inv"]*YR_TO_SEC)/(3*81*((self.calc["ts_inv"])**1.5)*4)
        self.calc["geif_inv"] = 1 - 0.97*np.exp(-1.0*((self.calc["flf_inv"])**0.4)*(1+0.3*self.calc["flf_inv"]**0.6))

        self.calc["Tfs_inv"] = (self.calc["Tr_inv"]*MU_t) / (self.get_dTR(self.data["n_inv"], self.calc["td_inv"])*MU_tej)
        self.calc["TR_inv"] = (MU_tej/MU_t) * self.calc["Tfs_inv"] * self.get_dTR(self.data["n_inv"], self.calc["td_inv"])
        self.calc["TeR_inv"] = (self.calc["TR_inv"]/MU_tej)/((1/(MU_Iej*self.calc["geif_inv"]))+(1/MU_eej))
        self.calc["TefS_inv"] = (self.calc["Tfs_inv"]/MU_t)/((1/(MU_I*self.calc["geif_inv"]))+(1/MU_e))

        self.calc["TfSav_predict"] = self.calc["Tfs_inv"] *self.get_dTF(self.data["n_inv"], self.calc["td_inv"])
        self.calc["EMF_predict"] = 16*((self.calc["n0_inv"])**2)*(MU_H/MU_e)*self.get_dEMF(self.data["n_inv"],self.calc["td_inv"])*((self.calc["Rs_ver"])**3)
        self.calc["TEf_predict"] = self.calc["TefS_inv"] * self.get_dTF(self.data["n_inv"], self.calc["td_inv"])
        self.calc["Rrev_predict"] = 0
        if((self.calc["tcn_inv"])<(self.calc["tch_inv"]*self.cnst.t_rev)):
            self.calculate_RCRr(self.calc["n0_inv"], self.calc["tcn_inv"], self.calc["E51c_inv"], self.data["n_inv"])
            self.calc["Rrev_predict"] = self.calc["RCRr_inv"]

        self.calc["Rs_ver"] = self.calc["Rs_ver"]/PC_TO_KM/10**5
        self.calc["EM_ver"] = self.calc["EM_ver"] /10**58
        self.calc["TE_ver"] = self.calc["TeR_inv"]/KEV_TO_ERG*BOLTZMANN

        self.calc["TEf_predict"] = self.calc["TEf_predict"]/KEV_TO_ERG*BOLTZMANN
        self.calc["EMF_predict"] = self.calc["EMF_predict"]/10**58
        self.calc["Rrev_predict"] = self.calc["Rrev_predict"]/PC_TO_KM

#######################################################################################################################
#######################################################################################################################
    def calculate_S2_Standard_Forward_values(self):
        self.calc["Te_inv"] = self.data["Te_f_inv"] * KEV_TO_ERG / BOLTZMANN
        self.calc["Rs_inv"] = self.data["R_f_inv"]*PC_TO_KM*10**5
        self.calc["Mej_inv"] = self.data["m_eject_inv"] * SOLAR_MASS_TO_GRAM
        self.calc["EM_inv"] = self.data["EM58_f_inv"]*10**58
        self.calc["vw_inv"] = 30
        self.calc["kmcm"] = 10**5
        self.calc["t_inv"] = 300*YR_TO_SEC
        self.calc["E51_inv"] = 1.0

        self.calc["q_inv"] = ((self.calc["Rs_inv"]*self.calc["EM_inv"]*MU_e*MU_H*(M_H**2))/(16*self.cnst.dEMf2))**0.5
        self.calculate_Vfs2(self.calc["t_inv"], self.calc["E51_inv"], self.data["n_inv"])

        self.calc["CC1_inv"] = ((1 - (3/self.data["n_inv"]))*self.calc["Mej_inv"])/((4/3)*np.pi*((self.calc["q_inv"]/self.cnst.a2)*((self.calc["Rs_inv"]/self.cnst.bbm)**(self.data["n_inv"]-2))))
        self.calc["CC2_inv"] = (3*(self.data["n_inv"]-3)*((self.calc["CC1_inv"])**(2/(3-self.data["n_inv"]))))/((10**18)*10*(self.data["n_inv"]-5))

        self.calc["t_inv"] = ((self.data["n_inv"]-3)*self.calc["Rs_inv"])/((self.data["n_inv"]-2)*self.calc["Vfs_inv"])
        self.calc["E51_inv"] = (self.calc["Mej_inv"]*self.calc["CC2_inv"]*(self.calc["t_inv"]**(-2)))/(10**33)

        for x in range (0,10):
            self.calculate_FS2_inv(self.calc["t_inv"], self.calc["E51_inv"], self.data["n_inv"])

#######################################################################################################################
    def calculate_FS2_inv(self, ty, E51, n):
        self.calculate_Vfs2(ty, E51, n)
        self.calc["t_inv"] = ((n-3)/(n-2))*(self.calc["Rs_inv"]/self.calc["Vfs_inv"])
        self.calc["E51_inv"] = self.calc["Mej_inv"]*self.calc["CC2_inv"]*(self.calc["t_inv"]**(-2))/(10**33)

#######################################################################################################################
    def calculate_Vfs2(self, ty, E51, n):

        #self.calc["Rch2_inv"] = (0.1*12.9*(PC_TO_KM*10**5)*(self.data["m_eject_inv"] * SOLAR_MASS_TO_GRAM))/(4*np.pi*(10**5)*YR_TO_SEC*self.calc["q_inv"]*self.calc["kmcm"]) #cm
        #self.calc["tch2_inv"] = (0.1*1770/(10**5))*((E51)**(-0.5))*(((self.data["m_eject_inv"] * SOLAR_MASS_TO_GRAM)**1.5)/(SOLAR_MASS_TO_GRAM**0.5))*(1/(4*np.pi*self.calc["q_inv"]*self.calc["kmcm"])) #s
        #self.calc["vcor2_inv"] = ((10*(self.data["n_inv"]-5))/(3*(self.data["n_inv"]-3)))**(0.5)
        #self.calc["tcor2_inv"] = (3/(self.cnst.phi_ed*4*np.pi*(self.data["n_inv"]-3)*self.data["n_inv"]))*(((3*(self.data["n_inv"]-3))/(10*(self.data["n_inv"]-5)))**(0.5))
        #self.calc["tcon_inv"] = ((((self.data["n_inv"]-3)/(self.data["n_inv"]-2))*((2*np.pi*0.75)**(0.5)))**((2*(self.data["n_inv"]-2))/(5-self.data["n_inv"])))*((self.calc["vcor2_inv"]*self.cnst.l_ed)**((3*(self.data["n_inv"]-2))/(5-self.data["n_inv"])))*((self.calc["tcor2_inv"])**(3/(5-self.data["n_inv"])))

        #self.calc["Vb2t_inv"] = 0
        #if(ty < (self.calc["tch2_inv"] * self.calc["tcon_inv"])):
        #    self.calc["Rbn2_inv"] = self.calc["Rch2_inv"]*(((3*((self.cnst.l_ed)**(self.data["n_inv"]-2))*((self.calc["vcor2_inv"])**(self.data["n_inv"]-3)))/(self.cnst.phi_ed*4*np.pi*self.data["n_inv"]*(self.data["n_inv"]-3)))**(1/(self.data["n_inv"]-2)))*((ty/self.calc["tch2_inv"])**((self.data["n_inv"]-3)/(self.data["n_inv"]-2)))
        #    self.calc["Vbn2_inv"] = ((self.data["n_inv"]-3)*self.calc["Rbn2_inv"])/((self.data["n_inv"]-2)*ty)
        #    self.calc["Vb2t_inv"] = self.calc["Vbn2_inv"]
        #else:
        #    self.calc["Rb2a_inv"] = self.calc["Rch2_inv"]*((((((3*((self.cnst.l_ed)**(self.data["n_inv"]-2)))/(self.cnst.phi_ed*4*np.pi*self.data["n_inv"]*(self.data["n_inv"]-3)))*((self.calc["tcon_inv"]*self.calc["vcor2_inv"])**(self.data["n_inv"]-3)))**(1.5/(self.data["n_inv"]-2)))+(((1.5/np.pi)**(0.5))*((ty/self.calc["tch2_inv"])-self.calc["tcon_inv"])))**(2/3))
        #    self.calc["Vb2a_inv"] = ((self.data["n_inv"]-3)*self.calc["Rb2a_inv"])/((self.data["n_inv"]-2)*ty)
        #    self.calc["Vb2t_inv"] = self.calc["Vb2a_inv"]

        #self.calc["Ts2_inv"] = (3/16)*MU_t*(M_H/BOLTZMANN)*(self.calc["Vb2t_inv"]**2)

        self.calc["gc_inv"] = ((1-(3/n))*self.calc["Mej_inv"])/(((4*np.pi)/3)*(((E51*(10**51)*10*(n-5))/(self.calc["Mej_inv"]*3*(n-3)))**((3-n)/2)))
        self.calc["RC_inv"] = (self.cnst.a2*self.calc["gc_inv"]/self.calc["q_inv"])**(1/(n-2))
        self.calc["Vf2_inv"] = ((n-3)/(n-2))*self.calc["RC_inv"]*self.cnst.bbm*((self.calc["t_inv"])**(-1/(n-2)))
        self.calc["Ts2_inv"] = (3/16)*MU_t*(M_H/BOLTZMANN)*(self.calc["Vf2_inv"]**2)
        self.calc["n2_inv"] = (8*self.calc["q_inv"])/(MU_e*M_H*(self.calc["Rs_inv"]**2))
        self.calc["lnLambda2_inv"] = np.log(1.2*(10**5)*0.5*(self.calc["Ts2_inv"]**1.5)/(2*(self.calc["n2_inv"]**0.5)))
        self.calc["fl2_inv"] = (5*self.calc["lnLambda2_inv"]*4*self.calc["n2_inv"]*self.calc["t_inv"])/(3*81*((self.calc["Ts2_inv"])**1.5)*4)
        self.calc["gei2_inv"] = 1 - 0.97*np.exp(-1.0*((self.calc["fl2_inv"])**0.4)*(1+0.3*self.calc["fl2_inv"]**0.6))
        if(self.data["model_inv"] == "standard_forward"):
            self.calc["Tfs_inv"] = ((self.calc["Te_inv"]/self.cnst.dTf2)*MU_t)*((1/(MU_I*self.calc["gei2_inv"]))+(1/MU_e))
        elif(self.data["model_inv"] == "standard_reverse"):
            self.calc["Tfs_inv"] = ((self.calc["Te_inv"]/self.cnst.dTr2)*MU_t)*((1/(MU_Iej*self.calc["gei2_inv"]))+(1/MU_eej))
        self.calc["Vfs_inv"] = ((16*BOLTZMANN*self.calc["Tfs_inv"])/(3*MU_t*M_H))**0.5

#######################################################################################################################
    def verify_S2_forward_generalT_n6to14(self):

        self.calc["gc_inv"] = ((1-(3/self.data["n_inv"]))*self.calc["Mej_inv"])/(((4*np.pi)/3)*(((self.calc["E51_inv"]*(10**51)*10*(self.data["n_inv"]-5))/(self.calc["Mej_inv"]*3*(self.data["n_inv"]-3)))**((3-self.data["n_inv"])/2)))
        self.calc["RC_inv"] = (self.cnst.a2*self.calc["gc_inv"]/self.calc["q_inv"])**(1/(self.data["n_inv"]-2))
        self.calc["Rf2_inv"] = self.calc["RC_inv"] * self.cnst.bbm * ((self.calc["t_inv"])**((self.data["n_inv"]-3)/(self.data["n_inv"]-2)))
        self.calc["Rs_ver"] = self.calc["Rf2_inv"]
        self.calc["EM_ver"] = (self.cnst.dEMf2*16*(self.calc["q_inv"]**2))/(MU_e*MU_H*M_H*M_H*self.calc["Rs_ver"])
        self.calc["TE_ver"] = (self.calc["Ts2_inv"]*self.cnst.dTf2/MU_t)*(((1/(MU_I*self.calc["gei2_inv"]))+(1/MU_e))**(-1))
        self.calc["EMR_predict"] = MU_ratej*16*(((self.calc["q_inv"])**2)*self.cnst.dEMr2)/(self.calc["Rs_ver"]*MU_e*MU_H*M_H*M_H)
        self.calc["Trav_predict"] = (MU_tej/MU_t)*self.calc["Ts2_inv"]*self.cnst.dTr2
        self.calc["Terav_predict"] = self.calc["Trav_predict"]/MU_tej/((1/(MU_Iej*self.calc["gei2_inv"]))+(1/MU_eej))
        self.calc["Rrev_predict"] = self.calc["Rf2_inv"]/self.cnst.l_ed

        self.calc["Rs_ver"] = self.calc["Rs_ver"]/PC_TO_KM/10**5
        self.calc["EM_ver"] = self.calc["EM_ver"] /10**58
        self.calc["TE_ver"] = self.calc["TE_ver"]/KEV_TO_ERG*BOLTZMANN

        self.calc["Terav_predict"] = self.calc["Terav_predict"]/KEV_TO_ERG*BOLTZMANN
        self.calc["EMR_predict"] = self.calc["EMR_predict"]/10**58
        self.calc["Rrev_predict"] = self.calc["Rrev_predict"]/PC_TO_KM/10**5

        self.calc["t_inv"] = self.calc["t_inv"]/YR_TO_SEC

#######################################################################################################################
#######################################################################################################################
    def calculate_S2_Standard_Reverse_values(self):
        self.calc["Te_inv"] = self.data["Te_f_inv"] * KEV_TO_ERG / BOLTZMANN
        self.calc["Rs_inv"] = self.data["R_f_inv"]*PC_TO_KM*10**5
        self.calc["Mej_inv"] = self.data["m_eject_inv"] * SOLAR_MASS_TO_GRAM
        self.calc["EM_inv"] = self.data["EM58_f_inv"]*10**58
        self.calc["vw_inv"] = 30
        self.calc["kmcm"] = 10**5
        self.calc["t_inv"] = 300*YR_TO_SEC
        self.calc["E51_inv"] = 1.0
        #self.calc["EM_ver"]=MU_ratej*16*((self.calc["n0_inv"])**2)*(MU_H/MU_e)*self.get_dEMR(self.data["n_inv"],self.calc["td_inv"])*((self.calc["Rs_inv"])**3)
        ##Supposed to be EMr below here, but no EMr known.
        self.calc["q_inv"] = ((self.calc["Rs_inv"]*self.calc["EM_inv"]*MU_e*MU_H*(M_H**2))/(16*MU_ratej*self.cnst.dEMr2))**0.5
        #self.calc["q_inv"] = 1.32964 * 10**13
        self.calculate_Vfs2(self.calc["t_inv"], self.calc["E51_inv"], self.data["n_inv"])

        self.calc["CC1_inv"] = ((1 - (3/self.data["n_inv"]))*self.calc["Mej_inv"])/((4/3)*np.pi*((self.calc["q_inv"]/self.cnst.a2)*((self.calc["Rs_inv"]/self.cnst.bbm)**(self.data["n_inv"]-2))))
        self.calc["CC2_inv"] = (3*(self.data["n_inv"]-3)*((self.calc["CC1_inv"])**(2/(3-self.data["n_inv"]))))/((10**18)*10*(self.data["n_inv"]-5))

        self.calc["t_inv"] = ((self.data["n_inv"]-3)*self.calc["Rs_inv"])/((self.data["n_inv"]-2)*self.calc["Vfs_inv"])
        self.calc["E51_inv"] = (self.calc["Mej_inv"]*self.calc["CC2_inv"]*(self.calc["t_inv"]**(-2)))/(10**33)

        self.calculate_FS2_inv(self.calc["t_inv"], self.calc["E51_inv"], self.data["n_inv"])

        for x in range (0,9):
            self.calculate_FS2_inv(self.calc["t_inv"], self.calc["E51_inv"], self.data["n_inv"])

#######################################################################################################################
    def verify_S2_reverse_generalT_n6to14(self):

        self.calc["gc_inv"] = ((1-(3/self.data["n_inv"]))*self.calc["Mej_inv"])/(((4*np.pi)/3)*(((self.calc["E51_inv"]*(10**51)*10*(self.data["n_inv"]-5))/(self.calc["Mej_inv"]*3*(self.data["n_inv"]-3)))**((3-self.data["n_inv"])/2)))
        self.calc["RC_inv"] = (self.cnst.a2*self.calc["gc_inv"]/self.calc["q_inv"])**(1/(self.data["n_inv"]-2))
        self.calc["Rf2_inv"] = self.calc["RC_inv"] * self.cnst.bbm * ((self.calc["t_inv"])**((self.data["n_inv"]-3)/(self.data["n_inv"]-2)))
        self.calc["Rs_ver"] = self.calc["Rf2_inv"]
        self.calc["EM_ver"] = MU_ratej*16*(((self.calc["q_inv"])**2)*self.cnst.dEMr2)/(self.calc["Rs_ver"]*MU_e*MU_H*M_H*M_H)
        self.calc["TE_ver"] = (self.calc["Ts2_inv"]*self.cnst.dTr2/MU_t)*(((1/(MU_Iej*self.calc["gei2_inv"]))+(1/MU_eej))**(-1))

        self.calc["EMF_predict"] = (self.cnst.dEMf2*16*(self.calc["q_inv"]**2))/(MU_e*MU_H*M_H*M_H*self.calc["Rs_ver"])
        self.calc["Terav_predict"] = (self.cnst.dTf2/MU_t)*self.calc["Ts2_inv"]*(((1/(MU_I*self.calc["gei2_inv"]))+(1/MU_e))**(-1))
        self.calc["Rrev_predict"] = self.calc["Rf2_inv"]/self.cnst.l_ed

        self.calc["Rs_ver"] = self.calc["Rs_ver"]/PC_TO_KM/10**5
        self.calc["EM_ver"] = self.calc["EM_ver"] /10**58
        self.calc["TE_ver"] = self.calc["TE_ver"]/KEV_TO_ERG*BOLTZMANN

        self.calc["Terav_predict"] = self.calc["Terav_predict"]/KEV_TO_ERG*BOLTZMANN
        self.calc["EMF_predict"] = self.calc["EMF_predict"]/10**58
        self.calc["Rrev_predict"] = self.calc["Rrev_predict"]/PC_TO_KM/10**5

        self.calc["t_inv"] = self.calc["t_inv"]/YR_TO_SEC


#######################################################################################################################
#######################################################################################################################
    def calculate_Cloudy_Forward_values(self):
        self.calc["Te_inv"] = self.data["Te_f_inv"] * KEV_TO_ERG / BOLTZMANN
        self.calc["Rs_inv"] = self.data["R_f_inv"]*PC_TO_KM*10**5
        self.calc["Mej_inv"] = self.data["m_eject_inv"] * SOLAR_MASS_TO_GRAM
        self.calc["EM_inv"] = self.data["EM58_f_inv"]*10**58
        self.calc["dEMw_inv"] = self.cloudyCnst.dEMc
        self.calc["dTw_inv"] = self.cloudyCnst.dTc
        self.calc["Kw_inv"] = self.cloudyCnst.Kc
        self.calc["E51_inv"] = 1.0

        self.calc["n0_inv"] = (((self.calc["EM_inv"]*MU_e)/(16*self.calc["dEMw_inv"]*MU_H*((self.calc["Rs_inv"])**3)))**(0.5))
        self.calc["Vfs1_inv"] = ((16*BOLTZMANN*self.calc["Te_inv"])/(3*MU_t*M_H*self.calc["dTw_inv"]*0.5))**0.5

        self.calc["t_inv"] = (self.calc["Rs_inv"]*1.2373*(10**10))/(self.calc["Vfs1_inv"]*0.3163*PC_TO_KM*(10**5)) # in years
        self.calc["E51_inv"] = (self.calc["n0_inv"]*((self.calc["Rs_inv"]/(0.3163*PC_TO_KM*10**5))**5))/(self.calc["Kw_inv"]*(self.calc["t_inv"]**2))
        self.calculate_nuSed_inv(self.calc["t_inv"], self.calc["E51_inv"])

        for x in range (0,13):
            self.calculate_nuSed_inv(self.calc["t_inv"], self.calc["E51_inv"])

#######################################################################################################################
    def calculate_Vfs_cloudy(self, ty, E51):

        self.calc["VSW_inv"] = 1.2373*(10**10)*((self.calc["E51_inv"]*self.calc["Kw_inv"]/self.calc["n0_inv"])**0.2)*((self.calc["t_inv"])**(-0.6))
        self.calc["Ts_inv"] = (3/16)*MU_t*(M_H/BOLTZMANN)*(self.calc["VSW_inv"]**2)
        self.calc["lnLambda2_inv"] = np.log(1.2*(10**5)*0.5*(self.calc["Ts_inv"]**1.5)/(2*(self.calc["n0_inv"]**0.5)))
        self.calc["fl2_inv"] = (5*self.calc["lnLambda2_inv"]*4*self.calc["n0_inv"]*self.calc["t_inv"]*YR_TO_SEC)/(3*81*((self.calc["Ts_inv"])**1.5)*4)
        self.calc["geif_inv"] = 1 - 0.97*np.exp(-1.0*((self.calc["fl2_inv"])**0.4)*(1+0.3*self.calc["fl2_inv"]**0.6))
        self.calc["Tfs_inv"] = ((self.calc["Te_inv"]/self.calc["dTw_inv"])*MU_t)*((1/(MU_I*self.calc["geif_inv"]))+(1/MU_e))
        self.calc["Vfs_cloudy_inv"] = ((16*BOLTZMANN*self.calc["Tfs_inv"])/(3*MU_t*M_H))**0.5

#######################################################################################################################
    def calculate_nuSed_inv(self, ty, E51):
        self.calculate_Vfs_cloudy(ty, E51)
        self.calc["t_inv"] = (self.calc["Rs_inv"]*1.2373*(10**10))/(self.calc["Vfs_cloudy_inv"]*0.3163*PC_TO_KM*(10**5))
        self.calc["E51_inv"] = (self.calc["n0_inv"]*((self.calc["Rs_inv"]/(0.3163*PC_TO_KM*10**5))**5))/(self.calc["Kw_inv"]*(self.calc["t_inv"]**2))

#######################################################################################################################
    def verify_forward_cloudyT_n6to14(self):
        self.calc["RSW_inv"] = 0.3163*PC_TO_KM*(10**5)*((self.calc["E51_inv"]*self.calc["Kw_inv"]/self.calc["n0_inv"])**0.2)*((self.calc["t_inv"])**(0.4))
        self.calc["Rs_ver"] = self.calc["RSW_inv"]
        self.calc["EM_ver"]=16*((self.calc["n0_inv"])**2)*(MU_H/MU_e)*self.calc["dEMw_inv"]*((self.calc["Rs_ver"])**3)

        self.calc["TeS_ver"] = self.calc["Ts_inv"]/MU_t/((1/(MU_I*self.calc["geif_inv"]))+(1/MU_e)) #K
        self.calc["TE_ver"] = self.calc["TeS_ver"]*self.calc["dTw_inv"]

        self.calc["Rs_ver"] = self.calc["Rs_ver"]/PC_TO_KM/10**5
        self.calc["EM_ver"] = self.calc["EM_ver"] /10**58
        self.calc["TE_ver"] = self.calc["TE_ver"]/KEV_TO_ERG*BOLTZMANN


#######################################################################################################################
#######################################################################################################################
    def calculate_Sedov_Forward_values(self):
        self.calc["Te_inv"] = self.data["Te_f_inv"] * KEV_TO_ERG / BOLTZMANN
        self.calc["Rs_inv"] = self.data["R_f_inv"]*PC_TO_KM*10**5
        self.calc["EM_inv"] = self.data["EM58_f_inv"]*10**58
        self.calc["dEMw_inv"] = 1
        self.calc["dTw_inv"] = 1
        self.calc["Kw_inv"] = 1

        self.calc["n0_inv"] = (((self.calc["EM_inv"]*MU_e)/(16*self.calc["dEMw_inv"]*MU_H*((self.calc["Rs_inv"])**3)))**(0.5))
        self.calc["Vfs1_inv"] = ((16*BOLTZMANN*self.calc["Te_inv"])/(3*MU_t*M_H*self.calc["dTw_inv"]))**0.5

        self.calc["t_inv"] = (self.calc["Rs_inv"]*1.237*(10**10))/(self.calc["Vfs1_inv"]*0.316*PC_TO_KM*(10**5)) # in years

        self.calc["E51_inv"] = (self.calc["n0_inv"]*((self.calc["Rs_inv"]/(0.3163*PC_TO_KM*10**5))**5))/(self.calc["Kw_inv"]*(self.calc["t_inv"]**2))

        self.calculate_nuSed_sedov_inv(self.calc["t_inv"], self.calc["E51_inv"])
        for x in range (0,15):
            self.calculate_nuSed_sedov_inv(self.calc["t_inv"], self.calc["E51_inv"])

#######################################################################################################################
    def calculate_nuSed_sedov_inv(self, ty, E51):
        self.calc["VSW_inv"] = 1.2373*(10**10)*((self.calc["E51_inv"]*self.calc["Kw_inv"]/self.calc["n0_inv"])**0.2)*((self.calc["t_inv"])**(-0.6))
        self.calc["Ts_inv"] = (3/16)*MU_t*(M_H/BOLTZMANN)*(self.calc["VSW_inv"]**2)
        self.calc["Vfs_sedov_inv"] = ((16*BOLTZMANN*self.calc["Ts_inv"])/(3*MU_t*M_H))**0.5

        self.calc["t_inv"] = (self.calc["Rs_inv"]*1.2373*(10**10))/(self.calc["Vfs_sedov_inv"]*0.3163*PC_TO_KM*(10**5))
        self.calc["E51_inv"] = (self.calc["n0_inv"]*((self.calc["Rs_inv"]/(0.3163*PC_TO_KM*10**5))**5))/(self.calc["Kw_inv"]*(self.calc["t_inv"]**2))

#######################################################################################################################
    def verify_forward_sedovT_n6to14(self):
        self.calc["RSW_inv"] = 0.3163*PC_TO_KM*(10**5)*((self.calc["E51_inv"]*self.calc["Kw_inv"]/self.calc["n0_inv"])**0.2)*((self.calc["t_inv"])**(0.4))
        self.calc["Rs_ver"] = self.calc["RSW_inv"]
        self.calc["EM_ver"]=16*((self.calc["n0_inv"])**2)*(MU_H/MU_e)*self.calc["dEMw_inv"]*((self.calc["Rs_ver"])**3)
        self.calc["TE_ver"] = self.calc["Ts_inv"]*self.calc["dTw_inv"]

        self.calc["Rs_ver"] = self.calc["Rs_ver"]/PC_TO_KM/10**5
        self.calc["EM_ver"] = self.calc["EM_ver"] /10**58
        self.calc["TE_ver"] = self.calc["TE_ver"]/KEV_TO_ERG*BOLTZMANN

#######################################################################################################################
#######################################################################################################################
    def get_dEMF(self, n, t):
        """Get dEMF from n and t values
        Returns:
            dEMF: dEMF value from input parameters
        """
        self.dEMFcnsts =dEMFInv_DICT[n]

        if (t < self.dEMFcnsts.t1ef):
           dEMF = self.dEMFcnsts.def0

        elif ((self.dEMFcnsts.t1ef <= t) and (t < self.dEMFcnsts.t2ef)):
            dEMF = self.dEMFcnsts.def0 * ((t/self.dEMFcnsts.t1ef)**(self.dEMFcnsts.a1ef))

        elif ((self.dEMFcnsts.t2ef <= t) and (t < self.dEMFcnsts.t3ef)):
            dEMF = (self.dEMFcnsts.def0 * ((self.dEMFcnsts.t2ef/self.dEMFcnsts.t1ef)**(self.dEMFcnsts.a1ef))
                                        * ((t/self.dEMFcnsts.t2ef)**(self.dEMFcnsts.a2ef)))

        elif ((self.dEMFcnsts.t3ef <= t) and (t < self.dEMFcnsts.t4ef)):
            dEMF = (self.dEMFcnsts.def0 * ((self.dEMFcnsts.t2ef/self.dEMFcnsts.t1ef)**(self.dEMFcnsts.a1ef))
                                        * ((self.dEMFcnsts.t3ef/self.dEMFcnsts.t2ef)**(self.dEMFcnsts.a2ef))
                                        * ((t/self.dEMFcnsts.t3ef)**(self.dEMFcnsts.a3ef)))

        elif (self.dEMFcnsts.t4ef <= t):
            dEMF = (self.dEMFcnsts.def0 * ((self.dEMFcnsts.t2ef/self.dEMFcnsts.t1ef)**(self.dEMFcnsts.a1ef))
                                        * ((self.dEMFcnsts.t3ef/self.dEMFcnsts.t2ef)**(self.dEMFcnsts.a2ef))
                                        * ((self.dEMFcnsts.t4ef/self.dEMFcnsts.t3ef)**(self.dEMFcnsts.a3ef))
                                        * ((t/self.dEMFcnsts.t4ef)**(self.dEMFcnsts.a4ef)))
        return dEMF


#######################################################################################################################
    def get_dTF(self, n, t):
        """Get dTF from n and t values
        Returns:
            dTF: dTF value from input parameters
        """
        self.dTFcnsts =dTFInv_DICT[n]

        if (t < self.dTFcnsts.t1tf):
           dTF = self.dTFcnsts.dtf

        elif ((self.dTFcnsts.t1tf <= t) and (t < self.dTFcnsts.t2tf)):
            dTF = self.dTFcnsts.dtf * ((t/self.dTFcnsts.t1tf)**(self.dTFcnsts.a1tf))

        elif (self.dTFcnsts.t2tf <= t):
            dTF = (self.dTFcnsts.dtf * ((self.dTFcnsts.t2tf/self.dTFcnsts.t1tf)**(self.dTFcnsts.a1tf))
                                        * ((t/self.dTFcnsts.t2tf)**(self.dTFcnsts.a2tf)))
        return dTF


#######################################################################################################################
    def get_dEMR(self, n, t):
        """Get dEMR from n and t values
        Returns:
            dEMR: dEMR value from input parameters
        """
        self.dEMRcnsts =dEMRInv_DICT[n]

        if (t < self.dEMRcnsts.t1er):
           dEMR = self.dEMRcnsts.der * ((t/self.dEMRcnsts.t1er)**(-1.0*self.dEMRcnsts.a1er))

        elif ((self.dEMRcnsts.t1er <= t) and (t < self.dEMRcnsts.t2er)):
            dEMR = self.dEMRcnsts.der * ((t/self.dEMRcnsts.t1er)**(-1.0*self.dEMRcnsts.a2er))

        elif ((self.dEMRcnsts.t2er <= t) and (t < self.dEMRcnsts.t3er)):
            dEMR = (self.dEMRcnsts.der * ((self.dEMRcnsts.t2er/self.dEMRcnsts.t1er)**(-1.0*self.dEMRcnsts.a2er))
                                        * ((t/self.dEMRcnsts.t2er)**(-1.0*self.dEMRcnsts.a3er)))

        elif (self.dEMRcnsts.t3er <= t):
            dEMR = (self.dEMRcnsts.der * ((self.dEMRcnsts.t2er/self.dEMRcnsts.t1er)**(-1.0*self.dEMRcnsts.a2er))
                                        * ((self.dEMRcnsts.t3er/self.dEMRcnsts.t2er)**(-1.0*self.dEMRcnsts.a3er))
                                        * ((t/self.dEMRcnsts.t3er)**(-1.0*self.dEMRcnsts.a4er)))
        return dEMR


#######################################################################################################################
    def get_dTR(self, n, t):
        """Get dTR from n and t values
        Returns:
            dTR: dTR value from input parameters
        """
        self.dTRcnsts =dTRInv_DICT[n]

        if (t < self.dTRcnsts.t1tr):
           dTR = self.dTRcnsts.dtr * ((t/self.dTRcnsts.t1tr)**(self.dTRcnsts.a1tr))

        elif ((self.dTRcnsts.t1tr <= t) and (t < self.dTRcnsts.t2tr)):
            dTR = self.dTRcnsts.dtr * ((t/self.dTRcnsts.t1tr)**(self.dTRcnsts.a2tr))

        elif ((self.dTRcnsts.t2tr <= t) and (t < self.dTRcnsts.t3tr)):
            dTR = (self.dTRcnsts.dtr * ((self.dTRcnsts.t2tr/self.dTRcnsts.t1tr)**(self.dTRcnsts.a2tr))
                                        * ((t/self.dTRcnsts.t2tr)**(self.dTRcnsts.a3tr)))

        elif (self.dTRcnsts.t3tr <= t):
            dTR = (self.dTRcnsts.dtr * ((self.dTRcnsts.t2tr/self.dTRcnsts.t1tr)**(self.dTRcnsts.a2tr))
                                        * ((self.dTRcnsts.t3tr/self.dTRcnsts.t2tr)**(self.dTRcnsts.a3tr))
                                        * ((t/self.dTRcnsts.t3tr)**(self.dTRcnsts.a4tr)))
        return dTR
