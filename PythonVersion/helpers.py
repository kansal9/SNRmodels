
import numpy as np
from collections import namedtuple
import os

PC_TO_KM = 3.0857e13  #km
KEV_TO_ERG = 1.60218e-9   #erg
BOLTZMANN = 1.38066e-16 #erg/K
PLANCK = 6.62608e-27 #erg/s
SOLAR_MASS_TO_GRAM = 1.9891e33 #g
M_H = 1.67262e-24
MU_H = 1.35862
MU_I = 1.2505
MU_e = 1.15197
MU_t = 0.59961
MU_ratej = 0.94105
MU_tej = 0.70033
MU_I2 = 1.53685
MU_e2 = 1.28663
MU_Iej = MU_I2
MU_eej = MU_e2
K_SED = 1.528
PHI_C = 0.5
A_VALS = {"H": 1, "He": 4, "C": 12, "O": 16, "Ne": 20, "N": 14, "Mg": 24, "Si": 28, "Fe": 56, "S": 32, "Ca": 42, "Ni": 60, "Na": 22, "Al": 26, "Ar": 36}
Z_VALS = {"H": 1, "He": 2, "C": 6, "O": 8, "Ne": 10, "N": 7, "Mg": 12, "Si": 14, "Fe": 26, "S": 16, "Ca": 20, "Ni": 28, "Na": 11, "Al": 13, "Ar": 18}
XI_0 = 2.026
YR_TO_SEC = 365.26 * 24 * 3600 #s
BETA = 2
PCyr_TO_KMs = PC_TO_KM / YR_TO_SEC # Converts from pc/yr to km/s (9.78e5 km/s)
# _rchg corresponds to values at which behaviour of reverse shock changes (typically t_core, t_rst for n=2)
# Note a_core = 2 * v_core / t_core
ValueSet = namedtuple("ValueSet", "l_ed phi_ed t_st r_st t_rev r_rev t_rchg r_rchg v_rchg a_rchg phi_eff f_n alpha")
VALUE_DICT_S0 = {
    0: ValueSet(1.10, 0.343, 0.495, 0.727, 2.582, 1.634, 0.495, 0.779, 0.533, 0.106, 0.0961, 3 / (4 * np.pi), 0.6),
    1: ValueSet(1.10, 0.343, 0.441, 0.703, 2.893, 1.724, 0.441, 0.524, 0.635, -0.005, 0.0960, 1 / (2 * np.pi), 0.5),
    2: ValueSet(1.10, 0.343, 0.387, 0.679, 4.141, 2.009, 0.387, 0.503, 0.686, -0.115, 0.0947, 1 / (4 * np.pi), 1 / 3),
    4: ValueSet(1.10, 0.343, 0.232, 0.587, 2.562, 1.666, 1.2, 0.775, 0.427, 0.712, 0.0791, 0.00645, 0.0746),
    6: ValueSet(1.3856, 0.3658, 1.04, 1.07, 2.805, 1.687, 0.5133, 0.541, 0.527, 0.112, None, None, None),
    7: ValueSet(1.2631, 0.4725, 0.732, 0.881, 2.691, 1.654, 0.3629, 0.469, 0.553, 0.116, None, None, None),
    8: ValueSet(1.2144, 0.5293, 0.605, 0.788, 2.682, 1.652, 0.2922, 0.413, 0.530, 0.139, None, None, None),
    9: ValueSet(1.1878, 0.5651, 0.523, 0.725, 2.726, 1.666, 0.2489, 0.371, 0.497, 0.162, None, None, None),
    10: ValueSet(1.171, 0.5897, 0.481, 0.687, 2.740, 1.670, 0.2204, 0.340, 0.463, 0.192, None, None, None),
    11: ValueSet(1.1595, 0.6075, 0.452, 0.661, 2.769, 1.679, 0.1987, 0.316, 0.433, 0.222, None, None, None),
    12: ValueSet(1.151, 0.6217, 0.424, 0.636, 2.740, 1.637, 0.1818, 0.293, 0.403, 0.251, None, None, None),
    13: ValueSet(1.1445, 0.6321, 0.406, 0.620, 2.827, 1.696, 0.1681, 0.276, 0.378, 0.264, None, None, None),
    14: ValueSet(1.1394, 0.6407, 0.389, 0.603, 2.866, 1.707, 0.1567, 0.259, 0.354, 0.277, None, None, None)
}

#DONT have proper values for everything except L_ed and phi_ed
ValueSet2 = namedtuple("ValueSet2", "l_ed phi_ed a2 bbm dTf2 dTr2 dEMf2 dEMr2")
VALUE_DICT_S2 = {
    0: ValueSet2(1.50, 0.025, 0, 0, 0, 0.8868, 0, 0),
    1: ValueSet2(1.50, 0.25, 0, 0, 0, 0.8868, 0, 0),
    2: ValueSet2(1.50, 0.25, 0, 0, 0, 0.8868, 0, 0),
    4: ValueSet2(1.50, 0.25, 0, 0, 0, 0.8868, 0, 0),
    6: ValueSet2(1.4362, 0.2469, 0.5331, 1.3767, 0.1417, 0.0413, 17.6005, 12.9549),
    7: ValueSet2(1.3389, 0.314, 0.2318, 1.2987, 0.0298, 0.0254, 99.3549, 47.8751),
    8: ValueSet2(1.2976, 0.3521, 0.131, 1.2671, 0.0533, 0.0169, 56.9161, 115.9666),
    9: ValueSet2(1.2745, 0.3764, 0.0846, 1.2498, 0.0677, 0.0119, 45.8373, 227.321),
    10: ValueSet2(1.2596, 0.3929, 0.0593, 1.239, 0.0515, 0.0088668, 62.7305, 391.5409),
    11: ValueSet2(1.2492, 0.4051, 0.0439, 1.2314, 0.042, 0.0068421, 79.1564, 619.3189),
    12: ValueSet2(1.2415, 0.4144, 0.0338, 1.2259, 0.036, 0.0054354, 94.6553, 920.7526),
    13: ValueSet2(1.2356, 0.4221, 0.0268, 1.2217, 0.0451, 0.0044209, 75.1756, 1307.5),
    14: ValueSet2(1.231, 0.4283, 0.0218, 1.2184, 0.0411, 0.0036663, 83.7641, 1788.5)
}

InverseCloudyConstants = namedtuple("InverseCloudyConstants", "dEMc dTc Kc")
VALUE_DICT_Cloudy = {
    0: InverseCloudyConstants(0.5164, 1.2896, 1.0),
    1: InverseCloudyConstants(0.7741, 1.3703, 0.746),
    2: InverseCloudyConstants(1.6088, 1.3693, 0.541),
    4: InverseCloudyConstants(6.9322, 1.3833, 0.270),
}

ValueSetdEMFInv = namedtuple("ValueSetdEMFInv", "def0 t1ef t2ef t3ef t4ef a1ef a2ef a3ef a4ef")
dEMFInv_DICT = {
    0: ValueSetdEMFInv(1.0343, 0.088968, 1.7973, 5.4523, 212.16, -0.21388, -0.062395, 0.019185, 0.006374),
    1: ValueSetdEMFInv(1.034, 0.069244, 1.5863, 5.286, 377.53, -0.2012, -0.063576, 0.015637, -0.010714),
    2: ValueSetdEMFInv(1.0333, 0.04247, 1.2248, 5.632, 102.52, -0.18355, -0.053807, 0.020978, -0.007918),
    4: ValueSetdEMFInv(1.0381, 0.007853, 0.29706, 12.873, 36.632, -0.14615, -0.042807, 0.041095, -0.001321),
    6: ValueSetdEMFInv(0.6746, 0.80586, 2.2704, 9.438, 45.697, -0.20999, -0.04777, 0.04576, -0.00481),
    7: ValueSetdEMFInv(0.7542, 0.53431, 1.7621, 9.7853, 47.795, -0.27137, -0.04595, 0.05211, -0.00631),
    8: ValueSetdEMFInv(0.8081, 0.40372, 1.1229, 3.9604, 240.78, -0.31972, -0.11315, 0.01795, -0.00708),
    9: ValueSetdEMFInv(0.8471, 0.33524, 1.1895, 8.9332, 40.988, -0.32353, -0.06303, 0.059833, -0.001358),
    10: ValueSetdEMFInv(0.8767, 0.25626, 1.2476, 7.2565, 41.692, -0.28106, -0.06753, 0.04489, 0.000056795),
    11: ValueSetdEMFInv(0.8998, 0.24002, 1.0709, 4.3649, 55.849, -0.29373, -0.10276, 0.02715, 0.0005721),
    12: ValueSetdEMFInv(0.9184, 0.20832, 1.1873, 6.0202, 51.346, -0.27723, -0.089841, 0.057657, -0.009586),
    13: ValueSetdEMFInv(0.9337, 0.20311, 1.1942, 7.4647, 53.164, -0.29323, -0.05229, 0.0341, 0.0056264),
    14: ValueSetdEMFInv(0.9465, 0.1928, 1.1496, 7.4809, 42.566, -0.28826, -0.07038, 0.05166, -0.00212)
}

ValueSetdTFInv = namedtuple("ValueSetdTFInv", "dtf t1tf t2tf a1tf a2tf")
dTFInv_DICT = {
    0: ValueSetdTFInv(1.0234, 0.03583, 7.9771, 0.03811, -0.00051559),
    1: ValueSetdTFInv(1.0239, 0.02592, 11.325, 0.03623, -0.00433),
    2: ValueSetdTFInv(1.0245, 0.01408, 11.053, 0.0336, -0.00503),
    4: ValueSetdTFInv(1.0239, 0.00537, 0.40537, 0.04366, 0.0024),
    6: ValueSetdTFInv(1.2614, 0.1, 1.1974, -0.00741, 0.00091974),
    7: ValueSetdTFInv(1.2227, 3.0116, 19.437, 0.02601, -0.00529),
    8: ValueSetdTFInv(1.1922, 2.5676, 17.852, 0.03346, -0.00243),
    9: ValueSetdTFInv(1.1694, 0.83038, 11.955, 0.02713, -0.00035205),
    10: ValueSetdTFInv(1.1518, 0.47077, 11.242, 0.02739, -0.00047925),
    11: ValueSetdTFInv(1.138, 0.4149, 13.72, 0.02805, -0.00028536),
    12: ValueSetdTFInv(1.127, 0.31582, 11.863, 0.03019, -0.00032846),
    13: ValueSetdTFInv(1.1179, 0.39999, 16.487, 0.035929, -0.004505),
    14: ValueSetdTFInv(1.1103, 0.24283, 8.6548, 0.03594, -0.00032298)
}

ValueSetdEMRInv = namedtuple("ValueSetdEMRInv", "der t1er t2er t3er a1er a2er a3er a4er")
dEMRInv_DICT = {
    0: ValueSetdEMRInv(0.84352, 0.68201, 2.5196, 6.5196, 3.5193, 3.0284, 1.5369, 1.9315),
    1: ValueSetdEMRInv(0.16079, 0.97003, 1.97, 5.97, 3.369, 2.8062, 1.7625, 1.9314),
    2: ValueSetdEMRInv(5.4852, 0.21484, 2.5712, 7.4783, 3.594, 2.6926, 1.257, 1.9692),
    4: ValueSetdEMRInv(0.4005, 0.12652, 2.4127, 16.011, 3.0886, 1.3641, 2.7001, 1.5229),
    6: ValueSetdEMRInv(0.1604, 0.61461, 2.8333, 4.4822, 0, 2.4651, 1.2471, 1.9194),
    7: ValueSetdEMRInv(0.625, 0.42728, 2.6576, 4.8889, 0, 2.6521, 1.3844, 1.9194),
    8: ValueSetdEMRInv(1.5485, 0.33823, 2.9531, 4.5135, 0, 2.7332, 0.78063, 1.9198),
    9: ValueSetdEMRInv(3.0872, 0.32215, 2.462, 4.9846, 0, 2.9551, 1.3352, 1.9209),
    10: ValueSetdEMRInv(5.3995, 0.23576, 2.7997, 4.8193, 0, 2.7187, 1.0087, 1.926),
    11: ValueSetdEMRInv(8.6343, 0.22097, 2.832, 4.8614, 0, 2.7817, 1.0238, 1.9272),
    12: ValueSetdEMRInv(12.98, 0.21044, 2.6659, 4.9225, 0, 2.8849, 0.98963, 1.9411),
    13: ValueSetdEMRInv(18.533, 0.19399, 2.73, 4.8213, 0, 2.8915, 1.0094, 1.9288),
    14: ValueSetdEMRInv(25.495, 0.17475, 2.6817, 4.8485, 0, 2.8823, 1.0067, 1.9338)
}

ValueSetdTRInv = namedtuple("ValueSetdTRInv", "dtr t1tr t2tr t3tr a1tr a2tr a3tr a4tr")
dTRInv_DICT = {
    0: ValueSetdTRInv(0.03458, 0.24804, 3.4223, 4.4223, 2.0715, 1.0448, 2.7094, 0.7271),
    1: ValueSetdTRInv(0.03838, 0.20222, 3.6594, 4.6594, 2.1888, 1.0786, 2.275, 0.73006),
    2: ValueSetdTRInv(0.03936, 0.12993, 2.865, 7.5695, 2.3071, 1.1531, 0.01, 0.83867),
    4: ValueSetdTRInv(0.33056, 0.11473, 0.61473, 2.6147, 1.7238, 0.76782, 0.44658, 0.77125),
    6: ValueSetdTRInv(0.8868, 0.10014, 1.376, 5.0236, 0, -0.03928, 1.2644, 0.71579),
    7: ValueSetdTRInv(0.5204, 0.22901, 1.229, 6.9959, 0, 0.07821, 1.1669, 0.71371),
    8: ValueSetdTRInv(0.3368, 0.43989, 2.0375, 4.6312, 0, 0.54803, 1.5622, 0.71316),
    9: ValueSetdTRInv(0.2347, 0.40168, 2.5233, 4.5042, 0, 0.65648, 1.8429, 0.719),
    10: ValueSetdTRInv(0.1723, 0.3075, 2.6861, 4.5201, 0, 0.6537, 1.9969, 0.71592),
    11: ValueSetdTRInv(0.1318, 0.26188, 2.4925, 4.5829, 0, 0.64362, 1.841, 0.72049),
    12: ValueSetdTRInv(0.1039, 0.23824, 2.7662, 4.5589, 0, 0.69647, 2.0949, 0.71098),
    13: ValueSetdTRInv(0.084, 0.21624, 2.8018, 4.5087, 0, 0.7265, 2.0792, 0.72111),
    14: ValueSetdTRInv(0.0693, 0.19662, 2.9253, 4.4577, 0, 0.76016, 2.2472, 0.72422)
}

#Min and max radius value at the start and end of the ChevParker Data files
S0_CHEV_RMIN = {6: 0.90606, 7: 0.93498, 8: 0.95023, 9: 0.95967, 10: 0.96609, 11: 0.97075, 12: 0.97428, 13: 0.97705, 14: 0.97928}
S0_CHEV_RMAX = {6: 1.25542, 7: 1.18099, 8: 1.15397, 9: 1.13993, 10: 1.13133, 11: 1.12554, 12: 1.12136, 13: 1.11822, 14: 1.11576}
S2_CHEV_RMIN = {6: 0.95849, 7: 0.96999, 8: 0.97649, 9: 0.98068, 10: 0.98360, 11: 0.98575, 12: 0.98740, 13: 0.98871, 14: 0.98978}
S2_CHEV_RMAX = {6: 1.37656, 7: 1.29871, 8: 1.26712, 9: 1.24986, 10: 1.23894, 11: 1.23142, 12: 1.22591, 13: 1.22170, 14: 1.21838}

K_DICT = {}

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the data.csv file
data_path = os.path.join(script_dir, 'data/WL91Parameters.csv')
cism_file = open(data_path, "r")
for line in cism_file:
    line = line.rstrip().split(",")
    K_DICT[float(line[0])] = float(line[1]) * 1.528
    
CISM_EM_WEIGHTED = {"beta": {0: 1.2896, 1: 1.370303, 2: 1.369303, 4: 1.08032}, #beta is ratio of em-weighted Tav to Tshock
                  "alpha": {0: 2.6193, 1: 2.476838394791874, 2: 2.637864262949064, 4: 8.0134}} #alpha is ratio of em-weighted n_av to n_0

#Note White&Long solution files are of the form: r/Rshock, P/Pshock, rho/rho_shock, v/Vshock (so vmax is 3/4), column emission measure
data_path2 = os.path.join(script_dir, 'data/WLsoln0.0_xfgh.txt')
lines = np.loadtxt(data_path2)
radius2 = lines[:, 0]
pressure2 = lines[:, 1]
density2 = lines[:, 2]
