import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve, minimize
from pyswarm import pso
import LLEaux, LLE_calc
import warnings
# Change here to the path of the unifac_input.py file
import unifac_input as ui
import LLE_plot
warnings.filterwarnings("ignore")

# Initial Feed conditions
T = 293.2 # System temperature
NP  = np.size(ui.z,0)   # Number of points inserted
xI  = np.zeros((NP,ui.NC))
xII = np.zeros((NP,ui.NC))
for i in range(NP):
    xI[i,:],xII[i,:] = LLE_calc.LLE_calc(ui.z[i,:], T, ui.NC, ui.NG, ui.v, ui.Rk, ui.Qk, ui.a)

# Concatenate arrays horizontally
    # ARRUMAR PARA ORDENAR AS FASES CORRETAMENTE
LLEresults = np.hstack((ui.z, xI, xII))
print(LLEresults)
# Save the combined array to a text file
np.savetxt('output1.txt', LLEresults, fmt='%.6f', delimiter='\t')
# Plot the Ternary Diagram
LLE_plot.LLE_plot()