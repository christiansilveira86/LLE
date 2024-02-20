import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve, minimize
from pyswarm import pso
import LLEaux
import warnings
warnings.filterwarnings("ignore")
# Phase stability flag:
#   1 - One phase
#   2 - Two phases

# LLE_calc receives the initial feed composition, temperature and number of components in the mixture
def LLE_calc(z, T, NC, NG, v, Rk, Qk, a):
    R = 8.3144 # (Pa.m^3)/(mol.K)
    # -----------------------------------------------------------------------------------------------
    # PHASE STABILITY TEST
    # -----------------------------------------------------------------------------------------------
    # Linear constraint (1): sum(x) = 1
    con1 = {'type':'eq', 'fun':LLEaux.constr1}
    con2 = {'type':'eq', 'fun':LLEaux.constr2}
    con3 = {'type':'eq', 'fun':LLEaux.constr3}
    # Random initial guesses for both phases
    xI0  = np.random.rand(NC)
    xII0 = np.random.rand(NC)
    xI0  = xI0/np.sum(xI0)
    xII0 = xII0/np.sum(xII0)
    x0   = np.concatenate((xI0,xII0),axis=0)
    # Boundaries for the variables (xI0 and xII0)
    lim = (0,1)
    bound = []
    for i in range(2*NC):
        bound.append(lim)
    bound = tuple(bound)
    # Phase stability test
    print('________________________________')
    print('Initial guesses:')
    print('xI = ',x0[:NC])
    print('xII =',x0[NC:])
    print('________________________________')
    result_stability = minimize(LLEaux.test_stability, x0, method='SLSQP', bounds=bound, constraints=[con2,con3], args=(z, T, NC, NG, v, Rk, Qk, a, R))
    OF = result_stability.fun
    if np.abs(OF)<1e-6:
        OF = 0
    if OF < 0:
        print('System is unstable and will split in 2 phases.')
        flg = 1
    else:
        print('System is stable as it is and won\'t split.')
        flg = 0
    print('OF = ',OF)
    print('________________________________')

    # -----------------------------------------------------------------------------------------------
    # PHASE COMPOSITION CALCULATION
    # -----------------------------------------------------------------------------------------------
    if flg == 0:
        # Calculate the activity coefficients and any other info and end
        print('No need for phase equilibria calculations.')
        xI = z
        xII = z
    else:
        # Phase equilibrium calculation
        # Constraints for the LLE Composition calculation
        con2 = {'type':'eq', 'fun':LLEaux.constr2}
        con3 = {'type':'eq', 'fun':LLEaux.constr3}
        con4 = {'type':'eq', 'fun':LLEaux.constr4, 'args':(T, NC, NG, v, Rk, Qk, a)}
        # Boundaries
        bound2 =[]
        for i in range(2*NC):
            bound2.append(lim)
        
        # Random initial guesses for both phases
        xI0  = np.random.rand(NC)
        xII0 = np.random.rand(NC)
        xI0  = xI0/np.sum(xI0)
        xII0 = xII0/np.sum(xII0)
        x0   = np.concatenate((xI0,xII0),axis=0)
        # Gibbs energy minimization
        # Particle Swarm Optimization - PSO
        print('\n Gibbs energy minimization is running, please wait.')
        x, OF = pso(LLEaux.LLEcomposition, [0,0,0,0,0,0],[1,1,1,1,1,1], ieqcons=[LLEaux.constr2a, LLEaux.constr2b, LLEaux.constr3a, LLEaux.constr3b], args=(z, T, NC, NG, v, Rk, Qk, a, R))
        print('________________________________')
        print('Particle Swarm Optimization')
        print('Objective Function:',OF)
        print('xI =',x[:NC])
        print('xII =',x[NC:])
        print('________________________________')

        x0 = x
        result_flash = LLEaux.flash_algorithm(x0, z, T, NC, NG, v, Rk, Qk, a, R)
        print(result_flash)
        xI = result_flash[0]
        xII = result_flash[1]
        print('________________________________')
        print('Equilibrium Composition - LLE')
        print('xI =',result_flash[0])
        print('xII =',result_flash[1])
        print('________________________________')
    return xI, xII
