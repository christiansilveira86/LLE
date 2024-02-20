import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from unifac import UNIFAC
import random
from scipy.optimize import fsolve, minimize
from pyswarm import pso
import LLEaux
import warnings
warnings.filterwarnings("ignore")

# Phase Stability test: Mass conservation constraint
def constr1(x):
    sum = 0
    for i in range(np.size(x)):
        sum += x[i]
    return sum-1

# LLE Composition Calc: Phase-1 molar fraction consistency
def constr2(x):
    NC = int(np.size(x)/2)
    xI    = x[0:NC]
    sumI  = 0
    for i in range(NC):
        sumI  += xI[i]
    return sumI - 1

# LLE Composition Calc: Phase-2 molar fraction consistency
def constr3(x):
    NC = int(np.size(x)/2)
    xII    = x[NC:]
    sumII  = 0
    for i in range(NC):
        sumII  += xII[i]
    return sumII - 1

# LLE Composition Calc: isofugacity criteria
def constr4(x, *args):
    T, NC, NG, v, Rk, Qk, a = args
    NC = int(np.size(x)/2)
    xI     = x[:NC]
    xII    = x[NC:]
    c      = np.ones(NC)*1000
    # Avoid zeros
    for i in range(NC):
        if xI[i] == 0:
            xI[i] = 1e-6
        if xII[i] == 0:
            xII[i] = 1e-6
    gammaI  = UNIFAC(xI, T, NC, NG, v, Rk, Qk, a)
    gammaII = UNIFAC(xII, T, NC, NG, v, Rk, Qk, a)
    for i in range(NC):
        c[i] = xI[i]*gammaI[i] - xII[i]*gammaII[i]
    return c

#------------- Experimental set of constraints for PSO --------------------------#
# As the PSO only accepts inequalities constraints, this is an attempt to adapt
# the equality constraints into inequalities constraints
def constr2a(x, *args):
    NC = int(np.size(x)/2)
    xI    = x[0:NC]
    sumI  = 0
    for i in range(NC):
        sumI  += xI[i]
    return -sumI + 1.0001

def constr2b(x, *args):
    NC = int(np.size(x)/2)
    xI    = x[0:NC]
    sumI  = 0
    for i in range(NC):
        sumI  += xI[i]
    return sumI - 0.9999

def constr3a(x, *args):
    NC = int(np.size(x)/2)
    xII    = x[NC:]
    sumII  = 0
    for i in range(NC):
        sumII  += xII[i]
    return -sumII + 1.0001

def constr3b(x, *args):
    NC = int(np.size(x)/2)
    xII    = x[NC:]
    sumII  = 0
    for i in range(NC):
        sumII  += xII[i]
    return sumII - 0.9999

def constr4a(x, *args):
    T, NC, NG, v, Rk, Qk, a = args
    NC = int(np.size(x)/2)
    xI     = x[:NC]
    xII    = x[NC:]
    c      = np.zeros(NC)
    gammaI  = UNIFAC(xI, T, NC, NG, v, Rk, Qk, a)
    gammaII = UNIFAC(xII, T, NC, NG, v, Rk, Qk, a)
    for i in range(NC):
        c[i] = -(xI[i]*gammaI[i] - xII[i]*gammaII[i]) + 0.0001
    return c

def constr4b(x, *args):
    T, NC, NG, v, Rk, Qk, a = args
    NC = int(np.size(x)/2)
    xI     = x[:NC]
    xII    = x[NC:]
    c      = np.zeros(NC)
    gammaI  = UNIFAC(xI, T, NC, NG, v, Rk, Qk, a)
    gammaII = UNIFAC(xII, T, NC, NG, v, Rk, Qk, a)
    for i in range(NC):
        c[i] = (xI[i]*gammaI[i] - xII[i]*gammaII[i]) + 0.0001
    return c
#--------------------------------------------------------------------------------#

# LLE Stability test
def test_stability(x,z, T, NC, NG, v, Rk, Qk, a, R):
    # Distribute the initial compositions for both phases
    NC = int(np.size(x)/2)
    xI = x[0:NC]
    xII = x[NC:2*NC]

    # Change this later. Make a more efficient test.
    for i in range(NC):
        if xI[i]<0 or xII[i]<0 or z[i]<0:
            print(xI,'\n',xII,'\n',z,'\n')
        if xI[i]>1 or xII[i]>1 or z[i]>1:
            print(xI,'\n',xII,'\n',z,'\n')
        if np.isnan(xI[i]) == 1 or np.isnan(xII[i]) == 1 or np.isnan(z[i]) == 1:
            print(xI,'\n',xII,'\n',z,'\n')

    # Calculates the activity coefficient of both phases
    gammaI  = UNIFAC(xI, T, NC, NG, v, Rk, Qk, a)
    gammaII = UNIFAC(xII, T, NC, NG, v, Rk, Qk, a)
    gammaZ  = UNIFAC(z, T, NC, NG, v, Rk, Qk, a)
    
    # Tests if the Gibbs energy for the initial feed composition (z) is more or less
    # stable than a two-phases mixtures with compositions xI and xII.
    # GZ,GI,GII = 0,0,0
    # for i in range(NC):
    #     GZ  += R*T*np.log(gammaZ[i])
    #     GI  += R*T*np.log(gammaI[i])
    #     GII += R*T*np.log(gammaII[i])
    
    # if GZ < (GI+GII) or GZ==(GI+GII):
    #     # System is stable as it is
    #     flg = 0
    # else:
    #     # System is unstable
    #     flg = 1
    OF = 0
    for i in range(NC):
        OF += R*T*(xI[i]*np.log(gammaI[i]) + xII[i]*np.log(gammaII[i]))
    for i in range(NC):
        OF -= R*T*(z[i]*np.log(gammaZ[i]))
    return OF

# Verify input arguments
def LLEcomposition(x, z, T, NC, NG, v, Rk, Qk, a, R):
    NP = 2                  # Number of phases (currently only available for 2)
    n  = np.zeros((NC,NP))  # Receives each component molar fraction in both phases
    xI = x[:NC]             # Phase-I molar fractions
    xII = x[NC:]            # Phase-II molar fractions
    rng = np.random.random(3)  # Generates random numbers for the phase-partitioning test
    for i in range(NC):
        n[i][0] = z[i]*rng[i] # Generates a guess for component 'i' in Phase-I from the feed composition
        # The step above is important to keep the values of both phases consistent to that of the feed
        n[i][NP-1] = z[i] - n[i][0]

    # Molar fractions of the 2-phase mixture
    sumI = np.sum(n,0)[0]
    sumII = np.sum(n,0)[1]
    for i in range(NC):
        xI[i] = n[i][0]/sumI
        xII[i] = n[i][1]/sumII
    
    # Avoid zeros
    for i in range(NC):
        if xI[i] == 0:
            xI[i] = 1e-6
        if xII[i] == 0:
            xII[i] = 1e-6

    # Phase fraction for phases I and II
    phaseI  = sumI
    phaseII = sumII
    # Phase I activity coefficients
    gammaI  = UNIFAC(xI, T, NC, NG, v, Rk, Qk, a)
    # Phase II activity coefficients
    gammaII  = UNIFAC(xII, T, NC, NG, v, Rk, Qk, a)

    # Objective function
    OF = 0
    for i in range(NC):
        # OF += phaseI*xI[i]*np.log(xI[i]*gammaI[i]) + phaseII*xII[i]*np.log(xII[i]*gammaII[i])
        OF += phaseI*xI[i]*np.log(gammaI[i]) + phaseII*xII[i]*np.log(gammaII[i])

    return OF

def flash(vv,K,z,T,NC):
    sum = 0
    for i in range(NC):
        sum += (K[i]*z[i])/(1+vv*(K[i]-1))
    return 1-sum

def flash_algorithm(x, z, T, NC, NG, v, Rk, Qk, a, R): 
    NP  = 2 # Number of phases (only 2 implemented)
    # Allocates the initial guesses to each phase
    xI  = x[:NC]
    xII = x[NC:]
    xa = np.zeros(NC)   # Temp variable for xI
    xb = np.zeros(NC)   # Temp variable for xII
    maxiter = 200       # Maximum number of iterations
    for j in range(maxiter):
        # Phase I activity coefficients
        gammaI  = UNIFAC(xI, T, NC, NG, v, Rk, Qk, a)
        # Phase II activity coefficients
        gammaII  = UNIFAC(xII, T, NC, NG, v, Rk, Qk, a)
        # Objective function calculation
        K = np.zeros(NC)
        for i in range(NC):
            K[i] = gammaII[i]/gammaI[i]
        
        # Separated fraction calculation
        # v0 = np.random.random()
        v0 = 0.3
        vv = fsolve(flash, v0, args=(K,z,T,NC))
        vv = vv[0]

        for i in range(NC):
            xa[i] = z[i]/((1-vv)/K[i] + vv)
        for i in range(NC):
            xb[i] = (z[i]-vv*xa[i])/(1-vv)

        xI = (xI+xa)/2
        xII = (xII+xb)/2

        if np.sum(np.abs(xI-xa))<1e-10 and np.sum(np.abs(xII-xb))<1e-10:
            # print('Molar fractions step is too small. Stop.')
            print('Converged.')
            print('Iterations steps smaller than tolerance.')
            break

    return xI,xII

