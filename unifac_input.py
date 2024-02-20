import numpy as np
# UNIFAC-LLE model parameters
NC = 3 # Number of components
NG = 6 # Number of functional groups

# Functional groups matrix of frequency:
# CH3, CH2, CH, H2O, CH3COO, COOH
v = np.array([[0,0,0,1,0,0],    # Water
              [1,0,0,0,0,1], 	# Acetic Acid
			  [2,1,1,0,1,0]])   # Isobutyl Acetate
# Parâmetros de área e volume dos grupos funcionais
Rk = np.array([0.9011, 0.6744, 0.4469, 0.9200, 1.9031, 1.3013])
Qk = np.array([0.8480, 0.5400, 0.2280, 1.4000, 1.7280, 1.2240])
# Parâmetros de interação binária entre os grupos funcionais
a = np.array([[0.0000, 0.0000, 0.0000, 1300.0, 972.40, 139.40],
              [0.0000, 0.0000, 0.0000, 1300.0, 972.40, 139.40],
			  [0.0000, 0.0000, 0.0000, 1300.0, 972.40, 139.40],
			  [342.40, 342.40, 342.40, 0.0000, -6.320, -465.7],
			  [-320.1, -320.1, -320.1, 385.90, 0.0000, 1417.0],
			  [1744.0, 1744.0, 1744.0, 652.30, -117.6, 0.0000]])

# Feed compositions
z = np.array([[0.7886,	0.0707,	0.1407],
            [0.7795,	0.0997,	0.1208],
            [0.7638,	0.1318,	0.1044],
            [0.7544,	0.1619,	0.0837],
            [0.7458,	0.195,	0.0592]])