'''
Christian Bunker
M^2QM at UF
July 2021

Runner file for prepping dot spin state with B field
'''

import ruojings_td_fci
import siam_current
import plot

import numpy as np
import matplotlib.pyplot as plt

##################################################################################
#### prepare dot in diff spin states

# top level
verbose = 4;
nleads = (3,2);
nelecs = (sum(nleads)+1,0); # half filling
get_data = False; # whether to run computations, if not data already exists

# phys params, must be floats
tl = 1.0;
th = 0.4;
Vb = -0.005;
mu = 0.0;
Vg = -0.5;
U = 1.0
Bs = [tl*5, tl*5, tl*5,tl*5,tl*5];
thetas = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi];

#time info
dt = 0.01;
tf = 20.0;

datafs = [];
if get_data: # must actually compute data

    for i in range(len(Bs)): # iter over B, theta inputs
        B, theta = Bs[i], thetas[i];
        params = tl, th, Vb, mu, Vg, U, B, theta;
        fname = siam_current.DotData(nleads, nelecs, tf, dt, phys_params=params, verbose = verbose);
        datafs.append(fname);

else: # already there
    datafs = ["dat/DotData/3_1_2_e6_B5.0_t0.0_Vg-0.5.npy","dat/DotData/3_1_2_e6_B5.0_t0.7_Vg-0.5.npy",
              "dat/DotData/3_1_2_e6_B5.0_t1.5_Vg-0.5.npy","dat/DotData/3_1_2_e6_B5.0_t2.3_Vg-0.5.npy",
              "dat/DotData/3_1_2_e6_B5.0_t3.1_Vg-0.5.npy"];

    #### plot comparing diff spin states
    plot.CompInitSpin(Bs,thetas,datafs);








