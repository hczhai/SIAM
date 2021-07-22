'''
Christian Bunker
M^2QM at UF
July 2021

Runner file for benchmarking td-FCI
'''

import ruojings_td_fci
import siam_current
import plot

import numpy as np
import matplotlib.pyplot as plt

##################################################################################
#### prepare dot in spin up state


verbose = 4;
nleads = (3,2);
nelecs = (nleads[0],nleads[0]); # half filling
nelecs_ASU = (sum(nelecs),0); # all spin up formalism
splots = ['Jtot','occ','delta_occ','Sz']; # which subplots to make

#time info
dt = 0.01;
tf = 8.0;

# benchmark with spin free code
# std inputs
if verbose: print("-"*80,"\nSpin free\n");
# saves data to .npy
fname = ruojings_td_fci.SpinfreeTest(nleads, nelecs, tf, dt, phys_params = None, verbose = verbose);
plot.PlotObservables(nleads, 0.4, fname, splots=splots); # plot results

# run ASU code for td FCI, std inputs + mag field
if verbose: print("-"*80,"\nASU\n");
B = 0.0 #4.0;
theta = 0.0 #np.pi/2
params = 1.0, 0.4, -0.005, 0.0, -0.5, 1.0, B, theta; # dot with B in z direction
# save data to .npy
fname_ASU = siam_current.DotData(nleads, nelecs_ASU, tf, dt, phys_params=None, verbose = verbose);
plot.PlotObservables(nleads, params[1], fname_ASU, splots=splots);

##################################################################################
#### test finite size effects ?








