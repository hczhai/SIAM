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


verbose = 3;
nleads = (2,1);
nelecs = (nleads[0],nleads[0]); # half filling
nelecs_ASU = (sum(nelecs),0); # all spin up formalism

#time info
dt = 0.01;
tf = 1.0;

# benchmark with spin free code
# std inputs
if verbose: print("Spin free\n","-"*80);
t, observables = ruojings_td_fci.TestRun(nleads, nelecs, tf, dt, phys_params = None, verbose = verbose);
plot.PlotObservables(nleads, t, observables, occ_only = False); # plot results

# run ASU code for td FCI, std inputs + mag field
if verbose: print("-"*80,"\nASU\n");
B = 0.0 #4.0;
theta = 0 #np.pi/2
params = 1.0, 0.4, -0.005, 0.0, -0.5, 1.0, B, theta; # dot with B in z direction
t, observables = siam_current.DotCurrentData(nleads, nelecs_ASU, tf, dt, phys_params=params, ret_results = True, verbose = verbose);
E, J, occ, Sz = observables; # unpack all data

# plot results
plot.PlotObservables(nleads, t, observables, occ_only = False);

##################################################################################
#### test finite size effects ?








