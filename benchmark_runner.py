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
#### replicate results from ruojing's code with siam_current module (ASU formalism)


verbose = 4;
nleads = (1,1);
nelecs = (2,1); # half filling
nelecs_ASU = (sum(nelecs),0); # all spin up formalism

#time info
dt = 0.01;
tf = 5.0;

# run test with spin free code
params = 1.0, 1.0, -0.005, 0.0, 0.0; # featureless dot
t, observables = ruojings_td_fci.TestRun(nleads, nelecs, tf, dt, phys_params = params, verbose = verbose);
E, J, occ, Sz = observables; # unpack all data

# plot results
plot.PlotObservables(nleads, t, observables);

del t, observables, E, J, occ, Sz

# run test with ASU code

#params_ASU = 1.0, 1.0, -0.005, 0.0, 0.0, 0.0, 0.0, 0.0; # featureless dot
#t, observables = siam_current.DotCurrentData(nleads, nelecs_ASU, tf, dt, phys_params=None, ret_results = True, verbose = verbose);
E, J, occ, Sz = observables; # unpack all data

# plot results
plot.PlotObservables(nleads, t, observables);

##################################################################################
#### test finite size effects ?








