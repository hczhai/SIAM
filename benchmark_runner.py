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
tf = 12.0;

# run test with spin free code
t, observables = ruojings_td_fci.TestRun(nleads, nelecs, tf, dt, verbose = verbose);
E, J, occ, Sz = observables; # unpack all data

# plot results
plot.PlotObservables(nleads, t, E, J, occ);

del t, observables, E, J, occ, Sz

# run test with ASU code
#t, observables = siam_current.DotCurrentData(nleads, nelecs_ASU, tf, dt, ret_results = True, verbose = verbose);
E, J, occ, Sz = observables; # unpack all data

# plot results
plot.PlotObservables(nleads, t, E, J, occ);

##################################################################################
#### test finite size effects ?








