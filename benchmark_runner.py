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

if False:

    psi_i = np.array([0,0,0,1,0,0,0,0,0]); # initial wf, eq eigenstate
    psi_c = []; # initial wf in coefs of noneq eigenstates
    t = 1.0; # dot lead hopping
    V = 0.0 # gate voltage
    b = -0.005; # bias
    U = 0.0 # hubbard
    h = np.array([[V,0,0,0,0,t,0,-t,0],
                  [0,V,0,0,t,-t,-t,t,0],
                  [0,0,V,0,-t,0,t,0,0],
                  [0,0,0,V+b,-t,-t,0,0,0],
                  [0,t,-t,-t,b/2,0,0,0,0],
                  [t,-t,0,-t,0,U+2*V+b/2,0,0,0],
                  [0,-t,t,0,0,0,U+2*V-b/2,0,-t],
                  [-t,t,0,0,0,0,0,-b/2,-t],
                  [0,0,0,0,0,0,-t,-t,V-b]]);
    print(h);
    evals, evecs = np.linalg.eigh(h);
    print("0. Exact diagonalization")
    for ei in range(len(evals)):
        psi_c.append(np.dot(psi_i, evecs[ei]) );
        print("- energy = ",evals[ei]," coeff = ", psi_c[ei]);
    print(np.dot(psi_c,psi_c));
    

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








