'''
Christian Bunker
M^2QM at UF
July 2021

Runner file for benchmarking td-FCI
'''

import td_fci
import td_dmrg
import siam_current
import plot

import numpy as np
import matplotlib.pyplot as plt


##################################################################################
#### 3_1_2 system is std benchmark

verbose = 4;
nleads = (2,2);
nelecs = (3,2); # half filling
nelecs_ASU = (sum(nelecs),0); # all spin up formalism
splots = ['Jtot','occ','delta_occ','Sz','E']; # which subplots to make

#time info
dt = 0.01;
tf = 0.2;

# benchmark with spin free, fci code, std inputs
fname = td_fci.SpinfreeTest(nleads, nelecs, tf, dt, phys_params = None, verbose = verbose);
#fname = "dat/SpinfreeTest/"+str(nleads[0])+"1"+str(nleads[1])+"_e"+str(sum(nelecs))+.npy"
plot.PlotObservables(fname, nleads = nleads, thyb = (1e-5,0.4), splots = splots);

# test ASU, dmrg code with std inputs
datafile = siam_current.DotDataDmrg(nleads,nelecs_ASU,tf,dt,phys_params = None, verbose = verbose);
#datafile = "dat/DotDataDMRG/"+str(nleads[0])+"_1_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_B0_t0_Vg-0.5.npy"
plot.PlotObservables(datafile, nleads = nleads, splots = splots);



