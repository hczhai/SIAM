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
splots = ['Jtot','occ','delta_occ','Sz']; # which subplots to make
B = 5.0
theta = 0.0
Rlead_pol = 1;

#time info
dt = 0.01;
tf = 5.0;

# benchmark with spin free, fci code, std inputs
#fname = td_fci.SpinfreeTest(nleads, nelecs, tf, dt, phys_params = None, verbose = verbose);
#fname = "dat/SpinfreeTest/"+str(nleads[0])+"1"+str(nleads[1])+"_e"+str(sum(nelecs))+".npy"
#plot.PlotObservables(fname, nleads = nleads, thyb = (1e-5,0.4), splots = splots);


# ASU fci code
datafile = "dat/DotData/spinpol/"+str(nleads[0])+"_1_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_B"+str(B)+"_t"+str(theta)+"_Vg-0.5.npy"
#plot.PlotObservables(datafile, nleads = nleads, splots = splots);

# test ASU, dmrg code with std inputs

params = 1.0, 0.4, -0.005, 0.0, -0.5, 1.0, 5.0, 0.0
#siam_current.DotDataDmrg(nleads,nelecs_ASU,tf,dt,phys_params = params, Rlead_pol = Rlead_pol, prefix="spinpol/", verbose = verbose);
datafile = "dat/DotDataDmrg/spinpol/"+str(nleads[0])+"_1_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_B"+str(B)+"_t"+str(theta)+"_Vg-0.5.npy"
plot.PlotObservables(datafile, nleads = nleads, splots = splots);





