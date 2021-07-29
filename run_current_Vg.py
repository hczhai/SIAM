'''
Christian Bunker
M^2QM at UF
July 2021

Runner file for getting, analyzing, plotting data for a dot
(1 site impurity w/ Vg, U) across sweep of Vg values

'''

import plot
import siam_current

import numpy as np

#################################################
#### inputs

# top level inputs
get_data = False;
plot_J = True;
plot_Fourier = False;
verbose = 5;
splots = ['Jtot','Sz'];

# time
tf = 5.0;
dt = 0.01;

# physical inputs
nleads = (2,2);
nimp = 1;
nelecs = (sum(nleads) + nimp, 0);
Vgs = [-0.5,-0.25, 0.0,0.25, 0.5];  # should be list
B = 5.0; # prep dot in down state always. NB starting thyb at 1e-5

#################################################
#### get data for current thru dot

if get_data:
    for Vg in Vgs:
        myparams = 1.0, 0.4, -0.005, 0.0, Vg, 1.0, B, 0.0; # std inputs except Vg
        siam_current.DotData(nleads, nelecs, tf, dt, phys_params = myparams, prefix = "VgSweep/", verbose = verbose);


#################################################
#### plot current data

# plot inputs
title = "Dot impurity:\n"+str(nleads[0])+" left sites, "+str(nleads[1])+" right sites, $t_{hyb} = 10^{-5}$ -> 0.4, B = "+str(B); 

if plot_J:
    # plot J vs t, E vs t, fourier freqs, ind'ly or across Vg, mu sweep
    folder = "dat/DotData/VgSweep/" # where data is stored
    plot.CurrentPlot(folder, nleads, nimp, nelecs, Vgs, B, 0.0, splots=splots, mytitle = title);


if plot_Fourier:
    # plot J vs t with energy spectrum, ∆E -> w, and fourier modes
    plot.FourierEnergyPlot(folder, nleads, nimp, nelecs, mus, Vgs, energies, mytitle = title);









