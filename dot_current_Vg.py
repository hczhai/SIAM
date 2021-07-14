'''
Christian Bunker
M^2QM at UF
July 2021

Runner file for getting, analyzing, plotting data for a dot
(1 site impurity w/ Vg, U) across sweep of Vg values

'''

import plot
import siam
import siam_current
import ruojings_td_fci

import numpy as np
import matplotlib.pyplot as plt

#################################################
#### inputs

# top level inputs
get_data = False;
plot_J = True;
plot_Fourier = True;
verbose = 5;
folder = "dat/DotCurrentData/" # where data is stored

# physical inputs
nleads = (3,3);
nimp = 1;
nelecs = (sum(nleads) + nimp, 0);
mu = [0]; # should be lists
Vg = [-0.75,-0.5,-0.25];
energies = [1,2,3];

#################################################
#### get data for current thru dot (INCOMPLETE, but data already mostly collected)

if get_data:
    siam_current.DotCurrentData(nleads, nelecs, timestop, deltat, mu, V_gate, prefix = "", verbose = verbose);


#################################################
#### plot current data

# plot inputs
title = "Dot with "+str(nleads[0])+" lead sites on each site"

if plot_J:
    # plot J vs t, E vs t, fourier freqs, ind'ly or across Vg, mu sweep
    plot.CurrentPlot(folder, nleads, nimp, nelecs, mu, Vg, mytitle = title);


if plot_Fourier:
    # plot J vs t with energy spectrum, ∆E -> w, and fourier modes
    plot.FourierEnergyPlot(folder, nleads, nimp, nelecs, [mu[0]], [Vg[0]], energies, mytitle = title);









