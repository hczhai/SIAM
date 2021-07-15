'''
Christian Bunker
M^2QM at UF
July 2021

Runner file for getting, analyzing, plotting data for a dot
(1 site impurity w/ Vg, U) across sweep of mu vals

'''

import plot
import siam
import siam_current
import siam_conductance
import ruojings_td_fci
import utils

import numpy as np
import matplotlib.pyplot as plt

#################################################
#### inputs

# top level inputs
compute_data = False;
plot_J = True;
plot_Fourier = False;
plot_conductance = True;
verbose = 5;
folder = "dat/DotCurrentData/" # where data is stored

# physical inputs
nleads = (3,3);
nimp = 1;
nelecs = (sum(nleads) + nimp, 0);
musweep = [-1.0, -0.875, -0.75, -0.625, -0.5, -0.375, -0.25, -0.125, 0.0]; # should be lists
Vg = [-0.5];

#################################################
#### get data for current thru dot (INCOMPLETE, but data already mostly collected)

if compute_data:
    siam_current.DotCurrentData(nleads, nelecs, timestop, deltat, musweep, V_gate, prefix = "", verbose = verbose);


#################################################
#### plot current data

# plot inputs
title = "Dot with "+str(nleads[0])+" lead sites on each side"

if plot_J:
    # plot J vs t, E vs t, fourier freqs, ind'ly or across mu sweep
    plot.CurrentPlot(folder, nleads, nimp, nelecs, musweep, Vg, mytitle = title, verbose = verbose);


if plot_Fourier:
    # plot J vs t with energy spectrum, ∆E -> w, and fourier modes
    # but for only one mu val
    mu_i = 0; # which mu to grab from list

    # get energies TODO: benchmark energies
    # location of data
    dataf = folder+str(nleads[0])+"_"+str(nimp)+"_"+str(nleads[1])+"_e"+str(nelecs[0])+"_mu"+str(musweep[mu_i])+"_Vg"+str(Vg[0])
    header = utils.TxtHeaderDict(dataf+"_J.txt"); # grabs dict of phys inputs from header
    params = header['V_leads'], header['V_imp_leads'], 0.0, header['mu'], header['V_gate'], header['U']; # unpacks dict
    h1e, g2e, H_imp = siam.dot_hams(nleads, nimp, nelecs, params, verbose = verbose); # ham arrays
    energies, dummy = siam.direct_FCI(h1e, g2e, 2*(sum(nleads)+nimp), nelecs, nroots = 50, verbose = verbose );

    # plot
    plot.FourierEnergyPlot(folder, nleads, nimp, nelecs, [musweep[mu_i]], [Vg[0]], energies, mytitle = title);


if plot_conductance:
    # plot fourier transform to get J(omega) then plot dJ(omega)/dmu
    siam_conductance.PlotConductance(folder, nleads, nimp, nelecs, musweep, Vg, mytitle = title);

    








