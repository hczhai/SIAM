'''
Plotting module for quick methods of making matplotlib plots in pyscf context
'''

from errors import *
import numpy as np
import matplotlib.pyplot as plt
import pyscf as ps

def BasisPlot(basisdict, labels, comparediff=False):
    '''
    Given a dict, with
        -keys being string for the basis used in the calc
        -vals being an array of indep var vs energy
    Make line plots comparing E vs indep var for each basis on same figure
    
    Args:
        basis dict, dict
        labels, tuple of strings for xlabel, ylabel, title
    '''
    
    # check inputs
    if( type(basisdict) != type(dict()) ): # check that basisdict is a dict
        raise PlotTypeError("plot.BasisPlot 1st arg must be dictionary.\n");
    if( type(labels) != type([]) ): # check that labels is a list
        raise PlotTypeError("plot.BasisPlot 2nd arg must be a list.\n");
    while( len(labels) < 3): # add dummy labels until we get to three
        labels.append('');
    
    
    # make figure
    # 2 axes required if doing an energy difference comparison
    if(comparediff):
        fig, axs = plt.subplots(2, 1, sharex=True);
        ax, ax1 = axs
    else:
        fig, ax = plt.subplots();
    
    # plot data for each entry in basis dict
    for b in basisdict: # keys are strings
    
        ax.plot(*basisdict[b], label = b);
        
    # plot E diff if asked
    if(comparediff):
    
        #choose baseline data
        baseline = list(basisdict)[0]; # essentially grabs a random key
        basedata = basisdict[baseline][1]; # get E vals that we subtract
    
        # now go thru and get E diff in each case
        for b in basisdict:
        
            data = basisdict[b][1];
            x, y = basisdict[b][0], data - basedata;
            ax1.plot(x, y, label = b);
            
        # format ax1
        ax1.set(xlabel = labels[0], ylabel = labels[1] + " Difference");
        
    # format and show
    ax.set(xlabel = labels[0], ylabel = labels[1], title=labels[2]);
    ax.legend();
    plt.show();

    return;
