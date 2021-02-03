'''
Plotting module for quick methods of making matplotlib plots in pyscf context
'''

import numpy as np
import matplotlib as plt
import pyscf as ps

def BasisPlot(basisdict, ind_var):
    '''
    Given a dict, with
        -keys being string for the basis used in the calc
        -vals being an array of indep var vs energy
    Make line plots comparing E vs indep var for each basis on same figure
    
    Args:
        basis dict, dict
        ind_var, string for x axis of plot
    '''

    return;
