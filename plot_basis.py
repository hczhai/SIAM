'''
playground.py

Mess around with pyscf

https://sunqm.github.io/pyscf/tutorial.html
'''

import plot
import utils

import numpy as np
import matplotlib.pyplot as plt
import pyscf as ps

#############################################################################
#### energy vs bond length in a given basis

def DiatomicEnergyVsR(atom, basis, Rvals):
    '''
    For a given diatomic molecule, find the ground state energy in a given basis set, over a range of bond lengths
    Args:
    -atom, string of atomic name to input to pyscf mol constructor
    -basis, string of basis name to input to pyscf mol constructor
    -Rvals, specifies range of R, can be list of vals to take, or tuple (Rmin, Rmax, # pts)
    Returns tuple of 1d np arrs: bond lengths (Rvals) and energies (Evals)
    '''
    
    # iter over different R vals
    #sort by data type:
    if( type(Rvals) == type( (1,1)) ): # make array from tuple
        Rvals = np.linspace(Rvals[0], Rvals[1], Rvals[2]);
        
    # return object is np arr
    Evals = np.zeros(len(Rvals) );
        
    # so Rvals is definitely a mesh of Rvals by now
    # can run thru it and get E's
    for i in range(len(Rvals)):
    
        R = Rvals[i];
        print( "R = "+str(R))
        mol = ps.gto.Mole(); # creates molecule object
        mol.verbose = 0; # how much printout
    
        # specify the geometry
        atomstring = atom+' 0 0 0; '+atom + ' 0 0 '+str(R); #watch spacing
        print("atomstring = ", atomstring);
        mol.atom = atomstring;
    
        # find HF energy
        m= ps.scf.RHF(mol);
        Evals[i] = m.kernel();
    
    return Rvals, Evals ; #### end diatomic energy
    
    
def GeneralEnergyVsR(geo, basis, coordkey, coordvals):
    '''
    For a given molecule, find the ground state energy in a given basis set, over range of different coordinates for a certain atom
    Args:
    -geo, standard dict of elems to cartesian coords
    -basis, string of basis name to input to pyscf mol constructor
    -coordkey, elem name string, tells which element in mol to vary position of
    
    Returns tuple of 1d np arrs: indep var coords and energies (Evals)
    '''

    #return object is np arr
    Evals = np.zeros(len(Rvals) );
    
    # construct atomstring from atoms and coords
    d = {'H' : np.ones(3), 'O' : np.zeros(3)}
    atomstring = utils.ParseGeoDict(d);
    print("atomstring = ", atomstring);
    mol = ps.gto.Mole(atom=atomstring);
    
    
#############################################################################
#### wrappers and test funcs

def DiatomicEnergyWrapper():

    print("Executing Diatomic Energy Vs R")

    # def inputs
    atom = 'H';
    Rvals = (0.5,1.5,20);
    
    # make dict to hold output E vs R data for each basis
    d=dict()
    
    # do multiple bases
    bases = ['ccpvdz', 'sto-3g', 'sto-6g'];
    for b in bases:
        
        print("inputs = ",atom,b,Rvals);
    
        # run func
        data = DiatomicEnergyVsR(atom, b, Rvals);
    
        # put results in dict
        d[b] = data;
    
    # call basisplot
    labels = ["Bond Length", "Energy", "Disassociation Curve by Basis Set"];
    labels = ["bad"];
    plot.BasisPlot(d, labels);
    
    
def GeneralEnergyWrapper():

    GeneralEnergyVsR();
    
    

#############################################################################
#### execute code

if __name__ == "__main__":

    #DiatomicEnergyWrapper();
    GeneralEnergyWrapper();

    
    
