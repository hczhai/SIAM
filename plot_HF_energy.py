'''
plot_energy.py

Routines for calculating HF approx energies, for a given molecular geometry
-against independent variables
-comparing different bases
'''

import plot
import utils

import numpy as np
import matplotlib.pyplot as plt
import pyscf as ps

#############################################################################
#### energy vs bond length in a given basis

def DiatomicEnergyVsR(atom, Rvals, basis):
    '''
    For a given diatomic molecule, find the ground state energy in a given basis set, over a range of bond lengths
    Args:
    -atom, string of atomic name to input to pyscf mol constructor
    -Rvals, specifies range of R, can be list of vals to take, or tuple (Rmin, Rmax, # pts)
    -basis, string of basis name to input to pyscf mol constructor (last arg for *construction)
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
    
        # specify the geometry
        R = Rvals[i];
        print( "R = "+str(R));
        atomstring = atom+' 0 0 0; '+atom + ' 0 0 '+str(R); #watch spacing
        print("atomstring = ", atomstring);
        
        mol = ps.gto.Mole(unit = "Bohr", atom = atomstring, basis = basis); # creates molecule object, using au
    
        # find HF energy
        EHF = ps.scf.RHF(mol);
        Evals[i] = EHF.kernel();
    
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
#### computing the energy across multiple basis sets

def EnergyByBasis(f, fargs, bases, labels):
    '''
    Using a given energy function, specifies the energy of the geo vs the indep var
    (as specified by fargs) for multiple basis, and plots comparison
    
    Args:
    f, function pointer, which returns tuple of 1d arrays indep var vals, E vals
        code checks that the return vals are ok, raises TypeError otherwise
    fargs, tuple of all args to pass to f for it to run
    bases, list of strings, with basis names to pass to ps
    labels, for passing to plotting routine
    
    returns none
    '''

    
    # make dict to hold output E vs R data for each basis
    d=dict()
    
    # compute energy for each basis
    for b in bases:
        
        print("inputs = ",fargs, "basis = ", b);
    
        # run func that gets energy
        data = f(*fargs, b);
        
        # check function has correct output
        if( type(data) != type( (1,1) ) ):
            raise TypeError("Energy function f in EnergyByBasis does not return tuple.");
        elif( type(data[0]) != type(np.ones(0) ) ):
            raise TypeError("Energy function f in EnergyByBasis does not return tuple of np arrays");
    
        # put results in dict
        d[b] = data;
    
    # call basisplot
    
    plot.BasisPlot(d, labels);
    
    return; #### end EnergyByBasis
    
    
#############################################################################
#### wrappers and test funcs

def DiatomicEnergyWrapper():

    # define inputs to EByBasis
    f=DiatomicEnergyVsR; # function to call for energies
    atom = 'H'; # ie make H2 molecule
    Rvals = (1.0,1.6,10); # Rmin, Rmax, # R pts
    fargs = atom, Rvals;
    bases = ["sto-3g", "sto-6g", "ccpvdz"]; # which bases to use
    labels = ["Bond Length (Bohr)", "Energy (Rydberg)", "Disassociation Curve by Basis Set"]; # for plot
    
    EnergyByBasis(f, fargs, bases, labels);
    
    return;#### end wrapper
    
    
def GeneralEnergyWrapper():

    GeneralEnergyVsR();
    

    
    

#############################################################################
#### execute code

if __name__ == "__main__":

    DiatomicEnergyWrapper();

    
    
