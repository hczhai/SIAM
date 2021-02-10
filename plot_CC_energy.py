'''
plot_energy.py

Routines for calculating CC approx energies, including correlation, for a given molecular geometry
'''

import plot
import utils

import numpy as np
import matplotlib.pyplot as plt
import pyscf as ps

#############################################################################
#### energy vs bond length in a given basis

def DiatomicCorrelEnergyVsR(atom, Rvals, basis):
    '''
    For a given diatomic molecule, find the ground state and correl over range of bond lengths
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
    corrvals= np.zeros(len(Rvals) );
        
    # so Rvals is definitely a mesh of Rvals by now
    # can run thru it and get E's
    for i in range(len(Rvals)):
    
        R = Rvals[i];
        print( "R = "+str(R))
        mol = ps.gto.Mole(unit = "Bohr"); # creates molecule object, using au
        mol.verbose = 0; # how much printout
    
        # specify the geometry
        atomstring = atom+' 0 0 0; '+atom + ' 0 0 '+str(R); #watch spacing
        print("atomstring = ", atomstring);
        mol.atom = atomstring;
    
        # find HF energy
        EHF = mol.RHF().run();
        Evals[i] = EHF.kernel();
        
        # find correl energy using CCSD
        ECC = EHF.CCSD().run();
        corrvals[i] = ECC.e_corr;
    
    return Rvals, Evals, Corrvals; #### end diatomic energy
    
    
#############################################################################
#### computing the energy across multiple basis sets

def EnergyWithCorrel(f, fargs, bases, labels):
    '''
    Using a given energy function, specifies the energy of the geo vs the indep var
    (as specified by fargs) for multiple basis, and plots comparison
    
    Args:
    f, function pointer, which returns tuple of 1d arrays indep var vals, E vals
        code checks that the return vals are ok, raises TypeError otherwise
    fargs, tuple of all args to pass to f for it to run
    labels, for passing to plotting routine
    
    returns none
    '''
    
    # for debugging
    fname = "EnergyWithCorrel"

    
    # make dict to hold output E vs R data for each basis
    d=dict()
    
    # compute energy for each basis
    for method in ["RHF","CCSD"]:
        
        print("inputs = ",fargs, "method = ", method);
    
        # run func that gets energy
        data = f(*fargs, b);
        
        # check function has correct output
        if( type(data) != type( (1,1) ) ):
            raise TypeError("Energy function f in "+fname+" does not return tuple.");
        elif( type(data[0]) != type(np.ones(0) ) ):
            raise TypeError("Energy function f in "+fname+" does not return tuple of np arrays");
    
        # put results (tuple of 3 np arrays) in dict
        d[method] = data;
    
    # call plotter
    plot.BasisPlot(d, labels);
    
    return; #### end EnergyByBasis
    
    
#############################################################################
#### wrappers and test funcs

def DiatomicWrapper():

    # define inputs
    f=DiatomicEnergyVsR; # function to call for energies
    atom = 'H'; # ie make H2 molecule
    Rvals = (1.2,1.6,10); # Rmin, Rmax, # R pts
    fargs = atom, Rvals;
    bases = ['sto-3g','ccpvdz']; # which bases to use
    labels = ["Bond Length (Bohr)", "Energy (Rydberg)", "Disassociation Curve by Basis Set"]; # for plot
    
    EnergyByBasis(f, fargs, bases, labels);
    
    return;#### end wrapper
    
    
def CCSDExample():

    print("\nSimple CCSD example with H2.");

    # set up geometry of the H2 molecule
    mol = ps.M(
        unit = 'Bohr',
        atom = 'H 0 0 0; H 0 0 1.4', # near gd state
        basis = 'ccpvdz',
        verbose = 3 ); #supress output for this example
    print("\nH2 geometry =\n",mol.atom_coords() );

    # use restricted HF to approx energy of HF gd state
    EHF = mol.RHF().run();
    print("\nEHF = mol.RHF().run() computes the HF gd state energy.",
    "\nAccess it using EHF.kernel(): E_HF = %.5g" %EHF.kernel() );

    # use coupled cluster singles doubles to approx correlation energy
    ECC = EHF.CCSD().run();
    print("\nECC = mf.CCSD().run() computes the correlation energy.",
    "\nAccess it using ECC.e_corr: ECC = %.5g" %ECC.e_corr );
    
    
    
    

#############################################################################
#### execute code

if __name__ == "__main__":

    DiatomicWrapper();

    
    
