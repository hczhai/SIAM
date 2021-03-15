'''
disassociation.py

Plotting the dissassociation energy (ie, energy of diatom - 2*energy of monatom)
vs R for different diatoms. Showing necessity of UHF methods
'''

import plot
import utils

import numpy as np
import matplotlib.pyplot as plt
import pyscf as ps

#############################################################################
#### energy by different methods

def DissEnergyRHF(atom, Rvals, basis):
    '''
    For a given diatomic molecule, find the ground state energy in a given basis set, over a range of bond lengths, with restricted hartree fock. Then
        subtract 2*monatomic to get diss energy
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
    
        # specify the diatom geometry
        R = Rvals[i];
        print( "R = "+str(R));
        diatom = atom+' 0 0 0; '+atom + ' 0 0 '+str(R); #watch spacing
        print("diatom = ", atomstring);
        
        mol = ps.gto.Mole(unit = "Bohr", atom = diatom, basis = basis); # creates molecule object, using au
    
        # find HF energy with restricted hartree fock
        E_diatom = ps.scf.RHF(mol).kernel();
        
        #monatom geometry
        monatom = atom+' 0 0 0;';
        mon = ps.gto.Mole(unit = "Bohr", atom = monatom, basis = basis); # creates molecule object, using au
        E_monatom = ps.scf.RHF(mon).kernel();
        
        # subtract for diss energy and store
        Evals[i] = Ediatom - 2*E_monatom;
    
    return Rvals, Evals ; #### end diatomic RHF
    
    
#############################################################################
#### wrappers and test funcs

def DiatomicEnergyWrapper():

    # define inputs to EByBasis
    f=DiatomicEnergyVsR; # function to call for energies
    atom = 'H'; # ie make H2 molecule
    Rvals = (1.2,1.6,10); # Rmin, Rmax, # R pts
    fargs = atom, Rvals;
    bases = ["sto-3g", "sto-6g", "ccpvdz"]; # which bases to use
    labels = ["Bond Length (Bohr)", "Energy (Rydberg)", "Disassociation Curve by Basis Set"]; # for plot
    
    EnergyByBasis(f, fargs, bases, labels);
    
    return;#### end wrapper
    
    

    
    

#############################################################################
#### execute code

if __name__ == "__main__":

    DiatomicEnergyWrapper();
