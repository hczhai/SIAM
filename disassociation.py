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

def DissEnergy(atom, Rvals, basis, UHF = True):
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
        print("diatom = ", diatom);
        
        mol = ps.gto.Mole(unit = "Bohr", atom = diatom, basis = basis); # creates molecule object, using au
    
        # find HF energy with (un)restricted hartree fock
        if(UHF):
            E_diatom = ps.scf.UHF(mol).kernel();
        else:
            E_diatom = ps.scf.RHF(mol).kernel();
        
        #monatom geometry
        monatom = atom+' 0 0 0;';
        mon = ps.gto.Mole(unit = "Bohr", atom = monatom, spin = 1, basis = basis); # creates molecule object, using au
        E_monatom = ps.scf.UHF(mon).kernel();
        
        # subtract for diss energy and store
        Evals[i] = E_diatom - 2*E_monatom;
    
    return Rvals, Evals ; #### end diss energy
    



    
    
#############################################################################
#### wrappers and test funcs

def DiatomicDissWrapper():

    # define inputs
    atom = 'F'; # ie make H2 molecule
    Rvals = (0.5,5,10); # Rmin, Rmax, # R pts
    labels = ["Bond Length (Bohr)", "Energy (Rydberg)", "Disassociation Curve by Basis Set"]; # for plot
    
    # get RHF data
    R, E_RHF = DissEnergy(atom, Rvals, "sto-6g", UHF= False);
    #plot.GenericPlot(R, E_RHF, labels);
    
    # get UHF data
    R, E_UHF = DissEnergy(atom, Rvals, "sto-6g");
    plot.GenericPlot(R, E_UHF, labels);
    
    return;#### end wrapper
    
    

    
    

#############################################################################
#### execute code

if __name__ == "__main__":

    DiatomicDissWrapper();
