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
    
    #monatom geometry
    monatom = atom+' 0 0 0;';
    mon = ps.gto.Mole(unit = "Bohr", atom = monatom, spin = 1, basis = basis, verbose = 0); # creates molecule object, using au
    mon.build();
    E_monatom = ps.scf.UHF(mon).kernel();
           
    # iter over Rs and get Es
    for i in range(len(Rvals)):
    
        # specify the diatom geometry
        R = Rvals[i];
        diatom = atom+' 0 0 0; '+atom + ' 0 0 '+str(R); #watch spacing
        print("diatom = ", diatom);
    
        # find HF energy with (un)restricted hartree fock
        mol = ps.gto.Mole(); #init molecule
        mol.unit = "Bohr";
        mol.atom = diatom;
        mol.basis = basis;
        mol.verbose = 0; #define traits
        
        if(UHF): #unrestricted case
            mol.verbose = 3;
            mol.spin=2;
            mol.build();
            E_diatom = ps.scf.UHF(mol).kernel();
        else: #restricted case
            mol.spin=0;
            mol.build();
            E_diatom = ps.scf.RHF(mol).kernel();
        
        # subtract monatom from diatom for diss energy and store
        Evals[i] = E_diatom - 2*E_monatom;
        
    if(UHF):
        print("UHF settles to RHF solution unless spin=2 set manually. See Szabo p. 225.\n");
    
    return Rvals, Evals ; #### end diss energy
    



    
    
#############################################################################
#### wrappers and test funcs

def DiatomicDissWrapper():

    # define inputs
    atom = 'H'; # ie make H2 molecule
    Rvals = (0.5,4,30); # Rmin, Rmax, # R pts
    labs = ["Bond Length (Bohr)", "Energy (Rydberg)", "RHF vs UHF dissassociation"]; # for plot
    
    # dashed hline at y=0 for ref
    hline = np.zeros(Rvals[2])
    styles = ['','','_']
    
    # legend handles
    legh = ["Restricted", "Unrestricted",""];
    
    # get energy data
    R, E_RHF = DissEnergy(atom, Rvals, "sto-3g", UHF= False); #restricted energies
    R, E_UHF = DissEnergy(atom, Rvals, "sto-3g"); # unrestricted
    energies = np.array([E_RHF, E_UHF, hline]); # combine y vals (energies)
    
    # pass bond lengths (x) energies (y) legend labels and plot labels to plotter
    plot.GenericPlot(R, energies, handles=legh, styles=styles, labels=labs);
    
    return;#### end diatomic dissassociation wrapper
    
    

    
    

#############################################################################
#### execute code

if __name__ == "__main__":

    DiatomicDissWrapper();
