'''
playground.py

Mess around with pyscf

https://sunqm.github.io/pyscf/tutorial.html
'''

import numpy as np
import matplotlib as plt
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
    '''

    mol = ps.gto.Mole(); # creates molecule object
    mol.verbose = 0; # how much printout
    
    # specify the geometry
    atomstring = atom+' 0 0 0; '+atom + ' 0 0 '+str(Rvals[0]); #watch spacing
    print("atomstring",atomstring);
    mol.atom = atomstring;
    
    # find HF energy
    m = ps.scf.RHF(mol);
    print('E(HF) = %g' % m.kernel())
    
    return; #### end diatomic energy
    
#############################################################################
#### wrappers and test funcs

def DiatomicEnergyWrapper():

    print("Executing Diatomic Energy Vs R")

    # def inputs
    atom = 'H';
    basis = 'ccpvdz';
    Rvals = (1, 2, 3);
    
    print("inputs = ",atom,basis,Rvals);
    
    # run func
    DiatomicEnergyVsR(atom, basis, Rvals);
    

#############################################################################
#### execute code

if __name__ == "__main__":

    DiatomicEnergyWrapper();

    #run in spin singlet and triplet states
    #mol.build(atom = '''O 0 0 0; O 0 0 1.2''', basis = 'ccpvdz');

    # HF energy
    #m = ps.scf.RHF(mol);
    #print('E(HF) = %g' % m.kernel())


    
    
