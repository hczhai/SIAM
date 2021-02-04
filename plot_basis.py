'''
playground.py

Mess around with pyscf

https://sunqm.github.io/pyscf/tutorial.html
'''

import plot

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
        print("atomstring",atomstring);
        mol.atom = atomstring;
    
        # find HF energy
        m= ps.scf.RHF(mol);
        Evals[i] = m.kernel();
    
    return Rvals, Evals ; #### end diatomic energy
    
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
    labels = "Bond Length", "Energy", "Disassociation Curve by Basis Set";
    labels = ["bad"];
    plot.BasisPlot(d, labels);
    
    

#############################################################################
#### execute code

if __name__ == "__main__":

    x = np.array([1,2,3,4]);
    y = np.ones(4);
    print(x-y);
    print(x);

    DiatomicEnergyWrapper();

    #run in spin singlet and triplet states
    #mol.build(atom = '''O 0 0 0; O 0 0 1.2''', basis = 'ccpvdz');

    # HF energy
    #m = ps.scf.RHF(mol);
    #print('E(HF) = %g' % m.kernel())


    
    
