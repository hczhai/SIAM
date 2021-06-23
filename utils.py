'''
Functions which are generally helpful in the pyscf enviro
'''

import numpy as np
import matplotlib.pyplot as plt
import pyscf as ps
from pyscf import fci

# useful reference - grab element by atomic number
PeriodicTable = [None,'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr']

##############################################################
#### measuring energy eigenstates

def Spin_exp(vs, norbs, nelecs):
    '''
    contract vector with S operator in h1e form to measure spin
    Args:
    - vs, array of fci vectors from FCI calc
    - norbs, int, num of spin orbs
    - nelecs, int, num of electrons
    returns <S>
    '''
    
    # return val
    S_exp = [];

    # make S operators
    Sx = np.zeros((norbs,norbs));
    for i in range(0,norbs,2): # i is spin up s=orb, i+1 is spin down
        Sx[i,i+1] = 1/2;
        Sx[i+1,i] = 1/2; # h.c.
    Sy = np.zeros((norbs,norbs));
    for i in range(0,norbs,2):
        Sy[i,i+1] = -1/2;
        Sy[i+1,i] = 1/2;
    print(Sy)
    Sz = np.zeros((norbs,norbs));
    for i in range(0,norbs,2):
        Sz[i,i] = 1/2;
        Sz[i+1,i+1] = -1/2;

    S_op = [Sx,Sy,Sz]; # spin operator

    # S squared operator
    S2 = np.zeros((norbs,norbs,norbs,norbs));
    S2[0,1,1,0] = 1;
    S2[1,0,0,1] = 1;
    S2[2,3,3,2] = 1;
    S2[3,2,2,3] = 1;
    
    # iter over vectors
    for v in vs:
    
        S_exp_val = np.zeros(3);

        #contract with each element of S operator
        for Si in range(len(S_op)):

            # use contract_1e function
            result = fci.direct_nosym.contract_1e(S_op[Si], v, norbs, nelecs);
            S_exp_val[Si] = np.dot(v.T, result);
          
        # done with this vector
        S_exp.append(S_exp_val);

    return S_exp; # end spin exp








##############################################################
#### constructing atom geometry dicts

# NB pyscf geometries default to angstrom, so always set unit="Bohr"
# however mole.atom_coords() returns geometry array in Bohr always

# reconfig since mole.atom takes lists, arrays !!

def MonatomicChain(el, n, R, axis = 'x'):
    '''
    Chain of same element atoms separated by distance R
    Args:
    - el, string of eement name
    - n, int, number of atoms in chain
    - R, double, distance between atoms
    - axis, string, tells which axis to run chain along
    '''
    
    # check inputs
    if(el not in PeriodicTable): # only take element names
        raise ValueError("Unsupported element passed to geo constructor\n");
    if(axis not in ['x','y','z']): # just reassign back to default
        axis = 'x';
        
    # make axis an int
    axis = ord(axis) - ord('x'); #map to ascii and subtract to get 0,1,2
    
    # return var is dict
    d = dict();
    
    for i in range(n):
    
        # construct coords based on which axis chain is along
        coords = np.zeros(3);
        coords[axis] = R*i; #update with correct number chain lengths
        d[el+str(i)] = coords
        
    return d;


##############################################################
#### interpreting atom geometry dicts

def ParseGeoDict(d):
    '''
    Given a dictionary of element names : cartesian coordinates np array,
    turn into a string formatted to pass to the mol.atom attribute
    '''
    
    # return value is string of geometry
    atomstring = '';
    
    # iter over elements aka keys
    for el in d:
    
        # get coords
        coords = d[el];
        coords = str(coords)[1:-1]; # get space seperated #s with [s deleted
        
        # update string
        atomstring += " "+el+" "+coords+";"
        
    return atomstring;


    
    
    
    
##############################################################
#### test code

def TestCode():

    print(MonatomicChain('H',5, 1));
    
    
if __name__ == "__main__":

    TestCode();

