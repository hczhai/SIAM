#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Template:
Solve FCI problem with given 1-electron and 2-electron Hamiltonian

Secific problem:
2 site hubbard model
'''

import numpy as np
from pyscf import fci

verbose = True;

# system inputs
# hamiltonian params must be floats
norbs = 2;
nelecs = 2;
epsilon = 0 # on site energy
t = 3.0 # hopping
U = 200.0; # hubbard repulsion strength


# implement h1e and h2e
# doing it this way forces floats which is a good fail safe
h1e = np.zeros((norbs,norbs));
h2e = np.full((norbs,norbs,norbs,norbs),U);

# put in on site energy
h1e[0,0], h1e[1,1] = epsilon, epsilon;

# hopping
h1e[0,1], h1e[1,0] = t, t;

# hubbard
h2e[0,0,0,0] += U;
h2e[1,1,1,1] += U;

# implement FCISolver object
cisolver = fci.direct_spin1.FCI();
cisolver.max_cycle = 100; # max number of iterations
cisolver.conv_tol = 1e-8; # energy convergence

# kernel takes (1e ham, 2e ham, num orbitals, num electrons
# returns eigvals and eigvecs of ham (in what basis?)
E_fci, v_fci = cisolver.kernel(h1e, h2e, norbs, nelecs); #since spin is none, nelecs=1,1
if(verbose):
    print("\n1. nelecs = 1,1"); # this matches analytical gd state
    print("FCI energies = ", E_fci);
    print(v_fci);
    
#### tests
    
# repeat w spin pol
# na, nb = 1,1 gives same as above but 2,0 gives E = 0
newnelecs = (2,1)
cisolver = fci.direct_spin1.FCI();
E_fci, v_fci = cisolver.kernel(h1e, h2e, norbs, newnelecs,nroots=4);
if(verbose):
    print("\n2. nelecs = ",newnelecs);
    print("FCI energies = ", E_fci);
    print(v_fci);

# should work same as first example
cisolver0 = fci.direct_spin0.FCI()
e_fci0, v_fci0 = cisolver.kernel(h1e, h2e, norbs, nelecs);
if(verbose):
    print("\n3. spin0 solver");
    print("FCI energies = ", e_fci0);
    print(v_fci0);
    
# still matches analytical case with multiple roots
cisolver0 = fci.direct_spin0.FCI()
e_fci0, v_fci0 = cisolver.kernel(h1e, h2e, norbs, nelecs, nroots =4);
if(verbose):
    print("\n4. extra roots");
    print("FCI energies = ", e_fci0-epsilon);
    print(v_fci0);
    

