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
norbs = 2;
nelecs = 2;
epsilon = 0 # on site energy
t = 3 # hopping
U = 2000; # hubbard repulsion strength


# set up 1e hamiltonian (on site and hopping)
# ie h_pq matrix elems
h1 = np.array([[epsilon, t],[t, epsilon]]);

# set up 2e hamiltonian (hubbard)
# ie v_pqrs matrix elems
h2 = np.zeros((norbs, norbs, norbs, norbs));
h2[0,0,0,0] = U;
h2[1,1,1,1] = U;

# implement FCISolver object
cisolver = fci.direct_spin1.FCI();
cisolver.max_cycle = 100; # max number of iterations
cisolver.conv_tol = 1e-8; # energy convergence

# kernel takes (1e ham, 2e ham, num orbitals, num electrons
# returns eigvals and eigvecs of ham (in what basis?)
E_fci, v_fci = cisolver.kernel(h1, h2, norbs, nelecs); #since spin is none, nelecs=1,1
if(verbose):
    print("\n1. nelecs = 1,1");
    print("FCI energies = ", E_fci);
    print(v_fci);
    
# repeat w spin pol
# na, nb = 1,1 gives same as above but 2,0 gives E = 0
cisolver = fci.direct_spin1.FCI();
E_fci, v_fci = cisolver.kernel(h1, h2, norbs, (2,0));
if(verbose):
    print("\n2. nelecs = 2,0");
    print("FCI energies = ", E_fci);
    print(v_fci);

# should work same as first example
cisolver0 = fci.direct_spin0.FCI()
e_fci0, v_fci0 = cisolver.kernel(h1, h2, norbs, nelecs);
if(verbose):
    print("\n3. spin0 solver");
    print("FCI energies = ", e_fci0);
    print(v_fci0);
    
# for some reason when nroots =2 both energies output to 0, why?
cisolver0 = fci.direct_spin0.FCI()
e_fci0, v_fci0 = cisolver.kernel(h1, h2, norbs, nelecs, nroots =2);
if(verbose):
    print("\n4. nroots = 2");
    print("FCI energies = ", e_fci0);
    print(v_fci0);

