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
norbs = 4;
nelecs = (1,1);
epsilon = 0 # on site energy
t = 3.0 # hopping # must be float
U = 20000.0; # hubbard repulsion strength

# init h1e and eri
h1e = np.zeros((norbs,norbs));
eri = np.zeros((norbs, norbs, norbs, norbs));

# input hopping
h1e[0,2]=t;
h1e[1,3]=t;
h1e[2,0]=t;
h1e[3,1]=t;

#input hubbard
eri[0,0,1,1]=U;
eri[1,1,0,0]=U;
eri[2,2,3,3]=U;
eri[3,3,2,2]=U;

# implement FCISolver object
cisolver = fci.direct_spin1.FCI();
cisolver.max_cycle = 100; # max number of iterations
cisolver.conv_tol = 1e-8; # energy convergence

# kernel takes (1e ham, eri, num orbitals, num electrons
# returns eigvals and eigvecs of ham (in what basis?)
E_fci, v_fci = cisolver.kernel(h1e, eri, norbs, nelecs, verbose=5); #since spin is none, nelecs=1,1
if(verbose):
    print("\n1. nelecs = ",nelecs);
    print("FCI energies = ", E_fci);
    print(v_fci);
