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
nelecs = (1,1);
epsilon = 5.0 # on site energy
t = 3.0 # hopping # must be float
U = 0.0; # hubbard repulsion strength

# init h1e and eri
h1e = np.zeros((norbs,norbs));
h2e = np.zeros((norbs, norbs, norbs, norbs));

# input on site
h1e[0,0] = epsilon;
h1e[1,1] = epsilon-1;

# implement FCISolver object
cisolver = fci.direct_spin1.FCI();
cisolver.max_cycle = 100; # max number of iterations
cisolver.conv_tol = 1e-8; # energy convergence

# kernel takes (1e ham, eri, num orbitals, num electrons
# returns eigvals and eigvecs of ham (in what basis?)
E_fci, v_fci = cisolver.kernel(h1e, h2e, norbs, nelecs, nroots = 4); #since spin is none, nelecs=1,1
if(verbose):
    print("\n1. nelecs = ",nelecs);
    print("FCI energies = ", E_fci);
    print(v_fci);
