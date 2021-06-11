#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Template:
Solve FCI problem with given 1-electron and 2-electron Hamiltonian

Specific case:
Solve 2 spin heisenberg hamiltonian
'''

import numpy as np
from pyscf import fci

verbose = True;

# system inputs
# ham params should be floats
nelecs = (1,1);
norbs = 2;
J = 1.0; #spin interaction strength

# implement h1e, h2e
h1e = np.zeros((norbs, norbs));
h2e = np.zeros((norbs, norbs,norbs,norbs));

# 1e terms

# all terms are 2e terms
h2e[0,0,1,1] = -J/4;
h2e[1,1,0,0] = -J/4;
h2e[0,1,1,0] = J/8;
h2e[1,0,0,1] = J/8;
h2e[0,0,0,0] = J/4;
h2e[1,1,1,1] = J/4;

# solve with FCISolver object
cisolver = fci.direct_spin1.FCI()
cisolver.max_cycle = 100
cisolver.conv_tol = 1e-8
E_fci, v_fci = cisolver.kernel(h1e, h2e, norbs, nelecs,nroots=4);
if(verbose):
    print("\n1. nelecs = ",nelecs); # this matches analytical gd state
    print("FCI energies = ", E_fci);
    print(v_fci);

