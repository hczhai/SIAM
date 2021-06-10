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

# system inputs
nelecs = 2; #always
norbs = 2;
J = 1; #spin interaction strength

# implement h1e
# all terms are 2e terms
h1e = np.zeros((norbs, norbs));

# h2e
h2e = np.zeros((norbs, norbs, norbs, norbs));

# solve with FCISolver object
cisolver = fci.direct_spin1.FCI()
cisolver.max_cycle = 100
cisolver.conv_tol = 1e-8
e, fcivec = cisolver.kernel(h1e, h2e, norbs, nelecs);

