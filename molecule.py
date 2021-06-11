#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Template:
Solve FCI problem with given 1-electron and 2-electron Hamiltonian

Specific case:
Solve molecule
'''

import numpy as np
from pyscf import fci

verbose = True;

# system inputs
# ham params should be floats
nelecs = (2,2);
norbs = 3;
alpha = 0.0;
D = 20;
E = 10;
U = 1000;

# implement h1e, h2e
h1e = np.zeros((norbs, norbs));
h2e = np.zeros((norbs, norbs,norbs,norbs));

# single electron terms
h1e[0,1] = 2*alpha;
h1e[1,2] = 2*alpha;
h1e[1,0] = 2*alpha;
h1e[2,1] = 2*alpha;
h1e[0,0] = -D;
h1e[2,2] = -D;
h1e[0,2] = 2*E;
h1e[2,0] = 2*E;

# double electron terms
h2e[0,0,0,0] = U;
h2e[1,1,1,1] = U;
h2e[2,2,2,2] = U;

# solve with FCISolver object
cisolver = fci.direct_spin1.FCI()
cisolver.max_cycle = 100
cisolver.conv_tol = 1e-8
E_fci, v_fci = cisolver.kernel(h1e, h2e, norbs, nelecs,nroots=15);
E_fci.sort();
if(verbose):
    np.set_printoptions(suppress=True); # no sci notatation printing
    print("\n1. nelecs = ",nelecs); # this matches analytical gd state
    print("FCI energies = ",E_fci - (U-2*D));
    
    
    
#### get correct energies

# from diag'd T sector
E_exact = np.array([0,U,0.5*U - 0.5*np.sqrt(U*U+64*E*E),0.5*U + 0.5*np.sqrt(U*U+64*E*E)]);

# from diag'd other sector
E_exact = np.append(E_exact,[-D+2*E,-D+2*E,-D-2*E,-D-2*E,-2*D+U]);

# sort and print
E_exact.sort();
if(verbose):
    print("\n2. Analytical solution")
    print("Exact energies = ",E_exact);
    print("Expected singlet energy, E/U-->0 = ",-16*E*E/U)



