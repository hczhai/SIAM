#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Template:
Solve FCI problem with given 1-electron and 2-electron Hamiltonian

Specific case:
- Silas' model of molecule (SOC and spatial anisotropy included)
- n=3, l=2 sector: m= -2,...2 (d orbital, 10 spin orbitals)
- aim for spin 1 object hence 2 unpaired e's, 8 total e's
- basis size: (5 choose 2)*4 + (5 choose 1) = 45

Formalism:
- h1e = (p|h|q) p,q spatial orbitals
- h2e = (pq|h|rs) chemists notation, <pr|h|qs> physicists notation
- all direct_x solvers assume 4fold symmetry from sum_{pqrs} (don't need to do manually)
- 1/2 out front all 2e terms, so contributions are written as 1/2(2*actual ham term)
- other symmetries to be aware of:
      hermicity: h_pqrs = h_qpsr
      E_pr,qs = E_rp,sq from properties of E
'''

import numpy as np
from pyscf import fci

verbose = True;
np.set_printoptions(suppress=True); # no sci notatation printing

# parameters in the hamiltonian
alpha = 0.1;
D = 20.0;
E = 10.0;
U = 1000.0;
if(verbose):
    print("\nInputs:","\nalpha = ",alpha,"\nD = ",D,"\nE = ",E,"\nU = ",U);
    print("E/U = ",E/U,"\nalpha/(E^2/U) = ", alpha*U/(E*E) );

#### get analytical energies

# diagonalize numerically
E_exact = np.array([]);

# sort and print
E_exact.sort();
if(verbose):
    print("\n0. Analytical solution not yet implemented");
    
    
#### solve with spin blind method
nelecs = (8,0);
norbs = 10; # spin orbs
nroots = 15; # size of full basis

# implement h1e, h2e
h1e = np.zeros((norbs, norbs));
h2e = np.zeros((norbs, norbs,norbs,norbs));

# single electron terms
# best practice to use += thru out since we may modify elements twice
h1e[0,0] += -4*D; # z axis anisotropy
h1e[1,1] += -4*D;
h1e[2,2] += -D;
h1e[3,3] += -D;
h1e[6,6] += -D;
h1e[7,7] += -D;
h1e[8,8] += -4*D;
h1e[9,9] += -4*D;
h1e[0,4] += 2*E; # xy plane anisotropy
h1e[4,0] += 2*E;
h1e[1,5] += 2*E;
h1e[5,1] += 2*E;
h1e[2,6] += 2*E;
h1e[6,2] += 2*E;
h1e[3,7] += 2*E;
h1e[57,3] += 2*E;
h1e[4,8] += 2*E;
h1e[8,4] += 2*E;
h1e[5,9] += 2*E;
h1e[9,5] += 2*E;

# double electron terms
h2e[0,0,1,1] = 2*U; # hubbard
h2e[2,2,3,3] = 2*U;
h2e[4,4,5,5] = 2*U;
h2e[6,6,7,7] = 2*U;
h2e[8,8,9,9] = 2*U;

# solve with FCISolver object
cisolver = fci.direct_spin1.FCI()
cisolver.max_cycle = 100
cisolver.conv_tol = 1e-8
E_fci, v_fci = cisolver.kernel(h1e, h2e, norbs, nelecs,nroots=nroots);
E_fci.sort();
if(verbose):
    print("\n1. Spin blind solution, nelecs = ",nelecs);
    print("FCI energies = ",E_fci - (U-2*D)); # overall shift of U-2D


