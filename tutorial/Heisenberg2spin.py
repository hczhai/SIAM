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

# hamiltonian parameters, must be floats
J = 1.0;

#### analytical solution
E_exact = np.array([-3*J/4,J/4,J/4,J/4]);
E_exact.sort();
if(verbose):
    print("\n0. Analytical solution");
    print("Exact energies = ", E_exact);
    
#### nelec constrained solution

# system inputs
nelecs = (1,1);
norbs = 2;

# implement h1e, h2e
h1e = np.zeros((norbs, norbs));
h2e = np.zeros((norbs, norbs,norbs,norbs));

# h2e terms
h2e[0,0,1,1] = -J/4;
h2e[1,1,0,0] = -J/4;
h2e[0,1,1,0] = -J/2; # not needed
h2e[1,0,0,1] = -J/2;


# solve with FCISolver object
cisolver = fci.direct_spin1.FCI()
cisolver.max_cycle = 100
cisolver.conv_tol = 1e-8
E_fci, v_fci = cisolver.kernel(h1e, h2e, norbs, nelecs,nroots=4);
if(verbose):
    print("\n1. Constrained solution, nelecs = ",nelecs);
    print("FCI energies = ", E_fci);

#### map solution onto Hubbard model

# all terms are 2e terms
h2e[0,0,1,1] = -J/4;
h2e[1,1,0,0] = -J/4;
h2e[0,1,1,0] = -J/2;
h2e[1,0,0,1] = -J/2;
h2e[0,0,0,0] = J/4; # mapping means this term is hubbard like
h2e[1,1,1,1] = J/4;

E_map, v_map = cisolver.kernel(h1e, h2e, norbs, nelecs,nroots=4);
if(verbose):
    print("\n2. Mapped solution, nelecs = ",nelecs);
    print("FCI energies = ", E_map);
    #print(v_map);
    

#### solve with direct_uhf method, still constrained basis

# need to reconfigure h arrays

# 1e part
# UHF takes h1e = (h1e alpha, h1e beta)
h1e_alpha = h1e;
h1e_beta = h1e; # still just empty

# 2e part
# UHF takes h2e = h2e alpha-alpha, h2e alpha-beta, h2e beta-beta
h2e_alpha_alpha = np.zeros((norbs,norbs,norbs,norbs));
h2e_alpha_beta = np.zeros((norbs,norbs,norbs,norbs));
h2e_beta_beta = np.zeros((norbs,norbs,norbs,norbs));

h2e_alpha_alpha[0,0,1,1] = J/4; # both these terms are not needed
h2e_beta_beta[1,1,0,0] = J/4; #NB alpha_alpha and beta_beta not ind'ly hermitian

h2e_alpha_beta[0,1,1,0] = -J/2; # this term not needed
h2e_alpha_beta[1,0,0,1] = -J/2; #NB alpha_beta is hermitian

h2e_alpha_beta[0,0,1,1] = -J/4;
h2e_alpha_beta[1,1,0,0] = -J/4;

# solve
cisolver = fci.direct_uhf.FCISolver();
cisolver.max_cycle = 100
cisolver.conv_tol = 1e-8
E_uhf, v_uhf = cisolver.kernel((h1e_alpha, h1e_beta), (h2e_alpha_alpha, h2e_alpha_beta, h2e_beta_beta), norbs, nelecs,nroots=4);
if(verbose):
    print("\n3. UHF solution, constrained basis, nelecs = ",nelecs);
    print("UHF energies = ", E_uhf);

