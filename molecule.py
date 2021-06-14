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
np.set_printoptions(suppress=True); # no sci notatation printing

# system inputs
# ham params should be floats
nelecs = (2,2);
norbs = 3;
alpha = 0.01;
D = 20.0;
E = 10.0;
U = 100.0;
full_basis = 15; # all 4e, 3o states
constrained_basis = 9; # when alpha = 2, beta=2
if(verbose):
    print("\nInputs:","\nalpha = ",alpha,"\nD = ",D,"\nE = ",E,"\nU = ",U);

#### get correct energies

# diagonalize numerically
H_T = np.array([[-2*alpha,0,-2*E,2*E], [0,2*alpha,2*E,-2*E],[-2*E,2*E,U,0],[2*E,-2*E,0,U]]); # T sector
H_O = np.array([[-D-alpha,0,0,2*E,0],[0,-D+alpha,2*E,0,0],[0,2*E,-D-alpha,0,0],[2*E,0,0,-D+alpha,0],[0,0,0,0,-2*D+U]]); # other sector
E_exact = np.append(np.linalg.eigh(H_T)[0], np.linalg.eigh(H_O)[0]);

# sort and print
E_exact.sort();
if(verbose):
    print("\n0. Analytical solution, constrained")
    print("Exact energies = ",E_exact);
    print("Expected singlet energy, E/U-->0 = ",-16*E*E/U)

# implement h1e, h2e
h1e = np.zeros((norbs, norbs));
h2e = np.zeros((norbs, norbs,norbs,norbs));

# single electron terms
#h1e[0,1] = 2*alpha;
#h1e[1,2] = 2*alpha;
#h1e[1,0] = 2*alpha;
#h1e[2,1] = 2*alpha;
h1e[0,0] = -D; # z anisotropy
h1e[2,2] = -D;
h1e[0,2] = 2*E; # xy plane anisotropy
h1e[2,0] = 2*E;

# double electron terms
h2e[0,0,0,0] = U; # onsite coulomb on each p orbital
h2e[1,1,1,1] = U;
h2e[2,2,2,2] = U;

# solve with FCISolver object
cisolver = fci.direct_spin1.FCI()
cisolver.max_cycle = 100
cisolver.conv_tol = 1e-8
E_fci, v_fci = cisolver.kernel(h1e, h2e, norbs, nelecs,nroots=constrained_basis);
E_fci.sort();
if(verbose):
    print("\n1. Constrained solution, nelecs = ",nelecs); # this matches analytical gd state
    print("FCI energies = ",E_fci - (U-2*D));
    

#### with diagonal SOC on, use UHF method

# add terms to h1e
h1e_alpha, h1e_beta = h1e, h1e;
h1e_alpha[0,0] += -alpha; # update values only!
h1e_alpha[2,2] += alpha;
h1e_beta[0,0] += alpha;
h1e_beta[2,2] += -alpha;

# hubbard is an alpha-beta interaction
h2e_alpha_alpha, h2e_alpha_beta, h2e_beta_beta = np.zeros((norbs, norbs,norbs,norbs)), np.zeros((norbs, norbs,norbs,norbs)), np.zeros((norbs, norbs,norbs,norbs));
h2e_alpha_beta[0,0,0,0] = U;
h2e_alpha_beta[1,1,1,1] = U;
h2e_alpha_beta[2,2,2,2] = U;

# solve
cisolver = fci.direct_uhf.FCISolver();
cisolver.max_cycle = 100
cisolver.conv_tol = 1e-8
E_uhf, v_uhf = cisolver.kernel((h1e_alpha, h1e_beta), (h2e_alpha_alpha, h2e_alpha_beta, h2e_beta_beta), norbs, nelecs, nroots=constrained_basis);
if(verbose):
    print("\n2. UHF solution, nelecs = ",nelecs); # this matches analytical gd state
    print("UHF energies = ", E_uhf - (U-2*D) );
    #print(v_fci);




