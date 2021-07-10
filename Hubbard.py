#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# Example modified by Christian Bunker, June 2021

'''
Template:
Solve FCI problem with given 1-electron and 2-electron Hamiltonian

pyscf/fci module:
- configuration interaction solvers of form fci.direct_x.FCI()
- diagonalize 2nd quant hamiltonians via the .kernel() method
- .kernel takes (1e hamiltonian, 2e hamiltonian, # spacial orbs, (# alpha e's, # beta e's))
- direct_nosym assumes only h_pqrs = h_rspq (switch r1, r2 in coulomb integral)
- direct_spin1 assumes h_pqrs = h_qprs = h_pqsr = h_qpsr

Formalism:
- h1e_pq = (p|h|q) p,q spatial orbitals
- h2e_pqrs = (pq|h|rs) chemists notation, <pr|h|qs> physicists notation
- all direct_x solvers assume 4fold symmetry from sum_{pqrs} (don't need to do manually)
- 1/2 out front all 2e terms, so contributions are written as 1/2(2*actual ham term)
- hermicity: h_pqrs = h_qpsr can absorb factor of 1/2

Specific problem:
2 site hubbard model
'''

import utils

import numpy as np
import scipy as sp
from pyscf import fci

verbose = 2;
np.set_printoptions(suppress=True); # no sci notatation printing

# system inputs
# hamiltonian params must be floats
epsilon1 = 0.0; # on site energy, site 1
epsilon2 = 0.0; # site 2
t = 2.0 # hopping
U = 1000.0 # hubbard repulsion strength
if(verbose):
    print("\nInputs:\nepsilon1 = ",epsilon1,"\nepsilon2 = ",epsilon2,"\nt = ",t,"\nU = ",U);
    #print("t/U = ",t/U);
    
# analytical solution by exact diag
H_exact = np.array([[U+2*epsilon1,0,-t,t],[0,U+2*epsilon2,t,-t],[-t,t,epsilon1+epsilon2,0],[t,-t,0,epsilon1+epsilon2]]);
E_exact = np.linalg.eigh(H_exact)[0];
E_exact = np.append(E_exact, [epsilon1 + epsilon2, epsilon1 + epsilon2])
E_exact.sort();
if(verbose):
    print("\n0. Analytical solution");
    print("Exact energies = ", E_exact);
    

######################################################################
#### use conventional (sum over spin) method to solve hubbard
norbs = 2;
nelecs = 1,1;
nroots = 6;

# implement h1e and h2e
# doing it this way forces floats which is a good fail safe
h1e = np.zeros((norbs,norbs));
h2e = np.zeros((norbs,norbs,norbs,norbs));

# put in on site energy
h1e[0,0] = epsilon1;
h1e[1,1] = epsilon2;

# hopping
h1e[0,1] = t;
h1e[1,0] = t;

# hubbard
h2e[0,0,0,0] += U;
h2e[1,1,1,1] += U;

# implement FCISolver object
cisolver = fci.direct_nosym.FCI();
cisolver.max_cycle = 100; # max number of iterations
cisolver.conv_tol = 1e-8; # energy convergence

# kernel takes (1e ham, 2e ham, num orbitals, num electrons
# returns eigvals and eigvecs of ham (in what basis?)
E_fci, v_fci = cisolver.kernel(h1e, h2e, norbs, nelecs, nroots = nroots);
if(verbose):
    print("\n1. nelecs = ",nelecs, " nroots = ",nroots); # this matches analytical gd state
    print("FCI energies = ", E_fci);
    if(verbose > 2):
        print(v_fci);

######################################################################
#### use spin blind method to solve hubbard

# spin blind = put all electrons as up and make p,q... spin orbitals
nelecs = (2,0); # put in all spins as spin up
norbs = 4;      # spin orbs ie 1alpha, 1beta, 2alpha, 2beta -> 0,1,2,3
nroots = 6;

# implement h1e and h2e
# doing it this way forces floats which is a good fail safe
h1e_sb = np.zeros((norbs,norbs));
h2e_sb = np.zeros((norbs,norbs,norbs,norbs));

# on site energy
h1e_sb[0,0] = epsilon1;
h1e_sb[1,1] = epsilon1;
h1e_sb[2,2] = epsilon2;
h1e_sb[3,3] = epsilon2;

# hopping
T = t/1e9
h1e_sb[0,2] = t+T;
h1e_sb[2,0] = t+T;
h1e_sb[1,3] = t-T;
h1e_sb[3,1] = t-T;

# hubbard: 1/2(2*2U) total contribution TODO: revise
if True:
    h2e_sb[0,0,1,1] = 2*U; # since pr,qs = 01,01 -> pr,qs^* = 01,01 can't absorb 1/2
    h2e_sb[2,2,3,3] = 2*U;
else: # this way also works, different shuffling of a's
    h2e_sb[0,1,1,0] = -U;  # use hermicity to absorb factor of 1/2
    h2e_sb[1,0,0,1] = -U;  # pr,qs = 01,10 -> pr,qs^* = 10,01
    h2e_sb[2,3,3,2] = -U;
    h2e_sb[3,2,2,3] = -U;

# implement FCISolver object
cisolver_sb = fci.direct_spin1.FCI();

# kernel takes (1e ham, 2e ham, num orbitals, num electrons
# returns eigvals and eigvecs of ham (in what basis?)
E_sb, v_sb = cisolver_sb.kernel(h1e_sb, h2e_sb, norbs, nelecs, nroots = nroots);
spinexps = utils.Spin_exp(v_sb,norbs,nelecs); # evals <S> using fci vecs
E_formatter = "{0:6.6f}"
if(verbose):
    print("\n2. nelecs = ",nelecs, " nroots = ",nroots);
    for i, v in enumerate(v_sb):
        Eform = E_formatter.format(E_sb[i]);
        print("- E = ",Eform, ", <S_x> = ", spinexps[i][1], "<S_z> = ", spinexps[i][2]);
        if(verbose > 2):
            print(v);
        

