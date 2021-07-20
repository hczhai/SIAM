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
- g2e_pqrs = (pq|h|rs) chemists notation, <pr|h|qs> physicists notation
- all direct_x solvers assume 4fold symmetry from sum_{pqrs} (don't need to do manually)
- 1/2 out front all 2e terms, so contributions are written as 1/2(2*actual ham term)
- particle interchange symmetry: g_pqrs = g_rspq usually absorbs factor of 2

Specific problem:
2 site hubbard model
'''

import utils
import numbers

import numpy as np
import scipy as sp
from pyscf import fci
from pyblock3 import fcidump, hamiltonian, algebra

# top level inputs
verbose = 2;
np.set_printoptions(suppress=True); # no sci notatation printing

# system inputs
# hamiltonian params must be floats
epsilon1 = 0.0; # on site energy, site 1
epsilon2 = 0.0; # site 2
t = 3.0 # hopping
U = 100.0 # hubbard repulsion strength
if(verbose):
    print("\nInputs:\nepsilon1 = ",epsilon1,"\nepsilon2 = ",epsilon2,"\nt = ",t,"\nU = ",U);
    
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
g2e = np.zeros((norbs,norbs,norbs,norbs));

# put in on site energy
h1e[0,0] = epsilon1;
h1e[1,1] = epsilon2;

# hopping
h1e[0,1] = t;
h1e[1,0] = t;

# hubbard
g2e[0,0,0,0] += U;
g2e[1,1,1,1] += U;

# implement FCISolver object
cisolver = fci.direct_spin1.FCI(); # # doesn't matter if nosym or spin1
cisolver.max_cycle = 100; # max number of iterations
cisolver.conv_tol = 1e-8; # energy convergence

# kernel takes (1e ham, 2e ham, num orbitals, num electrons
# returns eigvals and eigvecs of ham (in what basis?)
E_fci, v_fci = cisolver.kernel(h1e, g2e, norbs, nelecs, nroots = nroots);
if(verbose):
    print("\n1. Spin free formalism: nelecs = ",nelecs, " nroots = ",nroots); # this matches analytical gd state
    print("FCI energies = ", E_fci);
    if(verbose > 2):
        print(v_fci);

######################################################################
#### use all spin up formalism (ASU) method to solve hubbard

# ASU = put all electrons as up and make p,q... spin orbitals
nelecs = (2,0); # put in all spins as spin up
norbs = 4;      # spin orbs ie 1alpha, 1beta, 2alpha, 2beta -> 0,1,2,3
nroots = 6;

# implement h1e and h2e
# doing it this way forces floats which is a good fail safe
h1e_asu = np.zeros((norbs,norbs));
g2e_asu = np.zeros((norbs,norbs,norbs,norbs));

# on site energy
h1e_asu[0,0] = epsilon1;
h1e_asu[1,1] = epsilon1;
h1e_asu[2,2] = epsilon2;
h1e_asu[3,3] = epsilon2;

# 1 particle terms: hopping
T = t/1e9 # break degeneracy to make wfs right
h1e_asu[0,2] = t+T;
h1e_asu[2,0] = t+T;
h1e_asu[1,3] = t-T;
h1e_asu[3,1] = t-T;

# 2 particle terms: hubbard
g2e_asu[0,0,1,1] = U;
g2e_asu[1,1,0,0] = U;  # interchange particle labels
g2e_asu[2,2,3,3] = U;
g2e_asu[3,3,2,2] = U;

# implement FCISolver object
cisolver_asu = fci.direct_spin1.FCI(); # doesn't matter if nosym or spin1

# kernel takes (1e ham, 2e ham, num orbitals, num electrons
# returns eigvals and eigvecs of ham (in what basis?)
E_asu, v_asu = cisolver_asu.kernel(h1e_asu, g2e_asu, norbs, nelecs, nroots = nroots);
spinexps = utils.Spin_exp(v_asu,norbs,nelecs); # evals <S> using fci vecs
E_formatter = "{0:6.6f}"
if(verbose):
    print("\n2. All spin up formalism: nelecs = ",nelecs, " nroots = ",nroots);
    for i, v in enumerate(v_asu):
        Eform = E_formatter.format(E_asu[i]);
        print("- E = ",Eform, ", <S> = ", spinexps[i],);
        if(verbose > 2):
            print(v);
            
            
######################################################################
#### use dmrg to solve hubbard

import os
import pickle
from pyblock3.algebra.mpe import MPE, CachedMPE
from pyblock3.symbolic.expr import OpElement, OpNames
from pyblock3.algebra.symmetry import SZ

# top level inputs
bond_dim_i = 200;

if(verbose): print("\n3. DMRG (All spin up): nelecs = ",nelecs, " nroots = ",nroots);

# store hamiltonian matrices in fcidump
# syntax: point group, num MOs, total num elecs (int), 2S = na - nb, h1e, g2e
# I use ASU formalism so MOs are spin orbs
hdump = fcidump.FCIDUMP(pg = 'd2h', n_sites = norbs, n_elec = sum(nelecs), twos = nelecs[0] - nelecs[1], h1e = h1e_asu, g2e = g2e_asu)
if verbose: print("- Created fcidump");

# get hamiltonian from fcidump
# now instead of np arrays it is a pyblock3 Hamiltonian class
h = hamiltonian.Hamiltonian(hdump, True);
h_mpo = h.build_qc_mpo(); # hamiltonian as matrix product operator (DMRG lingo)
#mpo, error = h_mpo.compress(flat = True, left=True, cutoff=1E-9, norm_cutoff=1E-9)
if verbose: print("- Built H as MPO");

# initial ansatz and energy
psi_mps = h.build_mps(bond_dim_i); # multiplies as np array
if verbose: 
    print('MPO = ', h_mpo.show_bond_dims())
    print('MPS = ', psi_mps.show_bond_dims())
psi_sq = np.dot(psi_mps.conj(), psi_mps);
E_psi = np.dot(psi_mps.conj(), h_mpo @ psi_mps)/psi_sq; # initial exp value of energy
print("- Initial gd energy = ", E_psi);

# ground-state DMRG
# runs thru an MPE (matrix product expectation) class built from mpo, mps
MPE_obj = MPE(psi_mps, h_mpo, psi_mps);

# solve system by doing dmrg sweeps
# MPE.dmrg method takes list of bond dimensions, noises, threads defaults to 1e-7
bonddims = [bond_dim_i,bond_dim_i+100,bond_dim_i+200]; # increase
noises = [1e-4,1e-5,0]; # slowly turn off. limits num sweeps if shorter than bdims, but not vice versa
# can also control verbosity (iprint) sweeps (n_sweeps), conv tol (tol)
dmrg_obj = MPE_obj.dmrg(bdims=bonddims, noises=noises, iprint = 1);
E_dmrg = dmrg_obj.energies;
print("- Final gd energy = ", E_dmrg[-1]);
print("- Final energies = ", E_dmrg);

        

