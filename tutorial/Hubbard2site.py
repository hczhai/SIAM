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
np.set_printoptions(suppress=True); # no sci notatation printing

# system inputs
# hamiltonian params must be floats
nelecs = (2,0); # put in all spins as spin up
norbs = 4;      # now this means spin orbitals
nroots = 6;
epsilon = 0 # on site energy
t = 3.0 # hopping
U = 200.0; # hubbard repulsion strength
if(verbose):
    print("\nInputs:\nepsilon = ",epsilon,"\nt = ",t,"\nU = ",U);
    
# analytical solution
E_exact = np.array([0,U,U/2 - 1/2 *np.sqrt(16*t*t+U*U), U/2 + 1/2 *np.sqrt(16*t*t+U*U),0,0])
E_exact.sort();
if(verbose):
    print("\n0. Analytical solution");
    print("Exact energies = ", E_exact);


# implement h1e and h2e
# doing it this way forces floats which is a good fail safe
h1e = np.zeros((norbs,norbs));
h2e = np.zeros((norbs,norbs,norbs,norbs));

# hopping
h1e[0,2] = t;
h1e[2,0] = t;
h1e[1,3] = t;
h1e[3,1] = t;

# hubbard
if True:
    h2e[0,0,1,1] = 2*U;
    h2e[2,2,3,3] = 2*U;
else:
    h2e[0,1,1,0] = -U;
    h2e[1,0,0,1] = -U;
    h2e[2,3,3,2] = -U;
    h2e[3,2,2,3] = -U;

# implement FCISolver object
cisolver = fci.direct_nosym.FCI();
cisolver.max_cycle = 100; # max number of iterations
cisolver.conv_tol = 1e-8; # energy convergence

# kernel takes (1e ham, 2e ham, num orbitals, num electrons
# returns eigvals and eigvecs of ham (in what basis?)
E_fci, v_fci = cisolver.kernel(h1e, h2e, norbs, nelecs, nroots = nroots); #since spin is none, nelecs=1,1
if(verbose):
    print("\n1. nelecs = ",nelecs, " nroots = ",nroots); # this matches analytical gd state
    print("FCI energies = ", E_fci);
    #print(v_fci);
    
#### tests
    
# repeat w spin pol
# na, nb = 1,1 gives same as above but 2,0 gives E = 0
newnelecs = (2,1)
cisolver = fci.direct_spin1.FCI();
E_fci, v_fci = cisolver.kernel(h1e, h2e, norbs, newnelecs,nroots=4);
if(verbose):
    print("\n2. nelecs = ",newnelecs);
    print("FCI energies = ", E_fci);
    #print(v_fci);

# should work same as first example
cisolver0 = fci.direct_spin0.FCI()
e_fci0, v_fci0 = cisolver.kernel(h1e, h2e, norbs, nelecs, nroots = nroots);
if(verbose):
    print("\n3. spin0 solver, nelecs = ",nelecs, " nroots = ",nroots);
    print("FCI energies = ", e_fci0);
    #print(v_fci0);

  
  
#### solve with direct_uhf method

# need to reconfigure h arrays

# h1e = (h1e alpha, h1e beta)
h1e_alpha = h1e;
h1e_beta = h1e; # 1e parts do not change, since there is no spin involved

# h2e = h2e alpha-alpha, h2e alpha-beta, h2e beta-beta
h2e_alpha_alpha = np.zeros((norbs,norbs,norbs,norbs));
h2e_alpha_beta = np.zeros((norbs,norbs,norbs,norbs));
h2e_beta_beta = np.zeros((norbs,norbs,norbs,norbs));
h2e_alpha_beta[0,0,0,0] = U;
h2e_alpha_beta[1,1,1,1] = U;

    
# solve
cisolver = fci.direct_uhf.FCISolver();
cisolver.max_cycle = 100
cisolver.conv_tol = 1e-8
E_uhf, v_uhf = cisolver.kernel((h1e_alpha, h1e_beta), (h2e_alpha_alpha, h2e_alpha_beta, h2e_beta_beta), norbs, nelecs, nroots=nroots);
if(verbose):
    print("\n4. UHF solver"); # this matches analytical gd state
    print("FCI energies = ", E_uhf);
    #print(v_fci);

