#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Template:
Solve FCI problem with given 1-electron and 2-electron Hamiltonian

Specific case:
Solve 2 spin heisenberg hamiltonian

Formalism:
- h1e = (p|h|q) p,q spatial orbitals
- h2e = (pq|h|rs) chemists notation, <pr|h|qs> physicists notation
        thus hermicity means h_pqrs = h_qpsr
- as result of identity E_pq,rs = E_rs,pq, h_pq_rs = h_rs_pq
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
    
# call kernel to get warning print out here
warning = fci.direct_nosym.FCI();
warning.kernel(np.zeros((1,1)),np.zeros((1,1,1,1)),1,1);
    
#### nelec constrained solution

# system inputs
nelecs = [(2,0),(1,1), (0,2)]; # have to do 2,0; 1,1; and 0,2 separately
norbs = 2;

# init h1e, h2e
h1e = np.zeros((norbs, norbs));
h2e_11 = np.zeros((norbs, norbs,norbs,norbs)); # for nelecs = 1,1
h2e_20 = np.zeros((norbs, norbs,norbs,norbs)); # for nelecs= 2,0
h2e_02 = np.zeros((norbs, norbs,norbs,norbs)); # for nelecs = 0,2

if(verbose):
    print("\n1. Constrained solution \n - nelecs = ",nelecs[0]);

# h2e terms for 2,0 case
h2e_20[0,0,1,1] = J/4;
h2e_20[1,1,0,0] = J/4;

# solve with FCISolver object
cisolver20 = fci.direct_nosym.FCI();
cisolver20.max_cycle = 100
cisolver20.conv_tol = 1e-8
E_20, v_20 = cisolver20.kernel(h1e, h2e_20, norbs, nelecs[0],nroots=4);
if(verbose):
    print("FCI energies = ", E_20);
    
# h2e terms for 0,2 case
h2e_02[0,0,1,1] = J/4;
h2e_02[1,1,0,0] = J/4;

# solve
cisolver02 = fci.direct_nosym.FCI();
cisolver02.max_cycle = 100
cisolver02.conv_tol = 1e-8
E_02, v_02 = cisolver02.kernel(h1e, h2e_02, norbs, nelecs[2],nroots=4);
if(verbose):
    print("\n - nelecs = ",nelecs[2]);
    print("FCI energies = ", E_02);
    
# h2e terms for 1,1 case
h2e_11[0,0,1,1] = -J/4
h2e_11[1,1,0,0] = -J/4;
h2e_11[1,0,0,1] = J/2;
h2e_11[0,1,1,0] = J/2;
    
# solve
cisolver11 = fci.direct_nosym.FCI();
cisolver11.max_cycle = 100
cisolver11.conv_tol = 1e-8
E_11, v_11 = cisolver11.kernel(h1e, h2e_11, norbs, nelecs[1],nroots=4);
if(verbose):
    print("\n - nelecs = ",nelecs[1]);
    print("FCI energies = ", E_11);
    

#### map solution onto Hubbard model

# init h1e, h2e
h1e_hub = np.zeros((norbs, norbs));
h2e_hub = np.zeros((norbs, norbs,norbs,norbs)); # for nelecs = 1,1

# hubbard inputs
nelecs_hub = 1,1;

# all terms are 2e terms
h2e_hub[0,0,1,1] = -J/4;
h2e_hub[1,1,0,0] = -J/4;
h2e_hub[0,1,1,0] = J/2;
h2e_hub[1,0,0,1] = J/2;
h2e_hub[0,0,0,0] = J/4; # mapping means this term is hubbard like
h2e_hub[1,1,1,1] = J/4;

# solve
cisolver_hub = fci.direct_nosym.FCI();
cisolver_hub.max_cycle = 100
cisolver_hub.conv_tol = 1e-8
E_hub, v_hub = cisolver_hub.kernel(h1e_hub, h2e_hub, norbs, nelecs_hub,nroots=4);
if(verbose):
    print("\n2. Mapped solution, nelecs = ",nelecs_hub);
    print("FCI energies = ", E_hub);
    #print(v_map);
    
'''
#### solve with direct_uhf method, still constrained basis

# need to reconfigure h arrays

# 1e part
# UHF takes h1e = (h1e alpha, h1e beta)
h1e_alpha = np.zeros((norbs, norbs));
h1e_beta = np.zeros((norbs, norbs)); # still just empty

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
E_uhf, v_uhf = cisolver.kernel((h1e_alpha, h1e_beta), (h2e_alpha_alpha, h2e_alpha_beta, h2e_beta_beta), norbs, nelecs_hub,nroots=4);
if(verbose):
    print("\n3. UHF solution, constrained basis, nelecs = ",nelecs_hub);
    print("UHF energies = ", E_uhf);
    
'''

