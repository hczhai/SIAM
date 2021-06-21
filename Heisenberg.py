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
- all direct_x solvers assume 4fold symmetry from sum_{pqrs} (don't need to do manually)
- 1/2 out front all 2e terms, so contributions are written as 1/2(2*actual ham term)
- other symmetries to be aware of:
      hermicity: h_pqrs = h_qpsr
      E_pr,qs = E_rp,sq from properties of E
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
h2e_11[1,0,0,1] = -J/2; # technically - in non mapped case
h2e_11[0,1,1,0] = -J/2; # - a_1up^dagger a_2down^dagger a_1down a_2up, sum over spin
    
# solve
cisolver11 = fci.direct_nosym.FCI();
cisolver11.max_cycle = 100
cisolver11.conv_tol = 1e-8
E_11, v_11 = cisolver11.kernel(h1e, h2e_11, norbs, nelecs[1],nroots=6);
if(verbose):
    print("\n - nelecs = ",nelecs[1]);
    print("FCI energies = ", E_11);
    
    
#### spin blind solution
nelecs = (2,0); # spin blind means all up
norbs = 4; # spin orbs ie 1alpha, 1beta, 2alpha, 2beta -> 0,1,2,3
nroots = 6;

# init h1e, h2e
h1e_sb = np.zeros((norbs, norbs));
h2e_sb = np.zeros((norbs, norbs,norbs,norbs));

# all terms are 2 electron
if True:
    h2e_sb[0,1,3,2] += 2*J/2; # 1/2(2*2*J/2) contribution
    h2e_sb[1,0,2,3] += 2*J/2; # herms map into each other so can't absorb 1/2
else:
    h2e_sb[0,1,3,2] += J/2; # equivalently could absorb 1/2 using E_pq,rs = E_rp,sq identity
    h2e_sb[1,0,2,3] += J/2;
    h2e_sb[3,2,0,1] += J/2;
    h2e_sb[2,3,1,0] += J/2;
h2e_sb[0,0,2,2] += 2*J/4; # 1/2(2*2J/4)
h2e_sb[1,1,3,3] += 2*(J/4); # pqrs = 1133, pqrs^* = 1133 so can't absorb 1/2
h2e_sb[0,0,3,3] += 2*(-J/4); # likewise
h2e_sb[1,1,2,2] += 2*(-J/4);

# solve
cisolver_sb = fci.direct_nosym.FCI();
cisolver_sb.max_cycle = 100;
cisolver_sb.conv_tol = 1e-8;
E_sb, v_sb = cisolver_sb.kernel(h1e_sb, h2e_sb, norbs, nelecs, nroots = nroots);
if(verbose):
    print("\n2. Spin blind solution, nelecs = ",nelecs," nroots = ", nroots);
    print("FCI energies = ", E_sb);


#### map solution onto Hubbard model
norbs = 2;
nelecs = 1,1;

# init h1e, h2e
h1e_hub = np.zeros((norbs, norbs));
h2e_hub = np.zeros((norbs, norbs,norbs,norbs));

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
E_hub, v_hub = cisolver_hub.kernel(h1e_hub, h2e_hub, norbs, nelecs,nroots=6);
if(verbose):
    print("\n3. Mapped solution, nelecs = ",nelecs);
    print("FCI energies = ", E_hub);
    


