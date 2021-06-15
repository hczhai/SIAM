'''
Christian Bunker
M^2QM at UF
June 2021

Template:
Solve exact diag problem with given 1-electron and 2-electron Hamiltonian

Formalism:
- h1e_pq = (p|h|q) p,q spatial orbitals
- h2e_pqrs = (pq|h|rs) chemists notation, <pr|h|qs> physicists notation
- all direct_x solvers assume 4fold symmetry from sum_{pqrs} (don't need to do manually)
- 1/2 out front all 2e terms, so contributions are written as 1/2(2*actual ham term)
- hermicity: h_pqrs = h_qpsr can absorb factor of 1/2

Specific case:
- Silas' model of molecule (SOC and spatial anisotropy included)
- 5 L_z levels: m= -2,...2 (d orbital, 10 spin orbitals)
- aim for spin 1 object hence 2 unpaired e's, 8 total e's
- 8e basis: (10 choose 8) = 45 states
- analytical solution, 1e basis: (10 choose 1) = 10 states
'''

import numpy as np
from pyscf import fci

# global vars
verbose = True;
np.set_printoptions(suppress=True); # no sci notatation printing
check_analytical = False; # turn on to compare to analytical solutions
if(check_analytical): # implement simple 1e case that can be checked analytically
    nelecs = (1,0);
    nroots = 10; #1e, 10 orb basis
else:
    nelecs = (8,0);
    nroots = 45; # 8e, 10 orb basis

# parameters in the hamiltonian
alpha = 0.1;
D = 100.0;
E = 50.0;
U = 2000.0;
E_shift = (nelecs[0] - 2)/2 *U  # num paired e's/2 *U
if(verbose):
    print("\nInputs:","\nalpha = ",alpha,"\nD = ",D,"\nE = ",E,"\nU = ",U);
    print("E shift = ",E_shift,"\nE/U = ",E/U,"\nalpha/(E^2/U) = ", alpha*U/(E*E) );

#### get analytical energies

# diagonalize numerically
H_exact = np.zeros((10,10));
E_exact = np.linalg.eigh(H_exact)[0];

# sort and print
E_exact.sort();
if(verbose):
    print("\n0. Analytical solution not yet implemented");
    print("Exact energies = ",E_exact);
    
    
#### solve with spin blind method
norbs = 10; # spin orbs, d shell

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
h1e[7,3] += 2*E;
h1e[4,8] += 2*E;
h1e[8,4] += 2*E;
h1e[5,9] += 2*E;
h1e[9,5] += 2*E;
h1e[0,0] += -2*alpha; # diag SOC
h1e[1,1] += 2*alpha;
h1e[2,2] += -alpha;
h1e[3,3] += alpha;
h1e[6,6] += alpha;
h1e[7,7] += -alpha;
h1e[8,8] += 2*alpha;
h1e[9,9] += -2*alpha;
h1e[0,3] += alpha; # off diag SOC
h1e[3,0] += alpha;
h1e[2,5] += alpha;
h1e[5,2] += alpha;
h1e[4,7] += alpha;
h1e[7,4] += alpha;
h1e[6,9] += alpha;
h1e[9,6] += alpha;

# two electron terms
h2e[0,0,1,1] = 2*U; # hubbard
h2e[2,2,3,3] = 2*U;
h2e[4,4,5,5] = 2*U;
h2e[6,6,7,7] = 2*U;
h2e[8,8,9,9] = 2*U;

# pass ham to FCI solver kernel to diagonalize
cisolver = fci.direct_nosym.FCI()
myroots = 4; # don't print out 45
E_fci, v_fci = cisolver.kernel(h1e, h2e, norbs, nelecs,nroots=myroots);
E_fci.sort();
if(verbose):
    print("\n1. Spin blind solution, nelecs = ",nelecs," nroots = ",myroots);
    print("FCI energies = ",E_fci- E_shift);


