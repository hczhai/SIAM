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
- Silas' model of molecule (SOC and spatial anisotropy)
- 3 L_z levels: m=-1,0,1 (6 spin orbitals)
'''

import numpy as np
from pyscf import fci

verbose = True;
np.set_printoptions(suppress=True); # no sci notatation printing

# parameters in the hamiltonian
alpha = 0.001;
D = 400.0;
E = 60.0;
U = 800.0;
if(verbose):
    print("\nInputs:","\nalpha = ",alpha,"\nD = ",D,"\nE = ",E,"\nU = ",U);
    print("E/U = ",E/U,"\nalpha/(E^2/U) = ", alpha*U/(E*E),"\nalpha/D = ",alpha/D );

#### get analytical energies

# diagonalize numerically
H_T = np.array([[-2*alpha,0,-2*E,2*E], [0,2*alpha,2*E,-2*E],[-2*E,2*E,U,0],[2*E,-2*E,0,U]]); # T sector
H_O = np.array([[-D-alpha,0,0,2*E,0],[0,-D+alpha,2*E,0,0],[0,2*E,-D-alpha,0,0],[2*E,0,0,-D+alpha,0],[0,0,0,0,-2*D+U]]); # other sector
E_exact = np.append(np.linalg.eigh(H_T)[0], np.linalg.eigh(H_O)[0]);
H_eff = np.array([[-2*alpha- 8*E*E/U, 8*E*E/U], [8*E*E/U, 2*alpha - 8*E*E/U - 2*alpha*alpha/(alpha-D)] ])
E_S, E_T0 = np.linalg.eigh(H_eff)[0];
print(np.linalg.eigh(H_eff)[0]);

# sort and print
E_exact.sort();
if(verbose):
    print("\n0. Analytical solution, exact as alpha --> 0:")
    print("Exact energies = \n",E_exact);
    print("Expected energies as E/U, alpha/(E^2/U), alpha/D --> 0:\n- Singlet energy = ", E_S, "\n- T0 energy = ", E_T0, "\n- T+/- energy = ", alpha*alpha/(D-alpha) );
    
    
#### solve with spin blind method
nelecs = (4,0);
norbs = 6; # spin orbs
nroots = 15; # size of full basis

# implement h1e, h2e
h1e = np.zeros((norbs, norbs));
h2e = np.zeros((norbs, norbs,norbs,norbs));

# single electron terms
# best practice to use += thru out since we may modify elements twice
h1e[0,0] += -D; # z axis anisotropy
h1e[1,1] += -D;
h1e[4,4] += -D;
h1e[5,5] += -D;
h1e[0,4] += 2*E; # xy plane anisotropy
h1e[4,0] += 2*E;
h1e[1,5] += 2*E;
h1e[5,1] += 2*E;
h1e[0,0] += -alpha; # diagonal SOC
h1e[1,1] += alpha;
h1e[4,4] += alpha;
h1e[5,5] += -alpha;
h1e[0,3] += alpha; # off diagonal SOC
h1e[3,0] += alpha;
h1e[2,5] += alpha;
h1e[5,2] += alpha;

# double electron terms
h2e[0,0,1,1] = 2*U; # hubbard
h2e[2,2,3,3] = 2*U;
h2e[4,4,5,5] = 2*U;

# solve with FCISolver object
cisolver = fci.direct_nosym.FCI()
cisolver.max_cycle = 100
cisolver.conv_tol = 1e-8
E_fci, v_fci = cisolver.kernel(h1e, h2e, norbs, nelecs,nroots=nroots);
E_fci.sort();
if(verbose):
    print("\n1. Spin blind solution, nelecs = ",nelecs);
    print("FCI energies = ",E_fci - (U-2*D)); # overall shift of U-2D

