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
    if(verbose > 1):
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

# hubbard: 1/2(2*2U) total contribution
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
E_formatter = "{0:6.6f}"
if(verbose):
    print("\n2. nelecs = ",nelecs, " nroots = ",nroots);
    for i, v in enumerate(v_sb):
        Eform = E_formatter.format(E_sb[i]);
        print("- E = ",Eform);
        if(verbose > 1):
            print("  ",np.reshape(v, (1, v.size ) ) );
        

######################################################################
#### contract vector with Sz operator in h1e form to measure spin

# make S operators
Sx = np.zeros((norbs,norbs));
Sx[0,1] = 1/2;
Sx[1,0] = 1/2;
Sx[2,3] = 1/2;
Sx[3,2] = 1/2;
Sy = np.full((norbs, norbs), np.complex(0,0) );
Sy[0,1] = -np.complex(0,1)*1/2;
Sy[1,0] = np.complex(0,1)*1/2;
Sy[2,3] = -np.complex(0,1)*1/2;
Sy[3,2] = np.complex(0,1)*1/2;
Sz = np.zeros((norbs,norbs));
Sz[0,0] = 1/2;
Sz[1,1] = -1/2;
Sz[2,2] = 1/2;
Sz[3,3] = -1/2;
S2 = np.zeros((norbs,norbs,norbs,norbs));
S2[0,1,1,0] = 1;
S2[1,0,0,1] = 1;
S2[2,3,3,2] = 1;
S2[3,2,2,3] = 1;
#S2[0,0,0,0] = 2*1/4;
#S2[0,0,1,1] = 2*(-2/4);
#S2[1,1,1,1] = 2*1/4;
#S2[2,2,2,2] = 2*1/4;
#S2[2,2,3,3] = 2*(-2/4);
#S2[3,3,3,3] = 2*1/4;
S_op = [Sx,Sy,Sz]; # spin operator
S_exp = np.zeros(3); # <S> goes here
    
# iter over vectors and contract
print("\n3. Measure S on spin blind results");
for vi in range(len(v_sb)): # iter over vectors

    #contract with each element of S operator
    for Si in range(len(S_op)):

        # use contract_1e function
        result = fci.direct_nosym.contract_1e(S_op[Si], v_sb[vi], norbs, nelecs);
    
        S_exp[Si] = np.dot(np.reshape(v_sb[vi], (1,6)), result);
        
    S2_result = fci.direct_nosym.contract_2e(S2, v_sb[vi], norbs, nelecs); # act w S^2
    S2_result = np.dot(np.reshape(v_sb[vi], (1,6)), S2_result) + S_exp[2]*S_exp[2]/4 -S_exp[2]/2;
    
    if(verbose):
        Eform = E_formatter.format(E_sb[vi]); # delineate w/ corresponding energy
        print("E = ",Eform)
        print("- <S^2> = ", np.linalg.norm(S_exp) );
        print("- <S_z> = ", S_exp[2] );
    if(verbose > 1): # extra printing
        print("- <S> vector = ", S_exp );
        print("- <S^2> with alternate S2 = ",S2_result);


'''
#### rotate around singlet state
print("\n*******");

# the get singlet (the gd state) to rotate around (nonzero elems only)
singlet = v_sb[0];
triplet = v_sb[1];
singlet_rot = []
triplet_rot = []
for i,e in enumerate(triplet): # get nonzero elems from triplet
    if(abs(e) > 1e-2):
        singlet_rot.append(singlet[i]);
        triplet_rot.append(triplet[i]);
singlet_rot = np.array(singlet_rot); # already norm'd !
singlet_rot = singlet_rot.flatten();
triplet_rot = np.array(triplet_rot);
triplet_rot = triplet_rot.flatten();
print(triplet_rot);
#now norm of singlet_rot needs to be angle of rotation
angle_rot = np.arccos(triplet_rot[0]);
print(angle_rot);
singlet_rot = singlet_rot*angle_rot;
R_inst = sp.spatial.transform.Rotation.from_rotvec(singlet_rot); # encodes rot vector as Rotation instance
triplet_p = R_inst.apply(triplet_rot);
print(triplet_p);
'''
