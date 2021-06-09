#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Generate the entire FCI Hamiltonian for small system

example system = H2
'''

import numpy
from pyscf import fci, gto, scf

numpy.random.seed(2)
norb = 2  # psi_g and psi_u
nelec = (1,1) # 1 up e, 1 down e
h1 = numpy.random.random((norb,norb))
h2 = numpy.random.random((norb,norb,norb,norb))
# Restore permutation symmetry
h1 = h1 + h1.T
h2 = h2 + h2.transpose(1,0,2,3)
h2 = h2 + h2.transpose(0,1,3,2)
h2 = h2 + h2.transpose(2,3,0,1)

# pspace function computes the FCI Hamiltonian for "primary" determinants.
# Primary determinants are the determinants which have lowest expectation
# value <H>.  np controls the number of primary determinants.
# To get the entire Hamiltonian, np should be larger than the wave-function
# size.  In this example, a (8e,7o) FCI problem has 1225 determinants.
# = (7 choose 4)*(7 choose 4) bc we restrict ourselves to 4 up 4 down

# H2 is 2e, 2o so (2 choose 1)*(2 choose 1) = 4 determinants
H_fci = fci.direct_spin1.pspace(h1, h2, norb, nelec, np=4)[1]
e_diag, v_diag = numpy.linalg.eigh(H_fci) # eigvals and eigvecs

e_fci, v_fci = fci.direct_spin1.kernel(h1, h2, norb, nelec, nroots=3,
                                    max_space=30, max_cycle=100)

print('First root:')
print('energy: e_diag = ', e_diag[0],', e_fci = ', e_fci[0])
print('wfn overlap', v_diag[:,0].dot(v_fci[0].ravel()))

print('Second root:')
print('energy: e_diag = ', e_diag[1],', e_fci = ', e_fci[1])
print('wfn overlap', v_diag[:,1].dot(v_fci[1].ravel()))

print(v_diag, '\n',v_fci)

# comparison RHF calc for H2
H2mol = gto.M( atom = "H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="Bohr");
E_RHF = scf.RHF(H2mol).kernel(); # returns energy
print("E_RHF = ", E_RHF);

