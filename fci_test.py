'''
Test how implementing fci.direct_spin1 vs fci.direct_uhf compares
Consider behavior with both spin free and spin blind hamiltonian
'''

import numpy as np
from pyscf import fci, gto, ao2mo, scf
import functools

##################################################################################
#### implement 3 lead siam with dot impurity, half filling

# physical params
t = 1.0
td = 0.4
Vg = -0.5
U = 1.0

# decide whether to do spin free or spin blind
spin_free = False;
symmetry = 4; # perm symmetry of g2e. NB spin free uhf doesn't care about symmetry
                # but spin blind uhf closest with symmetry = 4
                
if spin_free: # spin free formalism means ham summed over spin, orbs are molecular orbs

    # top level params
    norb = 4; # 2 left sites, dot, right site
    nelec = (2,2); # half filling

    # make spin free hamiltonian
    h1e = np.array([ [0.0, -t, 0.0, 0.0],
                     [-t,  0.0, -td, 0.0],
                     [0.0, -td, Vg,  -td],
                     [0.0, 0.0, -td, 0.0],]);
    g2e = np.zeros((norb, norb, norb, norb));
    g2e[2,2,2,2] = U;
    print("\nSpin free calculation");
    
else:  # spin blind formalism means no sum over spin, orbs are spin orbs
       # input all electrons as up, and even spin orbs are up while odd are down
       
    # top level params
    norb = 8; # 2 left sites, dot, right site
    nelec = (4,0); # half filling

    # make spin free hamiltonian
    h1e = np.array([ [0.0, 0.0, -t, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, -t, 0.0, 0.0, 0.0, 0.0],
                     [-t, 0.0, 0.0, 0.0, -td, 0.0, 0.0, 0.0],
                     [0.0, -t, 0.0, 0.0, 0.0, -td, 0.0, 0.0],
                     [0.0, 0.0, -td, 0.0, Vg, 0.0, -td, 0.0],
                     [0.0, 0.0, 0.0, -td, 0.0, Vg, 0.0, -td],
                     [0.0, 0.0, 0.0, 0.0, -td, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, -td, 0.0, 0.0]]);
    g2e = np.zeros((norb, norb, norb, norb));
    g2e[4,4,5,5] = 2*U;
    print("\nSpin blind calculation")
    print(h1e)

# do direct fci
cisolver = fci.direct_spin1.FCI(); # needs to be spin1 not nosym
edirect, vdirect = cisolver.kernel(h1e,g2e,norb,nelec);
print("\n- direct fci gd state energy = ",edirect,"\n");

# do fci building from uhf
Pa = np.zeros(norb) # guess density matrices
Pa[::2] = 1.0
Pa = np.diag(Pa)
Pb = np.zeros(norb)
Pb[1::2] = 1.0
Pb = np.diag(Pb)
mol = gto.M() # make molecule obj
mol.incore_anyway = True
mol.nelectron = sum(nelec)
mf = scf.UHF(mol) # make scf obj
mf.get_hcore = lambda *args:h1e # pass hamiltonians to scf
mf.get_ovlp = lambda *args:np.eye(norb)
mf._eri = ao2mo.restore(symmetry, g2e, norb)
mf.kernel(dm0=(Pa,Pb)) # UHF calc of energy
mo_a = mf.mo_coeff[0]  # do fci on top of UHF
mo_b = mf.mo_coeff[1]
cisolver_uhf = fci.direct_uhf.FCISolver(mol)
h1e_a = functools.reduce(np.dot, (mo_a.T, mf.get_hcore(), mo_a)) # break up hams by alpha, beta
h1e_b = functools.reduce(np.dot, (mo_b.T, mf.get_hcore(), mo_b))
g2e_aa = ao2mo.incore.general(mf._eri, (mo_a,)*4, compact=False)
g2e_aa = g2e_aa.reshape(norb,norb,norb,norb)
g2e_ab = ao2mo.incore.general(mf._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
g2e_ab = g2e_ab.reshape(norb,norb,norb,norb)
g2e_bb = ao2mo.incore.general(mf._eri, (mo_b,)*4, compact=False)
g2e_bb = g2e_bb.reshape(norb,norb,norb,norb)
h1e_mo = (h1e_a, h1e_b)
g2e_mo = (g2e_aa, g2e_ab, g2e_bb)
euhf, vuhf = cisolver_uhf.kernel(h1e_mo, g2e_mo, norb, nelec) # fci calc of energy
print("\n- uhf gd state energy = ",euhf);
