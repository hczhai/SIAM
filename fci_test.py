'''
Test how implementing fci.direct_spin1 vs fci.direct_uhf compares
Consider behavior with both spin free and spin blind hamiltonian
'''

import numpy as np
from pyscf import fci, gto, ao2mo, scf
import functools

##################################################################################
#### implement 3 lead siam with dot impurity, half filling, spin free method
### # spin free formalism means ham summed over spin, orbs are molecular orbs

# physical params
t = 1.0
td = 0.4
Vg = -0.5
U = 1.0

# top level params
symmetry = 4; # perm symmetry of g2e. NB spin free uhf doesn't care about symmetry
lleads = 3;
rleads = 2;
norb = lleads + rleads + 1; # 2 left sites, dot, right site
nelec = (int(norb/2),int(norb/2)); # half filling

# make spin free hamiltonian
if (norb == 4): # leads = 2,1
    h1e = np.array([ [0.0, -t, 0.0, 0.0],
                 [-t,  0.0, -td, 0.0],
                 [0.0, -td, Vg,  -td],
                 [0.0, 0.0, -td, 0.0],]);
    g2e = np.zeros((norb, norb, norb, norb));
    g2e[lleads, lleads, lleads, lleads] = U;
    
elif (norb == 6):
    h1e = np.array([[0.0, -t, 0.0, 0.0, 0.0, 0.0],
                   [-t,  0.0, -t, 0.0, 0.0, 0.0],
                   [0.0, -t, 0.0,  -td, 0.0, 0.0],
                   [0.0, 0.0, -td, Vg, -td, 0.0],
                   [0.0, 0.0, 0.0, -td, 0.0, -t ],
                   [0.0, 0.0, 0.0, 0.0, -t, 0.0],
                   ]);
    g2e = np.zeros((norb, norb, norb, norb));
    g2e[lleads, lleads, lleads, lleads] = U;

def SpinFreeGdState():
    # does everything to compute gd state with fci direct, fci on uhf methods
    # not good code practice but I just want to be sure that none of the variables here
    # contaminate the spin blind run I do next

    # do direct fci
    cisolver = fci.direct_spin1.FCI(); # needs to be spin1 not nosym
    edirect, vdirect = cisolver.kernel(h1e,g2e,norb,nelec);
    print("\n- direct fci, norbs = ",norb, ", nelecs = ",nelec);
    print("- E_gd = ", edirect,"\n");

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
    print("\n- uhf fci, norbs = ",norb, ", nelecs = ",nelec);
    print("- E_gd = ",euhf);
    
print("\nSpin free calculation:");
SpinFreeGdState();
del h1e
del g2e

##################################################################################
#### implement 3 lead siam with dot impurity, half filling, spin blind method
# spin blind formalism means no sum over spin, orbs are spin orbs
# input all electrons as up, and even spin orbs are up while odd are down
   
# top level params
symmetry = 1; # spin blind uhf closest with symmetry = 4
norb = 2*(lleads + rleads + 1); # spin orbs now
print(norb)
nelec = (int(norb/2),0); # half filling

if (norb == 8):
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
    
elif (norb == 12):
    h1e = np.array([ [0.0, 0.0, -t, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, -t, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [-t, 0.0, 0.0, 0.0, -t, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                     [0.0, -t, 0.0, 0.0, 0.0, -t, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, -t, 0.0, 0.0, 0.0, -td, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, -t, 0.0, 0.0, 0.0, -td, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, -td, 0.0, Vg, 0.0, -td, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, -td, 0.0, Vg, 0.0, -td, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -td, 0.0, 0.0,0.0, -t, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -td, 0.0,0.0, 0.0, -t],
                     [0.0, 0.0, -t, 0.0, 0.0, 0.0, 0.0, 0.0, -t, 0.0,  0.0, 0.0],
                     [0.0, 0.0, -t, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -t,  0.0, 0.0] ]);
    g2e = np.zeros((norb, norb, norb, norb));
    g2e[2*lleads, 2*lleads, 2*lleads+1, 2*lleads+1] = 2*U;

def SpinBlindGdState():
    # does everything to compute gd state with fci direct, fci on uhf methods
    # now in spin blind formalism

    # do direct fci
    cisolver = fci.direct_spin1.FCI(); # needs to be spin1 not nosym
    edirect, vdirect = cisolver.kernel(h1e,g2e,norb,nelec);
    print("\n- direct fci, norbs = ",norb, ", nelecs = ",nelec);
    print("- E_gd = ", edirect,"\n");

    # do fci building from uhf
    Pa = np.zeros(norb) # guess density matrices
    Pa[::2] = 1.0
    Pa = np.diag(Pa)
    Pb = np.zeros(norb)
    #Pb[1::2] = 1.0
    Pb = np.diag(Pb)
    mol = gto.M() # make molecule obj
    mol.incore_anyway = True
    mol.nelectron = sum(nelec)
    mf = scf.UHF(mol) # make scf obj
    mf.get_hcore = lambda *args:h1e # pass hamiltonians to scf
    mf.get_ovlp = lambda *args:np.eye(norb)
    mf._eri = ao2mo.restore(symmetry, g2e, norb)
    mf.kernel(dm0=(Pa,Pb), max_cycle = 100) # UHF calc of energy
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
    # overwrite beta matrices with zeros
    if True: # shouldn't affect anything
        h1e_b = np.zeros((norb,norb));
        g2e_ab = np.zeros((norb, norb, norb, norb));
        g2e_bb = np.zeros((norb, norb, norb, norb));
    h1e_mo = (h1e_a, h1e_b)
    g2e_mo = (g2e_aa, g2e_ab, g2e_bb)
    euhf, vuhf = cisolver_uhf.kernel(h1e_mo, g2e_mo, norb, nelec) # fci calc of energy
    print("\n- uhf fci, norbs = ",norb, ", nelecs = ",nelec);
    print("- E_gd = ",euhf);


print("\nSpin blind calculation");
SpinBlindGdState();

