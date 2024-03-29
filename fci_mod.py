'''
Christian Bunker
M^2QM at UF
June 2021

fci_mod.py

Wrapper funcs for doing fci using pySCF

pyscf/fci module:
- configuration interaction solvers of form fci.direct_x.FCI()
- diagonalize 2nd quant hamiltonians via the .kernel() method
- .kernel takes (1e hamiltonian, 2e hamiltonian, # spacial orbs, (# alpha e's, # beta e's))
- direct_nosym assumes only h_pqrs = h_rspq (switch r1, r2 in coulomb integral)
- direct_spin1 assumes h_pqrs = h_qprs = h_pqsr = h_qpsr
'''

import ops

import numpy as np
import functools
from pyscf import fci, gto, scf, ao2mo


##########################################################################################################
####

def dot_hams(nleads, nsites, nelecs, physical_params, Rlead_pol=0, verbose = 0):
    '''
    Converts physical params into 1e and 2e parts of siam model hamiltonian, with
    Impurity hamiltonian:
    H_imp = H_dot = -V_g sum_i n_i + U n_{i uparrow} n_{i downarrow}
    where i are impurity sites
    
    Args:
    - nleads, tuple of ints of lead sites on left, right
    - nsites, int, num impurity sites
    - nelecs, tuple of number es, 0 due to All spin up formalism
    - physical params, tuple of t, thyb, Vbias, mu, Vgate, U, B, theta. if None gives defaults
    - Rlead_pol, int -1, 0, 1
        if +/- 1, will polarize right lead spins to up/down state
        if 0, does nothing (default)
        also does nothing if B=0 no matter what rlead_pol actually is
    
    Returns:
    h1e, 2d np array, 1e part of siam ham
    h2e, 2d np array, 2e part of siam ham ( same as g2e)
    input_str, string with info on all the phy params
    '''

    # unpack inputs
    norbs = 2*(sum(nleads)+nsites);
    dot_i = [2*nleads[0], 2*nleads[0]+1];
    V_leads, V_imp_leads, V_bias, mu, V_gate, U, B, theta = physical_params;
    
    input_str = "\nInputs:\n- Num. leads = "+str(nleads)+"\n- Num. impurity sites = "+str(nsites)+"\n- nelecs = "+str(nelecs)+"\n- V_leads = "+str(V_leads)+"\n- V_imp_leads = "+str(V_imp_leads)+"\n- V_bias = "+str(V_bias)+"\n- mu = "+str(mu)+"\n- V_gate = "+str(V_gate)+"\n- Hubbard U = "+str(U)+"\n- B = "+str(B)+"\n- theta = "+str(theta);
    if verbose: print(input_str);

    #### make full system ham from inputs

    # make, combine all 1e hamiltonians
    hl = ops.h_leads(V_leads, nleads); # leads only
    hc = ops.h_chem(mu, nleads);   # can addjust lead chemical potential
    hdl = ops.h_imp_leads(V_imp_leads, nsites); # leads talk to dot
    hd = ops.h_dot_1e(V_gate, nsites); # dot
    h1e = ops.stitch_h1e(hd, hdl, hl, hc, nleads, verbose = verbose); # syntax is imp, imp-leads, leads, bias
    h1e += ops.h_bias(V_bias, dot_i, norbs , verbose = verbose); # turns on bias
    h1e += ops.h_B(B, theta, dot_i, norbs, verbose = verbose); # prep dot state w/ magntic field in direction nhat (theta, phi=0)
    if(verbose > 1): print("\n- Full one electron hamiltonian = \n",h1e);

    # polarize the right lead if asked
    if(Rlead_pol == 1 or Rlead_pol == -1): # turn on mag field for right lead
        Rsites = np.arange(0,norbs, 1, dtype = int)[dot_i[-1]+1:];
        h1e += ops.h_B(-abs(B)*Rlead_pol,0.0,Rsites,norbs, verbose = verbose);
        
    # 2e hamiltonian only comes from impurity
    if(verbose > 1):
        print("\n- Nonzero h2e elements = ");
    hd2e = ops.h_dot_2e(U,nsites);
    h2e = ops.stitch_h2e(hd2e, nleads, verbose = verbose);

    return h1e, h2e, input_str; #end dot hams


def dot_model(h1e, g2e, norbs, nelecs, verbose = 0):
    '''
    Run whole SIAM machinery for given  model hamiltonian
    
    Args:
    - h1e, 2d np array, 1e part of siam ham
    - g2e, 2d np array, 2e part of siam ham
    - norbs, int, total num spin orbs
    - nelecs, tuple of number es, 0 due to All spin up formalism
    
    Returns: tuple of
    mol, gto.mol object which holds some physical params
    scf inst, holds physics: h1e, h2e, mo coeffs etc
    '''
    
    # initial guess density matrices
    Pa = np.zeros(norbs)
    Pa[::2] = 1.0
    Pa = np.diag(Pa)
    
    # put everything into UHF scf object
    if(verbose):
        print("\nUHF energy calculation")
    mol = gto.M(); # geometry is meaningless
    mol.incore_anyway = True
    mol.nelectron = sum(nelecs)
    mol.spin = nelecs[1] - nelecs[0]; # in all spin up formalism, mol is never spinless!
    scf_inst = scf.UHF(mol)
    scf_inst.get_hcore = lambda *args:h1e # put h1e into scf solver
    scf_inst.get_ovlp = lambda *args:np.eye(norbs) # init overlap as identity matrix
    scf_inst._eri = g2e # put h2e into scf solver
    scf_inst.kernel(dm0=(Pa, Pa)); # prints HF gd state but this number is meaningless
                                   # what matter is h1e, h2e are now encoded in this scf instance

    return mol, scf_inst;
    
    
def mol_model(nleads, nsites, norbs, nelecs, physical_params,verbose = 0):
    '''
    Run whole SIAM machinery, with impurity Silas' molecule
    returns np arrays: 1e hamiltonian, 2e hamiltonian, molecule obj, and scf object

    Args:
    - nleads, tuple of ints, left lead sites, right lead sites
    - nsites, int, num imp sites
    - norbs, num spin orbs (= 2*(nsites + nleads[0]+nleads[1]))
    - nelecs, tuple of up and down e's, 2nd must always be zero in spin up formalism
    - physical params, tuple of:
        lead hopping, imp hopping, bias voltage, chem potential, tuple of mol params specific to Silas' module (see molecule_5level.py)
    '''

    # checks
    assert norbs == 2*(nsites + nleads[0]+nleads[1]);
    assert nelecs[1] == 0;
    assert nelecs[0] <= norbs;

    # unpack inputs
    V_leads, V_imp_leads, V_bias, mu, mol_params = physical_params;
    D, E, alpha, U = mol_params;

    if(verbose): # print inputs
        try:
            print("\nInputs:\n- Num. leads = ",nleads,"\n- Num. impurity sites = ",nsites,"\n- nelecs = ",nelecs,"\n- V_leads = ",V_leads,"\n- V_imp_leads = ",V_imp_leads,"\n- V_bias = ",V_bias,"\n- mu = ",mu,"\n- D = ",D,"\n- E = ",E, "\n- alpha = ",alpha, "\n- U = ",U, "\n- E/U = ",E/U,"\n- alpha/D = ",alpha/D,"\n- alpha/(E^2/U) = ",alpha*U/(E*E),"\n- alpha^2/(E^2/U) = ",alpha*alpha**U/(E*E) );
        except: 
            print("\nInputs:\n- Num. leads = ",nleads,"\n- Num. impurity sites = ",nsites,"\n- nelecs = ",nelecs,"\n- V_leads = ",V_leads,"\n- V_imp_leads = ",V_imp_leads,"\n- V_bias = ",V_bias)
    #### make full system ham from inputs

    # make, combine all 1e hamiltonians
    hl = ops.h_leads(V_leads, nleads); # leads only
    hb = ops.h_chem(mu, nleads);   # chem potential on leads
    hdl = ops.h_imp_leads(V_imp_leads, nsites); # leads talk to dot
    hd = molecule_5level.h1e(nsites*2,D,E,alpha); # Silas' model
    h1e = ops.stitch_h1e(hd, hdl, hl, hb, nleads, verbose = verbose); # syntax is imp, imp-leads, leads, bias/chem potential
    if(verbose > 2):
        print("\n- Full one electron hamiltonian = \n",h1e);
        
    # 2e hamiltonian only comes from impurity
    if(verbose > 2):
        print("\n- Nonzero h2e elements = ");
    hd2e = molecule_5level.h2e(2*nsites, U);
    h2e = ops.stitch_h2e(hd2e, nleads, verbose = verbose);

    #### encode physics of dot model in an SCF obj

    # initial guess density matrices
    Pa = np.zeros(norbs)
    Pa[::2] = 1.0
    Pa = np.diag(Pa)

    # put everything into UHF scf object
    if(verbose):
        print("\nUHF energy calculation")
    mol = gto.M(); # geometry is meaningless
    mol.incore_anyway = True
    mol.nelectron = sum(nelecs)
    mol.spin = nelecs[1] - nelecs[0]; # in all spin up formalism, mol is never spinless!
    scf_inst = scf.UHF(mol)
    scf_inst.get_hcore = lambda *args:h1e # put h1e into scf solver
    scf_inst.get_ovlp = lambda *args:np.eye(norbs) # init overlap as identity matrix
    scf_inst._eri = h2e # put h2e into scf solver
    scf_inst.kernel(dm0=(Pa, Pa)); # prints HF gd state but this number is meaningless
                                   # what matter is h1e, h2e are now encoded in this scf instance
        
    return h1e, h2e, mol, scf_inst;


def direct_FCI(h1e, h2e, norbs, nelecs, nroots = 1, verbose = 0):
    '''
    solve gd state with direct FCI
    '''
    
    cisolver = fci.direct_spin1.FCI();
    E_fci, v_fci = cisolver.kernel(h1e, h2e, norbs, nelecs, nroots = nroots);
    if(verbose):
        print("\nDirect FCI energies, zero bias, norbs = ",norbs,", nelecs = ",nelecs);
        print("- E = ",E_fci);

    return E_fci, v_fci;


def scf_FCI(mol, scf_inst, nroots = 1, verbose = 0):
    '''
    '''

    # init ci solver with ham from molecule inst
    cisolver = fci.direct_uhf.FCISolver(mol);

    # get unpack from scf inst
    h1e = scf_inst.get_hcore(mol);
    norbs = np.shape(h1e)[0];
    nelecs = (mol.nelectron,0);

    # slater determinant coefficients
    mo_a = scf_inst.mo_coeff[0]
    mo_b = scf_inst.mo_coeff[1]
   
    # since we are in UHF formalism, need to split all hams by alpha, beta
    # but since everything is spin blind, all beta matrices are zeros
    h1e_a = functools.reduce(np.dot, (mo_a.T, h1e, mo_a))
    h1e_b = functools.reduce(np.dot, (mo_b.T, h1e, mo_b))
    h2e_aa = ao2mo.incore.general(scf_inst._eri, (mo_a,)*4, compact=False)
    h2e_aa = h2e_aa.reshape(norbs,norbs,norbs,norbs)
    h2e_ab = ao2mo.incore.general(scf_inst._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
    h2e_ab = h2e_ab.reshape(norbs,norbs,norbs,norbs)
    h2e_bb = ao2mo.incore.general(scf_inst._eri, (mo_b,)*4, compact=False)
    h2e_bb = h2e_bb.reshape(norbs,norbs,norbs,norbs)
    h1e_tup = (h1e_a, h1e_b)
    h2e_tup = (h2e_aa, h2e_ab, h2e_bb)
    
    # run kernel to get exact energy
    E_fci, v_fci = cisolver.kernel(h1e_tup, h2e_tup, norbs, nelecs, nroots = nroots)
    if(verbose):
        print("\nFCI from UHF, zero bias, norbs = ",norbs,", nelecs = ",nelecs);
        print("- E = ", E_fci);

    return E_fci, v_fci;


##########################################################################################################
#### exec code

if __name__ == "__main__":

    pass;
