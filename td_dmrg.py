'''
Christian Bunker
M^2QM at UF
July 2021

td_dmrg.py

Use Huanchen Zhai's DMRG code (pyblock3) to do time dependence in SIAM

Combine with fci code later
'''

import ops_dmrg

import numpy as np
from pyblock3 import hamiltonian, fcidump
from pyblock3.algebra.mpe import MPE

##########################################################################################################
#### compute observables

def compute_obs(op,mps):
    '''
    Compute expectation value of observable repped by given operator from MPS wf
    '''

    return np.dot(mps.conj(), op @ mps)/np.dot(mps.conj(),mps);


def compute_occ(site_i, mps, h):

    norbs = mps.n_sites

    # occupancy operator in dmrg
    occ_op = ops_dmrg.occ(site_i, norbs);
    occ_mpo = h.build_mpo(occ_op);
    return compute_obs(occ_mpo, mps);


def compute_current(site_i, mps, h):

    # current operator as h1e array
    norbs = mps.n_sites
    J = np.zeros((norbs,norbs));

    # to simplify code, set t_hyb = 1 and just multiply J vals by t_hyb later
    t=1;
    print("--> Setting t_hyb = 1");

    # iter over dot sites to fill current op
    for doti in range(site_i[0], site_i[-1]+1, 2): # doti is up, doti+1 is down # +1 because dot_i is incluse but range is exclusive
        J[doti - 2,doti] = -t/2;  # dot up spin to left up spin # left moving is -
        J[doti+1-2,doti+1] = -t/2; # down to down
        J[doti,doti - 2] =  t/2; # left up spin to dot up spin # hc of 2 above # right moving is +
        J[doti+1, doti+1-2] = t/2; # hc
        J[doti + 2,doti] = t/2;  # up spin to right up spin
        J[doti+1+2,doti+1] = t/2; # down to down
        J[doti,doti + 2] =  -t/2; # hc
        J[doti+1, doti+1+2] = -t/2; # hc

    # now convert current op from array to MPO
    J_mpo = h.build_mpo(J);

    return compute_obs(J_mpo, mps);
    
##########################################################################################################
#### time propagation

def kernel(mpo, h_obj, mps, tf, dt, i_dot, thyb, bdims, verbose = 0):
    '''
    Drive time prop for dmrg

    Args:
    - mpo, a matrix product operator form of the hamiltonian
    - h_obj, a pyblock3.hamiltonian.Hamiltonian form of the hailtonian
    '''

    assert(isinstance(bdims, list));

    N = int(tf/dt+1e-6); # num steps
    i_all = np.arange(0,h_obj.n_sites, 1, dtype = int);
    i_left = i_all[:i_dot[0] ];
    i_right = i_all[i_dot[-1]+1:];
    
    # return vals
    timevals = np.zeros(N+1);
    energyvals = np.zeros(N+1);
    currentvals = np.zeros((2, N+1));
    occvals = np.zeros( (3,N+1), dtype = complex );
    Szvals = np.zeros( (3,N+1), dtype = complex );

    # init mpe
    mpe_obj = MPE(mps, mpo, mps);

    # loop over time
    for i in range(N+1):

        # before any time stepping, get initial state
        if(i==0):
            occ_init, Sz_init = np.zeros(len(i_all), dtype = complex), np.zeros(len(i_all), dtype = complex);
            for sitej in i_all: # iter over sites
                
                if(sitej % 2 == 0): # spin up sites
                    occ_init[sitej] = compute_occ([sitej],mps, h_obj);
                    Sz_init[sitej] = 0.0
                else: # spin down sites
                    occ_init[sitej] = compute_occ([sitej], mps, h_obj);
                    Sz_init[sitej] = 0.0;

            initstatestr = "\nInitial state:"
            initstatestr += "\n    occ = "+str(np.real(occ_init));
            initstatestr += "\n    Sz = "+str(np.real(Sz_init));

        # mpe.tddmrg method does time prop, outputs energies but also modifies mpe obj
        energies = mpe_obj.tddmrg(bdims,-np.complex(0,dt),n_sweeps=1,iprint=0,normalize = False).energies

        # compute observables
        timevals[i] = i*dt;
        energyvals[i] = energies[-1];
        occvals[0][i] = compute_occ(i_left, mps, h_obj);
        occvals[1][i] = compute_occ(i_dot, mps, h_obj);
        occvals[2][i] = compute_occ(i_right, mps, h_obj);
        
        # update stdout        
        if(verbose>4): print("    time: ", i*dt);

    # return time and tuple of observables as functions of time, 1d arrays
    observables = [timevals, energyvals, thyb*currentvals[0], thyb*currentvals[1], occvals[0], occvals[1], occvals[2], Szvals[0], Szvals[1], Szvals[2]];
    return initstatestr, np.array(observables);


##########################################################################################################
#### exec code

if __name__ == "__main__":

    pass;
