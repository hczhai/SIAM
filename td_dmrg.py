'''
Christian Bunker
M^2QM at UF
July 2021

td_dmrg.py

Use Huanchen Zhai's DMRG code (pyblock3) to do time dependence in SIAM

Combine with fci code later
'''

import numpy as np
from pyblock3 import hamiltonian, fcidump
from pyblock3.algebra.mpe import MPE

##########################################################################################################
#### time propagation

def compute_obs(op,mps):
    '''
    Compute expectation value of observable repped by given operator from MPS wf
    '''

    return np.dot(mps.conj(), op @ mps)/np.dot(mps.conj(),mps);

def compute_current(site_i, mps, h):

    # current operator as h1e array
    norbs = mps.n_sites
    print(norbs);
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
    

def kernel(mpo, h_obj, mps, tf, dt, dot_i, bdims, verbose = 0):
    '''
    Drive time prop for dmrg

    Args:
    - mpo, a matrix product operator form of the hamiltonian
    - h_obj, a pyblock3.hamiltonian.Hamiltonian form of the hailtonian
    '''

    assert(isinstance(bdims, list));

    # return vals are observables
    N = int(tf/dt+1e-6); # num steps
    timevals = np.zeros(N+1);
    energyvals = np.zeros(N+1);
    currentvals = np.zeros(N+1);

    # init mpe
    mpe_obj = MPE(mps, mpo, mps);

    # loop over time
    for i in range(N+1):

        # mpe.tddmrg method does time prop, outputs energies but also modifies mpe obj
        energies = mpe_obj.tddmrg(bdims,-np.complex(0,dt),n_sweeps=1,iprint=1,normalize = False).energies

        # compute observables
        timevals[i] = i*dt;
        energyvals[i] = energies[-1];
        
        # update stdout        
        if(verbose > 4): print("    time: ", i*dt, " E = ",energies[-1]);

    # return time and tuple of observables as functions of time, 1d arrays
    return timevals, (energyvals, currentvals)



##########################################################################################################
#### exec code

if __name__ == "__main__":

    pass;
