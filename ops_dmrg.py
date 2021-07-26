'''
Christian Bunker
M^2QM at UF
June 2021

ops.py

Representation of operators, hamiltonian and otherwise, in DMRG
ie as generating functions (with yield statements) which are
then passed to the Hamiltonian.hamiltonian.build_mpo() method

pyscf formalism:
- h1e_pq = (p|h|q) p,q spatial orbitals
- h2e_pqrs = (pq|h|rs) chemists notation, <pr|h|qs> physicists notation
- all direct_x solvers assume 4fold symmetry from sum_{pqrs} (don't need to do manually)
- 1/2 out front all 2e terms, so contributions are written as 1/2(2*actual ham term)
- hermicity: h_pqrs = h_qpsr can absorb factor of 1/2

pyscf/fci module:
- configuration interaction solvers of form fci.direct_x.FCI()
- diagonalize 2nd quant hamiltonians via the .kernel() method
- .kernel takes (1e hamiltonian, 2e hamiltonian, # spacial orbs, (# alpha e's, # beta e's))
- direct_nosym assumes only h_pqrs = h_rspq (switch r1, r2 in coulomb integral)
- direct_spin1 assumes h_pqrs = h_qprs = h_pqsr = h_qpsr


'''

import numpy as np

#######################################################
####


def occ(site_i, norbs):
    '''
    Operator for the occupancy of sites specified by site_i
    ASU formalism only !!!
    Args:
    - site_i, list of (usually spin orb) site indices
    - norbs, total num orbitals in system
    '''

    def occ_yield(norbs, adag, a):

        # iter over site(s) we want occupancies of
        for i in site_i:
            spin = 0; # ASU formalism
            yield adag[i,spin]*a[i,spin]; # yield number operator of this site

    return occ_yield;


def Sz(site_i, norbs):
    '''
    Operator for the z spin of sites specified by site_i
    ASU formalism only !!!
    Args:
    - site_i, list of (usually spin orb) site indices
    - norbs, total num orbitals in system
    '''

    # check inputs
    assert( isinstance(site_i, list) or isinstance(site_i, np.ndarray));

    # create op array
    sz = np.zeros((norbs,norbs));

    # iter over all given sites
    for i in range(site_i[0], site_i[-1]+1, 2): # doti is up, doti+1 is down 
        sz[i,i] = 1/2; # spin up
        sz[i+1, i+1] = -1/2; # spin down

    return sz;


def Jup(site_i, norbs):
    '''
    Current of up spin e's thru sitei
    ASU formalism only !!!
    Args:
    - site_i, list of (usually spin orb) site indices
    - norbs, total num orbitals in system
    '''

    # check inputs
    assert( len(site_i) == 2); # should be only 1 site ie 2 spatial orbs

    # current operator (1e only)
    J = np.zeros((norbs,norbs));

    # even spin index is up spins
    upi = site_i[0];
    assert(upi % 2 == 0); # check even
    J[upi-2,upi] = -1/2;  # dot up spin to left up spin #left moving is negative current
    J[upi,upi-2] =  1/2; # left up spin to dot up spin # hc of above # right moving is +
    J[upi+2,upi] = 1/2;  # up spin to right up spin
    J[upi,upi+2] =  -1/2; # hc

    return J;


def Jdown(site_i, norbs):
    '''
    Current of down spin e's thru sitei
    ASU formalism only !!!
    Args:
    - site_i, list of (usually spin orb) site indices
    - norbs, total num orbitals in system
    '''

    # check inputs
    assert( len(site_i) == 2); # should be only 1 site ie 2 spatial orbs

    # current operator (1e only)
    J = np.zeros((norbs,norbs));

    # odd spin index is down spins
    dwi = site_i[1];
    assert(dwi % 2 == 1); # check even
    J[dwi-2,dwi] = -1/2;  # dot dw spin to left dw spin #left moving is negative current
    J[dwi,dwi-2] =  1/2; # left dw spin to dot dw spin # hc of above # right moving is +
    J[dwi+2,dwi] = 1/2;  # dot dw spin to right dw spin
    J[dwi,dwi+2] =  -1/2; # hc

    return J;


def h_B(B, theta, site_i, norbs, verbose=0):
    '''
    Turn on a magnetic field of strength B in the theta hat direction, on site i
    This has the effect of preparing the spin state of the site
    e.g. large, negative B, theta=0 yields an up electron

    Args:
    - B, float, mag field strength
    - theta, float, mag field direction
    - norbs, int, num spin orbs in whole system
    - site_i, list, spin indices (even up, odd down) of site that feels mag field

    Returns 2d np array repping magnetic field on given sites
    '''

    assert(isinstance(site_i, list) );

    hB = np.zeros((norbs,norbs));
    for i in range(site_i[0],site_i[-1],2): # i is spin up, i+1 is spin down
        hB[i,i+1] = B*np.sin(theta)/2; # implement the mag field, x part
        hB[i+1,i] = B*np.sin(theta)/2;
        hB[i,i] = B*np.cos(theta)/2;    # z part
        hB[i+1,i+1] = -B*np.cos(theta)/2;
        
    if (verbose > 3): print("h_B:\n", hB);
    return hB;


def alt_spin(B, norbs):
    '''
    Generally, alternate up and down spins on siam sites
    Specifically, prep a half filled siam system to be in a spinless state

    Can remove by putting in -B instead of B
    '''

    #return 0.0;

    # return var
    h_alt_spin = np.zeros((norbs,norbs))

    nsites = int(norbs/2);
    for sitei in range(0, nsites, 2): # consider sites in pairs

        # break up pair and convert to spin orb indices
        site1 = [2*sitei, 2*sitei+1]
        site2 = [2*(sitei+1), 2*(sitei+1)+1]

        # first site gets B that prefers up spins
        h_alt_spin += h_B(-B,0.0,site1,norbs)

        # second site prefers down spins
        h_alt_spin += h_B(B,0.0,site2,norbs);

    return h_alt_spin;

#####################################
#### wrapper functions, test code
    
#####################################
#### exec code

if(__name__ == "__main__"):

    pass;


