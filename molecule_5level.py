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

##########################################
#### create hamiltonians

def h1e(norbs,D,E,alpha):
    '''
    Create one electron part of Silas' model hamiltonian from physical params
    norbs = # spin orbs
    D = z axis spatial anisotropy
    E = xy plane spatial anisotropy
    alpha = SOC strength
    Returns: 2D np array
    '''
    
    # make empty 2d matrix
    # since code is not flexible right now (bad practice)
    my_norbs = 10; # spin orbs
    assert(norbs == my_norbs);
    h = np.zeros((norbs, norbs));

    # single electron terms
    # best practice to use += thru out since we may modify elements twice
    h[0,0] += -4*D; # z axis anisotropy
    h[1,1] += -4*D;
    h[2,2] += -D;
    h[3,3] += -D;
    h[6,6] += -D;
    h[7,7] += -D;
    h[8,8] += -4*D;
    h[9,9] += -4*D;
    h[0,4] += 2*E; # xy plane anisotropy
    h[4,0] += 2*E;
    h[1,5] += 2*E;
    h[5,1] += 2*E;
    h[2,6] += 2*E;
    h[6,2] += 2*E;
    h[3,7] += 2*E;
    h[7,3] += 2*E;
    h[4,8] += 2*E;
    h[8,4] += 2*E;
    h[5,9] += 2*E;
    h[9,5] += 2*E;
    h[0,0] += -2*alpha; # diag SOC
    h[1,1] += 2*alpha;
    h[2,2] += -alpha;
    h[3,3] += alpha;
    h[6,6] += alpha;
    h[7,7] += -alpha;
    h[8,8] += 2*alpha;
    h[9,9] += -2*alpha;
    h[0,3] += alpha; # off diag SOC
    h[3,0] += alpha;
    h[2,5] += alpha;
    h[5,2] += alpha;
    h[4,7] += alpha;
    h[7,4] += alpha;
    h[6,9] += alpha;
    h[9,6] += alpha;
    
    return h;
    
def h2e(norbs, U):
    '''
    Create two electron part of Silas' model hamiltonian from physical params
    norbs = # spin orbs
    U = hubbard repulsion
    Returns: 4D np array
    '''

    # make empty 4d matrix
    # since code is not flexible right now (bad practice)
    my_norbs = 10; # spin orbs
    assert(norbs == my_norbs);
    h = np.zeros((norbs, norbs,norbs,norbs));

    # hubbard terms
    h[0,0,1,1] = 2*U;
    h[2,2,3,3] = 2*U;
    h[4,4,5,5] = 2*U;
    h[6,6,7,7] = 2*U;
    h[8,8,9,9] = 2*U;

    return h;
    
    
##########################################
#### utils

def plot_DOS(energies, sigma, verbose = 0):

    if(verbose):
        print("Plotting DOS")

    # format plot
    fig, ax = plt.subplots();
    ax.set_xlim(energies[0],energies[-1])
    print(energies[0],energies[-1]);
 
 
##########################################
#### wrapper funcs, test code

def Test():

    # top level inputs
    verbose = True;
    np.set_printoptions(suppress=True); # no sci notatation printing
    
    #### solve with spin blind method
    norbs = 10; # spin orbs, d shell
    nelecs = (8,0);
    nroots = 45; # full 8e, 10 orb basis

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

    # implement h1e, h2e
    h1e_mat = h1e(norbs, D, E, alpha);
    h2e_mat = h2e(norbs, U);


    # pass ham to FCI solver kernel to diagonalize
    cisolver = fci.direct_nosym.FCI()
    myroots = 4; # don't print out 45
    E_fci, v_fci = cisolver.kernel(h1e_mat, h2e_mat, norbs, nelecs, nroots = myroots);
    E_fci.sort();
    if(verbose):
        print("\n1. Spin blind solution, nelecs = ",nelecs," nroots = ",myroots);
        print("FCI energies = ",E_fci- E_shift);
        
    # plot DOS
    sigma = 0.1 # gaussian smearing
    plot_DOS(E_fci,sigma, verbose = verbose);
    
    
##########################################
#### exec code

if __name__ == "__main__":

    Test();


