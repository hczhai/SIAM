'''
Christian Bunker
M^2QM at UF
June 2021

Template:
Use FCI exact diag to solve single impurity anderson model (siam)

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
'''

import numpy as np
from pyscf import fci

verbose = True;
np.set_printoptions(suppress=True); # no sci notatation printing

#### set up the generic parts of the hamiltonian
n_leads = 2;

# physical params
V_lead = 1;
V_dotlead = 1;
V_bias = 1; # bias voltage

def h_leads(V, N):
    '''
    create 1e hamiltonian for leads alone
    V is hopping between leads
    N is number of leads (on each side)
    '''
    
    h = np.zeros((4*N,4*N)); # N leads on each side, 2N total leads, 4N spin orbs
    
    # iter over lead sites
    for i in range(2*(N-1)): # i is spin up orb on left side, i+1 spin down

        h[i,i+2] = -V; # left side
        h[i+2,i] = -V; # h.c.
        h[4*N-1-i,4*N-1-(i+2)] = -V; # right side
        h[4*N-1-(i+2),4*N-1-i] = -V; # h.c.
        
    return h; # end h_leads;

def h_bias(V,N):
    '''
    create 1e hamiltonian for bias voltage
    V is hopping between leads
    N is number of leads (on each side)
    '''
    
    h = np.zeros((4*N,4*N)); # N leads on each side, 2N total leads, 4N spin orbs
    
    # iter over lead sites
    for i in range(2*N-1): # i is spin up orb on left side, i+1 spin down
    
        h[i,i] = V/2;
        h[i+1,i+1] = V/2;
        h[4*N-1-i,4*N-1-i] = -V/2;
        h[4*N-1-(i+1),4*N-1-(i+1)] = -V/2;
        
    return h; # end h bias
    
print(h_bias(V_bias, n_leads));
