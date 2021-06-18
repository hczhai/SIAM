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

#################################################
#### functions for creating hamiltonians

def h_leads(V, N):
    '''
    create 1e hamiltonian for leads alone
    V is hopping between leads
    N is number of leads (on each side)
    '''
    
    h = np.zeros((4*N,4*N)); # N leads on each side, 2N total leads, 4N spin orbs
    
    # iter over lead sites
    for i in range(2*(N-1)): # i is spin up orb on left side, i+1 spin down

        h[i,i+2] += -V; # left side
        h[i+2,i] += -V; # h.c.
        h[4*N-1-i,4*N-1-(i+2)] += -V; # right side
        h[4*N-1-(i+2),4*N-1-i] += -V; # h.c.
        
    return h; # end h_leads;


def h_bias(V,N):
    '''
    create 1e hamiltonian for bias voltage
    V is hopping between leads
    N is number of leads (on each side)
    '''
    
    h = np.zeros((4*N,4*N)); # N leads on each side, 2N total leads, 4N spin orbs
    
    # iter over lead sites
    for i in range(0,2*N-1,2): # i is spin up orb on left side, i+1 spin down
    
        h[i,i] += V/2;
        h[i+1,i+1] += V/2;
        h[4*N-1-i,4*N-1-i] += -V/2;
        h[4*N-1-(i+1),4*N-1-(i+1)] += -V/2;
        
    return h; # end h bias
    
    
def h_imp_leads(V,N):
    '''
    create 1e hamiltonian for e's hopping on and off impurity levels
    V is hopping between impurity, leads
    N is number of impurity levels
    '''
    
    h = np.zeros((2+2*N+2,2+2*N+2)); # 2N spin orbs on imp, 1st, last 2 are neighboring lead sites
    Liup,Lidown, Riup, Ridown = 0,1,2+2*N, 2+2*N + 1;
    
    # iter over dot sites
    for i in range(2,2+2*N,2): # i is spin up orb on imp, i+1 is spin down
    
        h[Liup,i] += -V;
        h[i,Liup] += -V; # h.c.
        h[Lidown,i+1] += -V;
        h[i+1,Lidown] += -V;
        h[Riup, i] += -V;
        h[i, Riup] += -V;
        h[Ridown,i+1] += -V;
        h[i+1,Ridown] += -V;
        
    return h; # end h imp leads
    
    
def h_dot_1e(V,N):
    '''
    create 1e part of dot hamiltonian
    dot is simple model of impurity
    V is gate voltage (ie onsite energy of dot sites)
    N is number of dot sites
    '''
    
    h = np.zeros((2*N,2*N));
    
    # on site energy for each dot site
    for i in range (2*N):
    
        h[i,i] = -V;
        
    return h; # end h dot 1e
    
def h_dot_2e(U,N):
    '''
    create 2e part of dot hamiltonian
    dot is simple model of impurity
    U is hubbard repulsion
    N is number of dot sites
    '''
    
    h = np.zeros((2*N,2*N,2*N,2*N));
    
    # hubbard repulsion when there are 2 e's on same MO
    for i in range(0,N,2): # i is spin up orb, i+1 is spin down
        h[i,i,i+1,i+1] = 2*U;
        
    return h; # end h dot 2e

#################################################
#### functions for manipulating hamiltonians
    
def stitch_h1e(h_imp, h_imp_leads, h_leads, h_bias, verbose = 0):
    '''
    stitch together the various parts of the 1e hamiltonian
    the structure of the final should be block diagonal:
    (Left leads)  V_dl
            V_il  (1e imp ham) V_il
                          V_dl (Right leads)
    '''
    
    # number of spin orbs
    assert(np.shape(h_leads)[0] % 2 == 0); # should be even bc spin orbs
    n_lead_sos = int(np.shape(h_leads)[0]/2);
    n_imp_sos = np.shape(h_imp)[0];
    n_spin_orbs = (2*n_lead_sos + n_imp_sos);
    
    # combine pure lead ham states
    assert(np.shape(h_leads) == np.shape(h_bias) );#should both be lead sites only
    h_leads = h_leads + h_bias;
    
    # widened ham has leads on outside, dot sites in middle
    h = np.zeros((n_spin_orbs,n_spin_orbs));
    
    # put pure lead elements on top, bottom block diag
    for i in range(n_lead_sos):
        for j in range(n_lead_sos):
            
            # the first 2*n leads indices are the left leads
            h[i,j] = h_leads[i,j];
            
            # last 2n_lead indices are right leads
            # these count up from n_spin_orbs - n_lead_sos
            h[n_spin_orbs-n_lead_sos+i,n_spin_orbs-n_lead_sos+j] = h_leads[i+n_lead_sos, j+n_lead_sos];
           
    # fill in imp and imp-lead elements in middle
    assert(n_imp_sos+4 == np.shape(h_imp_leads)[0]); # 2 spin orbs to left, right
    for i in range(n_imp_sos + 4):
        for j in range(n_imp_sos + 4):
            
            h[n_lead_sos - 2 + i, n_lead_sos - 2 + j] += h_imp_leads[i,j];
            if(i>1 and j>1 and i<n_imp_sos+2 and j< n_imp_sos+2): #skip first two, last two rows, columns
                h[n_lead_sos - 2 + i, n_lead_sos - 2 + j] += h_imp[i-2,j-2];
            
    if(verbose > 2):
        print("h_leads + h_bias:\n",h_leads,"\nh_imp_leads:\n",h_imp_leads,"\nh_imp:\n",h_imp);
    return h; # end stitch h1e
    
def stitch_h2e(h_imp,n_leads,verbose = 0):
    '''
    Put the 2e impurity hamiltonian in the center of the full leads+imp h2e matrix
    h_imp, 4D array, 2e part of impurity hamiltonian
    '''
    
    n_imp_sos = np.shape(h_imp)[0];
    n_lead_sos = 2*n_leads;
    n_spin_orbs = n_imp_sos + 2*n_lead_sos
    
    h = np.zeros((n_spin_orbs,n_spin_orbs,n_spin_orbs,n_spin_orbs));
    
    for i1 in range(n_imp_sos):
        for i2 in range(n_imp_sos):
            for i3 in range(n_imp_sos):
                for i4 in range(n_imp_sos):
                    h[n_lead_sos+i1,n_lead_sos+i2,n_lead_sos+i3,n_lead_sos+i4] = h_imp[i1,i2,i3,i4];
                    if(verbose > 1): # check 4D tensor by printing nonzero elems
                        if(h_imp[i1,i2,i3,i4] != 0):
                            print("  h_imp[",i1,i2,i3,i4,"] = ",h_imp[i1,i2,i3,i4]);
                        
    return h; # end stitch h2e


#####################################
#### wrapper functions, test code

def DotWrapper():
    '''
    Run whole SIAM machinery, with impurity a very simple dot model
    Impurity hamiltonian:
    H_imp = H_dot = -V_g sum_i n_i + U n_{i uparrow} n_{i downarrow}
    where i are impurity sites
    '''

    verbose = 2;
    np.set_printoptions(suppress=True); # no sci notatation printing

    #### set up the generic parts of the hamiltonian
    n_leads = 2;
    n_imp_sites = 1

    # physical params
    V_leads = 3; # hopping
    V_imp_leads = 4; # hopping
    V_bias = 1; # bias voltage
    V_gate = 10; # gate voltage on dot
    U = 50; # hubbard repulsion
    if(verbose): # print inputs
        print("\nInputs:\n- Num. leads = ",n_leads,"\n- Num. impurity sites = ",n_imp_sites,"\n- V_leads = ",V_leads,"\n- V_imp_leads = ",V_imp_leads,"\n- V_bias = ",V_bias,"\n- V_gate = ",V_gate, "\n- Hubbard U = ",U);

    # make, combine all 1e hamiltonians
    hl = h_leads(V_leads, n_leads);
    hb = h_bias(V_bias, n_leads);
    hdl = h_imp_leads(V_imp_leads, n_imp_sites);
    hd = h_dot_1e(V_gate, n_imp_sites);
    h1e = stitch_h1e(hd, hdl, hl, hb, verbose = verbose);
    if(verbose > 1):
        print("\n- Full one electron hamiltonian = \n",h1e);
        
    # 2e hamiltonian only comes from impurity
    if(verbose > 1):
        print("\n- Nonzero h2e elements = ");
    hd2e = h_dot_2e(U,n_imp_sites);
    h2e = stitch_h2e(hd2e, n_leads, verbose = verbose);
    
    # solve gd state of half filled system with FCI
    norbs = 2*(2*n_leads+n_imp_sites); # spin orbs
    nelecs = (int(norbs/2),0);
    nroots = 5;
    cisolver = fci.direct_nosym.FCI();
    E_fci, v_fci = cisolver.kernel(h1e, h2e, norbs, nelecs, nroots = nroots);
    if(verbose):
        print("**********\nFCI energies, norbs = ",norbs,", nelecs = ",nelecs,", nroots = ",nroots);
        for E in E_fci:
            print("- E = ",E);
        
    return; # end dot wrapper
    
#####################################
#### exec code

if(__name__ == "__main__"):

    # test machinery on garnet's simple dot model
    DotWrapper();


