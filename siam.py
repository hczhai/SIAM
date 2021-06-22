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
from pyscf import fci, gto, scf, ao2mo

#################################################
#### functions for creating hamiltonians

def h_leads(V, N):
    '''
    create 1e hamiltonian for leads alone
    V is hopping between leads
    N is number of leads (on each side)
    '''
    
    n_lead_sos = 2*N[0] + 2*N[1]; # 2 spin orbs per lead site
    h = np.zeros((n_lead_sos,n_lead_sos));
    
    # iter over lead sites
    for i in range(2*N[0]-2): # i is spin up orb on left side, i+1 spin down

        h[i,i+2] += -V; # left side
        h[i+2,i] += -V; # h.c.
        
    for i in range(2*N[1]-2):
        
        h[n_lead_sos-1-i,n_lead_sos-1-(i+2)] += -V; # right side
        h[n_lead_sos-1-(i+2),n_lead_sos-1-i] += -V; # h.c.
        
    return h; # end h_leads;


def h_bias(V,N):
    '''
    create 1e hamiltonian for bias voltage
    V is hopping between leads
    N is tuple of number of leads on each side
    '''
    
    n_lead_sos = 2*N[0] + 2*N[1]; # 2 spin orbs per lead site
    h = np.zeros((n_lead_sos,n_lead_sos));
    
    # iter over lead sites
    for i in range(2*N[0]): # i is spin up orb on left side, i+1 spin down

        h[i,i] += V/2; # left side
        
    for i in range(1,2*N[1]+1):
        h[n_lead_sos-i,n_lead_sos-i] += -V/2
        
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
    
        h[i,i] = V;
        
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

#######################################################
#### functions for manipulating basic hamiltonians
    
def stitch_h1e(h_imp, h_imp_leads, h_leads, h_bias, n_leads, verbose = 0):
    '''
    stitch together the various parts of the 1e hamiltonian
    the structure of the final should be block diagonal:
    (Left leads)  V_dl
            V_il  (1e imp ham) V_il
                          V_dl (Right leads)
    '''
    
    # number of spin orbs
    n_imp_sos = np.shape(h_imp)[0];
    n_lead_sos = 2*n_leads[0] + 2*n_leads[1];
    n_spin_orbs = (2*n_leads[0] + 2*n_leads[1] + n_imp_sos);
    
    # combine pure lead ham states
    assert(np.shape(h_leads) == np.shape(h_bias) );#should both be lead sites only
    h_leads = h_leads + h_bias;
    
    # widened ham has leads on outside, dot sites in middle
    h = np.zeros((n_spin_orbs,n_spin_orbs));
    
    # put pure lead elements on top, bottom block diag
    for i in range(2*n_leads[0]):
        for j in range(2*n_leads[0]):
            
            # the first 2*n leads indices are the left leads
            h[i,j] += h_leads[i,j];
            
    for i in range(2*n_leads[1]):
        for j in range(2*n_leads[1]):
        
            # last 2n_lead indices are right leads
            h[n_spin_orbs-1-i,n_spin_orbs-1-j] += h_leads[n_lead_sos-1-i, n_lead_sos-1-j];
        
    # fill in imp and imp-lead elements in middle
    assert(n_imp_sos+4 == np.shape(h_imp_leads)[0]); # 2 spin orbs to left, right
    for i in range(n_imp_sos + 4):
        for j in range(n_imp_sos + 4):
            
            h[2*n_leads[0] - 2 + i, 2*n_leads[0] - 2 + j] += h_imp_leads[i,j];
            if(i>1 and j>1 and i<n_imp_sos+2 and j< n_imp_sos+2): #skip first two, last two rows, columns
                h[2*n_leads[0] - 2 + i, 2*n_leads[0] - 2 + i] += h_imp[i-2,j-2];
            
    if(verbose > 1):
        print("h_leads + h_bias:\n",h_leads,"\nh_imp_leads:\n",h_imp_leads,"\nh_imp:\n",h_imp);
    return h; # end stitch h1e
    
    
def stitch_h2e(h_imp,n_leads,verbose = 0):
    '''
    Put the 2e impurity hamiltonian in the center of the full leads+imp h2e matrix
    h_imp, 4D array, 2e part of impurity hamiltonian
    '''
    
    n_imp_sos = np.shape(h_imp)[0];
    n_lead_sos = 2*n_leads[0] + 2*n_leads[1];
    i_imp = 2*n_leads[0]; # index where imp orbs start
    n_spin_orbs = n_imp_sos + n_lead_sos
    
    h = np.zeros((n_spin_orbs,n_spin_orbs,n_spin_orbs,n_spin_orbs));
    
    for i1 in range(n_imp_sos):
        for i2 in range(n_imp_sos):
            for i3 in range(n_imp_sos):
                for i4 in range(n_imp_sos):
                    h[i_imp+i1,i_imp+i2,i_imp+i3,i_imp+i4] = h_imp[i1,i2,i3,i4];
                    if(verbose > 1): # check 4D tensor by printing nonzero elems
                        if(h_imp[i1,i2,i3,i4] != 0):
                            print("  h_imp[",i1,i2,i3,i4,"] = ",h_imp[i1,i2,i3,i4]," --> h2e[",i_imp+i1,i_imp+i2,i_imp+i3,i_imp+i4,"]");
                        
    return h; # end stitch h2e
    
    

def DotModel(nleads, nsites, norbs, nelecs, physical_params,verbose = 0):
    '''
    Run whole SIAM machinery, with impurity a very simple dot model
    Impurity hamiltonian:
    H_imp = H_dot = -V_g sum_i n_i + U n_{i uparrow} n_{i downarrow}
    where i are impurity sites
    '''
    
    # unpack inputs
    V_leads, V_imp_leads, V_bias, V_gate, U = physical_params;
    
    if(verbose): # print inputs
        print("\nInputs:\n- Num. leads = ",nleads,"\n- Num. impurity sites = ",nsites,"\n- nelecs = ",nelecs,"\n- V_leads = ",V_leads,"\n- V_imp_leads = ",V_imp_leads,"\n- V_bias = ",V_bias,"\n- V_gate = ",V_gate, "\n- Hubbard U = ",U);

    #### make full system ham from inputs

    # make, combine all 1e hamiltonians
    hl = h_leads(V_leads, nleads); # leads only
    hb = h_bias(V_bias, nleads);   # bias leads only
    hdl = h_imp_leads(V_imp_leads, nsites); # leads talk to dot
    hd = h_dot_1e(V_gate, nsites); # dot
    h1e = stitch_h1e(hd, hdl, hl, hb, nleads, verbose = verbose); # syntax is imp, imp-leads, leads, bias
    if(verbose > 2):
        print("\n- Full one electron hamiltonian = \n",h1e);
        
    # 2e hamiltonian only comes from impurity
    if(verbose > 2):
        print("\n- Nonzero h2e elements = ");
    hd2e = h_dot_2e(U,nsites);
    h2e = stitch_h2e(hd2e, nleads, verbose = verbose);
    
    #### encode physics of dot model in an SCF obj
    
    # initial guess density matricex
    Pa = np.zeros(norbs)
    Pa[::2] = 1.0
    Pa = np.diag(Pa)
    
    # HF
    mol = gto.M(); # geometry is meaningless
    mol.incore_anyway = True
    mol.nelectron = sum(nelecs)
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args:h1e # put h1e into scf solver
    mf.get_ovlp = lambda *args:np.eye(norbs)
    symmetry = 8; # perm. symmetry of chemists integrals
    mf._eri = ao2mo.restore(symmetry, h2e, norbs) # h2e into scf solver
    mf.kernel(dm0=Pa); # runs to get HF gd state, prints result
    
    # solve gd state of half filled system with FCI
    nroots = 1;
    #h2e = np.zeros((norbs, norbs, norbs, norbs));
    cisolver = fci.direct_nosym.FCI();
    E_fci, v_fci = cisolver.kernel(h1e, h2e, norbs, nelecs, nroots = nroots);
    if(verbose):
        print("**********\nFCI energies, norbs = ",norbs,", nelecs = ",nelecs,", nroots = ",nroots);
        print("- E = ",E_fci);
        
    return;


#####################################
#### wrapper functions, test code

def CurrentWrapper():
    '''

    '''

    # top level inputs
    verbose = 5;
    np.set_printoptions(suppress=True); # no sci notatation printing

    # set up the hamiltonian
    n_leads = (2,1); # left leads, right leads
    n_imp_sites = 1
    norbs = 2*(n_leads[0]+n_leads[1]+n_imp_sites); # num spin orbs
    nelecs = (int(norbs/2),0);

    # physical params, should always be floats
    V_leads = 1.0; # hopping
    V_imp_leads = 0.4; # hopping
    V_bias = 0; # wait till later to turn on current
    V_gate = -0.5; # gate voltage on dot
    U = 1.0; # hubbard repulsion
    params = V_leads, V_imp_leads, V_bias, V_gate, U;

    DotModel(n_leads, n_imp_sites, norbs, nelecs, params, verbose = verbose);
    
def DotWrapper(std_inputs = True):
    '''
    Run whole SIAM machinery, with impurity a very simple dot model
    Impurity hamiltonian:
    H_imp = H_dot = -V_g sum_i n_i + U n_{i uparrow} n_{i downarrow}
    where i are impurity sites
    '''

    verbose = 5;
    np.set_printoptions(suppress=True); # no sci notatation printing

    #### set up the generic parts of the hamiltonian
    n_leads = (3,2); # left leads, right leads
    n_imp_sites = 1
    norbs = 2*(n_leads[0]+n_leads[1]+n_imp_sites); # num spin orbs
    nelecs = (int(norbs/2),0);

    # physical params, should always be floats
    if std_inputs: # input everything same as ruojing's code to compare
        V_leads = 0.0; # hopping
        V_imp_leads = 0.4; # hopping
        V_bias = 0.0; # bias voltage
        V_gate = -0.5; # gate voltage on dot
        U = 1.0; # hubbard repulsion
    else: # custom
        V_leads = 1.0; # hopping
        V_imp_leads = 0.4; # hopping
        V_bias = 0.00; # bias voltage
        V_gate = -0.5; # gate voltage on dot
        U = 1.0; # hubbard repulsion

    if(verbose): # print inputs
        print("\nInputs:\n- Num. leads = ",n_leads,"\n- Num. impurity sites = ",n_imp_sites,"\n- nelecs = ",nelecs,"\n- V_leads = ",V_leads,"\n- V_imp_leads = ",V_imp_leads,"\n- V_bias = ",V_bias,"\n- V_gate = ",V_gate, "\n- Hubbard U = ",U);

    # make, combine all 1e hamiltonians
    hl = h_leads(V_leads, n_leads); # leads only
    hb = h_bias(V_bias, n_leads);   # bias leads only
    hdl = h_imp_leads(V_imp_leads, n_imp_sites); # leads talk to dot
    hd = h_dot_1e(V_gate, n_imp_sites); # dot
    h1e = stitch_h1e(hd, hdl, hl, hb, n_leads, verbose = verbose); # syntax is imp, imp-leads, leads, bias
    if(verbose > 2):
        print("\n- Full one electron hamiltonian = \n",h1e);
        if False:
            eigvals, eigvecs = np.linalg.eigh(h1e);
            print("*****", eigvals);
    
    # 2e hamiltonian only comes from impurity
    if(verbose > 2):
        print("\n- Nonzero h2e elements = ");
    hd2e = h_dot_2e(U,n_imp_sites);
    h2e = stitch_h2e(hd2e, n_leads, verbose = verbose);

    # solve gd state of half filled system with FCI
    nroots = 1;
    #h2e = np.zeros((norbs, norbs, norbs, norbs));
    cisolver = fci.direct_nosym.FCI();
    cisolver.conv_tol = 1e-8
    cisolver.max_cycle = 100;
    E_fci, v_fci = cisolver.kernel(h1e, h2e, norbs, nelecs, nroots = nroots);
    if(verbose):
        print("**********\nFCI energies, norbs = ",norbs,", nelecs = ",nelecs,", nroots = ",nroots);
        print("- E = ",E_fci);
        
    return; # end dot wrapper

    
#####################################
#### exec code

if(__name__ == "__main__"):

    # test machinery on garnet's simple dot model
    DotWrapper();


