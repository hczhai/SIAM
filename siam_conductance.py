'''
Christian Bunker
M^2QM at UF
June 2021

Template:
Use FCI exact diag to solve single impurity anderson model (siam)


pyscf formalism:
- h1e_pq = (p|h|q) p,q spatial orbitals
- h2e_pqrs = (pq|h|rs) chemists notation, <pr|h|qs> physicists notation
- all direct_x solvers assume 4fold symmetry from sum_{pqrs} (don't need to do manually)
- 1/2 out front all 2e terms, so contributions are written as 1/2(2*actual ham term)
- hermicity: h_pqrs = h_qpsr can absorb factor of 1/2

pyscf fci module:
- configuration interaction solvers of form fci.direct_x.FCI()
- diagonalize 2nd quant hamiltonians via the .kernel() method
- .kernel takes (1e hamiltonian, 2e hamiltonian, # spacial orbs, (# alpha e's, # beta e's))
- direct_nosym assumes only h_pqrs = h_rspq (switch r1, r2 in coulomb integral)
- direct_spin1 assumes h_pqrs = h_qprs = h_pqsr = h_qpsr

siam.py
- outputs current vs time (...CurrentWrapper), current vs time vs lead chemical potential (...ConductWrapper) .txt files
'''

import plot

import numpy as np
import matplotlib.pyplot as plt


#################################################
#### manipulate conductance data

def DotConductWrapper():
    '''
    Extract conductance from multiple current runs at different chemical potentials
    '''

    # top level inputs
    verbose = 5; # passed along throughout to control printing
    np.set_printoptions(suppress=True); # no sci notatation printing

    # set up the hamiltonian
    n_leads = (3,3); # left leads, right leads
    n_imp_sites = 1 # code not flexible enough to change this
    imp_i = [n_leads[0]*2, n_leads[0]*2+1 ]; # should be list for generality
    norbs = 2*(n_leads[0]+n_leads[1]+n_imp_sites); # num spin orbs
    nelecs = (int(norbs/2),0);
    timestop, deltat = 10.0, 0.01 # time prop params

    # physical params, should always be floats
    V_leads = 1.0; # hopping
    V_imp_leads = 0.4; # hopping
    V_bias = 0; # wait till later to turn on current
    V_gate = -0.5; # gate voltage on dot
    murange = min(0,1.5*V_gate), max(0,1.5*V_gate); # which mu vals to sweep
    U = 1.0; # hubbard repulsion

    # hold results
    muvals = [];
    timevals = [];
    currentvals = [];


    for mu in np.linspace(murange[0], murange[1], 20):

        print("############################################\n\n mu = ",mu)

        # get h1e, h2e, and scf implementation of SIAM with dot as impurity
        params = V_leads, V_imp_leads, V_bias, mu, V_gate, U;
        h1e, h2e, mol, dotscf = dot_model(n_leads, n_imp_sites, norbs, nelecs, params, verbose = verbose);

        # from scf instance, do FCI
        E_fci, v_fci = scf_FCI(mol, dotscf, verbose = verbose);

        # prepare in dynamic state by turning on bias
        V_bias = -0.005;
        h1e = start_bias(V_bias, imp_i,h1e);
        if(verbose > 2):
            print(h1e)

        # from fci gd state, do time propagation
        time, energy, current = td.TimeProp(h1e, h2e, v_fci, mol, dotscf, timestop, deltat, imp_i, V_imp_leads, kernel_mode = "plot", verbose = verbose);

        # store results
        muvals.append(mu);
        timevals.append(time);
        currentvals.append(current);

    # write to ext file
    fstring = "DotConduct_t.txt"
    np.savetxt(fstring, timevals );
    fstring = "DotConduct_J.txt"
    np.savetxt(fstring, currentvals);
    fstring = "DotConduct_mu.txt"
    np.savetxt(fstring, muvals);

    return; # end conductance wrapper


def MolConductWrapper():
    '''
    Extract conductance from multiple current runs at different chemical potentials

    NOT ready
    '''

    # top level inputs
    verbose = 5; # passed along throughout to control printing
    np.set_printoptions(suppress=True); # no sci notatation printing

    # set up the hamiltonian
    n_leads = (2,2); # left leads, right leads
    n_imp_sites = 1 # code not flexible enough to change this
    imp_i = [n_leads[0]*2, n_leads[0]*2+1 ]; # should be list for generality
    norbs = 2*(n_leads[0]+n_leads[1]+n_imp_sites); # num spin orbs
    nelecs = (int(norbs/2),0);
    timestop, deltat = 10.0, 0.01 # time prop params

    # physical params, should always be floats
    # generic siam params
    V_leads = 1.0; # hopping
    V_imp_leads = 0.4; # hopping
    V_bias = 0; # wait till later to turn on current
    # molecule specific params
    D = 0.5
    E = 0.1
    alpha = 0.01
    U = 1.0; # hubbard repulsion
    murange = min(0,1.5*U), max(0,1.5*U); # which mu vals to sweep

    for mu in np.linspace(murange[0], murange[1], 20):

        print("############################################\n\n mu = ",mu)

        # get h1e, h2e, and scf implementation of SIAM with dot as impurity
        params = V_leads, V_imp_leads, V_bias, mu, V_gate, U;
        h1e, h2e, mol, dotscf = dot_model(n_leads, n_imp_sites, norbs, nelecs, params, verbose = verbose);

        # from scf instance, do FCI
        E_fci, v_fci = scf_FCI(mol, dotscf, verbose = verbose);

        # prepare in dynamic state by turning on bias
        V_bias = -0.005;
        h1e = start_bias(V_bias, imp_i,h1e);
        if(verbose > 2):
            print(h1e)

        # from fci gd state, do time propagation
        time, energy, current = td.TimeProp(h1e, h2e, v_fci, mol, dotscf, timestop, deltat, imp_i, V_imp_leads, kernel_mode = "plot", verbose = verbose);

        # store results
        muvals.append(mu);
        timevals.append(time);
        currentvals.append(current);

    # write to ext file
    fstring = "DotConductWrapper_t.txt"
    np.savetxt(fstring, timevals );
    fstring = "DotConductWrapper_J.txt"
    np.savetxt(fstring, currentvals);
    fstring = "DotConductWrapper_mu.txt"
    np.savetxt(fstring, muvals);

    return; # end conductance wrapper


    for mu in np.linspace(murange[0], murange[1], 20):

        print("############################################\n\n mu = ",mu)

        # get h1e, h2e, and scf implementation of SIAM with dot as impurity
        params = V_leads, V_imp_leads, V_bias, mu, V_gate, U;
        h1e, h2e, mol, dotscf = dot_model(n_leads, n_imp_sites, norbs, nelecs, params, verbose = verbose);

        # from scf instance, do FCI
        E_fci, v_fci = scf_FCI(mol, dotscf, verbose = verbose);

        # prepare in dynamic state by turning on bias
        V_bias = -0.005;
        h1e = start_bias(V_bias, imp_i,h1e);
        if(verbose > 2):
            print(h1e)

        # from fci gd state, do time propagation
        time, energy, current = td.TimeProp(h1e, h2e, v_fci, mol, dotscf, timestop, deltat, imp_i, V_imp_leads, kernel_mode = "plot", verbose = verbose);

        # store results
        muvals.append(mu);
        timevals.append(time);
        currentvals.append(current);

    # write to ext file
    fstring = "DotConductWrapper_t.txt"
    np.savetxt(fstring, timevals );
    fstring = "DotConductWrapper_J.txt"
    np.savetxt(fstring, currentvals);
    fstring = "DotConductWrapper_mu.txt"
    np.savetxt(fstring, muvals);

    return; # end conductance wrapper
        

#################################################
#### manipulate current data

def fourier_current():
  
  return;



#################################################
#### wrappers and test code

def Wrapper():
  
  # unpack data files
  dataf = "DotConductWrapper";
  timevals = np.loadtxt(dataf+"_t.txt");
  Jvals = np.loadtxt(dataf+"_J.txt");
  muvals = np.loadtxt(dataf+"_mu.txt");
  
  # truncate for now
  muvals = muvals[0:2];
  
  # iter over mu vals to FT
  for i, mu in enumerate(muvals):

    Jw_vals = np.fft.fft2(np.array([timevals, Jvals]));

  # test the fft2
  numpts = 1000
  x = np.linspace(0,10,numpts);
  y = np.zeros(numpts);
  square_freq = 1;
  for i in range(numpts):
    xval = x[i]
    if((xval%square_freq) < square_freq/2):
      y[i] = 100;
  yFT = np.fft.fft2(np.array([x,y]));
  plt.plot(x,y)
  plt.plot(x,yFT[1])
  plt.show();
  
  return; # end fourier current




#################################################
#### exec code

if __name__ == "__main__":
  
  Wrapper();
