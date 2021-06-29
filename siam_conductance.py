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

#################################################
#### manipulate current data

def fourier_current():
  
  return;



#################################################
#### wrappers and test code

def Wrapper():
  
  # unpack data files
  dataf = "";
  timevals = np.loadtxt(dataf+"_time.txt");
  currentvals = np.loadtxt(dataf+"_J.txt");
  muvals = np.loadtxt(dataf+"_mu.txt");
  
  # truncate for now
  muvals = muvals[0];
  
  # iter over mu vals to FT
  for i, mu in enumerate(muvals)
  
  return; # end fourier current



#################################################
#### exec code

if __name__ == "__main__":
  
  Wrapper();
