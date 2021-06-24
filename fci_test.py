'''
Test how implementing fci.direct_spin1 vs fci.direct_uhf compares
Consider behavior with both spin free and spin blind hamiltonian
'''

import numpy as np
from pyscf import fci

##################################################################################
#### implement 3 lead siam with dot impurity, half filling

# top level params
norbs = 4; # 2 left sites, dot, right site
nelecs = (2,2); # half filling

# physical params
t = 1.0
td = 0.4
Vg = -0.5
U = 1.0

# make spin free hamiltonian
h1e = np.array([ [0.0, -t, 0.0, 0.0],
                 [-t,  0.0, -td, 0.0],
                 [0.0, -td, Vg,  -td],
                 [0.0, 0.0, -td, 0.0],]);
h2e = np.zeros((norbs, norbs, norbs, norbs));
h2e[2,2,2,2] = U;

# do direct fci
cisolver = fci.direct_spin1();
edirect, vdirect = cisolver.kernel(h1e,h2e,norbs,nelecs);
print("\nSpin free, direct fci\n- gd state energy = ",edirect);
