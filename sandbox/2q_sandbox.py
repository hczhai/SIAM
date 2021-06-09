'''
2q_sandbox.py

play around with the second quantized functions
  des_a: removes up electron
  des_b: removes down electron
  cre_a: create up electron
  cre_b: create down electron
'''

import numpy as np
from pyscf import fci

verbose = True;

#### play with det strings
det = [0,3,9];
nelec = 1;
alldetstrings = fci.cistring.make_strings(det, nelec); # all binary string occ possibilites
                                                # encoded as (decimal) int
    
state = fci.cistring._gen_occslst(det, nelec);
if(verbose):
    print("input occupied orbital list = \n",state);

#### create 1e state from vaccuum

# input controls
norb = 3; # number molecular orbitals
nanb = 1,1; # number up, down electrons
orbi = 1; # which orbital to fill

# create e
for i in range(1,3):
    state = fci.addons.cre_a(state, norb, nanb, i ); # returns slater det

    if(verbose):
        print(i, "output occupied orbital list = \n", state);




