'''
playground.py

Mess around with pyscf

https://sunqm.github.io/pyscf/tutorial.html
'''

import pyscf

mol = pyscf.gto.Mole(); # creates molecule
mol.verbose = 0; # how much printout

#run in spin singlet and triplet states
for spin in [0,2]:
    try: # not all spins work
        mol.build(atom = '''O 0 0 0; O 0 0 1.2''', spin = spin, basis = 'ccpvdz');
    except RuntimeError as e:
        print(e);

    # HF energy
    m = pyscf.scf.RHF(mol);
    print('E(HF) = %g' % m.kernel())
    
