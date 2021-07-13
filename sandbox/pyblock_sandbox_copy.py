'''
pyblock/sandbox.py

Mess around with pyblock3

https://github.com/block-hczhai/pyblock3-preview
'''

import pyblock3.algebra
import block3

import numpy as np
import os
import pickle
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP
from pyblock3.algebra.mpe import MPE, CachedMPE
from pyblock3.symbolic.expr import OpElement, OpNames
from pyblock3.algebra.symmetry import SZ

# Physical Review B 79, 235336 (2009) Eqs. (1) and (2)
def siam(n, tp, t, vg, u, v=0):
    assert n % 2 == 0
    idximp = n // 2 - 1
    idxls = np.arange(0, idximp, dtype=int)
    idxrs = np.arange(idximp + 1, n, dtype=int)
    h1e = np.zeros((n, n))
    g2e = np.zeros((n, n, n, n))
    g2e[idximp, idximp, idximp, idximp] = u / 2
    h1e[idximp, idximp] = vg
    h1e[idximp, idxls[-1]] = h1e[idxls[-1], idximp] = -tp
    h1e[idximp, idxrs[0]] = h1e[idxrs[0], idximp] = -tp
    for il, ilp in zip(idxls, idxls[1:]):
        h1e[ilp, il] = h1e[il, ilp] = -t
    for ir, irp in zip(idxrs, idxrs[1:]):
        h1e[irp, ir] = h1e[ir, irp] = -t
    for il in idxls:
        h1e[il, il] = -v / 2
    for ir in idxrs:
        h1e[ir, ir] = v / 2
    return h1e, g2e, idximp

tp = 0.4
v = 0.005
gs_siam = siam(n=16, tp=tp, t=1, vg=-0.5, u=1)
td_siam = siam(n=16, tp=tp, t=1, vg=-0.5, u=1, v=v)

np.random.seed(1000)

h1e, g2e, idximp = gs_siam

# check symmetry
print(np.max(np.abs(h1e - h1e.T)))
print(np.max(np.abs(g2e - g2e.transpose((1, 0, 2, 3)))))
print(np.max(np.abs(g2e - g2e.transpose((0, 1, 3, 2)))))
print(np.max(np.abs(g2e - g2e.transpose((1, 0, 3, 2)))))

n_sites = len(h1e)
n_elec = n_sites
bond_dim = 500
bond_dim_init = 250
flat = True

fd = FCIDUMP(pg='c1', n_sites=len(h1e), n_elec=n_elec, h1e=h1e, g2e=g2e)
hamil = Hamiltonian(fd, flat=True)
mpo = hamil.build_qc_mpo()
mpo, error = mpo.compress(left=True, cutoff=1E-9, norm_cutoff=1E-9)
mps = hamil.build_mps(bond_dim_init)
print('MPO compression error = ', error)
print('MPO = ', mpo.show_bond_dims())
print('MPS = ', mps.show_bond_dims())

normsq = np.dot(mps.conj(), mps)
init_e = np.dot(mps.conj(), mpo @ mps) / normsq
print('MPS norm = ', np.sqrt(normsq))
print('initial energy = ', init_e)

bdims = [bond_dim_init, bond_dim // 2, bond_dim]
noises = [1E-4, 1E-5, 1E-6, 0]
davthrds = [1E-8] * 20
n_sweeps = 20
do_ground_state = True



