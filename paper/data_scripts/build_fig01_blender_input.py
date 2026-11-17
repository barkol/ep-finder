#!/usr/bin/env python3
"""Figure 1: input of the Blender renders -> data/fig01_blender.json.

For each panel, every eigenvalue branch of data/fig01_branches.npz (every 8th of the 2401 steps) is shifted by the
mean of the starting eigenvalues, scaled by the largest modulus on the orbit and stored as a list of (Re, Im) points."""
import json
import numpy as np
from _paths import data

z=np.load(data('fig01_branches.npz'))
out={}
for k in 'abcd':
    W=z['W_'+k]; W=W-W[0].mean(); W=W/np.abs(W).max()
    out[k]=[[[float(v.real),float(v.imag)] for v in W[::8,j]] for j in range(W.shape[1])]
json.dump(out,open(data('fig01_blender.json'),'w'))
print('wrote data/fig01_blender.json')
