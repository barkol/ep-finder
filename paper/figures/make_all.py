#!/usr/bin/env python3
"""Draw every figure of the paper and of the technical notes from the cached data (output/*.pdf)."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ['fig01_permutations.py', 'fig02_undetuned_ring.py', 'fig03_programming.py', 'fig04_spectrum_state_group.py',
           'fig05_interference.py', 'fig06_readout.py', 'figN1_readout_maps.py', 'figN2_ladders.py']
for s in SCRIPTS:
    print('==', s, flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, s)], check=True)
