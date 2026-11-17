#!/usr/bin/env python3
"""Rebuild the cached figure data in ../data from the result files in ../results (a few seconds).
The Blender renders of figure 1 are not rebuilt here; see render_fig01_blender.py."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for s in ['build_fig01_branches.py', 'build_fig01_blender_input.py', 'build_ring_dynamics.py', 'build_tables.py',
          'build_readout_certification.py', 'build_readout_maps.py']:
    print('==', s, flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, s)], check=True)
