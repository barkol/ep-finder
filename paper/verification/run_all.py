#!/usr/bin/env python3
"""Run the verification scripts V1-V6 (technical notes, section N1; about 40 s in total) and compare their output
with expected/v<k>.txt."""
import glob, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
bad = 0
for s in sorted(glob.glob(os.path.join(HERE, 'v[1-6]_*.py'))):
    k = os.path.basename(s)[:2]
    out = subprocess.run([sys.executable, s], capture_output=True, text=True, check=True).stdout
    same = out == open(os.path.join(HERE, 'expected', k + '.txt')).read()
    bad += not same
    print(f'{os.path.basename(s):40s} {"matches expected output" if same else "DIFFERS from expected output"}')
sys.exit(bad)
