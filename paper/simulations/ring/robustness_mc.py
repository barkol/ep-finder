#!/usr/bin/env python3
"""Figure 2(c,d): outcome of a feedback-phase orbit against disorder strength, N = 3, |c| = 1 -> results/robustness_mc.npz.
Disorder strength f = sigma / delta_min, from 0.02 to 30, so that the decay of P at large disorder is
visible. Classes det_re (real diagonal, equivalent to pure detunings) and coupling (ring_tools.draw).
NR draws per point. One resumable part file per (class, f) in work/robustness_parts, merged at the end.
Workers: environment variable RING_WORKERS (default 8); multiprocessing 'fork', i.e. Linux/macOS."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "1"
import sys, time
import multiprocessing as mp_
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ring_tools as L
from _paths import work, results

HERE = os.path.dirname(os.path.abspath(__file__))
PART = work("robustness_parts")
OUT = results("robustness_mc.npz")
SEED = 20261105; DMIN = np.sqrt(3.0); NR = 20000; CHUNK = 4000
FS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0])
CL = ("det_re", "coupling")

def unit(arg):
    ic, ifs = arg
    path = os.path.join(PART, f"c{ic}_f{ifs:02d}.npz")
    if os.path.exists(path): return path
    rng = np.random.default_rng(np.random.SeedSequence([SEED, ic, ifs]))
    cnt = np.zeros(4, int); un = 0
    for k in range(0, NR, CHUNK):
        D = L.draw(CL[ic], np.full(min(CHUNK, NR - k), FS[ifs] * DMIN), rng)
        perm, _, u = L.batch_monodromy(D)
        cnt += np.bincount(L.type_code(perm), minlength=4); un += u
    tmp = path + ".tmp.npz"; np.savez(tmp, cnt=cnt, unresolved=un); os.replace(tmp, path)
    return path

if __name__ == "__main__":
    os.makedirs(PART, exist_ok=True); t0 = time.time()
    jobs = [(c, i) for c in range(2) for i in range(len(FS))]
    with mp_.get_context("fork").Pool(int(os.environ.get("RING_WORKERS", "8"))) as pool:
        for i, _ in enumerate(pool.imap_unordered(unit, jobs)):
            print(f"{i+1}/{len(jobs)} {time.time()-t0:.0f} s", flush=True)
    cnt = np.zeros((2, len(FS), 4), int); un = 0
    for c, i in jobs:
        z = np.load(os.path.join(PART, f"c{c}_f{i:02d}.npz")); cnt[c, i] = z["cnt"]; un += int(z["unresolved"])
    assert (cnt.sum(axis=2) == NR).all()
    tmp = OUT + ".tmp.npz"; np.savez(tmp, fs=FS, cnt=cnt, classes=np.array(CL), codes=np.array(["correct", "reversed", "transposition", "identity"]), unresolved=un, NR=NR)
    os.replace(tmp, OUT); print("merged", OUT, "unresolved", un, flush=True)
