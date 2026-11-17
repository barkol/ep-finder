#!/usr/bin/env python3
"""Tabulated results behind figures 2 and N2 -> data/tables.npz (the plotting layer reads this file).
Sources (exact values): data/tab_radius.csv (radius of persistence and nearest collision, N=3..14),
data/tab_sweep_time.csv (adiabatic and orbit times), data/tab_link_model.csv (link model against kappa'),
data/tab_ladders.csv (ladders per cycle type, ideal drift matrix) and data/tab_link_devices.csv
(the same devices in the link model at kappa'=20)."""
import csv
import numpy as np
from _paths import data


def rows(name):
    with open(data(name), newline='') as f:
        return list(csv.DictReader(f))


R = rows('tab_radius.csv'); TN = rows('tab_sweep_time.csv'); K = rows('tab_link_model.csv')
P = rows('tab_ladders.csv'); V = {(r['N'], r['type']): r for r in rows('tab_link_devices.csv')}
fl = lambda rs, k: [float(r[k]) for r in rs]
gap = [r['gap'].split('--') for r in K]
win = [r['c_window'].split('--') for r in P]
link = lambda k: [float(V[(r['N'], r['type'])][k]) if (r['N'], r['type']) in V else np.nan for r in P]
out = dict(
    rN=fl(R, 'N'), rs=fl(R, 'sufficient'), rc=fl(R, 'nearest_collision'),
    tN=fl(TN, 'N'), td=fl(TN, 'delta_min'), ta=fl(TN, 'eT_adiab'), to=fl(TN, 'eT_orbit'), tg=fl(TN, 'mean_dGamma'),
    kk=fl(K, 'kappa'), kg0=[float(g[0]) for g in gap], kg1=[float(g[1]) for g in gap], kr=fl(K, 'rev_forw'),
    kl=fl(K, 'dlambda'), kS=fl(K, 'dS21'),
    pN=[int(r['N']) for r in P], pt=[r['type'] for r in P], pg=fl(P, 'g_loop'), pm=fl(P, 'g_max'), pT=fl(P, 'T_ad_us'),
    pd=fl(P, 'delta_max'), c0=[float(w[0]) for w in win], c1=[np.inf if w[1] == 'infty' else float(w[1]) for w in win],
    xd=fl(P, 'x99_det'), xc=fl(P, 'x99_cplx'), vg=link('g_loop'), vT=link('T_ad_us'), vm=link('delta_max'))
# checks against closed forms: sufficient radius sin(pi/N), gap 2 sin(pi/N)
assert np.allclose(out['rs'], np.sin(np.pi / np.array(out['rN'])), atol=6e-4)
assert np.allclose(out['td'], 2 * np.sin(np.pi / np.array(out['tN'])), atol=6e-4)
np.savez(data('tables.npz'), **out)
print('closed forms agree; ladder types also in the link model:', sum((r['N'], r['type']) in V for r in P))
print('wrote data/tables.npz')
