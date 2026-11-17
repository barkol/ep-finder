#!/usr/bin/env python3
"""Minimum gaps of figure 5(d) (|c| = eps, gap = smallest eigenvalue spacing over one orbit of the swept phase).
Reproduces the numbers typed into paper/data_scripts/build_readout_certification.py (about 2 min):
  network, ideal drift matrix, w = 0.5 eps, phi = 1 (type (3,1)) and phi = pi (type (4)), 2001 orbit points
      -> siec_i = [0.80, 1.19]
  network, link model kappa' = 40 eps, Gamma = 1.5 eps, same w and phi, adaptive tracking of the four slow modes
      -> siec_k = [1.03, 1.20]
  ring of N = 4 modes, ideal, detunings with |delta_k| <= 2 eps optimized by differential evolution for the largest
  min(g_loop, g_ray) (labeled ladder; g_ray is the gap on the labeling ray c = i t |c|, 0 <= t <= 1)
      -> dr_i = [0.75, 0.80]
  the same without the ray term (unlabeled ring)
      -> bez = [1.19, 1.41]
The labeled ladders at kappa' = 40 eps (dr_k = [0.80, 0.85]) come from the per-sigma optimization in the link model
(simulations of the programming ladders); they are not recomputed here."""
import numpy as np
import multiprocessing as mp
from scipy.optimize import differential_evolution
from link_model_types import link_model_type


def roots(d, c):
    return np.roots(np.r_[np.poly(d)[:-1], np.poly(d)[-1] - c])


def gap_loop(d, n=361):
    """Gap on the circle c = e^{i t} (radius |c| = 1)."""
    g = 9
    for t in np.linspace(0, 2 * np.pi, n):
        r = roots(d, np.exp(1j * t)); D = np.abs(r[:, None] - r[None, :]) + 9 * np.eye(len(r)); g = min(g, D.min())
    return g


def gap_ray(d, n=65):
    """Gap on the labeling ray c = i t, 0 <= t <= 1."""
    g = 9
    for t in np.linspace(0, 1, n):
        r = roots(d, 1j * t); D = np.abs(r[:, None] - r[None, :]) + 9 * np.eye(len(r)); g = min(g, D.min())
    return g


def ctype_from(d):
    """Cycle type from the critical values of prod(lambda - alpha_k) for real-rooted detuning ladders."""
    d = np.sort(d); p = np.poly(d); x = np.sort(np.roots(np.polyder(p)).real); v = np.polyval(p, x); cl = [1]
    for vv in v:
        if abs(vv) < 1: cl[-1] += 1
        else: cl.append(1)
    return tuple(sorted(cl, reverse=True))


def best_ring_gap(args):
    N, kind, ray = args
    def f(x):
        d = x - x.mean()
        if np.abs(d).max() > 2: return 1 + np.abs(d).max() - 2
        if ctype_from(d) != kind: return .5
        g = gap_loop(d, 121)
        if ray: g = min(g, gap_ray(d, 33))
        return -g
    r = differential_evolution(f, [(-2, 2)] * N, seed=3, popsize=18, maxiter=150, tol=1e-5, polish=False)
    d = np.sort(r.x - r.x.mean())
    return gap_loop(d, 721)


def network_ideal_gap(phi, w, c=1.):
    b = 1 + np.exp(1j * phi); dd = w * np.exp(1j * phi); g = 9
    for t in np.linspace(0, 2 * np.pi, 2001):
        r = np.roots([1, 0, 0, -c * np.exp(1j * t) * b, -c * np.exp(1j * t) * dd]); g = min(g, (np.abs(r[:, None] - r[None, :]) + 9 * np.eye(4)).min())
    return g


if __name__ == '__main__':
    print('network, ideal:      ', [round(float(network_ideal_gap(p, .5)), 2) for p in (1.0, np.pi)], ' typed [0.80, 1.19]')
    print('network, kappa\'=40:  ', [round(link_model_type(p, 40, w=0.5, c=1.0, Gam=1.5)[1], 2) for p in (1.0, np.pi)], ' typed [1.03, 1.20]')
    with mp.Pool(4) as P:
        g = P.map(best_ring_gap, [(4, (3, 1), True), (4, (4,), True), (4, (3, 1), False), (4, (4,), False)])
    print('ring, labeled:       ', [round(float(x), 2) for x in g[:2]], ' typed [0.75, 0.80]')
    print('ring, unlabeled:     ', [round(float(x), 2) for x in g[2:]], ' typed [1.19, 1.41]')
