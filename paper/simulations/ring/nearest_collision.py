#!/usr/bin/env python3
"""Radius on which the complex loop realizes the N-cycle (figure 2(e), appendix C) -> results/ring_nearest_collision.npz
and data/tab_radius.csv.

For the ring drift matrix M(alpha, c), |c| = 1, loop c = e^{i theta}: the sufficient radius is the Bauer-Fike bound
max_k |alpha_k| < sin(pi/N) (in units of |c|^{1/N}). The nearest collision, an upper bound on the true radius, is
min max_k |alpha_k| subject to M(alpha, e^{i theta}) having a double eigenvalue lambda0 (chi = chi' = 0) and
sum alpha = 0, found by SLSQP from 300 random starts per N. Validation: the monodromy is the N-cycle at 0.98 of the
collision and not at 1.02 of it. Resumable per N."""
import os, time, csv
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from _paths import results, data

OUT = results('ring_nearest_collision.npz')

def ring(alpha, c):
    N = len(alpha); M = np.diag(alpha).astype(complex)
    M[np.arange(N - 1), np.arange(1, N)] = 1.0; M[N - 1, 0] = c
    return M

def min_gap_on_loop(alpha, ngrid=256):
    """Minimal pairwise eigenvalue distance along c = e^{i theta}, grid + local refinement."""
    def gap(th):
        e = np.linalg.eigvals(ring(alpha, np.exp(1j * th)))
        d = np.abs(e[:, None] - e[None, :]); d[np.diag_indices_from(d)] = np.inf
        return d.min()
    ths = np.linspace(0, 2 * np.pi, ngrid, endpoint=False)
    g = np.array([gap(t) for t in ths]); k = int(np.argmin(g))
    r = minimize_scalar(gap, bounds=(ths[k] - 2 * np.pi / ngrid, ths[k] + 2 * np.pi / ngrid),
                        method="bounded", options={"xatol": 1e-10})
    return min(r.fun, g[k])

def perm_type_is_Ncycle(alpha, nth=720):
    """Track the eigenvalues once around the loop and test whether the monodromy is an N-cycle."""
    from scipy.optimize import linear_sum_assignment
    N = len(alpha); ths = np.linspace(0, 2 * np.pi, nth + 1)
    lam = np.linalg.eigvals(ring(alpha, 1.0)); start = lam.copy()
    for th in ths[1:]:
        e = np.linalg.eigvals(ring(alpha, np.exp(1j * th)))
        _, c = linear_sum_assignment(np.abs(lam[:, None] - e[None, :])); lam = e[c]
    _, p = linear_sum_assignment(np.abs(lam[:, None] - start[None, :]))
    seen, n, j = set(), 0, 0
    while j not in seen:
        seen.add(j); j = p[j]; n += 1
    return n == N

def unpack(x, N):
    a = x[:N-1] + 1j * x[N-1:2*N-2]; a = np.append(a, -a.sum())
    lam = x[2*N-2] + 1j * x[2*N-1]; th = x[2*N]; s = x[2*N+1]
    return a, lam, th, s

def cons(x, N):
    a, lam, th, s = unpack(x, N)
    p = np.poly(ring(a, np.exp(1j * th)))
    v = np.polyval(p, lam); d = np.polyval(np.polyder(p), lam)
    return np.array([v.real, v.imag, d.real, d.imag])

def nearest(N, nstart, rng):
    best = None
    ncon = {"type": "eq", "fun": lambda x: cons(x, N)}
    icon = {"type": "ineq", "fun": lambda x: x[2*N+1]**2 - np.abs(unpack(x, N)[0])**2}
    for _ in range(nstart):
        r0 = rng.uniform(0.3, 3.0)
        a0 = r0 * (rng.normal(size=N) + 1j * rng.normal(size=N)) / np.sqrt(2)
        th0 = rng.uniform(0, 2*np.pi); lam0 = np.exp(1j * th0 / N) * rng.uniform(0.2, 1.2) * np.exp(2j*np.pi*rng.integers(N)/N)
        x0 = np.concatenate([a0[:N-1].real, a0[:N-1].imag, [lam0.real, lam0.imag, th0, np.abs(a0).max()]])
        r = minimize(lambda x: x[2*N+1], x0, constraints=[ncon, icon], method="SLSQP",
                     options={"maxiter": 500, "ftol": 1e-12})
        if not r.success or np.abs(cons(r.x, N)).max() > 1e-8:
            continue
        a, lam, th, s = unpack(r.x, N); s = np.abs(a).max()
        if best is None or s < best[0]:
            best = (s, a, th, lam)
    return best

def run_all(out=OUT, Ns=range(3, 15), nstart=300):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = dict(np.load(out, allow_pickle=True)) if os.path.exists(out) else {}
    rng = np.random.default_rng(20260923)
    for N in Ns:
        if f"upper_{N}" in done:
            print(f"N={N}: cached", flush=True); continue
        t0 = time.time(); s, a, th, lam = nearest(N, nstart, rng)
        below, above = perm_type_is_Ncycle(0.98 * a), perm_type_is_Ncycle(1.02 * a)
        done.update({f"upper_{N}": s, f"lower_{N}": np.sin(np.pi / N), f"alpha_{N}": a, f"theta_{N}": th,
                     f"valid_{N}": np.array([below, not above])})
        np.savez(out + ".tmp.npz", **done); os.replace(out + ".tmp.npz", out)
        print(f"N={N}: sin(pi/N)={np.sin(np.pi/N):.4f} <= rho <= {s:.4f}  validated={below and not above} "
              f"({time.time()-t0:.0f} s)", flush=True)

def write_table(out=OUT):
    D = np.load(out, allow_pickle=True)
    with open(data('tab_radius.csv'), 'w', newline='') as f:
        c = csv.writer(f); c.writerow(['N', 'sufficient', 'nearest_collision', 'ratio'])
        for N in range(3, 15):
            lo, up = float(D[f"lower_{N}"]), float(D[f"upper_{N}"])
            assert bool(np.all(D[f"valid_{N}"])), f"N={N} not validated"
            c.writerow([N, f"{lo:.3f}", f"{up:.3f}", f"{up/lo:.2f}"])
    print('wrote data/tab_radius.csv')


if __name__ == "__main__":
    run_all()
    write_table()
