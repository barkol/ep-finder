"""V3: counting step of Theorem 1 for a rational R = A/B (deg B < deg A = n, no common factor).
U = components of {|R|<|c|}, V = components of {|R|>|c|} (on the Riemann sphere), K = number of critical
points of R (zeros of A'B - AB', with multiplicity) in {|R|<|c|}, m = number of cycles of the phase loop.
Claims: m = #U + #V - 1 (tree), m = 2#U - deg R + K (Riemann-Hurwitz), every U contains a zero and
every V a pole; polynomial R: #V = 1, m = n - K.  m from tracking, #U and #V from grid labeling."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import ndimage
rng = np.random.default_rng(7)
def track_cycles(A, B, cabs, nst=3000):
    th = np.linspace(0, 2*np.pi, nst+1)
    r0 = np.roots(np.polysub(A, cabs*B)); cur = r0.copy()
    for t in th[1:]:
        r = np.roots(np.polysub(A, cabs*np.exp(1j*t)*B))
        _, col = linear_sum_assignment(np.abs(cur[:, None]-r[None, :])); cur = r[col]
    _, perm = linear_sum_assignment(np.abs(cur[:, None]-r0[None, :]))
    seen = set(); cyc = []
    for s in range(len(r0)):
        if s in seen: continue
        L = 0; u = s
        while u not in seen: seen.add(u); u = perm[u]; L += 1
        cyc.append(L)
    return sorted(cyc)
res = []; skipped = 0
while len(res) < 120:
    n = rng.integers(2, 6); q = rng.integers(0, n)  # deg B = q < n
    A = np.concatenate([[1], rng.normal(size=n)+1j*rng.normal(size=n)])
    B = rng.normal(size=q+1)+1j*rng.normal(size=q+1)
    if np.abs(np.roots(A)[:, None]-np.roots(B)[None, :]).min(initial=9) < 1e-2: continue
    W = np.polysub(np.polymul(np.polyder(A), B), np.polymul(A, np.polyder(B)))
    cp = np.roots(W); cv = np.abs(np.polyval(A, cp)/np.polyval(B, cp))
    cabs = np.exp(rng.uniform(np.log(0.05), np.log(30)))
    if len(cv) and np.min(np.abs(np.log(cv/cabs))) < 0.15: skipped += 1; continue
    K = int(np.sum(cv < cabs))
    m = len(track_cycles(A, B, cabs))
    # grid labeling on a box that contains {|R|<=|c|}
    pts = np.concatenate([np.roots(A), np.roots(B) if q else []])
    Rb = 2*np.abs(pts).max()+2
    while True:
        z = Rb*np.exp(1j*np.linspace(0, 2*np.pi, 400))
        if np.abs(np.polyval(A, z)/np.polyval(B, z)).min() > 1.5*cabs: break
        Rb *= 1.5
    for npx in (1400, 4000):  # refine once if the grid count is inconsistent (thin necks)
        g = np.linspace(-Rb, Rb, npx); Z = g[None, :]+1j*g[:, None]
        absR = np.abs(np.polyval(A, Z)/np.polyval(B, Z))
        nU = ndimage.label(absR < cabs)[1]; nV = ndimage.label(absR > cabs)[1]
        if m == nU+nV-1: break
    if m != nU+nV-1:
        # a V component around a pole p is at least the disc where |R|>|c|; report its size against the pixel
        pix = 2*Rb/(len(g)-1); ring = pix*np.exp(1j*np.linspace(0, 2*np.pi, 64))
        small = sum(1 for p in (np.roots(B) if q else []) if (np.abs(np.polyval(A, p+ring)/np.polyval(B, p+ring))).min() < cabs)
        print('tree mismatch: n,q,K,m,#U,#V =', n, q, K, m, nU, nV, '; poles whose V-disc is below one pixel:', small)
    res.append((n, q, K, m, nU, nV))
res = np.array(res)
n, q, K, m, nU, nV = res.T
print("cases:", len(res), "skipped (critical value within 15% of |c|):", skipped)
print("m == #U + #V - 1:", np.sum(m == nU+nV-1), "/", len(res))
print("m == 2#U - deg R + K:", np.sum(m == 2*nU-n+K), "/", len(res))
pol = q == 0
print("polynomial R: #V == 1:", np.sum(nV[pol] == 1), "/", pol.sum(), "; m == n - K:", np.sum((m == n-K)[pol]), "/", pol.sum())
print("nontrivial (m < n):", np.sum(m < n))
