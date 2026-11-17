#!/usr/bin/env python3
"""Tools for the ring: exact discriminant sign counts and eigenvalue continuation. No side effects on import.

Exact part: sign changes of the discriminant D(phi) on the real contour
alpha_1 = r cos(phi), c = r sin(phi), offsets alpha_{m+2} = s (m+1), alpha_N = -sum,
through the Weierstrass substitution u = tan(phi/2), square-free decomposition over Z
and Sturm sequences. Nothing here is floating point.

Numerical part: adaptive eigenvalue continuation. A step is accepted only when every
root moves by less than 0.2 of the smallest gap at both ends, otherwise it is halved.
"""
from fractions import Fraction
import numpy as np
from flint import fmpq_mpoly_ctx, fmpq_poly, fmpz_poly, fmpq, Ordering
import mpmath as mp

# ---------------------------------------------------------------- exact counting
def _q(x):
    x = Fraction(x); return fmpq(x.numerator, x.denominator)

def offsets_exact(N, s):
    return [Fraction(s) * (m + 1) for m in range(N - 2)]

def disc_terms(N, s, b=None):
    """Discriminant of chi = (l^2 + B l - w) Q(l) - c as a list ((i, j), coeff): w^i c^j, w = a(a+B).
    Offsets are s(m+1), or the explicit list b of N-2 Fractions."""
    ctx = fmpq_mpoly_ctx.get(('l', 'w', 'c'), Ordering.lex)
    l, w, c = ctx.gens()
    b = offsets_exact(N, s) if b is None else list(b); B = sum(b, Fraction(0))
    Q = l ** 0
    for x in b:
        Q = Q * (l - ctx.constant(_q(x)))
    chi = (l * l + ctx.constant(_q(B)) * l - w) * Q - c
    D = chi.discriminant('l')
    out = []
    for mon, co in D.to_dict().items():
        assert mon[0] == 0
        out.append(((int(mon[1]), int(mon[2])), co))
    return out, B

def disc_numerator(terms, B, r):
    """Numerator N(u) of D(phi) = N(u) / (1+u^2)^K; returns (N, K)."""
    r = _q(r); B = _q(B)
    one = fmpq_poly([1]); u = fmpq_poly([0, 1]); den = one + u * u
    W = (one - u * u) * r * ((one - u * u) * r + den * B)
    C = u * 2 * r
    K = max(2 * i + j for (i, j), _ in terms)
    imax = max(i for (i, _), _ in terms); jmax = max(j for (_, j), _ in terms)
    Wp = [one]; Cp = [one]; Dp = [one]
    for _ in range(imax): Wp.append(Wp[-1] * W)
    for _ in range(jmax): Cp.append(Cp[-1] * C)
    for _ in range(K): Dp.append(Dp[-1] * den)
    Nu = fmpq_poly([0])
    for (i, j), co in terms:
        Nu += Wp[i] * Cp[j] * Dp[K - 2 * i - j] * co
    return Nu, K

def _primitive(p):
    """fmpq_poly -> primitive fmpz_poly with the same sign of the leading coefficient."""
    z = fmpz_poly(p.numer().coeffs())
    g = z.content()
    return fmpz_poly([x // g for x in z.coeffs()]) if g != 1 else z

def sturm_count(f):
    """Number of distinct real roots of the square-free fmpz_poly f."""
    if f.degree() <= 0: return 0
    seq = [fmpq_poly(f.coeffs()), fmpq_poly(f.derivative().coeffs())]
    while seq[-1].degree() > 0:
        r = -(seq[-2] % seq[-1])
        if r.degree() < 0: break
        c = _primitive(r)
        # keep the sign of r when clearing denominators and contents (both positive)
        lead_r = r.coeffs()[-1]; lead_c = c.coeffs()[-1]
        c = c if (lead_r > 0) == (lead_c > 0) else fmpz_poly([-x for x in c.coeffs()])
        seq.append(fmpq_poly(c.coeffs()))
    lead = [p.coeffs()[-1] for p in seq]
    def var(signs):
        v = 0; prev = 0
        for x in signs:
            if x == 0: continue
            if prev != 0 and (x > 0) != (prev > 0): v += 1
            prev = x
        return v
    plus = var(lead)
    minus = var([x if p.degree() % 2 == 0 else -x for x, p in zip(lead, seq)])
    return minus - plus

def sign_changes(terms, B, r, cross_check=False):
    """Exact number of sign changes of D(phi) on phi in [0, 2 pi): odd-multiplicity zeros."""
    Nu, K = disc_numerator(terms, B, r)
    z = _primitive(Nu)
    _, fac = z.factor_squarefree()
    n = 0
    for f, m in fac:
        if int(m) % 2 == 1:
            k = sturm_count(f)
            if cross_check:
                assert k == sum(1 for x, _ in f.complex_roots() if x.imag == 0), "Sturm disagrees with arb isolation"
            n += k
    if (2 * K - z.degree()) % 2 == 1:      # zero of odd order at u = infinity, that is phi = pi
        n += 1
    assert n % 2 == 0, "a sign function on a circle changes sign an even number of times"
    return n

# ---------------------------------------------------------------- critical values
def critical_values(alpha, dps=60):
    """Critical values P(mu_j), P = prod(l - alpha_k), mu_j the roots of P'. High precision."""
    mp.mp.dps = dps
    co = [mp.mpf(1)]
    for a in alpha:
        a = mp.mpc(a) if isinstance(a, complex) else mp.mpf(a.numerator) / a.denominator if isinstance(a, Fraction) else mp.mpf(a)
        new = co + [mp.mpf(0)]
        for i in range(len(co)):
            new[i + 1] -= a * co[i]
        co = new
    n = len(co) - 1
    dco = [co[i] * (n - i) for i in range(n)]
    mus = mp.polyroots(dco, maxsteps=2000, extraprec=800)
    return [mp.polyval(co, m) for m in mus]

# ---------------------------------------------------------------- tracking
def ring(alpha, c):
    alpha = np.asarray(alpha, complex); N = len(alpha)
    M = np.diag(alpha)
    M[np.arange(N - 1), np.arange(1, N)] = 1.0
    M[N - 1, 0] = c
    return M

def _gap(e):
    d = np.abs(e[:, None] - e[None, :]); np.fill_diagonal(d, np.inf)
    return d.min()

def track(Mfun, t0=0.0, t1=1.0, h0=1 / 256, hmin=1e-11, frac=0.2):
    """Continue eigenvalues of Mfun(t) on [t0, t1]. Returns (start, end, W, min_gap, ok).
    end[k] is the continuation of start[k]; W is the winding of the discriminant."""
    t = t0; e = np.linalg.eigvals(Mfun(t)); start = e.copy(); h = h0
    arg = 0.0; mg = _gap(e); ok = True
    iu = np.triu_indices(len(e), 1)
    while t < t1 - 1e-15:
        h = min(h, t1 - t)
        f = np.linalg.eigvals(Mfun(t + h))
        idx = np.argmin(np.abs(e[:, None] - f[None, :]), axis=1)
        g = min(_gap(e), _gap(f))
        good = len(set(idx.tolist())) == len(e) and np.max(np.abs(f[idx] - e)) < frac * g
        if not good:
            if h > hmin:
                h /= 2; continue
            ok = False
            if len(set(idx.tolist())) != len(e):      # unresolved step: assign optimally
                from scipy.optimize import linear_sum_assignment
                _, idx = linear_sum_assignment(np.abs(e[:, None] - f[None, :]))
        fn = f[idx]
        d0 = e[iu[0]] - e[iu[1]]; d1 = fn[iu[0]] - fn[iu[1]]
        arg += 2 * np.angle(d1 / d0).sum()
        e = fn; t += h; mg = min(mg, _gap(e))
        h = min(h * 1.5, h0)
    return start, e, arg / (2 * np.pi), mg, ok

def perm_of(start, end):
    idx = np.argmin(np.abs(end[:, None] - start[None, :]), axis=1)
    return idx

def cycle_type(p):
    p = list(p); seen = set(); out = []
    for i in range(len(p)):
        if i in seen: continue
        n = 0; j = i
        while j not in seen:
            seen.add(j); j = int(p[j]); n += 1
        out.append(n)
    return tuple(sorted(out, reverse=True))

# ---------------------------------------------------------------- batch Monte Carlo, N = 3 by default
TYPE_NAMES = {0: "correct", 1: "reversed", 2: "transposition", 3: "identity"}

def _assign_path(evs, e_start=None):
    """evs: (n, m, N) eigenvalues on a mesh. Nearest-neighbour continuation with a validity test.
    Returns the end values ordered as the start, min gap and a flag for unresolved samples."""
    n, m, N = evs.shape
    e = evs[:, 0] if e_start is None else e_start
    bad = np.zeros(n, bool); mg = np.full(n, np.inf)
    rows = np.arange(n)[:, None]
    tri = np.triu_indices(N, 1)
    for k in range(0 if e_start is not None else 1, m):
        f = evs[:, k]
        d = np.abs(e[:, :, None] - f[:, None, :])
        idx = np.argmin(d, axis=2)
        moved = np.abs(f[rows, idx] - e).max(axis=1)
        ge = np.abs(e[:, tri[0]] - e[:, tri[1]]).min(axis=1)
        gf = np.abs(f[:, tri[0]] - f[:, tri[1]]).min(axis=1)
        g = np.minimum(ge, gf)
        distinct = (np.sort(idx, axis=1) == np.arange(N)).all(axis=1)
        bad |= ~distinct | (moved >= 0.25 * g)
        mg = np.minimum(mg, g)
        e = f[rows, idx]
    return e, mg, bad

THETA0 = 1.0    # base point of the loop; theta = 0 would force label collisions for real dM

def nominal_labels(N):
    return np.exp(1j * (THETA0 + 2 * np.pi * np.arange(N)) / N)

def ring_stack(N, c):
    """c: array of shape (m,). Returns (m, N, N)."""
    M = np.zeros((len(c), N, N), complex)
    M[:, np.arange(N - 1), np.arange(1, N)] = 1.0
    M[:, N - 1, 0] = c
    return M

def batch_monodromy(dM, m=192, m_s=48):
    """Monodromy of the loop c = exp(i theta), M = M_ring(0, c) + dM, for a stack dM of shape (n, N, N).
    Labels are carried from the unperturbed vertices by the straight path s dM at theta = THETA0. Returns perm (n, N) in those labels: perm[k] is the label reached by label k; the
    nominal N-cycle is k -> k + 1. Unresolved samples are redone with the adaptive tracker.
    Also returns the number of samples for which even the adaptive tracker reported an unresolved step."""
    n, N = dM.shape[0], dM.shape[1]
    th = THETA0 + np.linspace(0, 2 * np.pi, m + 1)
    M0 = ring_stack(N, np.exp(1j * th))
    ev = np.linalg.eigvals(M0[None] + dM[:, None])
    ss = np.linspace(0, 1, m_s + 1)[1:]
    M00 = ring_stack(N, np.array([np.exp(1j * THETA0)]))
    evs = np.linalg.eigvals(M00[None] + ss[None, :, None, None] * dM[:, None])
    lab0 = np.broadcast_to(nominal_labels(N), (n, N))
    e_lab, _, bad_s = _assign_path(evs, e_start=lab0)
    e_end, mg, bad_l = _assign_path(ev)
    # e_end[j] continues ev[:, 0, j]; e_lab[l] is the eigenvalue at theta=0 with label l
    start = ev[:, 0]
    rows = np.arange(n)[:, None]
    lab_of = np.argmin(np.abs(start[:, :, None] - e_lab[:, None, :]), axis=2)      # start index -> label
    back = np.argmin(np.abs(e_end[:, :, None] - start[:, None, :]), axis=2)         # start index j -> start index reached
    perm = np.zeros((n, N), int)
    perm[rows, lab_of] = lab_of[rows, back]
    bad = bad_s | bad_l | ~(np.sort(lab_of, axis=1) == np.arange(N)).all(axis=1) | ~(np.sort(back, axis=1) == np.arange(N)).all(axis=1)
    unresolved = 0
    for i in np.flatnonzero(bad):
        D = dM[i]
        s0, s1, _, _, ok1 = track(lambda t: ring(np.zeros(N), np.exp(1j * THETA0)) + t * D, t0=0.0, t1=1.0, h0=1 / 64)
        order = np.argmin(np.abs(s0[:, None] - nominal_labels(N)[None, :]), axis=1)  # s0[j] has label order[j]
        lab = np.empty(N, complex); lab[order] = s1
        st, en, _, g, ok2 = track(lambda t: ring(np.zeros(N), np.exp(1j * (THETA0 + 2 * np.pi * t))) + D)
        lo = np.argmin(np.abs(st[:, None] - lab[None, :]), axis=1)
        bk = np.argmin(np.abs(en[:, None] - st[None, :]), axis=1)
        p = np.zeros(N, int); p[lo] = lo[bk]
        perm[i] = p; mg[i] = g
        unresolved += int(not (ok1 and ok2))
    return perm, mg, unresolved

def type_code(perm):
    """N = 3 only: 0 nominal 3-cycle, 1 reversed 3-cycle, 2 transposition, 3 identity."""
    n = perm.shape[0]; lab = np.arange(3)
    c = np.full(n, 2)
    c[(perm == (lab + 1) % 3).all(axis=1)] = 0
    c[(perm == (lab + 2) % 3).all(axis=1)] = 1
    c[(perm == lab).all(axis=1)] = 3
    return c

CLASSES = ("det_re", "det_c", "coupling", "general")

def draw(cls, x, rng, N=3):
    """Perturbation stack for strengths x (array, in absolute units): entry standard deviation x.
    det_re: real diagonal. det_c: complex diagonal. coupling: real entries on the three positions
    (0,2), (1,0), (2,1), which the ring leaves empty (figure 2(c,d)). general: complex, all entries."""
    n = len(x); D = np.zeros((n, N, N), complex); x = np.asarray(x, float)
    if cls == "det_re":
        D[:, np.arange(N), np.arange(N)] = x[:, None] * rng.standard_normal((n, N))
    elif cls == "det_c":
        D[:, np.arange(N), np.arange(N)] = x[:, None] * (rng.standard_normal((n, N)) + 1j * rng.standard_normal((n, N))) / np.sqrt(2)
    elif cls == "coupling":
        assert N == 3
        for (i, j) in [(0, 2), (1, 0), (2, 1)]:
            D[:, i, j] = x * rng.standard_normal(n)
    elif cls == "general":
        D = x[:, None, None] * (rng.standard_normal((n, N, N)) + 1j * rng.standard_normal((n, N, N))) / np.sqrt(2)
    else:
        raise ValueError(cls)
    return D
