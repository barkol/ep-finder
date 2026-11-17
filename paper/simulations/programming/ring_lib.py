"""Core library for the detuning-programmed ring (figure 3, ladders). No side effects on import.

Conventions (units: cascade coupling eps = 1, modes indexed 0..N-1):
  M[k,k+1]=1, M[N-1,0]=c, M[k,k]=alpha_k.  chi(lam)=P(lam)-c, P=prod(lam-alpha_k).
  Detuning form: alpha_k=-1j*delta_k, lam=-1j*mu, P=(-1j)^N q(mu), q=prod(mu-delta_k), ctilde=c*1j^N.
  Label k = branch that tends to alpha_k as c->0 along the straight ray from 0 to the base c_b.
  sigma[k]=l means lam_k(theta0+2pi)=lam_l(theta0), for theta increasing.
"""
import numpy as np


def Mring(al, c):
    al = np.asarray(al, complex)
    N = len(al)
    M = np.diag(al) + np.diag(np.ones(N - 1, complex), 1)
    M[N - 1, 0] = c
    return M


def pgap(l):
    d = np.abs(l[:, None] - l[None, :])
    d[np.diag_indices(len(l))] = np.inf
    return d.min()


def track(l0, Mf, h0=1 / 300, frac=0.1, hmin=1e-11):
    """Adaptive nearest-neighbour tracker on t in [0,1].
    Returns (l_end, min_gap, ok). ok=False when the step floor is reached."""
    l = l0.copy()
    N = len(l)
    t = 0.0
    h = h0
    mg = pgap(l)
    while t < 1 - 1e-14:
        h = min(h, 1 - t)
        e = np.linalg.eigvals(Mf(t + h))
        C = np.abs(l[:, None] - e[None, :])
        idx = C.argmin(1)
        g = min(pgap(l), pgap(e))
        if len(set(idx.tolist())) < N or C[np.arange(N), idx].max() > frac * g:
            h /= 2
            if h < hmin:
                return l, mg, False
            continue
        l = e[idx]
        t += h
        mg = min(mg, pgap(l))
        h = min(1.5 * h, h0)
    return l, mg, True


def labelled_sigma(al, cabs, theta0):
    """Labelled monodromy by direct tracking. Returns (sigma, g_ray, g_loop, ok)."""
    al = np.asarray(al, complex)
    cb = cabs * np.exp(1j * theta0)
    l, g_ray, ok1 = track(al.copy(), lambda t: Mring(al, t * cb))
    if not ok1:
        return None, g_ray, 0.0, False
    l1, g_loop, ok2 = track(l.copy(), lambda t: Mring(al, cb * np.exp(2j * np.pi * t)))
    if not ok2:
        return None, g_ray, g_loop, False
    C = np.abs(l1[:, None] - l[None, :])
    q = C.argmin(1)
    if len(set(q.tolist())) < len(al):
        return None, g_ray, g_loop, False
    return tuple(int(x) for x in q), g_ray, g_loop, True


def cycles(s):
    seen = set()
    out = []
    for i in range(len(s)):
        if i in seen:
            continue
        c = []
        j = i
        while j not in seen:
            seen.add(j)
            c.append(j)
            j = s[j]
        out.append(c)
    return out


def ctype(s):
    return tuple(sorted((len(c) for c in cycles(s)), reverse=True))


def crit_data(al):
    """Critical points and values of P (with multiplicity, from np.roots of P')."""
    p = np.poly(np.asarray(al, complex))
    x = np.roots(np.polyder(p))
    return x, np.polyval(p, x)


def base_theta(N):
    """Base phase with ctilde=+i|c|: c_b=(-i)^N * i * |c|."""
    return float(np.mod(np.pi / 2 * (1 - N), 2 * np.pi))


def sigma_rank_formula(d, R):
    """Exact rule for real detunings d (strictly increasing), base ctilde=+iR.
    Returns the permutation of ranks: sigma=t_plus o t_minus (first t_minus).
    t_minus: adjacent swaps (r,r+1) with v_r<0 and |v_r|<R; t_plus: v_r>0 and |v_r|<R."""
    d = np.asarray(d, float)
    N = len(d)
    p = np.poly(d)
    xs = np.sort(np.roots(np.polyder(p)).real)
    v = np.polyval(p, xs)
    tm = list(range(N))
    tp = list(range(N))
    for r in range(N - 1):
        if abs(v[r]) < R:
            t = tm if v[r] < 0 else tp
            t[r], t[r + 1] = t[r + 1], t[r]
    return tuple(tp[tm[k]] for k in range(N)), v


def assign_modes(rank_sigma, target):
    """Find f: rank->mode with f o rank_sigma o f^-1 = target.  Same cycle type required."""
    cr = sorted(cycles(rank_sigma), key=len)
    ct = sorted(cycles(target), key=len)
    if [len(c) for c in cr] != [len(c) for c in ct]:
        raise ValueError("cycle types differ")
    f = {}
    for a, b in zip(cr, ct):
        for x, y in zip(a, b):
            f[x] = y
    return [f[r] for r in range(len(target))]


def layout_to_d(comp, s, D):
    """Uniform layout: clusters of sizes comp, internal spacing s, edge-to-edge spacing D, zero mean."""
    d = []
    x = 0.0
    for i, n in enumerate(comp):
        if i:
            x += D
        for j in range(n):
            if j:
                x += s
            d.append(x)
    d = np.array(d)
    return d - d.mean()


def compositions(N):
    if N == 0:
        yield ()
        return
    for k in range(1, N + 1):
        for r in compositions(N - k):
            yield (k,) + r


def comps_mod_reflection(N):
    seen = set()
    for c in compositions(N):
        if c[::-1] in seen:
            continue
        seen.add(c)
        yield c


def partition_of(comp):
    return tuple(sorted(comp, reverse=True))


def rank_cluster_mask(comp):
    """expected inside-mask of the N-1 critical points for a layout."""
    m = []
    for i, n in enumerate(comp):
        m += [True] * (n - 1)
        if i < len(comp) - 1:
            m.append(False)
    return np.array(m, bool)


def _roots_batch(d, ct):
    """Roots mu of q(mu)=ct for an array ct (companion batch)."""
    p = np.poly(d)
    N = len(d)
    A = np.zeros((len(ct), N, N), complex)
    A[:, np.arange(1, N), np.arange(0, N - 1)] = 1.0
    A[:, 0, :] = -p[1:]
    A[:, 0, N - 1] = -(p[N] - ct)
    return np.linalg.eigvals(A)


def _gap_batch(r):
    d = np.abs(r[:, :, None] - r[:, None, :])
    i = np.arange(r.shape[1])
    d[:, i, i] = np.inf
    return d.min(axis=(1, 2))


def gaps(d, R=1.0, nloop=721, nray=65, refine=True):
    """(g_loop, g_ray) for real detunings d on the circle |c|=R and the ray ctilde=i t R."""
    d = np.asarray(d, float)
    th = np.linspace(0, np.pi, nloop // 2 + 1)  # conj symmetry for real d
    g = _gap_batch(_roots_batch(d, R * np.exp(1j * th)))
    gl = g.min()
    if refine:
        i = int(g.argmin())
        a, b = th[max(i - 1, 0)], th[min(i + 1, len(th) - 1)]
        for _ in range(4):
            tt = np.linspace(a, b, 41)
            gg = _gap_batch(_roots_batch(d, R * np.exp(1j * tt)))
            j = int(gg.argmin())
            gl = min(gl, gg[j])
            a, b = tt[max(j - 1, 0)], tt[min(j + 1, 40)]
    t = np.linspace(0, 1, nray)
    gr = _gap_batch(_roots_batch(d, 1j * R * t)).min()
    return float(gl), float(gr)


def eig_rate(d, R=1.0, n=361):
    """max Re lam and max |Im lam| over the loop for alpha=-i d."""
    th = np.linspace(0, 2 * np.pi, n)
    mu = _roots_batch(np.asarray(d, float), R * np.exp(1j * th))
    lam = -1j * mu
    return float(lam.real.max()), float(np.abs(lam.imag).max())


def adiab_S(al, cabs=1.0, n=361):
    """S=max_{theta,n} sum_{m!=n} ||Pi_m dM/dtheta Rhat_n||^2/|lam_m-lam_n|^4 on the loop.
    First-order adiabatic estimate P(T)=(2pi/T)^2 S for a constant sweep. At alpha=0, |c|=1
    this is (1/N^2) sum_m [2 sin(pi m/N)]^-4, the V3 formula."""
    al = np.asarray(al, complex)
    N = len(al)
    best = 0.0
    for th in np.linspace(0, 2 * np.pi, n)[:-1]:
        c = cabs * np.exp(1j * th)
        M = Mring(al, c)
        w, R = np.linalg.eig(M)
        Ri = np.linalg.inv(R)
        dM = np.zeros((N, N), complex)
        dM[N - 1, 0] = 1j * c
        nr = np.linalg.norm(R, axis=0)
        for k in range(N):
            s = 0.0
            rn = R[:, k] / nr[k]
            am = Ri @ (dM @ rn)
            for m in range(N):
                if m != k:
                    s += (abs(am[m]) * nr[m]) ** 2 / abs(w[m] - w[k]) ** 4
            best = max(best, s)
    return float(best)


def full_link_matrix(delta, theta, kappa, gamma0=0.0, cabs=1.0, gamma_link=1.0, dphi=0.0):
    """Full link model: N modes + N reservoir modes (Metelmann-Clerk link), eps=gamma_link=1.
    Link l transfers from mode (l+1 mod N) into mode l, the corner link from 0 into N-1 carries theta.
    Each link: coherent J=Gamma/2 with phase phi, reservoir z with loss kappa, couplings g=sqrt(Gamma*kappa)/2.
    In the limit kappa->inf: da_to=Gamma e^{i th} a_from, no back action, local loss Gamma/2 per link end."""
    delta = np.asarray(delta, float)
    N = len(delta)
    M = np.zeros((2 * N, 2 * N), complex)
    M[np.arange(N), np.arange(N)] = -1j * delta - gamma0 / 2
    for l in range(N):
        if l < N - 1:
            to, fr, th, G = l, l + 1, 0.0, gamma_link
        else:
            to, fr, th, G = N - 1, 0, theta, gamma_link * cabs
        J = G / 2
        phi = th + np.pi / 2 + dphi
        M[to, fr] += -1j * J * np.exp(1j * phi)
        M[fr, to] += -1j * J * np.exp(-1j * phi)
        z = N + l
        g = np.sqrt(G * kappa) / 2
        # g_to g_from^* = g^2 e^{i chi}, chi=th+pi  : put the phase on g_to
        gto = g * np.exp(1j * (th + np.pi))
        gfr = g
        M[to, z] += -1j * gto
        M[z, to] += -1j * np.conj(gto)
        M[fr, z] += -1j * gfr
        M[z, fr] += -1j * np.conj(gfr)
        M[z, z] = -kappa / 2
    return M
