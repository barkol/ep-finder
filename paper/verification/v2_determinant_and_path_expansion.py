"""V2: (a) det(lam - M0 - c e_i e_j^T) = A - c B, A = det(lam - M0), B = adj(lam - M0)_{ji};
(b) path expansion B = sum over directed paths P from i to j (edge l->k when M0[k,l] != 0) of
    w(P) * det(lam - M0) restricted to the modes not on P, w(P) = product of the entries along P;
(c) ring: B = prod eps_k; back-coupling at (i,j), i>j, in the bidiagonal chain: B = prod_{k outside j..i}(lam-alpha_k),
    i.e. a common factor of A and B."""
import itertools, random, sympy as sp
lam, c = sp.symbols('lambda c')
random.seed(3)
def paths(M, s, t, N):
    out = []
    def rec(path):
        u = path[-1]
        if u == t: out.append(list(path)); return
        for k in range(N):
            if k not in path and M[k, u] != 0:
                rec(path+[k])
    rec([s]); return out
nok = 0; ntest = 0
for trial in range(60):
    N = random.randint(2, 6)
    M0 = sp.zeros(N, N)
    for r in range(N):
        M0[r, r] = sp.Rational(random.randint(-3, 3), random.randint(1, 3)) + sp.I*random.randint(-2, 2)
        for s in range(N):
            if r != s and random.random() < 0.45:
                M0[r, s] = sp.Rational(random.randint(-4, 4), random.randint(1, 3)) + sp.I*random.randint(-2, 2)
    i, j = random.sample(range(N), 2)
    E = sp.zeros(N, N); E[i, j] = 1
    X = lam*sp.eye(N) - M0
    A = X.det(); B = X.adjugate()[j, i]
    lhs = (lam*sp.eye(N) - M0 - c*E).det()
    ok_a = sp.expand(lhs - (A - c*B)) == 0
    Bp = 0
    for P in paths(M0, i, j, N):
        w = sp.prod([M0[P[s+1], P[s]] for s in range(len(P)-1)])
        rest = [k for k in range(N) if k not in P]
        Bp += w*(X.extract(rest, rest).det() if rest else 1)
    ok_b = sp.expand(B - Bp) == 0
    ntest += 1; nok += ok_a and ok_b
print(f"(a,b) random networks N=2..6: {nok}/{ntest} pass")
# (c) chain with single back-coupling b at (i,j)
for N in range(2, 7):
    al = sp.symbols(f'alpha1:{N+1}'); ep = sp.symbols(f'eps1:{N}')
    M0 = sp.zeros(N, N)
    for k in range(N):
        M0[k, k] = al[k]
        if k < N-1: M0[k, k+1] = ep[k]
    X = lam*sp.eye(N) - M0
    good = True
    for i in range(N):
        for j in range(i):
            B = X.adjugate()[j, i]
            expect = sp.prod(ep[j:i])*sp.prod([lam-al[k] for k in range(N) if k < j or k > i])
            good &= sp.expand(B - expect) == 0
    print(f"(c) N={N}: B = prod eps on sub-chain * prod_(outside)(lam-alpha_k) for all i>j: {good}")
