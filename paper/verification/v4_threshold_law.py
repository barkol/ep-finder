"""V4: threshold law as the critical value of R = lam^{L'}/(b lam^m + d) (Theorem 1 with the common factor
lam^{N-L'} cancelled).  (a) symbolic: critical points of F = z^{L'}/(z^m+1) besides z=0 solve L z^m + L' = 0,
|F| there = (L/m)(L'/L)^{L'/m}; (b) numeric: the critical values of R for random b, d all have modulus
|c|_th of equation (20) of the paper; (c) diamond L=3, m=1: 256/27 |d|^3/|b|^4, lam_* = -4d/(3b); general N: N^N/(N-1)^{N-1};
(d) phi_c at |c|=1 for w=1 and w=0.5; gap at phi=pi equals sqrt(2)(|c|w)^{1/4}."""
import numpy as np, sympy as sp
z = sp.symbols('z'); 
for L in range(1, 5):
    for m in range(1, 4):
        Lp = L+m; F = z**Lp/(z**m+1)
        num = sp.factor(sp.numer(sp.together(sp.diff(F, z))))
        ok = sp.simplify(num - z**(Lp-1)*(L*z**m+Lp)) == 0
        zs = sp.Rational(-Lp, L)**sp.Rational(1, m) if m == 1 else None
        crit = [complex(r) for r in sp.Poly(L*z**m+Lp, z).nroots()]
        vals = [abs(complex(F.subs(z, r))) for r in crit]
        sc = (L/m)*(Lp/L)**(Lp/m)
        assert ok and np.allclose(vals, sc), (L, m)
print("(a) F' numerator = z^{L'-1}(L z^m + L') and |F(crit)| = (L/m)(L'/L)^{L'/m}: L=1..4, m=1..3 ok")
rng = np.random.default_rng(5); worst = 0
for t in range(300):
    L = rng.integers(1, 5); m = rng.integers(1, 4); Lp = L+m
    b = rng.normal()+1j*rng.normal(); d = rng.normal()+1j*rng.normal()
    A = np.zeros(Lp+1, complex); A[0] = 1                      # lam^{L'}
    B = np.zeros(m+1, complex); B[0] = b; B[-1] = d            # b lam^m + d
    W = np.polysub(np.polymul(np.polyder(A), B), np.polymul(A, np.polyder(B)))
    cp = np.roots(W); cp = cp[np.abs(cp) > 1e-6]
    cv = np.abs(np.polyval(A, cp)/np.polyval(B, cp))
    th = L/(m*abs(d))*(Lp*abs(d)/(L*abs(b)))**(Lp/m)
    worst = max(worst, np.abs(cv/th-1).max())
print("(b) max relative deviation of nonzero critical values from |c|_th:", worst)
b, d, lam = sp.symbols('b d lambda'); N = sp.symbols('N', positive=True)
R = lam**4/(b*lam+d)
print("(c) critical points:", sp.solve(sp.numer(sp.together(sp.diff(R, lam))), lam), " R(lam_*) =", sp.simplify(R.subs(lam, -4*d/(3*b))))
for n in range(2, 8):
    L, m = n-1, 1
    th = sp.Rational(L, m)*(sp.Rational(n, L))**n
    assert sp.simplify(th - sp.Integer(n)**n/sp.Integer(n-1)**(n-1)) == 0
print("(c) general N: (L/m)(L'/L)^{L'/m} with L=N-1, m=1 equals N^N/(N-1)^{N-1}: ok")
from scipy.optimize import brentq
for w in (1.0, 0.5):
    f = lambda p: 256/27*w**3/abs(1+np.exp(1j*p))**4 - 1
    print(f"(d) phi_c(w={w}, |c|=1) = {brentq(f, 0.01, 3.1):.4f} rad")
cw = 0.37*0.5; r = np.roots([1, 0, 0, 0, cw]); gaps = [abs(r[i]-r[j]) for i in range(4) for j in range(i)]
print("(d) phi=pi: min gap", min(gaps), "vs sqrt2 (|c|w)^{1/4} =", np.sqrt(2)*cw**0.25)
