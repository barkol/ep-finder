"""V1: first moments of the master equation give the drift matrix of one link,
and adiabatic elimination of a reservoir mode gives the link-model entries.
(a) H = sum h_mn a_m^dag a_n, L = sum l_n a_n  =>  d<a>/dt = (-i h - 1/2 l^* l^T)<a>.
    Checked against the exact Lindblad action on a truncated two-mode Fock space.
(b) For equation (4) of the paper: M_jk = eps e^{i phi}, M_kj = 0, M_jj = M_kk = -eps/2.
(c) Reservoir: modes (a_j, a_k, r), coherent H_{j<-k} plus H_res = J' r^dag(a_j - e^{i phi}a_k)+h.c.,
    L = sqrt(kappa') r, J' = sqrt(eps kappa')/2. Eliminating r at frequency lam (Schur complement)
    gives M_jk = eps/2 e^{i phi}(1+g), M_kj = eps/2 e^{-i phi}(g-1), M_jj=M_kk=-eps/2 g,
    g = 1/(1+2 lam/kappa'); g -> 1 for kappa' -> inf reproduces (b)."""
import numpy as np, sympy as sp
rng = np.random.default_rng(1)
# (a) exact Lindblad on truncated Fock space, 2 modes, check d<a_m>/dt on random states with low photon number
nc = 6
a = np.diag(np.sqrt(np.arange(1, nc)), 1); I = np.eye(nc)
A = [np.kron(a, I), np.kron(I, a)]
eps, phi = 0.7, 0.9
h = np.array([[0.3, 1j*eps/2*np.exp(1j*phi)], [-1j*eps/2*np.exp(-1j*phi), -0.2]])
l = np.sqrt(eps)*np.array([1.0, -np.exp(1j*phi)])
H = sum(h[m, n]*A[m].conj().T@A[n] for m in range(2) for n in range(2))
L = sum(l[n]*A[n] for n in range(2))
worst = 0
for t in range(20):
    # random state supported on <=2 photons in total, so truncation is exact for first moments
    psi = np.zeros(nc*nc, complex)
    for n1 in range(3):
        for n2 in range(3-n1):
            psi[n1*nc+n2] = rng.normal()+1j*rng.normal()
    psi /= np.linalg.norm(psi); rho = np.outer(psi, psi.conj())
    drho = -1j*(H@rho-rho@H) + L@rho@L.conj().T - 0.5*(L.conj().T@L@rho + rho@L.conj().T@L)
    lhs = np.array([np.trace(A[m]@drho) for m in range(2)])
    M = -1j*h - 0.5*np.outer(l.conj(), l)
    rhs = M@np.array([np.trace(A[m]@rho) for m in range(2)])
    worst = max(worst, np.abs(lhs-rhs).max())
print("(a) max |d<a>/dt - M<a>| on truncated Fock space:", worst)
# (b) symbolic entries of equation (4) of the paper, j = 0, k = 1, coherent part only for h
e, p = sp.symbols('varepsilon phi', positive=True)
hs = sp.Matrix([[0, sp.I*e/2*sp.exp(sp.I*p)], [-sp.I*e/2*sp.exp(-sp.I*p), 0]])
ls = sp.Matrix([sp.sqrt(e), -sp.sqrt(e)*sp.exp(sp.I*p)])
Ms = sp.simplify(-sp.I*hs - sp.Rational(1, 2)*ls.conjugate()*ls.T)
print("(b) drift of one link:", Ms.tolist())
# (c) reservoir elimination
kp = sp.symbols("kappa'", positive=True); lam = sp.symbols('lambda')
Jp = sp.sqrt(e*kp)/2
# modes (a_j, a_k, r); Hamiltonian matrix h3 with coherent link and reservoir coupling
h3 = sp.zeros(3, 3)
h3[0, 1] = sp.I*e/2*sp.exp(sp.I*p); h3[1, 0] = sp.conjugate(h3[0, 1])
h3[2, 0] = Jp; h3[2, 1] = -Jp*sp.exp(sp.I*p); h3[0, 2] = Jp; h3[1, 2] = -Jp*sp.exp(-sp.I*p)
l3 = sp.Matrix([0, 0, sp.sqrt(kp)])
M3 = -sp.I*h3 - sp.Rational(1, 2)*l3.conjugate()*l3.T
M3 = M3.subs({sp.conjugate(e): e, sp.conjugate(p): p})
Maa, Mar, Mra, Mrr = M3[:2, :2], M3[:2, 2], M3[2, :2], M3[2, 2]
Meff = sp.simplify(Maa + Mar*Mra/(lam - Mrr))
g = 1/(1+2*lam/kp)
target = sp.Matrix([[-e/2*g, e/2*sp.exp(sp.I*p)*(1+g)], [e/2*sp.exp(-sp.I*p)*(g-1), -e/2*g]])
print("(c) Schur complement minus link-model formula:", sp.simplify(Meff-target).tolist())
print("(c) limit kappa'->oo:", [[sp.limit(Meff[r,s], kp, sp.oo) for s in range(2)] for r in range(2)])
