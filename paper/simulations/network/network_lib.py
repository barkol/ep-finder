"""Shared tools: monodromy tracking with an adaptive step (failures raise an error, never pass silently),
the ideal drift matrix and the link model with one reservoir mode per link."""
import numpy as np
from scipy.optimize import linear_sum_assignment as lsa
class TrackingError(Exception): pass
def cycle_type(p):
    seen=set(); out=[]
    for s in range(len(p)):
        if s in seen: continue
        k=0; x=s
        while x not in seen: seen.add(x); x=p[x]; k+=1
        out.append(k)
    return tuple(sorted(out,reverse=True))
def track(ev, n0=400, tol=0.25, max_halvings=14):
    """ev(theta) -> eigenvalues. The step is halved until every branch moves by less than tol * the smallest spacing.
    Returns (permutation, minimum gap, eigenvalues at theta=0)."""
    l0=ev(0.0); dim=len(l0)
    def min_spacing(x):
        d=np.abs(x[:,None]-x[None,:]); np.fill_diagonal(d,np.inf); return d.min()
    l=l0.copy(); t=0.0; h=2*np.pi/n0; mg=min_spacing(l)
    if not np.isfinite(mg): raise TrackingError('infinite spacing')
    while t<2*np.pi-1e-12:
        h=min(h,2*np.pi-t); w=ev(t+h); C=np.abs(l[:,None]-w[None,:]); a,b=lsa(C); step=C[a,b].max(); g=min(min_spacing(l),min_spacing(w))
        if step>tol*g:
            h/=2
            if h<2*np.pi/n0/2**max_halvings: raise TrackingError(f't={t:.5f} step={step:.3e} gap={g:.3e}')
            continue
        l=w[b]; t+=h; mg=min(mg,g); h=min(h*1.5,2*np.pi/n0)
    a,b=lsa(np.abs(l[:,None]-l0[None,:]))
    return list(b), mg, l0
def m_ideal(l, Gam=0.0, delta=None):
    """l: list of (j,k,amp) with complex amp: mode k drives mode j. N follows from the largest index."""
    N=1+max(max(j,k) for j,k,_ in l); M=np.zeros((N,N),complex)
    for j,k,a in l: M[j,k]+=a
    M-=Gam*np.eye(N)
    if delta is not None: M-=1j*np.diag(delta)
    return M
def link_model_matrix(l, kappa, Gam, delta=None, errors=None, allow_overload=False):
    """Full link model: every link has its own reservoir mode (paper, section on directional links). l: (j,k,eps,phi).
    errors(idx) -> (dJ, dphi): relative amplitude error and lock-phase error of the reservoir coupling of link idx.
    Every mode is equalized to the uniform loss Gam; Gam below the link load raises ValueError unless allow_overload."""
    N=1+max(max(x[0],x[1]) for x in l); n=len(l); M=np.zeros((N+n,N+n),complex); load=np.zeros(N)
    for q,(j,k,e,phi) in enumerate(l):
        dJ,dph=(0.0,0.0) if errors is None else errors(q)
        J=np.sqrt(e*kappa)/2*(1+dJ); ph=phi+dph; r=N+q
        M[j,k]+=e/2*np.exp(1j*phi); M[k,j]+=-e/2*np.exp(-1j*phi)
        M[r,r]+=-kappa/2; M[r,j]+=-1j*J; M[r,k]+=1j*J*np.exp(1j*ph); M[j,r]+=-1j*J; M[k,r]+=1j*J*np.exp(-1j*ph)
        load[j]+=e/2; load[k]+=e/2
    if np.any(load>Gam+1e-12) and not allow_overload: raise ValueError(f'Gam {Gam} below the link load {load}')
    for i in range(N): M[i,i]-=Gam-load[i]
    if delta is not None:
        for i in range(N): M[i,i]-=1j*delta[i]
    return M,N
def diamond(phi,w,c,th):
    """Four-mode network: 3->1->0, 3->2->0 (phase phi), 1->2 (w), feedback 0->3 c e^{i th}. Entries (j,k,eps,phase)."""
    return [(0,1,1.0,0.0),(1,3,1.0,0.0),(2,3,1.0,0.0),(0,2,1.0,phi),(2,1,w,0.0),(3,0,c,th)]
