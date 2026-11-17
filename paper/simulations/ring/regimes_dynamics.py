"""Figures 2 and 4 (state dynamics) -> results/regimes_dynamics.npz.
Integration with 80 significant digits, stepwise propagators, independent noise eta*sqrt(dt) after every step
(fixed seed). Computes (a) the spectral braid at Gamma = 1.4 and (b) the largest pairwise infidelity of the inputs
and their common label over six orbits. Checkpoint per run in work/regimes/<key>.npz (computed if missing).
Run with the argument 'test' for two short runs."""
import numpy as np, mpmath as mp, os, sys, time
from scipy.optimize import linear_sum_assignment
from _paths import work, results
N=3; DPS=80
CK=work('regimes')
os.makedirs(CK,exist_ok=True)
def props(T,nst,order=30):
    mp.mp.dps=DPS; T=mp.mpf(T); Om=2*mp.pi/T; dt=T/nst; out=[]
    for s in range(nst):
        g=[mp.exp(1j*Om*s*dt)]
        for n in range(order): g.append(g[-1]*1j*Om/(n+1))
        P=mp.zeros(N)
        for col in range(N):
            y=[[mp.mpf(1) if r==col else mp.mpf(0)] for r in range(N)]
            for n in range(order):
                for r in range(N):
                    v=y[r+1][n] if r<N-1 else sum(g[m]*y[0][n-m] for m in range(n+1))
                    y[r].append(v/(n+1))
            for r in range(N):
                v=mp.mpf(0)
                for n in range(order,-1,-1): v=v*dt+y[r][n]
                P[r,col]=v
        out.append(P)
    return out
def Rm(th):
    """branches tracked analytically: k -> lam=e^{i(th+2 pi k)/3}; columns R, rows L (M is normal)"""
    return [mp.matrix([mp.exp(1j*(th+2*mp.pi*k)/N*j)/mp.sqrt(N) for j in range(N)]) for k in range(N)]
def run(T,eta,norb,seed=11,nperorb=240):
    key=os.path.join(CK,f"T{T}_eta{eta}_s{seed}_n{norb}_p{nperorb}.npz")
    if os.path.exists(key): return dict(np.load(key))
    t0=time.time(); P=props(T,nperorb); mp.mp.dps=DPS
    rng=np.random.default_rng(seed); dt=T/nperorb
    psi=[Rm(mp.mpf(0))[k].copy() for k in range(N)]; ns=norb*nperorb
    inf=np.zeros(ns+1); lab=np.zeros((ns+1,N),int)
    for i in range(ns+1):
        th=2*mp.pi*i/nperorb; R=Rm(th)
        for a in range(N):
            ov=[abs(sum(mp.conj(x)*y for x,y in zip(R[k],psi[a]))) for k in range(N)]
            lab[i,a]=int(np.argmax([float(mp.log(o+mp.mpf(10)**(-DPS))) for o in ov]))
        F=[]
        for a,b in ((0,1),(0,2),(1,2)):
            F.append(1-abs(sum(mp.conj(x)*y for x,y in zip(psi[a],psi[b])))**2)
        inf[i]=float(max(F))
        if i==ns: break
        Pi=P[i%nperorb]
        for a in range(N):
            v=Pi*psi[a]
            if eta>0:
                z=(rng.normal(size=N)+1j*rng.normal(size=N))*eta*np.sqrt(dt/2)
                nv=mp.sqrt(sum(abs(x)**2 for x in v))
                v=mp.matrix([x/nv+mp.mpc(float(q.real),float(q.imag)) for x,q in zip(v,z)])
            v=v/mp.sqrt(sum(abs(x)**2 for x in v)); psi[a]=v
    d=dict(inf=inf,lab=lab,T=T,eta=eta,norb=norb,nperorb=nperorb)
    np.savez(key+'.tmp.npz',**d); os.replace(key+'.tmp.npz',key); print(key,time.time()-t0,flush=True)
    return d
def braid(G=1.4):
    TH=np.linspace(0,2*np.pi,361); W=np.zeros((361,N))
    def M(t): return np.array([[-G,1,0],[0,-G,1],[np.exp(1j*t),0,-G]],complex)
    lam=np.linalg.eigvals(M(TH[0])); o=np.argsort(-lam.imag); prev=lam[o]; W[0]=(-lam.imag)[o]
    for i in range(1,361):
        lam=np.linalg.eigvals(M(TH[i])); C=np.abs(prev[:,None]-lam[None,:]); r,c=linear_sum_assignment(C)
        W[i]=(-lam.imag)[c]; prev=lam[c]
    return TH,W
if __name__=='__main__':
    small=len(sys.argv)>1 and sys.argv[1]=='test'
    runs=[(20,0.,3),(20,1e-8,3)] if small else [(20,0.,6),(20,1e-12,6),(20,1e-8,6),(20,1e-4,6),(110,0.,6),(110,1e-16,6),(110,1e-12,6),(110,1e-8,6),(110,1e-4,6)]
    out={}
    for T,eta,no in runs:
        d=run(T,eta,no,nperorb=(120 if T==20 else 240))
        k=f"T{T}_e{eta}"; out[k+'_inf']=d['inf']; out[k+'_lab']=d['lab']
        lab=d['lab']; nper=int(d['nperorb']); dis=[len(set(l)) for l in lab]
        sw=[i/nper for i in range(1,len(lab)) if lab[i,0]!=lab[i-1,0]]
        print(k,'1-F at orbit ends',[f"{d['inf'][nper*o]:.2e}" for o in range(1,no+1)],'label at orbit ends',[int(lab[nper*o,0]) for o in range(1,no+1)],'label switches of input 0',[round(x,3) for x in sw][:12],'times with >1 distinct label after orbit 2:',sum(1 for x in dis[2*nper:] if x>1),flush=True)
    TH,W=braid(); np.savez(results('regimes_dynamics.npz'),TH=TH,W=W,**out); print('wrote results/regimes_dynamics.npz')
