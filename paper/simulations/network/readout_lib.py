"""Read-out of the four-mode network by model fitting (technical notes, section N2). Data come from the full link model;
the fit uses the link model with the reservoirs eliminated exactly (kappa' known), 24 spectra along one orbit and a
probe on the modes. Rules: reduced chi^2 > 1 + 3 sqrt(2/n) rejects; a parameter on a window edge is undecided;
sampling test: at least 95% of 24 samples from the fit covariance give the same type, otherwise undecided.
Verdict strings stored in the result files: 'pewne' = certified, 'odrzut' = rejected, 'nierozstrz' = undecided."""
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import numpy as np
from scipy.optimize import least_squares
from network_lib import *
OM=np.linspace(-6,6,501); TH=2*np.pi*np.arange(24)/24
N=4; U=np.ones(N)/np.sqrt(N)
PORTS=[(U,U)]   # list of (input, output) probe vectors; default: uniform probe. Several ports = spectra stacked along the phase axis
def links(phi,w,c,th): return diamond(phi,w,c,th)
def response_full(om,th,phi,w,c,G,k,delta=None,errors=None,pert=None):
    """Resolvent of the full 10-mode link model (data). pert(L, i): modifies the link list for spectrum i."""
    Y=np.zeros((len(PORTS)*len(th),len(om)),complex)
    for i,t in enumerate(th):
        L=links(phi,w,c,t)
        if pert is not None: L=pert(L,i)
        M,n=link_model_matrix(L,k,G,delta=delta,errors=errors); d=M.shape[0]; A=(-1j*om)[:,None,None]*np.eye(d)-M
        for ip,(pa,pb) in enumerate(PORTS):
            W=np.zeros(d,complex); W[:n]=pa
            Y[ip*len(th)+i]=np.linalg.solve(A,np.broadcast_to(W,(len(om),d))[...,None])[...,0][:,:n]@pb
    return Y
_E=[(0,1),(1,3),(2,3),(0,2),(2,1),(3,0)]
def response_eliminated(om,th,phi,w,c,G,k,delta):
    """Slow 4x4 model with the reservoirs eliminated exactly at each frequency; vectorized. Equal to response_full for
    the same parameters (fit model)."""
    om=np.asarray(om); th=np.asarray(th); T,F=len(th),len(om)
    g=1/(1+2*(-1j*om)/k)
    M=np.zeros((T,F,4,4),complex)
    eps=[1.,1.,1.,1.,w,c]; load=np.zeros(4)
    for q,(j,kk) in enumerate(_E):
        ph=phi if q==3 else 0.0
        e=eps[q]; load[j]+=e/2; load[kk]+=e/2
        if q==5:
            f=np.exp(1j*th)[:,None]; M[:,:,j,kk]+=e/2*f*(1+g); M[:,:,kk,j]+=e/2*np.conj(f)*(g-1)
        else:
            M[:,:,j,kk]+=e/2*np.exp(1j*ph)*(1+g); M[:,:,kk,j]+=e/2*np.exp(-1j*ph)*(g-1)
        M[:,:,j,j]+=-e/2*g; M[:,:,kk,kk]+=-e/2*g
    for i in range(4): M[:,:,i,i]+=-(G-load[i])-1j*delta[i]
    A=(-1j*om)[None,:,None,None]*np.eye(4)-M
    return np.concatenate([np.linalg.solve(A,np.broadcast_to(pa,(T,F,4))[...,None])[...,0]@pb for pa,pb in PORTS],axis=0)
def device_type(phi,w,c,G,k,delta=None,errors=None,pert=None,th0=0.0):
    def ev(t):
        L=links(phi,w,c,t+th0)
        if pert is not None: L=pert(L,0)
        M,n=link_model_matrix(L,k,G,delta=delta,errors=errors,allow_overload=True); e=np.linalg.eigvals(M); return e[np.argsort(-e.real)][:n]   # a fit may end slightly below the link load
    p,g,_=track(ev,n0=300); return cycle_type(p),float(g)
# p=[th0,G,c,s,phi,w,d0,d1,d2,d3]
WINDOW=dict(G=(0.9,3.),c=(0.3,2.),w=(0.2,2.))
def fit(y,sig,k,dloc,dwindow,rng,nst=10,om=OM,th=TH,sd_pr=None):
    pr=(lambda x: np.zeros(0)) if sd_pr is None else (lambda x:(x[6:10]-dloc)/sd_pr)
    def model(p,o=om,t=th): return p[3]*np.abs(response_eliminated(o,t+p[0],p[4],p[5],p[2],p[1],k,p[6:10]))**2
    lo=np.r_[-np.inf,WINDOW['G'][0],WINDOW['c'][0],1e-9,-np.inf,WINDOW['w'][0],dloc-dwindow]; hi=np.r_[np.inf,WINDOW['G'][1],WINDOW['c'][1],1e9,np.inf,WINDOW['w'][1],dloc+dwindow]
    Yc=np.concatenate([y[i*len(th)::1][:len(th)][::2,::4] for i in range(len(PORTS))],axis=0); oc=om[::4]; tc=th[::2]; res=[]
    for s in range(nst):
        q=np.r_[2*np.pi*s/nst,rng.uniform(1.1,2.5),np.exp(rng.uniform(np.log(.4),np.log(1.8))),1.,2*np.pi*rng.random(),np.exp(rng.uniform(np.log(.3),np.log(1.8))),dloc+rng.normal(size=4)*min(dwindow,0.5)*0.5]
        q[6:10]=np.clip(q[6:10],lo[6:10]+1e-6,hi[6:10]-1e-6)
        m=model(q,oc,tc); q[3]=max((m*Yc).sum()/(m*m).sum(),1e-8)
        r=least_squares(lambda x:np.r_[((model(x,oc,tc)-Yc)/sig).ravel(),pr(x)],q,bounds=(lo,hi),x_scale=np.r_[1,1,1,q[3],1,1,1,1,1,1],max_nfev=25,diff_step=1e-5)
        res.append((r.cost,r.x))
    res.sort(key=lambda z:z[0]); best=None
    for _,q in res[:2]:
        r=least_squares(lambda x:np.r_[((model(x)-y)/sig).ravel(),pr(x)],np.clip(q,lo+1e-9,hi-1e-9),bounds=(lo,hi),x_scale=np.r_[1,1,1,q[3],1,1,1,1,1,1],max_nfev=60,diff_step=1e-5)
        if best is None or r.cost<best.cost: best=r
    return best,lo,hi
def verdict(r,lo,hi,G_,k,rng,n):
    p=r.x; chi=2*r.cost/n
    if chi>1+3*np.sqrt(2/n): return 'odrzut',None,chi
    at_edge=any(abs(p[i]-lo[i])<1e-3 or abs(p[i]-hi[i])<1e-3 for i in (1,2,5,6,7,8,9))
    try: t,_=device_type(p[4],p[5],p[2],p[1],k,delta=p[6:10],th0=p[0])
    except (TrackingError,ValueError,np.linalg.LinAlgError): return 'nierozstrz',None,chi
    if at_edge: return 'nierozstrz',t,chi
    J=r.jac; ww,VV=np.linalg.eigh(J.T@J); ww=np.clip(ww,1e-10*ww.max(),None); L=VV*np.sqrt(1/ww); agree=0   # covariance in the full space, also when degenerate
    for _ in range(24):
        q=p+L@rng.normal(size=len(p))
        if q[5]<=0 or q[2]<=0: continue    # a sample outside the physical region counts as disagreeing
        try:
            tq,_=device_type(q[4],q[5],q[2],q[1],k,delta=q[6:10],th0=q[0]); agree+= tq==t
        except (TrackingError,ValueError,np.linalg.LinAlgError): pass
    return ('pewne' if agree>=0.95*24 else 'nierozstrz'),t,chi
def simulate_data(phi,w,k,snr,rng,errors=None,pert=None,G=1.5):
    Y=np.abs(response_full(OM,TH,phi,w,1.0,G,k,errors=errors,pert=pert))**2
    sig=10**(-snr/20)*Y.max(); return Y+sig*rng.normal(size=Y.shape),sig
