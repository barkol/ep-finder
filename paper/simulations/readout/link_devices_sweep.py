"""Read-out of the calibrated ring devices in the link model at kappa'=20 (technical notes, section N2, programmed devices, figure N1(a)):
(1) the type of the true device with its static errors is computed, which separates a change of the device from a read-out error;
(2) detunings are known from a local calibration of each mode, delta_cal = delta + noise(sd), window +-0.3, which fixes the mode
assignment; (3) error levels ideal, e1, e2, e5 (ERR).
Output: results/readout/link_devices/<device>__<error>__<snr>.npz, one file per configuration (resumable).
python link_devices_sweep.py [trials=100] [processes=8]; --test runs two trials per device."""
import os,sys,time,zlib
for _v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[_v]="1"
import numpy as np
from _paths import work
from scipy.optimize import least_squares,linear_sum_assignment as lsa
from link_model import Mfull,spec,slow_type,label_sigma,devices,OM,unif
from readout_lib import ctype
FIELDS=['typeD','typeL','typeLD','elemL','chi2L','fL','atbL','thL','cL','gL','costL','costT','n']
ERR={'ideal':(0,0,0,0,0,.02),'e1':(.01,.02,.01,.01,.02,.02),'e2':(.02,.05,.01,.02,.05,.02),'e5':(.05,.05,.02,.05,.1,.05)}  # link G, link phase, pump |c|, probe amplitude, probe phase, sd of the detuning calibration
def stype(d,th,e,n=480):
    N=len(d);F=lambda t:np.linalg.eigvals(Mfull(d,th+t,0.,1.,**e));st=F(0.);lam=st.copy();t=0.;T=2*np.pi;h0=T/n;h=h0
    def gp(x): y=np.abs(x[:,None]-x[None,:]);np.fill_diagonal(y,9e9);return y.min()
    while t<T-1e-12:
        h=min(h,T-t);x=F(t+h);x=x[lsa(np.abs(lam[:,None]-x[None,:]))[1]];g=min(gp(lam),gp(x))
        if np.abs(x-lam).max()>.2*g and h>h0/2**14: h/=2;continue
        lam=x;t+=h;h=min(1.5*h,h0)
    p=lsa(np.abs(lam[:,None]-st[None,:]))[1];idx=sorted(np.argsort(st.real)[-N:].tolist());m={a:b for b,a in enumerate(idx)}
    if not all(int(p[i]) in m for i in idx): return (-1,)
    return ctype(tuple(m[int(p[i])] for i in idx))
def trial(cfg):
    rng=np.random.default_rng(cfg['seed']);d0=np.array(cfg['d']);N=len(d0);th0=cfg['th0'];M=24;sg=10**(-cfg['snr']/20)
    eG,eP,eC,pA,pP,sd=ERR[cfg['err']];TH=np.arange(M)*2*np.pi/M;w=v=unif(N)
    es=dict(G=1+eG*rng.normal(size=N),dph=eP*rng.normal(size=N));dw=(1+pA*rng.normal(size=N))*np.exp(1j*pP*rng.normal(size=N));dv=(1+pA*rng.normal(size=N))*np.exp(1j*pP*rng.normal(size=N))
    dc=eC*rng.normal(size=M);dth=eP*rng.normal(size=M)
    tD=stype(d0,th0,es)
    Y=spec(OM,th0+TH+dth,d0,0.,1.,w*dw,v*dv,dc=dc,**es);Y=Y/Y.max()+rng.normal(0,sg,Y.shape);y=Y.ravel();n=y.size
    dcal=d0+sd*rng.normal(size=N);lo=np.r_[-.3,.3,dcal-.3];hi=np.r_[1.5,2.,dcal+.3];LO=np.r_[-np.inf,lo];HI=np.r_[np.inf,hi]
    def res(p,th,om,Yd):
        m=spec(om,p[0]+th,p[3:],p[1],p[2],w,v).ravel();return m*(m@Yd)/(m@m)-Yd
    Yc=Y[::3,::2].ravel();ths,oms=TH[::3],OM[::2];cand=[]
    S=[np.r_[2*np.pi*k/8,rng.uniform(0,.5),np.exp(rng.uniform(np.log(.7),np.log(1.4))),dcal+.1*rng.normal(size=N)] for k in range(8)]
    for q in S:
        try: a=least_squares(res,np.clip(q,LO+1e-6,HI-1e-6),args=(ths,oms,Yc),bounds=(LO,HI),x_scale='jac',max_nfev=20);cand.append((a.cost,a.x))
        except Exception: pass
    cand.sort(key=lambda t:t[0]);best=None
    for c_,q in cand[:3]:
        a=least_squares(res,q,args=(TH,OM,y),bounds=(LO,HI),x_scale='jac',max_nfev=40)
        if best is None or a.cost<best.cost: best=a
    p=best.x;tt=slow_type(p[3:],p[0],p[1],p[2])[0];tn=cfg['type'];el=label_sigma(p[3:],p[0],p[1],p[2])==tuple(cfg['sigma'])
    chi=2*best.cost/(n-len(p)-1)/sg**2
    C=np.linalg.pinv(best.jac.T@best.jac,rcond=1e-12)*sg**2;C=(C+C.T)/2;ev,U=np.linalg.eigh(C);Lc=U*np.sqrt(np.clip(ev,0,None));same=0
    for _ in range(24):
        x=np.clip(p+Lc@rng.normal(size=len(p)),LO,HI);same+=int(slow_type(x[3:],x[0],x[1],x[2],n=240)[0]==tt)
    atb=any(p[1:3]<lo[:2]+.01*(hi[:2]-lo[:2]))or any(p[1:3]>hi[:2]-.01*(hi[:2]-lo[:2]))or any(p[3:]<lo[2:]+.005)or any(p[3:]>hi[2:]-.005)
    ct=res(np.r_[th0,0.,1.,d0],TH,OM,y);ct=.5*ct@ct
    return [float(tD==tn),float(tt==tn),float(tt==tD),float(el),chi,same/24,float(atb),float(np.angle(np.exp(1j*(p[0]-th0)))),p[2],p[1],best.cost,ct,n]
def F(A,n): return A[:,FIELDS.index(n)]
def verdict(A):
    """correct/false certificates with respect to the type of the true device"""
    rej=F(A,'chi2L')>1+3*np.sqrt(2/F(A,'n'));und=(~rej)&((F(A,'atbL')>0)|(F(A,'fL')<.95));dec=(~rej)&(~und);return rej,und,dec&(F(A,'typeLD')>0),dec&(F(A,'typeLD')==0)
def summ(A):
    r,u,o,b=verdict(A);return f"device_kept={F(A,'typeD').mean():.3f} fit_device={F(A,'typeLD').mean():.3f} elem={F(A,'elemL').mean():.3f} cert_ok={o.mean():.3f} cert_false={b.mean():.3f} undecided={u.mean():.3f} rejected={r.mean():.3f} chi2={np.median(F(A,'chi2L')):.3f} rms_th={np.sqrt((F(A,'thL')**2).mean()):.3f} glob={np.mean(F(A,'costL')<=F(A,'costT')+1e-9):.3f}"
def main():
    n=int(sys.argv[1]) if len(sys.argv)>1 else 100;np_=int(sys.argv[2]) if len(sys.argv)>2 else 8;C=[]
    for D in devices():
        for er in ERR:
            for s in(20,26,32,40): C.append(dict(key=f"{D['name']}__{er}__{s}".replace('+','p'),base=dict(D,snr=s,err=er)))
    from multiprocessing import Pool
    os.makedirs(work('link_devices'),exist_ok=True);todo=[c for c in C if not os.path.exists(work(f"link_devices/{c['key']}.npz"))];print('total',len(C),'to compute',len(todo),flush=True)
    with Pool(np_) as P:
        for i,c in enumerate(todo):
            t0=time.time();s0=zlib.crc32(c['key'].encode())%2**31
            A=np.array(P.map(trial,[dict(c['base'],seed=s0+s) for s in range(n)],chunksize=1),float)
            tmp=work(f"link_devices/.{c['key']}.tmp.npz");np.savez_compressed(tmp,A=A,fields=np.array(FIELDS));os.replace(tmp,work(f"link_devices/{c['key']}.npz"))
            print(f"[{i+1}/{len(todo)}] {c['key']} {summ(A)} {time.time()-t0:.0f}s",flush=True)
if __name__=='__main__':
    if '--test' in sys.argv:
        for D in devices():
            assert stype(D['d'],D['th0'],{})==D['type'],D['name']
            A=np.array([trial(dict(D,snr=40,err='ideal',seed=s)) for s in range(2)]);print(D['name'],summ(A),flush=True)
    else: main()
