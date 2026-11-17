"""Read-out protocol of the permutation: joint fit of M spectra along the orbit, free alpha, bounded multistart, blind starts.
No side effects on import. One trial: trial(cfg). Result: a list of numbers (FIELDS), stored by run_configs."""
import numpy as np,os,pickle,time,zlib
from readout_lib import *
OMJ=np.linspace(-5,5,501)
FIELDS=['W_hat','W_true','type_ok','elem_ok','chi2','f','msig','G_hat','c_hat','th_err','at_bound','stable','cost','cost_truth','mgn_hat','mgn_true','n','dchi_b']
def err_model(name,N,rng,M):
    """calibration errors not known to the fit model. Returns a dict for simulate"""
    e={}
    if name in('casc5','all'): e['eps']=1+.05*rng.normal(size=N-1)
    if name=='casc10': e['eps']=1+.10*rng.normal(size=N-1)
    if name in('probe5','all'):
        e['dw']=(1+.05*rng.normal(size=N))*np.exp(1j*.1*rng.normal(size=N));e['dv']=(1+.05*rng.normal(size=N))*np.exp(1j*.1*rng.normal(size=N))
    if name in('pump','all'): e['dth']=.05*rng.normal(size=M);e['dc']=.02*rng.normal(size=M)
    if name=='pump2': e['dth']=.15*rng.normal(size=M);e['dc']=.05*rng.normal(size=M)
    return e
def calib_window(om,al,Gam,cm,snr,rng,w,v,kappa=.5,cal_db=2.,span_db=6.):
    """Physical calibration: spectrum with the pump of link c switched off (triangular matrix, poles alpha_k-Gamma_off).
    Gamma_off=Gamma-kappa*|c| (the reservoir link loads the modes). |c| from the pump power: c_nom=|c|*10^(g/20), g~N(0,cal_db), window +-span_db."""
    N=len(al);Goff=Gam-kappa*cm
    y=np.abs(amp(om,np.zeros(1),Goff,1e-12,al,w,v))**2;pk=y.max();sig=10**(-snr/20)
    y=(y/pk+rng.normal(0,sig,y.shape)).ravel()
    def res(q):
        a=np.r_[q[2:2+N-1],0.]+1j*q[2+N-1:];a[N-1]=a[N-1]-q[2:2+N-1].sum()
        return q[1]*(np.abs(amp(om,np.zeros(1),q[0],1e-12,a,w,v))**2).ravel()-y
    best=None
    for k in range(6):
        q0=np.r_[rng.uniform(.3,2.5),1.,rng.normal(size=N-1)*.1,rng.normal(size=N)*[.2,1.,2.,2.,1.,.5][k]]
        m=res(np.r_[q0[0],1.,q0[2:]])+y;q0[1]=max(m@y/(m@m),1e-6)
        try: r=least_squares(res,q0,bounds=(np.r_[.02,1e-9,[-2.5]*(N-1),[-6]*N],np.r_[6,1e9,[2.5]*(N-1),[6]*N]),x_scale='jac',max_nfev=60)
        except Exception: continue
        if best is None or r.cost<best.cost:best=r
    Ghat=best.x[0];c_nom=cm*10**(rng.normal(0,cal_db)/20);f=10**(span_db/20)
    clo,chi=c_nom/f,c_nom*f
    win=((max(.02,Ghat-.2),Ghat+.2+kappa*1.5*chi),(clo,chi))
    x0=[np.r_[0.,Ghat+kappa*c_nom,c_nom,1.,best.x[2:]]]
    x0=[np.r_[2*np.pi*k/4,Ghat+kappa*c_nom,c_nom,1.,best.x[2:]] for k in range(4)]
    return win,x0,Ghat
def trial(cfg):
    """cfg: dict(al,N,G,cm,snr,M,seed,win,err,probe,nst,cal). None of the true values enters the starts, only the data."""
    rng=np.random.default_rng(cfg['seed']);N=len(cfg['al']);G,cm,snr,M=cfg['G'],cfg['cm'],cfg['snr'],cfg['M']
    al0=canon(cfg['al']);shift=rng.uniform(-.3,.3);al=al0+1j*shift;th0=rng.uniform(0,2*np.pi)
    TH=np.arange(M)*2*np.pi/M;om=cfg.get('om',OMJ)
    w=v=unif(N) if cfg.get('probe','u')=='u' else port(N)
    e=err_model(cfg.get('err','ideal'),N,rng,M)
    wt=w*e['dw'] if 'dw' in e else w;vt=v*e['dv'] if 'dv' in e else v
    y,sg,pk=simulate(om,TH,al,G,cm,rng,snr,th0=th0,w=wt,v=vt,eps=e.get('eps'),dth=e.get('dth'),dc=e.get('dc'))
    win=cfg.get('win',WIN['V4']);x0=None
    if cfg.get('cal'): win,x0,_=calib_window(om,al,G,cm,snr,rng,w,v)
    r,J=fit(om,TH,N,y,win,rng,nst=cfg.get('nst',16),w=w,v=v,x0=x0)
    p=J.full(r.x);ah=canon(p2al(p,N))
    ph,_,_,sth=monodromy(ah,p[2],th0=p[0]);pt,_,_,stt=monodromy(al0,cm,th0=th0)
    mp=linear_sum_assignment(np.abs(sth[:,None]-stt[None,:]))[1]
    elem=all(mp[ph[i]]==pt[mp[i]] for i in range(N))
    Wh,Wt=W_of(ah,p[2]),W_of(al0,cm)
    n=y.size;chi=2*r.cost/(n-len(r.x))/sg**2
    f,ms=laplace_stab(r,J,sg,rng=rng)
    dchi=lrt_boundary(r,J,sg,win,rng)
    (g0,g1),(c0,c1)=win;atb=(p[1]<g0+.01*(g1-g0))or(p[1]>g1-.01*(g1-g0))or(p[2]<c0+.01*(c1-c0))or(p[2]>c1-.01*(c1-c0))
    # diagnostics (not part of the protocol): cost when starting from the truth in a widened window, to compare the minima
    q=np.r_[th0,G,cm,pk**-1,al.real[:-1],al.imag]
    try: rt=least_squares(J.res,q,jac=J.jac,x_scale='jac',max_nfev=40,bounds=bounds_for(N,((.01,20),(.01,20))));ct=rt.cost
    except Exception: ct=np.nan
    return [Wh,Wt,ctype(ph)==ctype(pt),elem,chi,f,ms,p[1],p[2],np.angle(np.exp(1j*(p[0]-th0))),atb,stable(p,N),r.cost,ct,margin(ah,p[2]),margin(al0,cm),n,dchi]
def key_of(c): return c['key']
def run_configs(cfgs,outdir,nproc=10,log=None):
    """cfgs: list of {key,n,base(dict)}; checkpoint: one npz file per configuration (atomic write); a restart skips finished ones."""
    from multiprocessing import Pool
    os.makedirs(outdir,exist_ok=True);lg=open(log,'a') if log else None
    todo=[c for c in cfgs if not os.path.exists(f"{outdir}/{c['key']}.npz")]
    print(f"total {len(cfgs)}, to compute {len(todo)}",flush=True)
    with Pool(nproc) as P:
        for i,c in enumerate(todo):
            t0=time.time();b=c['base'];s0=zlib.crc32(c['key'].encode())%(2**31)
            A=P.map(trial,[dict(b,seed=s0+s) for s in range(c['n'])],chunksize=max(1,c['n']//(4*nproc)))
            A=np.array(A,float);tmp=f"{outdir}/.{c['key']}.tmp.npz";np.savez_compressed(tmp,A=A,fields=np.array(FIELDS));os.replace(tmp,f"{outdir}/{c['key']}.npz")
            msg=f"[{i+1}/{len(todo)}] {c['key']} {summ(A)} {time.time()-t0:.0f}s"
            print(msg,flush=True)
            if lg: lg.write(msg+"\n");lg.flush()
def F(A,n): return A[:,FIELDS.index(n)]
def verdict(A,d0=9.,k=3.):
    """decision rule. Returns masks: rej (model rejected), und (undecided), ok (correct certificate), bad (false certificate)"""
    n=F(A,'n');thr=1+k*np.sqrt(2/n);rej=F(A,'chi2')>thr;ab=F(A,'at_bound')>0
    und=(~rej)&(ab|(F(A,'dchi_b')<d0));dec=(~rej)&(~und);ok=dec&(F(A,'type_ok')>0);bad=dec&(F(A,'type_ok')==0)
    return rej,und,ok,bad
def summ(A):
    rej,und,ok,bad=verdict(A)
    return f"P(type)={F(A,'type_ok').mean():.3f} P(elem)={F(A,'elem_ok').mean():.3f} cert_ok={ok.mean():.3f} cert_false={bad.mean():.3f} dchi={np.median(F(A,'dchi_b')):.0f} undecided={und.mean():.3f} rejected={rej.mean():.3f} rms_th={np.sqrt((F(A,'th_err')**2).mean()):.3f} chi2={np.median(F(A,'chi2')):.3f} G^={np.median(F(A,'G_hat')):.2f} c^={np.median(F(A,'c_hat')):.2f}"
