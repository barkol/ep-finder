"""Per-sigma calibrated detunings in the full link model at finite kappa': start from the ideal layout and optimise the
ladder of detunings, with the rank->mode assignment fixed and the base-phase offset free, requiring the labelled
sigma and maximising min(loop gap, ray gap), |delta|<=cap.
Resumable: results/programming/link_sigma_k<kappa>.jsonl, one line per sigma.
Usage: link_per_sigma.py kappa Nmin Nmax  (shipped: kappa=20 and 40, N=2..5)"""
import os
os.environ["MKL_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, json, sys, time, itertools, multiprocessing as mp
from scipy.optimize import differential_evolution, minimize
from ring_lib import *
from _paths import data, results, work
from link_design import slow_sigma
from link_all_sigma import design
def mk(x,f,N):
    d=np.concatenate(([0],np.cumsum(x[:N-1])));d=d-d.mean();delta=np.zeros(N);delta[f]=d;return d,delta
def obj(x,f,N,s,kappa,th0,cap):
    d,delta=mk(x,f,N)
    if np.abs(d).max()>cap:return 1+np.abs(d).max()-cap
    r=slow_sigma(delta,kappa,th0+x[N-1])
    if r is None or r[0]!=s:return .5
    return -min(r[1],r[2])
def job(a):
    s,d0,kappa=a
    N=len(s);d0=np.array(d0);rs,_=sigma_rank_formula(d0,1.0);f=assign_modes(rs,s);th0=base_theta(N)
    cap=2.0
    for capi,it in ((2.0,30),(3.0,60)):
        x0=np.concatenate((np.clip(np.diff(d0),.05,2*capi),[0.]))
        bounds=[(.05,2*capi)]*(N-1)+[(-1.5,1.5)]
        r=differential_evolution(obj,bounds,args=(f,N,s,kappa,th0,capi),x0=x0,seed=sum((k+1)*v for k,v in enumerate(s)),popsize=8,maxiter=it,tol=1e-3,polish=False,init='sobol',updating='immediate')
        if r.fun<0:cap=capi;break
        cap=capi
    x=r.x
    if r.fun<0:
        r2=minimize(obj,x,args=(f,N,s,kappa,th0,cap),method='Nelder-Mead',options=dict(maxiter=150,xatol=1e-3,fatol=1e-4))
        if r2.fun<r.fun:x=r2.x
    d,delta=mk(x,f,N)
    R=slow_sigma(delta,kappa,th0+x[N-1],h0=1/400,frac=.08)
    ok=bool(R is not None and R[0]==s)
    return dict(kappa=kappa,sigma=list(s),type=list(ctype(s)),ok=ok,delta=delta.tolist(),th_off=float(x[N-1]),dmax=float(np.abs(d).max()),g_loop=None if not ok else float(R[2]),g_ray=None if not ok else float(R[1]),maxRe=None if not ok else float(R[3]))
if __name__=='__main__':
    kappa=float(sys.argv[1]);Nmin,Nmax=int(sys.argv[2]),int(sys.argv[3]);best,_=design(kappa)
    CK=results(f'link_sigma_k{int(kappa)}.jsonl');done=set()
    if os.path.exists(CK):
        for ln in open(CK):done.add(tuple(json.loads(ln)['sigma']))
    jobs=[(s,best[(N,ctype(s))]['d'],kappa) for N in range(Nmin,Nmax+1) for s in itertools.permutations(range(N)) if s not in done]
    print('jobs',len(jobs),flush=True);t0=time.time();n=nok=0
    with mp.get_context('fork').Pool(12) as pool,open(CK,'a') as fo:
        for o in pool.imap_unordered(job,jobs):
            fo.write(json.dumps(o)+'\n');fo.flush();n+=1;nok+=o['ok']
            if n%20==0 or not o['ok']:print(n,'ok',nok,o['sigma'],o['ok'],o['g_loop'],round(time.time()-t0),flush=True)
    print('done',n,nok,round(time.time()-t0),flush=True)
