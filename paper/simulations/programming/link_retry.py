"""(1) Heavier search for the sigma that link_per_sigma.py did not reach at kappa'=20 (N=5). (2) Stratified sample at
N=6, kappa'=20: 6 sigma per cycle type, or all if fewer. Resumable: results/programming/link_sigma_retry_k20.jsonl."""
import os
os.environ["MKL_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, json, sys, time, itertools, multiprocessing as mp
from scipy.optimize import differential_evolution, minimize
from ring_lib import *
from _paths import data, results, work
from link_design import slow_sigma
from link_all_sigma import design
from link_per_sigma import mk, obj, job
def heavy(a):
    s,d0,kappa=a
    N=len(s);d0=np.array(d0);rs,_=sigma_rank_formula(d0,1.0);f=assign_modes(rs,s);th0=base_theta(N);cap=3.0
    bestx=None;bf=9
    for seed in (1,2,3):
        bounds=[(.05,2*cap)]*(N-1)+[(-3.1,3.1)]
        r=differential_evolution(obj,bounds,args=(f,N,s,kappa,th0,cap),seed=seed,popsize=20,maxiter=150,tol=1e-3,polish=False,init='sobol',updating='immediate')
        if r.fun<bf:bf=r.fun;bestx=r.x
        if bf<0:break
    d,delta=mk(bestx,f,N);R=slow_sigma(delta,kappa,th0+bestx[N-1],h0=1/400,frac=.08);ok=bool(R is not None and R[0]==s)
    return dict(kind='heavy',kappa=kappa,sigma=list(s),type=list(ctype(s)),ok=ok,delta=delta.tolist(),th_off=float(bestx[N-1]),dmax=float(np.abs(d).max()),g_loop=None if not ok else float(R[2]),g_ray=None if not ok else float(R[1]),maxRe=None if not ok else float(R[3]))
def wrap(j):
    if j[0]=='h':return heavy(j[1])
    o=job(j[1]);o['kind']='n6';return o
if __name__=='__main__':
    kappa=20.;best,_=design(kappa);CK=results('link_sigma_retry_k20.jsonl');done=set()
    if os.path.exists(CK):
        for ln in open(CK):done.add(tuple(json.loads(ln)['sigma']))
    fails=[tuple(json.loads(l)['sigma']) for l in open(results('link_sigma_k20.jsonl')) if not json.loads(l)['ok']]
    jobs=[('h',(s,best[(len(s),ctype(s))]['d'],kappa)) for s in fails if s not in done]
    rng=np.random.default_rng(66);cl={}
    for s in itertools.permutations(range(6)):cl.setdefault(ctype(s),[]).append(s)
    for t,v in sorted(cl.items()):
        idx=rng.permutation(len(v))[:6]
        jobs+=[('n',(v[i],best[(6,t)]['d'],kappa)) for i in idx if v[i] not in done]
    print('jobs',len(jobs),flush=True);t0=time.time()
    with mp.get_context('fork').Pool(14) as pool,open(CK,'a') as fo:
        for o in pool.imap_unordered(wrap,jobs):
            fo.write(json.dumps(o)+'\n');fo.flush();print(o['kind'],o['sigma'],o['ok'],o['g_loop'],o['dmax'],round(time.time()-t0),flush=True)
    print('done',round(time.time()-t0),flush=True)
