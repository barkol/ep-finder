"""Redesign of the sigma where label_check.py disagrees (the coarse tracker stepped across a narrow passage of the ray).
Same objective with a finer tracker; each candidate must also pass label_check.mono. Appends to
results/programming/link_sigma_retry_k20.jsonl with kind='fix'."""
import os
os.environ["MKL_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, json, multiprocessing as mp
from scipy.optimize import differential_evolution
from ring_lib import *
from _paths import data, results, work
from link_design import slow_sigma
from link_all_sigma import design
from link_per_sigma import mk
from label_check import mono
kappa=20.
def obj(x,f,N,s,th0,cap):
    d,delta=mk(x,f,N)
    if np.abs(d).max()>cap:return 1+np.abs(d).max()-cap
    r=slow_sigma(delta,kappa,th0+x[N-1],h0=1/500,frac=.05)
    if r is None or r[0]!=s:return .5
    return -min(r[1],r[2])
def job(s):
    N=len(s);best,_=design(kappa);d0=np.array(best[(N,ctype(s))]['d']);rs,_=sigma_rank_formula(d0,1.);f=assign_modes(rs,s);th0=base_theta(N);cap=2.5
    for seed in (11,12,13,14):
        r=differential_evolution(obj,[(.05,2*cap)]*(N-1)+[(-3.1,3.1)],args=(f,N,s,th0,cap),seed=seed,popsize=12,maxiter=60,tol=1e-3,polish=False,init='sobol')
        if r.fun<0:
            d,delta=mk(r.x,f,N);o=r.x[N-1]
            g=mono(delta,kappa,th0+o)
            if g[0]==s:
                R=slow_sigma(delta,kappa,th0+o,h0=1/800,frac=.05)
                return dict(kind='fix',kappa=kappa,sigma=list(s),type=list(ctype(s)),ok=True,delta=delta.tolist(),th_off=float(o),dmax=float(np.abs(d).max()),g_loop=float(R[2]),g_ray=float(R[1]),maxRe=float(R[3]))
    return dict(kind='fix',kappa=kappa,sigma=list(s),type=list(ctype(s)),ok=False,delta=delta.tolist(),th_off=float(o) if r.fun<0 else 0.,dmax=0.,g_loop=None,g_ray=None,maxRe=None)
if __name__=='__main__':
    R=json.load(open(results('label_check.json')));bad=[tuple(r['sigma']) for r in R if r['k']==20 and not r['same']]
    print(bad,flush=True)
    with mp.get_context('fork').Pool(3) as p,open(results('link_sigma_retry_k20.jsonl'),'a') as fo:
        for o in p.imap_unordered(job,bad):
            fo.write(json.dumps(o)+'\n');fo.flush();print(o['sigma'],o['ok'],o['g_loop'],o['g_ray'],o['dmax'],flush=True)
