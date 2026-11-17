"""Ladders: maximal minimum gap for real detunings, per layout (composition of N up to reflection) and cap on |delta|.
Objective: maximise G=min(g_loop,g_ray) at |c|=1, with the inside/outside pattern of the critical values imposed;
differential evolution and a Nelder-Mead polish. Also the two-parameter uniform scheme (s,D) on a grid.
Resumable: one JSON line per finished job in results/programming/ladder_optimum.jsonl.
Usage: ladder_optimize.py Nmax [--test]  (shipped results: Nmax=6)"""
import os
os.environ["MKL_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1";os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np, json, sys, time, multiprocessing as mp
from scipy.optimize import differential_evolution, minimize
from ring_lib import *
from _paths import data, results, work
CK=results('ladder_optimum.jsonl')
def d_from_x(x):
    d=np.concatenate(([0],np.cumsum(x)));return d-d.mean()
def G_of(d,mask,cap,fine=False):
    p=np.poly(d);xs=np.sort(np.roots(np.polyder(p)).real);v=np.abs(np.polyval(p,xs))
    pen=0.0
    viol=np.where(mask,np.maximum(v-1,0),np.maximum(1-v,0))
    pen+=viol.sum()
    over=max(np.abs(d).max()-cap,0)
    if pen>0 or over>0:
        return -1.0-pen-over
    gl,gr=gaps(d,1.0,nloop=(1441 if fine else 241),nray=(129 if fine else 33),refine=fine)
    return min(gl,gr)
def job(a):
    N,comp,cap,seed=a
    mask=rank_cluster_mask(comp)
    bounds=[(0.02,min(2*cap,14.0))]*(N-1)
    f=lambda x:-G_of(d_from_x(x),mask,cap)
    best=None
    # uniform scheme grid
    bu=None
    for s in np.linspace(.1,3,59):
        for D in np.linspace(.1,10,100):
            d=layout_to_d(comp,s,D)
            g=G_of(d,mask,cap)
            if g>0 and (bu is None or g>bu[0]):bu=(g,s,D)
            if len(comp)==1:break
    x0=None
    r=differential_evolution(f,bounds,seed=seed,popsize=12,maxiter=120 if N<=5 else 160,tol=1e-5,polish=False,init='sobol',updating='immediate')
    r2=minimize(f,r.x,method='Nelder-Mead',options=dict(maxiter=1500,xatol=1e-4,fatol=1e-5))
    x=r2.x if r2.fun<r.fun else r.x
    if bu is not None:
        xu=np.diff(layout_to_d(comp,bu[1],bu[2]))
        r3=minimize(f,xu,method='Nelder-Mead',options=dict(maxiter=2000,xatol=1e-4,fatol=1e-5))
        if r3.fun<f(x):x=r3.x
    d=d_from_x(x);G=G_of(d,mask,cap,fine=True)
    if G>0:gl,gr=gaps(d,1.0,nloop=2881,nray=257)
    else:gl,gr=(0,0)
    out=dict(N=N,comp=list(comp),part=list(partition_of(comp)),cap=cap,G=float(G),g_loop=gl,g_ray=gr,d=d.tolist(),
             uniform=None if bu is None else dict(G=float(bu[0]),s=float(bu[1]),D=float(bu[2])))
    return out
if __name__=='__main__':
    Nmax=int(sys.argv[1]);test='--test' in sys.argv
    caps=[2.0,3.0,5.0] if not test else [3.0]
    done=set()
    if os.path.exists(CK) and not test:
        for ln in open(CK):
            o=json.loads(ln);done.add((o['N'],tuple(o['comp']),o['cap']))
    jobs=[]
    for N in range(2,Nmax+1):
        for comp in comps_mod_reflection(N):
            for cap in caps:
                if (N,comp,cap) not in done:jobs.append((N,comp,cap,1000*N+int(cap)))
    print('jobs',len(jobs),flush=True);t0=time.time()
    with mp.get_context('fork').Pool(14) as pool, open(CK if not test else work('ladder_optimum_test.jsonl'),'a') as fo:
        for o in pool.imap_unordered(job,jobs):
            fo.write(json.dumps(o)+'\n');fo.flush()
            print(o['N'],o['comp'],o['cap'],'G=%.4f gl=%.4f gr=%.4f'%(o['G'],o['g_loop'],o['g_ray']),'dmax=%.2f'%max(abs(x) for x in o['d']),o['uniform'],round(time.time()-t0),flush=True)
