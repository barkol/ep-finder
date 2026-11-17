"""Design in the full link model (reservoir mode per link) at finite kappa' (eps=1): for each cycle type maximise
min(slow loop gap, slow ray gap) over static detunings, requiring the target labelled sigma. Labels: corner-link
strength s->0 at fixed base phase, the link-model version of the c->0 rule. Also provides slow_sigma(), used by the
other link-model scripts. Resumable: work/link_design.jsonl. Usage: link_design.py kappa [--test]"""
import os
os.environ["MKL_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, json, sys, time, multiprocessing as mp
from scipy.optimize import differential_evolution, minimize, linear_sum_assignment as lsa
from ring_lib import *
from _paths import data, results, work
CK=work('link_design.jsonl')
def fmat(delta,theta,kappa,s=1.0):
    # the corner link also carries local loss s/2 on modes 0 and N-1; compensate it so that only c changes on the ray (the c->0 rule)
    g=np.zeros(len(delta));g[0]=g[-1]=(1-s)
    return full_link_matrix(delta,theta,kappa,gamma0=g,cabs=s)
def slow_sigma(delta,kappa,th0,h0=1/160,frac=.15):
    """returns (sigma,g_ray,g_loop,maxRe) or None. Tracks the slow subset only, assignment among all 2N eigenvalues."""
    N=len(delta)
    # ray: s from small to 1 at th0
    s0=1e-4
    w=np.linalg.eigvals(fmat(delta,th0,kappa,s0))
    target=-1j*delta-1.0   # ideal; matched only at s->0
    C=np.abs(target[:,None]-w[None,:]);r,c=lsa(C)
    l=w[c].copy();mr=-np.inf
    def adv(l,F,h0):
        t=0.;h=h0;mg=pgap(l);mx=l.real.max()
        while t<1-1e-13:
            h=min(h,1-t);e=np.linalg.eigvals(F(t+h));C=np.abs(l[:,None]-e[None,:]);idx=C.argmin(1)
            g=min(pgap(l),pgap(e[idx]))
            if len(set(idx.tolist()))<N or C[np.arange(N),idx].max()>frac*g:
                h/=2
                if h<1e-7:return None,mg,mx
                continue
            l=e[idx];t+=h;mg=min(mg,pgap(l));mx=max(mx,l.real.max());h=min(1.5*h,h0)
        return l,mg,mx
    lb,gr,_=adv(l,lambda t:fmat(delta,th0,kappa,s0+(1-s0)*t),h0)
    if lb is None:return None
    le,gl,mx=adv(lb,lambda t:fmat(delta,th0+2*np.pi*t,kappa,1.0),h0)
    if le is None:return None
    q=np.abs(le[:,None]-lb[None,:]).argmin(1)
    if len(set(q.tolist()))<N:return None
    # gap to t->0 end excluded: include label separation at s0
    return tuple(int(x) for x in q),gr,gl,mx
def obj(x,kappa,target,th0,cap):
    d=np.concatenate(([0],np.cumsum(x)));d=d-d.mean()
    if np.abs(d).max()>cap:return 1+np.abs(d).max()-cap
    r=slow_sigma(d,kappa,th0)
    if r is None or r[0]!=target:return 0.5
    return -min(r[1],r[2])
def job(a):
    kappa,N,part,comp,d0,cap=a
    d0=np.array(d0);target,_=sigma_rank_formula(d0,1.0);th0=base_theta(N)
    bounds=[(0.05,2*cap)]*(N-1)
    x0=np.clip(np.diff(d0),.05,2*cap)
    r=differential_evolution(obj,bounds,args=(kappa,target,th0,cap),x0=x0,seed=N*7+len(comp),popsize=8,maxiter=45,tol=1e-3,polish=False,init='sobol',updating='immediate')
    r2=minimize(obj,r.x,args=(kappa,target,th0,cap),method='Nelder-Mead',options=dict(maxiter=250,xatol=1e-3,fatol=1e-4))
    x=r2.x if r2.fun<r.fun else r.x
    d=np.concatenate(([0],np.cumsum(x)));d=d-d.mean()
    R=slow_sigma(d,kappa,th0,h0=1/400,frac=.08)
    R0=slow_sigma(d0,kappa,th0,h0=1/400,frac=.08)
    o=dict(kappa=kappa,N=N,part=list(part),comp=list(comp),d=d.tolist(),same=bool(R is not None and R[0]==target),g_ray=None if R is None else R[1],g_loop=None if R is None else R[2],maxRe=None if R is None else R[3],
           ideal_d_same=bool(R0 is not None and R0[0]==target),ideal_d_gloop=None if R0 is None else R0[2],ideal_d_gray=None if R0 is None else R0[1])
    # ideal-model monodromy at the redesigned d, for the record
    return o
if __name__=='__main__':
    kappa=float(sys.argv[1]);test='--test' in sys.argv
    best={}
    for ln in open(results('ladder_optimum.jsonl')):
        o=json.loads(ln)
        if o['cap']!=2.0:continue
        k=(o['N'],tuple(o['part']))
        if k not in best or o['G']>best[k]['G']:best[k]=o
    done=set()
    if os.path.exists(CK):
        for ln in open(CK):
            o=json.loads(ln);done.add((o['kappa'],o['N'],tuple(o['part'])))
    jobs=[(kappa,k[0],k[1],o['comp'],o['d'],2.0) for k,o in sorted(best.items()) if (kappa,k[0],k[1]) not in done and (not test or k[0]<=3)]
    print('jobs',len(jobs),flush=True);t0=time.time()
    with mp.get_context('fork').Pool(12) as pool,open(work('link_design_test.jsonl') if test else CK,'a') as fo:
        for o in pool.imap_unordered(job,jobs):
            fo.write(json.dumps(o)+'\n');fo.flush()
            print(o['N'],o['part'],'same',o['same'],'gl',o['g_loop'],'gr',o['g_ray'],'maxRe',o['maxRe'],'ideal-d same',o['ideal_d_same'],o['ideal_d_gloop'],'dmax %.2f'%max(abs(x) for x in o['d']),round(time.time()-t0),flush=True)
