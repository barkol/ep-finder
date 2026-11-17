"""Finite kappa' breaks the permutation symmetry of chi, so the ring position of each detuning matters.
For every labelled sigma, N=2..6: assign the link_design.py layout (given kappa') to the modes, run the full model
with c->0 labels, record success and slow gaps; slow-sector adiabatic S of the worst sigma of each class.
Also provides design() and Sfull(). Resumable per N in work/. Usage: link_all_sigma.py kappa"""
import os
os.environ["MKL_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, json, sys, time, itertools, multiprocessing as mp
from ring_lib import *
from _paths import data, results, work
from link_design import slow_sigma
def design(kappa):
    best={};fm={}
    for ln in open(results('ladder_optimum.jsonl')):
        o=json.loads(ln)
        if o['cap']!=2.0:continue
        k=(o['N'],tuple(o['part']))
        if k not in best or o['G']>best[k]['G']:best[k]=o
    for ln in (open(work('link_design.jsonl')) if os.path.exists(work('link_design.jsonl')) else []):
        o=json.loads(ln)
        if o['kappa']==kappa:fm[(o['N'],tuple(o['part']))]=o
    return best,fm
def one(a):
    s,d0,d,kappa=a
    N=len(s);rs,_=sigma_rank_formula(np.array(d0),1.0);f=assign_modes(rs,s)
    delta=np.zeros(N);delta[f]=d
    r=slow_sigma(delta,kappa,base_theta(N),h0=1/300,frac=.1)
    if r is None:return dict(sigma=list(s),delta=delta.tolist(),ok=False,g_loop=None,g_ray=None,maxRe=None,got=None)
    return dict(sigma=list(s),delta=delta.tolist(),ok=bool(r[0]==tuple(s)),g_loop=r[2],g_ray=r[1],maxRe=r[3],got=list(r[0]))
def Sfull(delta,kappa,n=121):
    """slow-sector S with spectral projectors of the full matrix; dM/dtheta by central difference"""
    N=len(delta);best=0.
    for th in np.linspace(0,2*np.pi,n)[:-1]:
        M=full_link_matrix(delta,th,kappa);w,R=np.linalg.eig(M);Ri=np.linalg.inv(R)
        h=1e-5;dM=(full_link_matrix(delta,th+h,kappa)-full_link_matrix(delta,th-h,kappa))/(2*h)
        slow=np.argsort(-w.real)[:N];nr=np.linalg.norm(R,axis=0)
        for k in slow:
            am=Ri@(dM@(R[:,k]/nr[k]));s=sum((abs(am[m])*nr[m])**2/abs(w[m]-w[k])**4 for m in range(2*N) if m!=k)
            best=max(best,s)
    return float(best)
if __name__=='__main__':
    kappa=float(sys.argv[1]);best,fm=design(kappa);t0=time.time()
    missing=sorted(k for k in best if k not in fm)
    if missing:sys.exit(f'run link_design.py {sys.argv[1]} first: no link-model design for {len(missing)} cycle types, e.g. {missing[0]}')
    res={}
    with mp.get_context('fork').Pool(14) as pool:
        for N in range(2,7):
            ck=work(f'link_all_sigma_k{int(kappa)}_N{N}.json')
            if os.path.exists(ck):R=json.load(open(ck))
            else:
                jobs=[(s,best[(N,ctype(s))]['d'],fm[(N,ctype(s))]['d'],kappa) for s in itertools.permutations(range(N))]
                R=pool.map(one,jobs,chunksize=6);json.dump(R,open(ck,'w'))
            cl={}
            for r in R:
                c=ctype(tuple(r['sigma']));e=cl.setdefault(str(c),dict(n=0,ok=0,gmin=np.inf,grmin=np.inf,maxRe=-9,worst=None))
                e['n']+=1;e['ok']+=int(r['ok'])
                if r['ok']:
                    if r['g_loop']<e['gmin']:e['gmin']=r['g_loop'];e['worst']=r['delta']
                    e['grmin']=min(e['grmin'],r['g_ray']);e['maxRe']=max(e['maxRe'],r['maxRe'])
            for c,e in cl.items():
                if e['worst'] is not None:e['S']=Sfull(np.array(e['worst']),kappa)
                print(N,c,{k:(round(v,4) if isinstance(v,float) else v) for k,v in e.items() if k!='worst'},flush=True)
            res[N]=cl
            print(f'N={N}: {sum(r["ok"] for r in R)}/{len(R)} t={time.time()-t0:.0f}',flush=True)
    json.dump(res,open(work(f'link_all_sigma_k{int(kappa)}.json'),'w'),indent=0)
