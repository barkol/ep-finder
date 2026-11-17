"""Detunings of every labelled sigma, N=2..6: static detunings from the optimal layout of its cycle type (cap |delta|<=2),
the exact rule, and an independent tracker check; the uniform (s,D) scheme per N.
Output: results/programming/sigma_detunings.npz, uniform_layout.json; per-N checkpoints in work/."""
import os
os.environ["MKL_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, json, itertools, time, multiprocessing as mp, sys
from ring_lib import *
from _paths import data, results, work
CAP=2.0
def load_best():
    best={}
    for ln in open(results('ladder_optimum.jsonl')):
        o=json.loads(ln)
        if o['cap']!=CAP:continue
        k=(o['N'],tuple(o['part']))
        if k not in best or o['G']>best[k]['G']:best[k]=o
    return best
def one(a):
    s,d,comp,th0=a
    N=len(s)
    rs,v=sigma_rank_formula(d,1.0)
    f=assign_modes(rs,s)
    delta=np.zeros(N)
    for r in range(N):delta[f[r]]=d[r]
    tr,gr,gl,ok=labelled_sigma(-1j*delta,1.0,th0)
    return dict(sigma=list(s),delta=delta.tolist(),tracked=list(tr) if ok else None,g_ray=gr,g_loop=gl,ok=bool(ok and tuple(tr)==tuple(s)))
if __name__=='__main__':
    best=load_best();t0=time.time()
    out={}
    with mp.get_context('fork').Pool(16) as pool:
        for N in range(2,7):
            ck=work(f'program_sigma_N{N}.json')
            if os.path.exists(ck):R=json.load(open(ck))
            else:
                jobs=[]
                for s in itertools.permutations(range(N)):
                    b=best[(N,ctype(s))]
                    jobs.append((s,np.array(b['d']),b['comp'],base_theta(N)))
                R=pool.map(one,jobs,chunksize=8)
                json.dump(R,open(ck,'w'))
            n_ok=sum(r['ok'] for r in R)
            print(f'N={N}: {len(R)} sigma, realised exactly {n_ok}, min tracked g_loop={min(r["g_loop"] for r in R):.4f}, min g_ray={min(r["g_ray"] for r in R):.4f}, max|delta|={max(max(abs(x) for x in r["delta"]) for r in R):.3f} t={time.time()-t0:.0f}',flush=True)
            out[N]=R
    # one (s,D) pair per N, all non-identity types (identity uses D alone)
    print('universal pair per N (non-identity layouts, best composition of each type)')
    uni={}
    for N in range(3,7):
        comps=[c for c in comps_mod_reflection(N) if len(c)<N]
        bestu=None
        for s in np.arange(.2,1.21,.02):
            for D in np.arange(.5,3.01,.05):
                g={}
                for c in comps:
                    d=layout_to_d(c,s,D);p=np.poly(d);xs=np.sort(np.roots(np.polyder(p)).real);v=np.abs(np.polyval(p,xs));m=rank_cluster_mask(c)
                    if np.all(v[m]<1) and np.all(v[~m]>1):
                        gl,gr=gaps(d,1.0,nloop=181,nray=17,refine=False);gg=min(gl,gr)
                    else:gg=0
                    k=partition_of(c);g[k]=max(g.get(k,0),gg)
                mn=min(g.values())
                if bestu is None or mn>bestu[0]:bestu=(mn,s,D,{str(k):v for k,v in g.items()})
        uni[N]=dict(G=bestu[0],s=float(bestu[1]),D=float(bestu[2]),per=bestu[3]);print(N,uni[N],flush=True)
    json.dump(uni,open(results('uniform_layout.json'),'w'),indent=0)
    # polygon reference: complex alpha, N-cycle, N=3: rho^3=1/2
    rho=.5**(1/3);a=rho*np.exp(2j*np.pi*np.arange(3)/3);s,gr,gl,ok=labelled_sigma(a,1.0,.37);print('complex polygon N=3 rho=%.4f:'%rho,s,'g_ray %.4f g_loop %.4f (needs Re alpha spread %.3f)'%(gr,gl,a.real.max()-a.real.min()))
    np.savez(results('sigma_detunings.npz'),**{f'N{N}_sigma':np.array([r['sigma'] for r in out[N]]) for N in out},**{f'N{N}_delta':np.array([r['delta'] for r in out[N]]) for N in out},**{f'N{N}_gloop':np.array([r['g_loop'] for r in out[N]]) for N in out},**{f'N{N}_gray':np.array([r['g_ray'] for r in out[N]]) for N in out},**{f'N{N}_ok':np.array([r['ok'] for r in out[N]]) for N in out})
