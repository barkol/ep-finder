"""Budget per cycle type at eps=1e8/s for the optimal layout (cap 2): |delta|/eps, gaps, adiabatic time T_ad(P=1e-3),
the loss Gamma_min = max Re lambda, the |c| window from the critical values, and the full 2N-mode link model at
kappa'/eps=20, 40, 1000 (labelled sigma, slow gap). Also a check of the first-order P estimate against integration.
Output: results/programming/budget.json."""
import os
os.environ["MKL_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, json, time, multiprocessing as mp
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment as lsa
from ring_lib import *
from _paths import data, results, work
EPS=1e8; Pstar=1e-3
def best_layouts():
    best={}
    for ln in open(results('ladder_optimum.jsonl')):
        o=json.loads(ln)
        if o['cap']!=2.0:continue
        k=(o['N'],tuple(o['part']))
        if k not in best or o['G']>best[k]['G']:best[k]=o
    return best
def slow_track(delta,kappa,th0,g0=0.0,dphi=0.0):
    """labelled monodromy of the slow sector of the full model. Labels: at theta0 the N slowest eigenvalues are matched to the ideal ones
    (-1-i mu) by assignment; the ideal ones carry the c->0 labels from the ideal ray. Returns sigma, slow gap, match error."""
    N=len(delta)
    al=-1j*delta
    cb=np.exp(1j*th0)
    li,gr,ok=track(al.copy(),lambda t:Mring(al,t*cb))
    if not ok:return None,0,0,'ideal'
    F=lambda t:full_link_matrix(delta,th0+2*np.pi*t,kappa,gamma0=g0,dphi=dphi)
    w=np.linalg.eigvals(F(0))
    C=np.abs((li-1.0)[:,None]-w[None,:]);r,c=lsa(C);err=C[r,c].max()
    idx=np.array(c)
    l=w.copy()
    # track all 2N, follow the slow subset
    t=0;h=1/400;mg=np.inf
    cur=w.copy()
    while t<1-1e-14:
        h=min(h,1-t);e=np.linalg.eigvals(F(t+h));CC=np.abs(cur[:,None]-e[None,:]);rr,cc=lsa(CC)
        g=min(pgap(cur),pgap(e))
        if CC[rr,cc].max()>.1*g:
            h/=2
            if h<1e-10:return None,mg,err,'floor'
            continue
        cur=e[cc];t+=h;h=min(1.5*h,1/400);mg=min(mg,pgap(cur[idx]))
    # where do the slow branches end
    C2=np.abs(cur[idx][:,None]-w[idx][None,:]);q=C2.argmin(1)
    # do slow branches return into the slow set?
    d_all=np.abs(cur[idx][:,None]-w[None,:]);closure=d_all.argmin(1)
    inset=all(k in set(idx.tolist()) for k in closure.tolist())
    return tuple(int(x) for x in q),float(mg),float(err),('ok' if inset else 'leak')
def one(a):
    N,part,o=a
    d=np.array(o['d']);rs,v=sigma_rank_formula(d,1.0);th0=base_theta(N)
    al=-1j*d
    S=adiab_S(al,1.0)
    T=2*np.pi*np.sqrt(S/Pstar)
    gl,gr=gaps(d,1.0,nloop=2881,nray=257)
    mr,mi=eig_rate(d)
    mask=rank_cluster_mask(tuple(o['comp']))
    vv=np.abs(v);rlo=float(vv[mask].max()) if mask.any() else 0.0;rhi=float(vv[~mask].min()) if (~mask).any() else float('inf')
    row=dict(N=N,part=list(part),comp=o['comp'],d=d.tolist(),dmax=float(np.abs(d).max()),g_loop=gl,g_ray=gr,S=S,T_ad=T,T_ad_us=T/EPS*1e6,T20P=(2*np.pi/20)**2*S,
             Gamma_min=mr,win=[rlo,rhi],sigma_rank=list(rs),maxIm=mi)
    for kap in (20.,40.,1000.):
        s,mg,err,st=slow_track(d,kap,th0)
        row[f'k{int(kap)}']=dict(same=(s==tuple(rs)),type=None if s is None else list(ctype(s)),gap=mg,match=err,st=st)
    return row
def Pcheck():
    print('P estimate vs integration (share of non-dominant branches at the end of a constant sweep, unit-norm eigenvectors)')
    out=[]
    for d in ([-1.051,0,1.051],[-1.48,-.5,1.0],[-.5,0,.5]):
        d=np.array(d,float);d=d-d.mean();al=-1j*d;N=3;th0=.4
        Mf=lambda th:Mring(al,np.exp(1j*th))
        w,R=np.linalg.eig(Mf(th0));Ri=np.linalg.inv(R);nr=np.linalg.norm(R,axis=0)
        for T in (80.,160.,320.,640.):
            w0,R0=np.linalg.eig(Mf(th0-2*np.pi));k0=int(w0.real.argmax())
            s=solve_ivp(lambda t,y:Mf(th0-2*np.pi+2*np.pi*t/T)@y,(0,T),R0[:,k0]/np.linalg.norm(R0[:,k0]),method='DOP853',rtol=1e-10,atol=1e-30*0+1e-13)
            y=s.y[:,-1];y=y/np.linalg.norm(y)
            a=Ri@y*nr;p=np.abs(a)**2;k=int(p.argmax());Pn=1-p[k]/p.sum()
            dM=np.zeros((N,N),complex);dM[N-1,0]=1j*np.exp(1j*th0);am=Ri@(dM@(R[:,k]/nr[k]))
            est=(2*np.pi/T)**2*sum((abs(am[m])*nr[m])**2/abs(w[m]-w[k])**4 for m in range(N) if m!=k)
            print(f' d={np.round(d,3)} T={T}: integrated {Pn:.3e}, first-order {est:.3e}, ratio {Pn/est:.3f}',flush=True)
            out.append(dict(d=d.tolist(),T=T,num=Pn,est=est))
    return out
if __name__=='__main__':
    best=best_layouts();t0=time.time()
    jobs=[(k[0],k[1],o) for k,o in sorted(best.items())]
    # sanity: the kappa->inf limit at alpha=0 reproduces the ideal gap
    print('alpha=0 N=3, S',adiab_S(np.zeros(3),1.0),'formula',(1/9)*sum((2*np.sin(np.pi*m/3))**-4 for m in (1,2)))
    with mp.get_context('fork').Pool(16) as pool:
        rows=pool.map(one,jobs)
    rows0=[]
    for N in (3,4,5,6):
        al=np.zeros(N);S=adiab_S(al,1.0);rows0.append(dict(N=N,S=S,T_ad=2*np.pi*np.sqrt(S/Pstar),g=2*np.sin(np.pi/N)))
    json.dump(dict(rows=rows,alpha0=rows0,Pcheck=Pcheck(),Pstar=Pstar,EPS=EPS),open(results('budget.json'),'w'),indent=0)
    for r in rows:
        print(r['N'],r['part'],r['comp'],'dmax %.2f gl %.3f gr %.3f S %.2f Tad %.0f (%.2f us) P(T=20) %.1e Gmin %.2f win %.3f..%.3f'%(r['dmax'],r['g_loop'],r['g_ray'],r['S'],r['T_ad'],r['T_ad_us'],r['T20P'],r['Gamma_min'],r['win'][0],r['win'][1]),*[f"k{k}:{'Y' if r['k%d'%k]['same'] else 'N'},{r['k%d'%k]['gap']:.3f},{r['k%d'%k]['st']},{r['k%d'%k]['match']:.3f}" for k in (20,40,1000)])
    for r in rows0:print(r)
    print('t',time.time()-t0)
