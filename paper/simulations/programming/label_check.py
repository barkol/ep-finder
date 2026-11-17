"""Independent check of the link-model labels and sigma. Labels are carried from the ideal limit: at s=1e-5, 1/kappa'
is lowered from kappa'=1e5 to the target on a fixed logarithmic grid with one-to-one assignment, then s is raised to 1
at the base phase, then one orbit. Compares with the stored sigma for all N<=5 (kappa'=20, 40) and the N=6 sample.
Output: results/programming/label_check.json."""
import os
os.environ["MKL_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, json, csv, multiprocessing as mp
from scipy.optimize import linear_sum_assignment as lsa
from ring_lib import full_link_matrix
from _paths import data, results, work
def F(delta,th,kap,s):
    g=np.zeros(len(delta));g[0]=g[-1]=1-s;return full_link_matrix(delta,th,kap,gamma0=g,cabs=s)
def step(cur,M):
    e=np.linalg.eigvals(M);C=np.abs(cur[:,None]-e[None,:]);r,c=lsa(C);return e[c],C[r,c].max()
def mono(delta,kap,th0,n=(1500,2000,6000)):
    N=len(delta);s0=1e-5;kk=np.geomspace(1e5,kap,n[0])
    w=np.linalg.eigvals(F(delta,th0,kk[0],s0));C=np.abs((-1j*delta-1)[:,None]-w[None,:]);r,c=lsa(C);cur=w[c];err0=C[r,c].max();mg=9e9;mdisp=0
    def rec(l):
        d=np.abs(l[:,None]-l[None,:])+np.eye(N)*1e9;return d.min()
    for k in kk[1:]:
        cur,dd=step(cur,F(delta,th0,k,s0));mg=min(mg,rec(cur));mdisp=max(mdisp,dd/max(rec(cur),1e-12))
    lab0=cur.copy();glab=rec(cur)
    for s in np.linspace(s0,1,n[1])[1:]:
        cur,dd=step(cur,F(delta,th0,kap,s));mg=min(mg,rec(cur));mdisp=max(mdisp,dd/max(rec(cur),1e-12))
    base=cur.copy()
    for t in np.linspace(0,2*np.pi,n[2])[1:]:
        cur,dd=step(cur,F(delta,th0+t,kap,1.));mg=min(mg,rec(cur));mdisp=max(mdisp,dd/max(rec(cur),1e-12))
    q=np.abs(cur[:,None]-base[None,:]).argmin(1)
    return tuple(int(x) for x in q),float(mg),float(mdisp),float(err0),float(glab)
def job(a):
    k,r=a
    N=int(r['N']);sg=tuple(map(int,r['sigma'].split()));de=np.array(list(map(float,r['delta_over_eps'].split())));o=float(r['base_phase_offset_rad'])
    s,mg,md,e0,gl=mono(de,float(k),np.pi/2*(1-N)+o)
    return dict(k=k,N=N,sigma=list(sg),got=list(s),same=s==sg,mg=mg,mdisp=md,glab=gl,g_loop=None if r['g_loop']=='None' else float(r['g_loop']),stored_ok=r['ok']=='1')
if __name__=='__main__':
    jobs=[]
    for k in (20,40):
        for r in csv.DictReader(open(results(f'link_sigma_table_k{k}.csv'))):
            if r['ok']=='1':jobs.append((k,r))
    with mp.get_context('fork').Pool(16) as p:R=p.map(job,jobs,chunksize=4)
    json.dump(R,open(results('label_check.json'),'w'),indent=0)
    for k in (20,40):
        for rng,lab in (((2,5),'N<=5'),((6,6),'N=6')):
            X=[r for r in R if r['k']==k and rng[0]<=r['N']<=rng[1]]
            if X:print(k,lab,'same',sum(r['same'] for r in X),'/',len(X),'worst step/gap',max(r['mdisp'] for r in X))
    for r in R:
        if not r['same']:print('DIFF',r['k'],r['sigma'],r['got'],'mg %.3f glab %.3f disp %.3f'%(r['mg'],r['glab'],r['mdisp']))
