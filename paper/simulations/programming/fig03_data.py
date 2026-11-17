"""Figure 3 data: labelled sigma on the zero-trace detuning plane (N=3, |c|=1), the loop gap there, and the per-type
summary (ideal gaps from ladder_optimize.py, link-model gaps at kappa'=20 from link_budget.py).
Output: data/fig03_programming.npz.
The shipped data/fig03_programming.npz was produced by this script with the shipped results/programming files
(ladder_optimum.jsonl and link_model_budget.json from the second link_budget.py run). A full rerun of
this script with these inputs (about 2 min) reproduces the shipped file byte for byte."""
import os
os.environ.setdefault("MKL_NUM_THREADS","4")
import numpy as np, json
from ring_lib import *
from _paths import data, results, work
perms=[(0,1,2),(1,0,2),(2,1,0),(0,2,1),(1,2,0),(2,0,1)]
n=601;L=3.2
u=np.linspace(-L,L,n)
# coordinates: x=(delta_1-delta_0)/sqrt2... orthonormal basis of the zero-trace plane
e1=np.array([1,-1,0])/np.sqrt(2);e2=np.array([1,1,-2])/np.sqrt(6)
S=np.full((n,n),-1,int);G=np.zeros((n,n))
def rule(delta,R=1.):
    o=np.argsort(delta);d=delta[o]
    rs,_=sigma_rank_formula(d,R);s=[0]*3
    for r in range(3):s[o[r]]=int(o[rs[r]])
    return tuple(s)
for i,y in enumerate(u):
    for j,x in enumerate(u):
        de=x*e1+y*e2
        if np.diff(np.sort(de)).min()<1e-9:continue
        S[i,j]=perms.index(rule(de))
        G[i,j]=gaps(np.sort(de),1.0,nloop=121,nray=3,refine=False)[0]
# direct tracker spot checks on the grid
rng=np.random.default_rng(5);ok=tot=0
for _ in range(400):
    i,j=rng.integers(n,size=2);de=u[j]*e1+u[i]*e2
    s,gr,gl,fl=labelled_sigma(-1j*de,1.0,base_theta(3))
    if fl and gl>.02 and S[i,j]>=0:
        tot+=1;ok+=int(perms.index(s)==S[i,j])
print('tracker vs map on random grid points:',ok,'/',tot)
best={}
for ln in open(results('ladder_optimum.jsonl')):
    o=json.loads(ln)
    if o['cap']!=2.0:continue
    k=(o['N'],tuple(o['part']))
    if k not in best or o['G']>best[k]['G']:best[k]=o
fm={(r['N'],tuple(r['part'])):r for r in json.load(open(results('link_model_budget.json'))) if r['kappa']==20}
keys=[k for k in sorted(best,key=lambda k:(k[0],-len(k[1]))) if k[0]>=3]
names=['+'.join(str(x) for x in k[1]) for k in keys]
Ns=np.array([k[0] for k in keys]);gi=np.array([best[k]['g_loop'] for k in keys])
g20=np.array([fm[(k[0],k[1])]['g'] for k in keys])
pts={}
d3=np.array(best[(3,(3,))]['d']);d21=np.array(best[(3,(2,1))]['d'])
np.savez(data('fig03_programming.npz'),u=u,S=S,G=G,e1=e1,e2=e2,perm=np.array(perms),names=np.array(names),Ns=Ns,gi=gi,g20=g20,d3=d3,d21=d21,check=np.array([ok,tot]))
