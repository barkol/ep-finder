"""Monte Carlo tolerance per cycle type (gap and error law depend only on the multiset of detunings; channel B draws a
random sigma of the class in each trial). Channel A: real detuning errors N(0,x^2) on each delta_k and relative error
N(0,x^2) on |c|, exact rule, 20000 trials per level. Channel B: complex errors on alpha (detuning and loss, each
N(0,x^2)) and on |c|, independent tracker at fixed base phase, 1000 trials per level. Labelled success = same sigma;
type success = same cycle type. The level x99 of figure N2(d) is the largest x with success >= 0.99.
Resumable: results/programming/error_mc.jsonl, one line per (N, type, channel, x). Usage: error_mc.py [--test]"""
import os
os.environ["MKL_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, json, sys, time, multiprocessing as mp, zlib
from ring_lib import *
from _paths import data, results, work
LEVELS=[.0025,.005,.01,.02,.03,.05,.07,.1,.15,.2,.3]
CK=results('error_mc.jsonl')
def seed_of(*a):return zlib.crc32(repr(a).encode())
def best_layouts():
    best={}
    for ln in open(results('ladder_optimum.jsonl')):
        o=json.loads(ln)
        if o['cap']!=2.0:continue
        k=(o['N'],tuple(o['part']))
        if k not in best or o['G']>best[k]['G']:best[k]=o
    return best
def rule_sigma_modes(delta,R):
    """exact rule for real delta in mode labels, base ctilde=+iR"""
    o=np.argsort(delta);d=delta[o]
    if np.diff(d).min()<1e-12:return None
    rs,_=sigma_rank_formula(d,R)
    N=len(d);s=[0]*N
    for r in range(N):s[o[r]]=int(o[rs[r]])
    return tuple(s)
def chA(a):
    N,part,d,x,n=a
    rng=np.random.default_rng(seed_of('A',N,part,x));d=np.array(d)
    s0,_=sigma_rank_formula(d,1.0)
    lab=typ=0;gs=[]
    for i in range(n):
        dd=d+x*rng.normal(size=N);R=1+x*rng.normal()
        if R<=0:continue
        s=rule_sigma_modes(dd,R)
        if s is None:continue
        if s==s0:lab+=1
        if ctype(s)==ctype(s0):typ+=1
        if i<300:gs.append(gaps(np.sort(dd),R,nloop=181,nray=9,refine=False)[0])
    return dict(N=N,part=list(part),ch='A',x=x,n=n,lab=lab/n,typ=typ/n,g05=float(np.percentile(gs,5)),g50=float(np.median(gs)))
def chB(a):
    N,part,d,x,n=a
    rng=np.random.default_rng(seed_of('B',N,part,x));d=np.array(d)
    rs0,_=sigma_rank_formula(d,1.0);th0=base_theta(N)
    lab=typ=fl=0;
    for i in range(n):
        tgt=[int(k) for k in rng.permutation(N)]
        # random sigma in the class: conjugate rs0 by f
        f=tgt
        sig=[0]*N
        for r in range(N):sig[f[r]]=f[rs0[r]]
        delta=np.zeros(N);delta[f]=d
        al=-1j*delta+x*(rng.normal(size=N)+1j*rng.normal(size=N));R=1+x*rng.normal()
        if R<=0:continue
        s,gr,gl,ok=labelled_sigma(al,R,th0)
        if not ok:fl+=1;continue
        if s==tuple(sig):lab+=1
        if ctype(s)==ctype(sig):typ+=1
    return dict(N=N,part=list(part),ch='B',x=x,n=n,lab=lab/n,typ=typ/n,flag=fl)
def wrap(j):
    return chA(j[1]) if j[0]=='A' else chB(j[1])
if __name__=='__main__':
    test='--test' in sys.argv
    best=best_layouts();done=set()
    if os.path.exists(CK) and not test:
        for ln in open(CK):
            o=json.loads(ln);done.add((o['N'],tuple(o['part']),o['ch'],o['x']))
    jobs=[]
    for (N,part),o in sorted(best.items()):
        for x in LEVELS:
            if (N,part,'A',x) not in done:jobs.append(('A',(N,part,o['d'],x,200 if test else 20000)))
            if (N,part,'B',x) not in done:jobs.append(('B',(N,part,o['d'],x,20 if test else 1000)))
    print('jobs',len(jobs),flush=True);t0=time.time()
    with mp.get_context('fork').Pool(16) as pool,open(work('error_mc_test.jsonl') if test else CK,'a') as fo:
        for i,o in enumerate(pool.imap_unordered(wrap,jobs)):
            fo.write(json.dumps(o)+'\n');fo.flush()
            if i%40==0:print(i,o,round(time.time()-t0),flush=True)
    print('done',round(time.time()-t0))
