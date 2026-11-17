#!/usr/bin/env python3
"""Four-mode network in the full link model (one reservoir mode per link): cycle type of the slow modes against phi
and kappa' (w = eps, |c| = eps, Gamma = 1.5 eps). Resumable; output work/link_model_types.jsonl."""
import numpy as np, json, os
from network_lib import *
from _paths import work
Gam=1.5; c=1.0; w=1.0; OUT=work('link_model_types.jsonl')
def link_model_type(phi,k,w=w,c=c,Gam=Gam,errors=None,delta=None):
    """Tracks only the N slow modes (largest Re); the separation of slow and fast modes is checked. The reservoir
    modes are nearly degenerate and carry no information."""
    sep=[np.inf]
    def ev(th):
        M,N=link_model_matrix(diamond(phi,w,c,th),k,Gam,errors=errors,delta=delta); e=np.linalg.eigvals(M); e=e[np.argsort(-e.real)]
        sep[0]=min(sep[0],e[N-1].real-e[N].real); return e[:N]
    p,g,l0=track(ev,n0=500)
    if sep[0]<=0: raise TrackingError('slow and fast modes mix')
    return cycle_type(p),float(g),float(sep[0]),float(l0.real.max())
if __name__=='__main__':
    os.makedirs(os.path.dirname(OUT),exist_ok=True); got={}
    if os.path.exists(OUT):
        for l in open(OUT):
            x=json.loads(l); got[(x['k'],round(x['phi'],5))]=x
    with open(OUT,'a') as f:
        for k in (20,40,80,160,320,1000):
            for phi in np.r_[np.linspace(0,np.pi,9),0.8,1.0,1.2,1.4,1.6,1.8]:
                if (k,round(phi,5)) in got: continue
                try: t,g,sp,rm=link_model_type(phi,k); st='ok'
                except TrackingError as e: t,g,sp,rm=None,None,None,None; st=str(e)[:50]
                f.write(json.dumps(dict(k=k,phi=float(phi),typ=t,gap=g,sep=sp,remax=rm,st=st))+'\n'); f.flush()
    W=[json.loads(l) for l in open(OUT)]
    for k in (20,40,80,160,320,1000):
        r=sorted([x for x in W if x['k']==k],key=lambda x:x['phi'])
        print(k,' '.join(f"{x['phi']:.2f}:{''.join(map(str,x['typ'])) if x['typ'] else 'P'}/{(x['gap'] or 0):.2f}" for x in r),'sep',round(min((x['sep'] or 0) for x in r),1))
