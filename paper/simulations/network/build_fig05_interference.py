#!/usr/bin/env python3
"""Figure 5 data -> paper/data/fig05_interference.npz (computed only if the file is missing; about 10 s).
fi, gi, gl: minimum gap over the orbit against phi at w = 0.5 eps, |c| = eps, ideal drift matrix (gi) and link model
at kappa' = 40 eps, Gamma = 1.5 eps (gl). fc1, fc05: threshold phase phi_c between the types (3,1) and (4) at |c| = eps
in the link model for kappa' = 22, 40, 80, 160, 1000 eps (k), w = eps and w = 0.5 eps, by 14 bisection steps."""
import numpy as np, os
from link_model_types import link_model_type
from network_lib import *
from _paths import data
OUT=data('fig05_interference.npz')
if not os.path.exists(OUT):
    fi=np.linspace(0,np.pi,41)
    def ideal(phi,w):
        b=1+np.exp(1j*phi); d=w*np.exp(1j*phi)
        ev=lambda th:(lambda ce:np.roots([1,0,0,-ce*b,-ce*d]))(np.exp(1j*th))
        try: p,g,_=track(ev,n0=1500); return cycle_type(p),g
        except TrackingError: return None,0.0
    gi=np.array([ideal(f,0.5)[1] for f in fi]); gl=[]
    for f in fi:
        try: gl.append(link_model_type(f,40,w=0.5,c=1.0,Gam=1.5)[1])
        except TrackingError: gl.append(0.0)
    gl=np.array(gl); thr={}
    for w in (1.0,0.5):
        fc=[]
        for k in (22,40,80,160,1000):
            a,b=0.0,np.pi
            for _ in range(14):
                m=(a+b)/2
                try: t=link_model_type(m,k,w=w,c=1.0,Gam=1.5)[0]
                except TrackingError: t=None
                if t==(3,1): a=m
                elif t==(4,): b=m
                else: b=m if m>(a+b)/2 else m
            fc.append((a+b)/2)
        thr[w]=fc
    np.savez(OUT,fi=fi,gi=gi,gl=gl,k=[22,40,80,160,1000],fc1=thr[1.0],fc05=thr[0.5])
d=np.load(OUT)
print('fi',np.round(d['fi'][[0,10,20,26,30,40]],2)); print('gap, ideal, w=0.5',np.round(d['gi'][[0,10,20,26,30,40]],3)); print('gap, link model',np.round(d['gl'][[0,10,20,26,30,40]],3))
print('threshold w=1',np.round(d['fc1'],3),'w=0.5',np.round(d['fc05'],3),'kappa',d['k'])
