#!/usr/bin/env python3
"""Sweep time against N (figure 2(f), technical data of appendix 'Sweep time') -> data/tab_sweep_time.csv.
For each N the orbit time T is chosen so that the first-order off-target population P(T) equals its value at N = 3,
T = 20/eps; P(T) is then checked by high-precision integration of one orbit (P_int). Also: delta_min = 2 sin(pi/N),
eps*T_adiab = 2 pi/delta_min^2, mean dGamma and T in ns for eps = 1e8 1/s.
Checkpoint per N in work/sweep_time.json (computed if missing)."""
import numpy as np, json, os, time, csv
from high_precision_propagator import *
from _paths import work, data
CK=work('sweep_time.json'); res=json.load(open(CK)) if os.path.exists(CK) else {}
def Pf(N,T): return (2*np.pi/(N*T))**2*sum((2*np.sin(np.pi*m/N))**-4 for m in range(1,N))
P0=Pf(3,20.)
def Poff(N,T,ramp=False):
    th0=-mp.pi; U=U_of(N,T,th0=th0,ramp=ramp); v=U*R(N,th0,0); v=v/vn(v)
    return float(1-abs(sum(mp.conj(a)*b for a,b in zip(R(N,th0+2*mp.pi,0),v)))**2)
t0=time.time()
NS=(3,4,5,6,8,10,14)
for N in NS:
    k=str(N)
    if k not in res:
        T=2*np.pi/N*np.sqrt(sum((2*np.sin(np.pi*m/N))**-4 for m in range(1,N))/P0)
        P=Poff(N,float(T)); d=2*np.sin(np.pi/N); Tad=2*np.pi/d**2; m=(2*N/np.pi)*np.sin(np.pi/N)*(1-np.cos(np.pi/N))
        res[k]=dict(T=float(T),Pint=P,ratio=P/P0,delta=d,Tad=Tad,mean=m,dG=m*T,ns=T/1e8*1e9)
        json.dump(res,open(CK+'.tmp','w'),indent=0); os.replace(CK+'.tmp',CK)
    print(N,res[k],time.time()-t0,flush=True)
with open(data('tab_sweep_time.csv'),'w',newline='') as f:
    c=csv.writer(f); c.writerow(['N','delta_min','eT_adiab','eT_orbit','P_int','mean_dGamma','T_orbit_ns'])
    for N in NS:
        r=res[str(N)]
        c.writerow([N,f"{r['delta']:.3f}",f"{r['Tad']:.2f}",f"{r['T']:.1f}",f"{r['Pint']*1e3:.2f}e-3",f"{r['mean']:.3f}",f"{r['ns']:.0f}"])
print('wrote data/tab_sweep_time.csv')
