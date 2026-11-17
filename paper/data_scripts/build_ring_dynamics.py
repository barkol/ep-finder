#!/usr/bin/env python3
"""Data of figures 2 and 4 -> data/ring_dynamics.npz. Reads the cached Monte-Carlo robustness counts
(results/robustness_mc.npz) and dynamics (results/regimes_dynamics.npz); only the gap on the real loop is computed here."""
import numpy as np
from _paths import data, results
def M(a,c): return np.array([[a,1,0],[0,0,1],[c,0,-a]],dtype=complex)
th=np.sort(np.concatenate([np.linspace(0,2*np.pi,400),np.pi*np.array([.1055,.8945,1.1055,1.8945])]))
dre=np.array([min(abs(e[i]-e[j]) for i in range(3) for j in range(i+1,3)) for e in (np.linalg.eigvals(M(np.cos(p),np.sin(p))) for p in th)])
Z=np.load(results('robustness_mc.npz')); P=Z['cnt']/int(Z['NR']); assert P[0,:,1].sum()==0
D=np.load(results('regimes_dynamics.npz'))
kl=['TH','W']+[k+s for k in ('T20_e0.0','T20_e1e-08','T20_e0.0001','T110_e0.0001') for s in ('_inf',)]+[k+'_lab' for k in ('T110_e0.0','T110_e1e-16','T110_e0.0001')]
kw={k.replace('.','p').replace('-','m'):D[k] for k in kl}
np.savez(data('ring_dynamics.npz'),th=th,dre=dre,fs=Z['fs'],P=P,**kw)
print('complex-loop gap',2*np.sin(np.pi/3),'min real-loop gap',dre.min(),'P(0.1)',P[0,np.argmin(abs(Z['fs']-.1)),0],'1-F T20 2 orbits',D['T20_e0.0_inf'][240])
print('wrote data/ring_dynamics.npz')
