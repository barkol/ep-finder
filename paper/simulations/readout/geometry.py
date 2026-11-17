"""Geometric quantities of figure 6(a,b) and of the read-out: branches, set of peaks, derivatives. No randomness.
Output: results/readout/geom.npz and a printed summary."""
from _paths import work
import numpy as np
from readout_lib import *
N,G,CM=3,1.4,1.
TH=np.linspace(0,2*np.pi,241);OM=np.linspace(-2.4,2.4,420);w=unif(3)
Sg=np.abs(amp(OM,TH,G,CM,np.zeros(3),w,w))**2;peak=Sg.max();Sg/=peak
W=np.zeros((241,3));GG=np.zeros((241,3));prev=None
for i,t in enumerate(TH):
    lam=np.linalg.eigvals(ringM(-G*np.ones(3),CM*np.exp(1j*t)))
    if prev is None: lam=lam[np.argsort(-lam.imag)]
    else: lam=lam[linear_sum_assignment(np.abs(prev[:,None]-lam[None,:]))[1]]
    prev=lam;W[i]=-lam.imag;GG[i]=2*np.abs(lam.real)
gs=GG.min(1);dmin=2*np.sin(np.pi/3)
npk=np.array([maxim(s) for s in Sg])
om=lambda t,k:-np.sin((t+2*np.pi*k)/3)
br=np.sort([abs(om(2*np.pi/3,k)-om(0,k)) for k in range(3)])
T=np.linspace(0,2*np.pi,721)
A=np.sort([[om(t,k) for k in range(3)] for t in T],axis=1);B=np.sort([[om(t+2*np.pi/3,k) for k in range(3)] for t in T],axis=1)
dset=np.abs(B-A).max(1)
dw=np.abs(np.cos(2*np.pi*np.arange(3)/3))/3
np.savez(work('geom.npz'),TH=TH,OM=OM,Sg=Sg,W=W,G=GG,gsharp=gs,dmin=dmin,npeaks=npk,peak=peak,branches=br,dset_min=dset.min(),dset_max=dset.max(),dset_mean=dset.mean(),dwdth0=dw,
  mean_width=GG.mean(),gs_min=gs.min(),gs_max=gs.max(),ratio_best=dmin/gs.min(),ratio_mean=dmin/gs.mean(),ratio_worst=dmin/gs.max(),comb=np.diff(np.sort(-np.sin(2*np.pi*np.arange(3)/3))))
print("delta_min",dmin,"mean FWHM",GG.mean(),"sharp",gs.min(),gs.max(),"ratios",dmin/gs.min(),dmin/gs.mean(),dmin/gs.max(),"max maxima",npk.max(),"peak",peak)
print("branches",br,"change of the peak set",dset.min(),dset.max(),dset.mean(),"dw/dth(0)",dw,"comb",np.diff(np.sort(-np.sin(2*np.pi*np.arange(3)/3))))
