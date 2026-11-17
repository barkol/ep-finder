"""Fits of a single spectrum (figure 6(c), curve 'one spectrum'): 4 parameters, one spectrum, grid of 420 points with |om|<=2.4,
window V4, blind starts. Also two oracle fits for comparison: start near the truth with a fixed scale 1 ('orakul_kod') and with the
correct scale ('orakul_skala'); these names are file keys. Output: results/readout/fig3c/<kind>_<window>_<snr>_<mode>.npy, one file per
configuration (resumable). python single_spectrum_fits.py [trials=400] [processes=8]"""
from _paths import work
import sys,os,time,numpy as np
from multiprocessing import Pool
from readout_lib import *
OM=np.linspace(-2.4,2.4,420);G0,C0=1.4,1.0;N=3;w=unif(3);P3=[0,2*np.pi/3,4*np.pi/3]
TH=np.linspace(0,2*np.pi,241)
NORM=(np.abs(amp(OM,TH,G0,C0,np.zeros(3),w,w))**2).max()
def tr(a):
    kind,win,snr,mode,seed=a
    rng=np.random.default_rng(seed);th=P3[seed%3] if mode=='P3' else rng.uniform(0,2*np.pi)
    y=(np.abs(amp(OM,np.array([th]),G0,C0,np.zeros(3),w,w))**2/NORM+rng.normal(0,10**(-snr/20),OM.shape)).ravel()
    J=Joint(OM,np.zeros(1),3,y,fixal=True)
    if kind=='blind':
        r,_=fit(OM,np.zeros(1),3,y,WIN[win] if win else ((.01,60),(.01,120)),rng,nst=8,fixal=True)
    elif kind=='orakul_kod':   # oracle: LM, abs(), start th+N(0,.15), scale 1 while the true one is 1/NORM
        def f(q): return abs(q[3])*(np.abs(amp(OM,np.array([q[0]]),abs(q[1]),abs(q[2]),np.zeros(3),w,w))**2).ravel()-y
        r=least_squares(f,[th+rng.normal(0,.15),G0,C0,1.0],method='lm',max_nfev=4000)
    elif kind=='orakul_skala':
        def f(q): return abs(q[3])*(np.abs(amp(OM,np.array([q[0]]),abs(q[1]),abs(q[2]),np.zeros(3),w,w))**2).ravel()-y
        r=least_squares(f,[th+rng.normal(0,.15),G0,C0,1/NORM],method='lm',max_nfev=4000)
    e=np.angle(np.exp(1j*(r.x[0]-th)));return e,abs(r.x[1]),abs(r.x[2]),abs(r.x[3]),2*r.cost/(420-4)*10**(snr/10),th
def main():
    n=int(sys.argv[1]) if len(sys.argv)>1 else 400;np_=int(sys.argv[2]) if len(sys.argv)>2 else 8
    os.makedirs(work('fig3c'),exist_ok=True)
    SN=list(np.arange(6,36,2.))
    CFG=[('blind','V4',s,'P3') for s in SN]+[('blind','V4',s,'U') for s in SN]+[('blind','do5',s,'P3') for s in (14,20,26,40)]+[('blind','do10',s,'P3') for s in (14,20,26,40)]+[('blind',None,s,'P3') for s in (14,20,26,40)]+[('blind','ciasne',s,'P3') for s in (14,20,26,40)]+[(k,None,s,'P3') for k in('orakul_kod','orakul_skala') for s in SN]
    with Pool(np_) as P:
        for c in CFG:
            fn=work(f"fig3c/{c[0]}_{c[1]}_{int(c[2])}_{c[3]}.npy")
            if os.path.exists(fn): continue
            t0=time.time();R=np.array(P.map(tr,[c+(100000*int(c[2])+s,) for s in range(n)],chunksize=8))
            np.save(fn+'.tmp.npy',R);os.replace(fn+'.tmp.npy',fn)
            e=R[:,0];print(c,f"P={np.mean(np.abs(e)<np.pi/3):.4f} rms={np.sqrt((e**2).mean()):.4f} medG={np.median(R[:,1]):.2f} medc={np.median(R[:,2]):.2f} {time.time()-t0:.0f}s",flush=True)
if __name__=='__main__': main()
