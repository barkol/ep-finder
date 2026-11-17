#!/usr/bin/env python3
"""Read-out of the four-mode network with probe errors unknown to the fit (amplitude and phase on every mode, fixed
per device, independent at input and output); technical notes, section N2, and figure N1(b).
Levels: e = errors of links, reservoirs and pump as in run_readout.py (phases 2e rad); ep, fp = amplitude and phase
error of the probe. Output paper/results/network_readout_phased_probe.jsonl (or argv[1]). Resumable; seeds from a
SeedSequence of the task key. Several hours on 20 cores."""
import sys, json, os
import readout_lib
from readout_lib import *
import multiprocessing as mp
from _paths import results
PHASED=np.exp(1j*np.array([0,1.57,3.14,4.71]))/2
def _job(z):
    rng=np.random.default_rng(np.random.SeedSequence([7,int(1000*z['phi']),int(100*z['w']),z['snr'],z['t'],int(1000*z['e']),int(1000*z['ep']),int(1000*z['fp'])]))
    phi,w,snr,e,ep,fp=z['phi'],z['w'],z['snr'],z['e'],z['ep'],z['fp']; k=40; pert=None; errors=None
    if e>0:
        de=rng.normal(size=6)*e; dp=rng.normal(size=6)*2*e; dr=rng.normal(size=6)*e; dq=rng.normal(size=6)*2*e; dth=rng.normal(size=24)*2*e; dc=rng.normal(size=24)*e
        pert=lambda L,i:[(j,kk,ee*(1+de[q])*(1+(dc[i] if q==5 else 0)),ph+dp[q]+(dth[i] if q==5 else 0)) for q,(j,kk,ee,ph) in enumerate(L)]
        errors=lambda q:(dr[q],dq[q])
    G=1.6 if e>0 else 1.5
    draw_probe=lambda:PHASED*(1+ep*rng.normal(size=4))*np.exp(1j*fp*rng.normal(size=4))
    readout_lib.PORTS=[(draw_probe(),draw_probe())]; y,s=simulate_data(phi,w,k,snr,rng,errors=errors,pert=pert,G=G); readout_lib.PORTS=[(PHASED,PHASED)]
    true_type=[int(x) for x in device_type(phi,w,1.0,G,k,errors=errors,pert=pert)[0]]
    r,lo,hi=fit(y,s,k,0.02*rng.normal(size=4),0.3,rng,sd_pr=0.02); ver,t,chi=verdict(r,lo,hi,1.5,k,rng,y.size)
    return dict(z,prawda=true_type,wer=ver,typ=[int(x) for x in t] if t else None,chi=float(chi))
def job(z):
    try: return _job(z)
    except Exception as ex: return dict(z,prawda=None,wer='ERROR:'+repr(ex)[:80],typ=None,chi=None)
LEVELS=[(0.0,0.01,0.02,100),(0.0,0.02,0.05,60),(0.0,0.05,0.1,60),(0.01,0.01,0.02,100),(0.02,0.02,0.05,60)]   # (e, ep, fp, trials)
DEVICES=[(.5,0.),(.5,1.),(.5,1.7),(.5,2.6),(.5,float(np.pi)),(1.,2.),(1.,float(np.pi))]                   # (w, phi)
if __name__=='__main__':
    OUT=sys.argv[1] if len(sys.argv)>1 else results('network_readout_phased_probe.jsonl'); done=set()
    if os.path.exists(OUT):
        for l in open(OUT):
            q=json.loads(l); done.add((q['phi'],q['w'],q['snr'],q['e'],q['ep'],q['fp'],q['t']))
    Z=[dict(phi=float(p),w=w,snr=s,e=e,ep=ep,fp=fp,t=t) for w,p in DEVICES for e,ep,fp,n in LEVELS for s in (26,32) for t in range(n)]
    Z=[z for z in Z if (z['phi'],z['w'],z['snr'],z['e'],z['ep'],z['fp'],z['t']) not in done]; print('tasks',len(Z),flush=True)
    with mp.Pool(min(20,os.cpu_count())) as P, open(OUT,'a') as f:
        for w in P.imap_unordered(job,Z,chunksize=2): f.write(json.dumps(w)+'\n'); f.flush()
    print('done',flush=True)
