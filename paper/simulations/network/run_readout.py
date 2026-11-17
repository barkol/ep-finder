#!/usr/bin/env python3
"""Read-out of the four-mode network at kappa' = 40 eps (technical notes, section N2; figures 5(e) and N1(b)).
Usage: run_readout.py [full|pilot] [output.jsonl]
  full  (default): phased probe (phases 0, 1.57, 3.14, 4.71 on the modes), w = eps and 0.5 eps, six phases phi each,
        fit variants 'prior' (local detunings as a Gaussian constraint, 0.02 eps), 'okno' (detunings in a +-0.3 eps
        window) and 'slepy' (blind, +-2.5 eps); output paper/results/network_readout_full.jsonl
  pilot: uniform probe, phi = 0 and pi; output work/network_readout_pilot.jsonl
Resumable: tasks already in the output file are skipped. Each task draws its random numbers from a SeedSequence
built from its key, so results do not depend on the order of execution. Several hours on 20 cores."""
import sys, json, os
from readout_lib import *
import multiprocessing as mp
import readout_lib
from _paths import results, work
MODE=sys.argv[1] if len(sys.argv)>1 else 'full'
OUT=sys.argv[2] if len(sys.argv)>2 else (results('network_readout_full.jsonl') if MODE=='full' else work('network_readout_pilot.jsonl'))
PHASED=np.exp(1j*np.array([0,1.57,3.14,4.71]))/2
if MODE=='full': readout_lib.PORTS=[(PHASED,PHASED)]   # probe selected among twelve configurations on pilot data; new seeds here
def task_key(z): return [int(1000*z['phi']),int(100*z['w']),z['snr'],z['t'],{'okno':1,'prior':2,'slepy':3}[z['war']],int(z['e']*1000)]
def job(z):
    try: return _job(z)
    except Exception as ex: return dict(z,prawda=None,wer='ERROR:'+repr(ex)[:80],typ=None,chi=None)   # counted and reported, never lost
def _job(z):
    rng=np.random.default_rng(np.random.SeedSequence(task_key(z))); phi,w,snr,variant,e=z['phi'],z['w'],z['snr'],z['war'],z['e']; k=40
    pert=None; errors=None
    if e>0:   # error level e: amplitudes of links and reservoirs, phases 2e rad; fixed per device, feedback per spectrum
        de=rng.normal(size=6)*e; dp=rng.normal(size=6)*2*e; dr=rng.normal(size=6)*e; dq=rng.normal(size=6)*2*e; dth=rng.normal(size=24)*2*e; dc=rng.normal(size=24)*e
        pert=lambda L,i:[(j,kk,ee*(1+de[q]) *(1+(dc[i] if q==5 else 0)),ph+dp[q]+(dth[i] if q==5 else 0)) for q,(j,kk,ee,ph) in enumerate(L)]
        errors=lambda q:(dr[q],dq[q])
    y,s=simulate_data(phi,w,k,snr,rng,errors=errors,pert=pert,G=(1.6 if e>0 else 1.5))
    Gd=1.6 if e>0 else 1.5   # with errors the links load the modes more; 1.6 keeps the device passive
    try: true_type=[int(x) for x in device_type(phi,w,1.0,Gd,k,errors=errors,pert=pert)[0]]   # type of this particular device, including its errors
    except TrackingError: true_type=None
    d=np.zeros(4); dl=d+0.02*rng.normal(size=4)
    if variant=='okno': r,lo,hi=fit(y,s,k,dl,0.3,rng)
    elif variant=='prior': r,lo,hi=fit(y,s,k,dl,0.3,rng,sd_pr=0.02)
    else: r,lo,hi=fit(y,s,k,d,2.5,rng,nst=16)
    n=y.size
    ver,t,chi=verdict(r,lo,hi,1.5,k,rng,n)
    return dict(z,prawda=true_type,wer=ver,typ=[int(x) for x in t] if t else None,chi=float(chi),phi_f=float(r.x[4]%(2*np.pi)),w_f=float(r.x[5]))
if __name__=='__main__':
    Z=[]
    if MODE=='pilot':
        for w in (1.,.5):
            for phi in (0.,np.pi):
                for snr in (20,26,32,40):
                    for variant in ('okno','prior'):
                        for t in range(30): Z.append(dict(phi=float(phi),w=w,snr=snr,war=variant,e=0.0,t=t))
    if MODE=='full':
        for w,fs in ((1.,(0.,.6,1.2,1.5,2.,float(np.pi))),(.5,(0.,1.,1.7,2.2,2.6,float(np.pi)))):
            for phi in fs:
                for t in range(100):
                    for snr in (20,26,32,40): Z.append(dict(phi=float(phi),w=w,snr=snr,war='prior',e=0.0,t=t))
                    for snr in (26,32,40): Z.append(dict(phi=float(phi),w=w,snr=snr,war='prior',e=0.01,t=t))
                for t in range(60):
                    for snr in (26,32):
                        Z.append(dict(phi=float(phi),w=w,snr=snr,war='okno',e=0.0,t=t)); Z.append(dict(phi=float(phi),w=w,snr=snr,war='slepy',e=0.0,t=t))
    os.makedirs(os.path.dirname(OUT),exist_ok=True); done=set()
    if os.path.exists(OUT):
        for l in open(OUT):
            q=json.loads(l); done.add((q['phi'],q['w'],q['snr'],q['war'],q['e'],q['t']))
    Z=[z for z in Z if (z['phi'],z['w'],z['snr'],z['war'],z['e'],z['t']) not in done]; print('tasks',len(Z),flush=True)
    with mp.Pool(min(20,os.cpu_count())) as P, open(OUT,'a') as f:
        for i,w in enumerate(P.imap_unordered(job,Z,chunksize=2)): f.write(json.dumps(w)+'\n'); f.flush()
    print('done',flush=True)
