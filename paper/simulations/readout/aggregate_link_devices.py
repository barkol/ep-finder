"""Aggregation without physics: reads results/readout/link_devices/*.npz and writes ../../results/readout_link_devices.json, one entry per
device, error level and SNR. The field names are keys of the result file: zach (device keeps its type), fit (fitted type equals the
device type), elem (labeled sigma correct), ok (certified with the true type), zle (false certificate), nieps (undecided),
odrz (rejected), glob (fit reaches the cost of the truth), rms (r.m.s. phase error). Prints a summary per error level."""
import numpy as np,glob,os,json
from link_devices_sweep import FIELDS,F,verdict
from _paths import work,results
O={}
for f in sorted(glob.glob(work('link_devices/*.npz'))):
    k=os.path.basename(f)[:-4];d,e,s=k.split('__');A=np.load(f)['A'];r,u,o,b=verdict(A)
    O[k]=dict(dev=d,err=e,snr=int(s),n=len(A),zach=F(A,'typeD').mean(),fit=F(A,'typeLD').mean(),elem=F(A,'elemL').mean(),ok=o.mean(),zle=b.mean(),nieps=u.mean(),odrz=r.mean(),glob=np.mean(F(A,'costL')<=F(A,'costT')+1e-9),rms=float(np.sqrt((F(A,'thL')**2).mean())))
json.dump(O,open(results('readout_link_devices.json'),'w'),indent=0,default=float)
for e in('ideal','e1','e2','e5'):
    print('==',e,'  device: kept | fit@20,26,32,40 | elem | ok | false')
    for d in dict.fromkeys(v['dev'] for v in O.values()):
        R=[O.get(f'{d}__{e}__{s}') for s in(20,26,32,40)]
        if None in R: print(d,'missing');continue
        g=lambda x:' '.join(f"{r[x]:.2f}" for r in R)
        print(f"{d:14s} {np.mean([r['zach'] for r in R]):.2f} | {g('fit')} | {g('elem')} | {g('ok')} | {g('zle')}")
