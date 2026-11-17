#!/usr/bin/env python3
"""Figure N1 of the technical notes: read-out maps -> data/figN1_readout_maps.npz and data/tab_readout_maps.csv.
Reads result files in ../results only.
(a) calibrated rings, link model at kappa'=20: certified with the true type (none, e1), labeled sigma correct (sigma),
    device keeps its type (kept, mean over SNR 20-40 dB) for error levels e1, e2, e5.
(b) four-mode network at kappa'=40 with phased probe, w=0.5, error classes A-E at 26 and 32 dB."""
import json, csv, numpy as np
from _paths import data, results
A=json.load(open(results('readout_link_devices.json')))
# rows of panel (a): N, cycle type, worst-case loop gap g_loop/eps of the calibrated device
ROWS=[('3','2+1','1.10'),('3','3','1.27'),('4','2+2','0.70'),('4','3+1','0.87'),('4','4','0.93'),
      ('5','3+2','0.36'),('5','2+2+1','0.37'),('5','5','0.61'),('5','1+1+1+1+1','0.72')]
K={'3|2+1':'N3_2p1','3|3':'N3_3','4|2+2':'N4_2p2','4|3+1':'N4_3p1','4|4':'N4_4','5|3+2':'N5_3p2','5|2+2+1':'N5_2p2p1','5|5':'N5_5','5|1+1+1+1+1':'N5_1p1p1p1p1'}
D=np.full((9,8,4),np.nan); typy=[]; g=[]
for i,w in enumerate(ROWS):
    d=K[w[0]+'|'+w[1]]; typy.append(w[1]); g.append(w[2])
    q=lambda e,s:A[f'{d}__{e}__{s}']
    vals=[q('ideal',26)['ok'],q('ideal',32)['ok'],q('ideal',40)['elem'],q('e1',26)['ok'],q('e1',32)['ok']]+[np.mean([q(e,s)['zach'] for s in (20,26,32,40)]) for e in ('e1','e2','e5')]
    fpos=[q('ideal',26)['zle'],q('ideal',32)['zle'],0,q('e1',26)['zle'],q('e1',32)['zle'],0,0,0]
    rejected=[q('ideal',26)['odrz'],q('ideal',32)['odrz'],0,q('e1',26)['odrz'],q('e1',32)['odrz'],0,0,0]
    for j in range(8): D[i,j]=(100*vals[j],100*fpos[j],100*rejected[j],100)
# network
Wp=[json.loads(l) for l in open(results('network_readout_phased_probe.jsonl'))]; Wf=[json.loads(l) for l in open(results('network_readout_full.jsonl'))]
Fi=[0,1.0,1.7,2.6,np.pi]
def tally(r):
    n=len(r); true_type=lambda x:tuple(x['typ'] or ())==tuple(x['prawda'])
    return (100*sum(x['wer']=='pewne' and true_type(x) for x in r)/n,100*sum(x['wer']=='pewne' and not true_type(x) for x in r)/n,100*sum(x['wer']=='odrzut' for x in r)/n,n)
def select(z,f,s,**kw): return [x for x in z if x['w']==.5 and abs(x['phi']-f)<1e-3 and x['snr']==s and all(x[k]==v for k,v in kw.items())]
Z={'A':lambda f,s:select(Wf,f,s,war='prior',e=0.0),'B':lambda f,s:select(Wf,f,s,war='prior',e=.01),'C':lambda f,s:select(Wp,f,s,e=0.0,ep=.01,fp=.02),'D':lambda f,s:select(Wp,f,s,e=.01,ep=.01,fp=.02),'E':lambda f,s:select(Wp,f,s,e=.02,ep=.02,fp=.05)}
C=np.full((5,10,4),np.nan)
for i,f in enumerate(Fi):
    for j,(e,s) in enumerate((e,s) for e in 'ABCDE' for s in (26,32)):
        r=Z[e](f,s); assert r,(f,e,s); C[i,j]=tally(r)
np.savez(data('figN1_readout_maps.npz'),D=D,C=C,typy=typy,g=g); print('wrote data/figN1_readout_maps.npz')
COLS_A=['none26','none32','sigma40','e1_26','e1_32','kept_e1','kept_e2','kept_e5']
COLS_B=[f'{e}{s}' for e in 'ABCDE' for s in (26,32)]
with open(data('tab_readout_maps.csv'),'w',newline='') as f:
    c=csv.writer(f); c.writerow('panel,row,column,value_pct,false_pct,rejected_pct,trials'.split(','))
    for i,t in enumerate(typy):
        for j,col in enumerate(COLS_A): c.writerow(['rings',t,col]+[round(float(x),1) for x in D[i,j]])
    for i,phi in enumerate(['0','1.0','1.7','2.6','pi']):
        for j,col in enumerate(COLS_B): c.writerow(['network',phi,col]+[round(float(x),1) for x in C[i,j]])
print('wrote data/tab_readout_maps.csv')
