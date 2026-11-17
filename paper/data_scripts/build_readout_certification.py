#!/usr/bin/env python3
"""Certified fractions of the read-out maps in figures 5(e) and 6(d,e) -> data/readout_certification.npz and
data/tab_readout_certification.csv. Reads result files in ../results only and computes no fits.
Selected cells are checked against reference values (within one percentage point); any difference stops the script."""
import json, csv, numpy as np, sys
from _paths import data, results
T=json.load(open(results('readout_protocol_tables.json'))); A=json.load(open(results('readout_link_devices.json')))
def cell(ok,mismatches,rejected,n): return (100*ok,100*mismatches,100*rejected,n)
# (a) drift-matrix fit
Ra=[('$N{=}3$, $\\bm\\alpha{=}\\bm0$','nominal'),('$N{=}4$, $\\bm\\alpha{=}\\bm0$','nominal4'),('$(2,1,1)$','2(-2.2,-.2,.3,2.4)i'),('$(2,2)$','22(-1.5,-1.3,1.3,1.5)i')]
Ca=[('A',s) for s in (14,17,20,26,30,40)]+[('Eall',s) for s in (26,40)]
a=np.full((len(Ra),len(Ca),4),np.nan)
for i,(_,k) in enumerate(Ra):
    for j,(e,s) in enumerate(Ca):
        x=T.get(f'{e}__{k}__{s}')
        if x: a[i,j]=cell(x['cert_ok'],x['cert_bad'],x['rej'],x['n'])
# (b) calibrated devices, link model kappa'=20
Rb=[('N3_2p1','$(2,1)$'),('N3_3','$(3)$'),('N4_2p2','$(2,2)$'),('N4_3p1','$(3,1)$'),('N4_4','$(4)$'),('N5_3p2','$(3,2)$'),('N5_2p2p1','$(2,2,1)$'),('N5_5','$(5)$'),('N5_1p1p1p1p1','$(1^5)$')]
Cb=[('ideal',26),('ideal',32),('e1',26),('e1',32)]
b=np.full((len(Rb),len(Cb),4),np.nan)
for i,(k,_) in enumerate(Rb):
    for j,(e,s) in enumerate(Cb):
        x=A[f'{k}__{e}__{s}']; b[i,j]=cell(x['ok'],x['zle'],x['odrz'],x['n'])
# (c) four-mode network, link model kappa'=40, phased probe, w=0.5
Wp=[json.loads(l) for l in open(results('network_readout_phased_probe.jsonl'))]; Wf=[json.loads(l) for l in open(results('network_readout_full.jsonl'))]
Fi=[0,1.0,1.7,2.6,np.pi]
def tally(r):
    n=len(r); true_type=lambda x:tuple(x['typ'] or ())==tuple(x['prawda'])
    return cell(sum(x['wer']=='pewne' and true_type(x) for x in r)/n,sum(x['wer']=='pewne' and not true_type(x) for x in r)/n,sum(x['wer']=='odrzut' for x in r)/n,n)
def select(source,phi,s,**kw):
    return [x for x in source if x['w']==.5 and abs(x['phi']-phi)<1e-3 and x['snr']==s and all(x[k]==v for k,v in kw.items())]
Cc=[('A',s) for s in (26,32)]+[('D',s) for s in (26,32)]+[('E',s) for s in (26,32)]
c=np.full((len(Fi),len(Cc),4),np.nan)
for i,f in enumerate(Fi):
    for j,(e,s) in enumerate(Cc):
        r={'A':lambda:select(Wf,f,s,war='prior',e=0.0),'D':lambda:select(Wp,f,s,e=.01,ep=.01,fp=.02),'E':lambda:select(Wp,f,s,e=.02,ep=.02,fp=.05)}[e]()
        assert r,(f,e,s); c[i,j]=tally(r)
# --- check against reference values: (array, row, col, field, value)
pr=lambda *q:q
K=[pr(a,0,2,0,76),pr(a,0,3,0,100),pr(a,0,6,0,97),pr(a,1,6,0,56),pr(a,3,6,0,33),pr(a,3,6,1,5),pr(a,2,0,1,12),pr(a,2,1,1,12),pr(a,2,3,1,8),pr(a,2,4,1,0),
   pr(b,0,2,0,99),pr(b,0,3,0,96),pr(b,2,3,0,94),pr(b,5,0,0,17),pr(b,5,1,0,88),pr(b,5,2,0,15),pr(b,5,3,0,50),pr(b,5,3,1,7),pr(b,6,2,0,64),pr(b,6,3,0,89),pr(b,7,3,0,95),pr(b,8,3,0,92),
   pr(c,0,0,0,100),pr(c,1,0,0,99),pr(c,0,2,0,75),pr(c,0,3,0,9),pr(c,1,2,0,97),pr(c,1,3,0,60),pr(c,2,3,0,84),pr(c,3,3,0,96),pr(c,4,2,0,99),pr(c,4,3,0,95),pr(c,0,4,0,8),pr(c,0,4,1,2),pr(c,1,4,0,57),pr(c,2,4,1,5),pr(c,4,4,0,90),pr(c,4,5,0,18)]
mismatches=[(q[1],q[2],q[3],q[4],float(q[0][q[1],q[2],q[3]])) for q in K if not abs(q[0][q[1],q[2],q[3]]-q[4])<=1.0]
print("check against reference values:",len(K),"cells, mismatches:",len(mismatches)); [print(' ',z) for z in mismatches]
if mismatches: sys.exit(1)
# Minimum gaps of figure 5(d), in units of eps, for the types (3,1) and (4) (first and second entry); gap = smallest
# eigenvalue spacing over one orbit of the swept phase at |c| = eps. Entered as numbers rounded to 0.01:
#   siec_i  four-mode network, ideal drift matrix, w = 0.5 eps, phi = 1 (type (3,1)) and phi = pi (type (4)),
#           roots of chi on 2001 orbit points: 0.802, 1.189
#   siec_k  the same network in the link model, kappa' = 40 eps, Gamma = 1.5 eps, adaptive tracking of the four slow
#           modes (simulations/network/link_model_types.py): 1.028, 1.196
#   dr_i    labeled ring of N = 4 modes, ideal drift matrix: detunings |delta_k| <= 2 eps chosen by differential
#           evolution (seed 3, population 18, 150 generations) to maximize min(g_loop, g_ray), g_ray being the gap on
#           the labeling ray c = i t |c|, 0 <= t <= 1; reported g_loop on 721 orbit points: 0.748, 0.795
#   dr_k    labeled rings at kappa' = 40 eps, ladders optimized per labeled sigma in the link model; smallest g_loop
#           over the sigma of each type (8 sigma of type (3,1), 6 of type (4)): 0.802, 0.854
#           (simulations/programming/link_per_sigma.py 40, results in results/programming/link_sigma_k40.jsonl)
#   bez     unlabeled ring, ideal: as dr_i without the ray term: 1.189, 1.414
# simulations/network/min_gaps_fig5d.py recomputes siec_i, siec_k, dr_i and bez. The network values are also compared
# with data/fig05_interference.npz (nearest grid phase) below.
z=np.load(data('fig05_interference.npz')); g=lambda k,p:float(z[k][np.argmin(abs(z['fi']-p))])
np.savez(data('readout_certification.npz'),a=a,b=b,c=c,Ra=[r[0] for r in Ra],Rb=[r[1] for r in Rb],Ca=[f'{e}{s}' for e,s in Ca],Cb=[f'{e}{s}' for e,s in Cb],Cc=[f'{e}{s}' for e,s in Cc],
   siec_i=[.80,1.19],siec_k=[1.03,1.20],dr_i=[.75,.80],dr_k=[.80,.85],bez=[1.19,1.41])
print('wrote data/readout_certification.npz; network gaps in fig05 data:',g('gi',1.0),g('gi',np.pi),g('gl',1.0),g('gl',np.pi))
d=np.load(data('readout_certification.npz'))
with open(data('tab_readout_certification.csv'),'w',newline='') as f:
    c=csv.writer(f); c.writerow('panel,row,column,certified_pct,false_pct,rejected_pct,trials'.split(','))
    for nm,D,rw,cl in (('ring_drift',d['a'],d['Ra'],d['Ca']),('ring_k20',d['b'],d['Rb'],d['Cb']),('network_k40',d['c'],['0','1.0','1.7','2.6','pi'],d['Cc'])):
        for i in range(D.shape[0]):
            for j in range(D.shape[1]): c.writerow([nm,rw[i],cl[j]]+[round(float(x),1) for x in D[i,j]])
print('wrote data/tab_readout_certification.csv')
