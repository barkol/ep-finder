"""Tables from the ladder results (read only; no computation beyond eigenvalues on the loop):
data/sigma_table.csv (detunings of every labelled sigma, N<=6) and data/tab_ladders.csv (figure N2: layout,
max|delta|, g_loop, g_max, T_ad, |c| window, x99 for detuning and complex errors).
Inputs: ladder_optimum.jsonl, sigma_detunings.npz, error_mc.jsonl, budget.json in results/programming."""
import numpy as np, json, os, csv, itertools
from ring_lib import *
from _paths import data, results, work
EPS=1e8
best={}
uni={}
for ln in open(results('ladder_optimum.jsonl')):
    o=json.loads(ln)
    k=(o['N'],tuple(o['part']))
    if o['cap']==2.0 and (k not in best or o['G']>best[k]['G']):best[k]=o
    if o['cap']==3.0 and o['uniform'] and (k not in uni or o['uniform']['G']>uni[k]['G']):uni[k]=dict(o['uniform'],comp=o['comp'])
B=json.load(open(results('budget.json')));rows={(r['N'],tuple(r['part'])):r for r in B['rows']}
mc={}
for ln in open(results('error_mc.jsonl')):
    o=json.loads(ln);mc.setdefault((o['N'],tuple(o['part']),o['ch']),[]).append(o)
def x99(lst,key):
    lst=sorted(lst,key=lambda o:o['x']);x=0
    for o in lst:
        if o[key]>=.99:x=o['x']
        else:break
    return x
S=[]
for k in sorted(best,key=lambda k:(k[0],-len(k[1]))):
    N,part=k;r=rows[k];d=np.array(best[k]['d'])
    th=np.linspace(0,np.pi,2001);from ring_lib import _roots_batch,_gap_batch
    gm=_gap_batch(_roots_batch(d,np.exp(1j*th))).max()
    a=mc.get((N,part,'A'),[]);b=mc.get((N,part,'B'),[])
    S.append(dict(N=N,part=list(part),comp=best[k]['comp'],d=d.tolist(),dmax=r['dmax'],g_loop=r['g_loop'],g_max=float(gm),g_ray=r['g_ray'],S=r['S'],T_ad=r['T_ad'],T_ad_us=r['T_ad_us'],Gamma_min=r['Gamma_min'],win=r['win'],
        xA_lab=x99(a,'lab'),xA_typ=x99(a,'typ'),xB_lab=x99(b,'lab'),xB_typ=x99(b,'typ'),uniform=uni.get(k)))
# sigma table
with open(data('sigma_table.csv'),'w',newline='') as f:
    w=csv.writer(f);w.writerow(['N','sigma','cycle_type','delta_over_eps(mode 0..N-1)','g_ray','g_loop','tracked_ok'])
    Z=np.load(results('sigma_detunings.npz'))
    for N in range(2,7):
        for s,de,gl,gr,ok in zip(Z[f'N{N}_sigma'],Z[f'N{N}_delta'],Z[f'N{N}_gloop'],Z[f'N{N}_gray'],Z[f'N{N}_ok']):
            w.writerow([N,' '.join(map(str,s)),'+'.join(map(str,ctype(tuple(int(x) for x in s)))),' '.join('%.4f'%x for x in de),'%.4f'%gr,'%.4f'%gl,int(ok)])
def fmtpart(p):return '+'.join(map(str,p))
# ladders per cycle type (figure N2), N>=3
with open(data('tab_ladders.csv'),'w',newline='') as f:
    w=csv.writer(f);w.writerow(['N','type','layout','delta_max','g_loop','g_max','T_ad_us','c_window','x99_det','x99_cplx'])
    for r in S:
        if r['N']<3:continue
        hi=r['win'][1]
        w.writerow([r['N'],fmtpart(r['part']),fmtpart(r['comp']),f"{r['dmax']:.2f}",f"{r['g_loop']:.2f}",f"{r['g_max']:.2f}",f"{r['T_ad_us']:.1f}",
                    f"{r['win'][0]:.2f}--"+('infty' if hi>1e9 else format(hi,'.2f')),f"{r['xA_lab']:g}",f"{r['xB_lab']:g}"])
print('wrote data/sigma_table.csv and data/tab_ladders.csv')
for r in S:print(r['N'],fmtpart(r['part']),'uniform',r['uniform'] and (round(r['uniform']['G'],2),round(r['uniform']['s'],2),round(r['uniform']['D'],2)),'xA type',r['xA_typ'],'xB type',r['xB_typ'],'Gamma_min %.2f'%r['Gamma_min'])
