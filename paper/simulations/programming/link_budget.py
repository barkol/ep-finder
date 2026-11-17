"""Link-model budget per cycle type at kappa'=20 and 40 with per-sigma calibration (link_per_sigma.py, link_retry.py,
link_fix.py): success, worst loop gap, max|delta|, max Re lambda, slow-sector adiabatic S and T_ad(P=1e-3) of the
worst-gap sigma, physical units at eps=1e8/s. Output: results/programming/link_model_budget.json,
link_sigma_table_k20.csv, link_sigma_table_k40.csv, and data/tab_link_devices.csv (kappa'=20, N=3..5; figure N2).
Order of the link-model steps: link_per_sigma -> link_retry -> link_budget -> label_check -> link_fix -> link_budget
(the shipped link_model_budget.json comes from this second link_budget run)."""
import os
os.environ["MKL_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, json, csv, multiprocessing as mp
from ring_lib import *
from _paths import data, results, work
from link_all_sigma import Sfull
EPS=1e8;P=1e-3
def load(k):
    R={}
    for ln in open(results(f'link_sigma_k{k}.jsonl')):
        o=json.loads(ln);R[tuple(o['sigma'])]=o
    if k==20:
        for ln in open(results('link_sigma_retry_k20.jsonl')):
            o=json.loads(ln);s=tuple(o['sigma'])
            if s not in R or (o['ok'] and not R[s]['ok']) or o.get('kind')=='fix':R[s]=o
    return R
def work(a):
    k,key,o=a
    S=Sfull(np.array(o['delta']),float(k));return k,key,S
if __name__=='__main__':
    out={};jobs=[]
    for k in (20,40):
        R=load(k);cl={}
        for s,o in R.items():
            key=(len(s),tuple(o['type']));e=cl.setdefault(key,dict(n=0,ok=0,g=np.inf,gr=np.inf,dmax=0,maxRe=-9,worst=None))
            e['n']+=1;e['ok']+=int(o['ok'])
            if o['ok']:
                if o['g_loop']<e['g']:e['g']=o['g_loop'];e['worst']=o
                e['gr']=min(e['gr'],o['g_ray']);e['dmax']=max(e['dmax'],o['dmax']);e['maxRe']=max(e['maxRe'],o['maxRe'])
        out[k]=cl
        for key,e in cl.items():jobs.append((k,key,e['worst']))
        with open(results(f'link_sigma_table_k{k}.csv'),'w',newline='') as f:
            w=csv.writer(f);w.writerow(['N','sigma','type','ok','delta_over_eps','base_phase_offset_rad','g_ray','g_loop','maxRe'])
            for s,o in sorted(R.items(),key=lambda x:(len(x[0]),x[0])):
                w.writerow([len(s),' '.join(map(str,s)),'+'.join(map(str,o['type'])),int(o['ok']),' '.join('%.4f'%x for x in o['delta']),'%.4f'%o['th_off'],o['g_ray'],o['g_loop'],o['maxRe']])
    with mp.get_context('fork').Pool(10) as pool:
        for k,key,S in pool.imap_unordered(work,jobs):
            e=out[k][key];e['S']=S;e['T_ad']=2*np.pi*np.sqrt(S/P);e['T_ad_us']=e['T_ad']/EPS*1e6
    js=[]
    for k in out:
        for key,e in sorted(out[k].items()):
            r=dict(kappa=k,N=key[0],part=list(key[1]),n=e['n'],ok=e['ok'],g=e['g'],gr=e['gr'],dmax=e['dmax'],maxRe=e['maxRe'],S=e['S'],T_ad=e['T_ad'],T_ad_us=e['T_ad_us'],wc=e['worst']['sigma']);js.append(r)
            print(k,key,e['ok'],'/',e['n'],'g %.3f gr %.3f dmax %.2f maxRe %.2f S %.2f T %.2f us'%(e['g'],e['gr'],e['dmax'],e['maxRe'],e['S'],e['T_ad_us']))
    json.dump(js,open(results('link_model_budget.json'),'w'),indent=0,default=float)
    # figure N2: link-model devices at kappa'=20, N=3..5
    with open(data('tab_link_devices.csv'),'w',newline='') as f:
        w=csv.writer(f);w.writerow(['N','type','reached_tried','g_loop','delta_max','T_ad_us'])
        for r in sorted((r for r in js if r['kappa']==20 and 3<=r['N']<=5),key=lambda r:(r['N'],r['part'])):
            w.writerow([r['N'],'+'.join(map(str,r['part'])),f"{r['ok']}/{r['n']}",f"{r['g']:.2f}",f"{r['dmax']:.2f}",f"{r['T_ad_us']:.1f}"])
    print('wrote data/tab_link_devices.csv')
