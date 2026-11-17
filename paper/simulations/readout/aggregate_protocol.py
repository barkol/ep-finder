"""Aggregation without physics: reads results/readout/geom.npz, results/readout/fig3c/ and results/readout/proto/ and writes ../../data/fig06_readout.npz
(data of figure 6) and ../../results/readout_protocol_tables.json (one entry per configuration of the protocol sweep)."""
import numpy as np,os,glob,json
from protocol import FIELDS,F,verdict
from _paths import work,data,results
SN=np.arange(6,36,2.)
def load3(k,w,m,S=SN):
    rms=[];P=[];G=[];c=[]
    for s in S:
        fn=work(f"fig3c/{k}_{w}_{int(s)}_{m}.npy")
        if not os.path.exists(fn): rms.append(np.nan);P.append(np.nan);continue
        R=np.load(fn);rms.append(np.sqrt((R[:,0]**2).mean()));P.append(np.mean(np.abs(R[:,0])<np.pi/3))
    return np.array(rms),np.array(P)
g=dict(np.load(work('geom.npz')))
out=dict(g)
out['snr']=SN
for nm,(k,w,m) in dict(blind_P3=('blind','V4','P3'),blind_U=('blind','V4','U'),orakul_kod=('orakul_kod','None','P3'),orakul_skala=('orakul_skala','None','P3')).items():
    out['rms_'+nm],out['P_'+nm]=load3(k,w,m)
PS=[14,17,20,23,26,30,40];out['snr_proto']=np.array(PS,float)
PN=['nominal','a0.8i','1.5i','tr(-1.05,-.75,1.8)i','kol0.98','kol1.02','nominal4','4cyk(.9,.3,-.3,-.9)i','3cyk(-1,-.6,-.2,2.4)i','22(-1.5,-1.3,1.3,1.5)i','2(-2.2,-.2,.3,2.4)i','id(-2.7,-.9,.9,2.7)i']
T={}
def rec(key):
    fn=work(f"proto/{key}.npz")
    if not os.path.exists(fn): return None
    A=np.load(fn)['A'];rej,und,ok,bad=verdict(A)
    return dict(n=len(A),Ptyp=float(F(A,'type_ok').mean()),Pelem=float(F(A,'elem_ok').mean()),cert_ok=float(ok.mean()),cert_bad=float(bad.mean()),und=float(und.mean()),rej=float(rej.mean()),
      rms_th=float(np.sqrt((F(A,'th_err')**2).mean())),G=float(np.median(F(A,'G_hat'))),c=float(np.median(F(A,'c_hat'))),chi2=float(np.median(F(A,'chi2'))),
      atb=float(F(A,'at_bound').mean()),wrong_lower=float(np.mean((F(A,'type_ok')==0)&(F(A,'cost')<F(A,'cost_truth')-1e-9))))
for fn in sorted(glob.glob(work('proto/*.npz'))):
    k=os.path.basename(fn)[:-4];T[k]=rec(k)
def tab(name,field):
    return np.array([T.get(f"A__{name}__{s}",{}).get(field,np.nan) if T.get(f"A__{name}__{s}") else np.nan for s in PS])
for i,n in enumerate(PN):
    out[f'Ptyp_{i}']=tab(n,'Ptyp');out[f'cert_{i}']=tab(n,'cert_ok');out[f'bad_{i}']=tab(n,'cert_bad');out[f'rmsj_{i}']=tab(n,'rms_th')
out['proto_names']=np.array(PN)
np.savez(data('fig06_readout.npz'),**out)
json.dump(T,open(results('readout_protocol_tables.json'),'w'),indent=0)
print("protocol entries",len(T))
