"""Read-out of the calibrated ring devices in the full link model (2N modes, kappa'=20); detunings and base phase of each device
from inputs/link_model_sigma_table_k20.csv (ladder optimization in the link model, worst-gap sigma of each type).
Two analyses of the same data: L = fit with the link model (known kappa'), I = fit with the ideal drift matrix (model mismatch).
Blind start: calibration with the swept link switched off plus random starts; the truth enters only the cost diagnostics.
This module provides the link model, the slow-sector monodromy and the labeled sigma for link_devices_sweep.py.
python link_model.py [n] [nproc] [stage] runs the blind analysis into results/readout/link_blind/ (resumable); --test checks the devices."""
import os,sys,csv,time,zlib
for _v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[_v]="1"
import numpy as np
from _paths import work,inputs
from scipy.optimize import least_squares,linear_sum_assignment as lsa
from readout_lib import fit as fit_ideal,monodromy as mono_ideal,ctype,canon,p2al,unif,WIN
KAP=20.;OM=np.linspace(-5,5,401);CSV=inputs('link_model_sigma_table_k20.csv')
FIELDS=['typeL','elemL','chi2L','fL','atbL','stabL','thL','cL','gL','costL','costT','typeI','chi2I','n','slowL']
def Mfull(d,th,g0=0.,c=1.,kap=KAP,G=None,dph=None):
    """full link-model drift matrix (2N modes, one reservoir per link), with per-link errors G (factor) and dph (rad)"""
    d=np.asarray(d,float);N=len(d);M=np.zeros((2*N,2*N),complex);M[np.arange(N),np.arange(N)]=-1j*d-np.asarray(g0)/2
    for l in range(N):
        e=1. if G is None else G[l];ph=0. if dph is None else dph[l]
        if l<N-1: to,fr,t,Gg=l,l+1,0.,e
        else: to,fr,t,Gg=N-1,0,th,e*c
        J=Gg/2;phi=t+np.pi/2+ph;M[to,fr]+=-1j*J*np.exp(1j*phi);M[fr,to]+=-1j*J*np.exp(-1j*phi)
        z=N+l;g=np.sqrt(Gg*kap)/2;gto=g*np.exp(1j*(t+np.pi))
        M[to,z]+=-1j*gto;M[z,to]+=-1j*np.conj(gto);M[fr,z]+=-1j*g;M[z,fr]+=-1j*np.conj(g);M[z,z]=-kap/2
    return M
def spec(om,ths,d,g0,c,w,v,**k):
    """|v^T(-i om - M)^-1 w|^2, shape (len(ths),len(om))"""
    N=len(d);W=np.r_[w,np.zeros(N)];V=np.r_[v,np.zeros(N)];O=[]
    for i,t in enumerate(ths):
        kk={a:(b[i] if a=='dc' else b) for a,b in k.items()};dc=kk.pop('dc',0.)
        L,U=np.linalg.eig(Mfull(d,t,g0,c*(1+dc),**kk));R=np.linalg.solve(U,W);A=V@U
        O.append(np.abs(((A*R)[None,:]/(-1j*om[:,None]-L[None,:])).sum(1))**2)
    return np.array(O)
def slow_type(d,th0,g0=0.,c=1.,n=480):
    """monodromy of the slow sector (the N eigenvalues with largest Re at the base point), adaptive tracking of all 2N. Returns (type, slow sector closed, min gap)"""
    N=len(d);F=lambda t:np.linalg.eigvals(Mfull(d,th0+t,g0,c));st=F(0.);lam=st.copy();t=0.;T=2*np.pi;h0=T/n;h=h0;mg=9e9
    def gp(e): x=np.abs(e[:,None]-e[None,:]);np.fill_diagonal(x,9e9);return x.min()
    while t<T-1e-12:
        h=min(h,T-t);e=F(t+h);e=e[lsa(np.abs(lam[:,None]-e[None,:]))[1]];g=min(gp(lam),gp(e))
        if np.abs(e-lam).max()>.2*g and h>h0/2**14: h/=2;continue
        lam=e;t+=h;mg=min(mg,g);h=min(1.5*h,h0)
    p=lsa(np.abs(lam[:,None]-st[None,:]))[1];sl=set(np.argsort(st.real)[-N:].tolist())
    okslow=all(int(p[i]) in sl for i in sl)
    idx=sorted(sl);m={a:b for b,a in enumerate(idx)}
    q=tuple(m.get(int(p[i]),0) for i in idx)
    return (ctype(q) if okslow else (-1,)),okslow,mg
def label_sigma(d,th0,g0=0.,c=1.,n=(500,600,2400)):
    """labels as in the ladder calculation: from the ideal limit at c->0 with loss compensation, then the orbit. Returns sigma."""
    N=len(d);s0=1e-5
    def F(th,kap,s):
        g=np.full(N,float(g0));g[0]+=1-s;g[-1]+=1-s;return Mfull(d,th,g,s,kap=kap)
    def st(cur,M): e=np.linalg.eigvals(M);return e[lsa(np.abs(cur[:,None]-e[None,:]))[1]]
    kk=np.geomspace(1e5,KAP,n[0]);w=np.linalg.eigvals(F(th0,kk[0],s0));cur=w[lsa(np.abs((-1j*np.asarray(d)-1-g0/2)[:,None]-w[None,:]))[1]]
    for k in kk[1:]: cur=st(cur,F(th0,k,s0))
    for s in np.linspace(s0,c,n[1])[1:]: cur=st(cur,F(th0,KAP,s))
    b=cur.copy()
    for t in np.linspace(0,2*np.pi,n[2])[1:]:
        g=np.full(N,float(g0));g[0]+=1-c;g[-1]+=1-c;cur=st(cur,Mfull(d,th0+t,g,c))
    return tuple(int(x) for x in np.abs(cur[:,None]-b[None,:]).argmin(1))
def err_model(name,N,rng,M):
    if name!='all': return {},np.ones(N),np.ones(N)
    e=dict(G=1+.05*rng.normal(size=N),dph=.05*rng.normal(size=N),dc=.02*rng.normal(size=M))
    return e,(1+.05*rng.normal(size=N))*np.exp(1j*.1*rng.normal(size=N)),(1+.05*rng.normal(size=N))*np.exp(1j*.1*rng.normal(size=N))
def trial(cfg):
    rng=np.random.default_rng(cfg['seed']);d0=np.array(cfg['d']);N=len(d0);th0=cfg['th0'];M=cfg.get('M',24);snr=cfg['snr']
    TH=np.arange(M)*2*np.pi/M;w=v=unif(N);e,dw,dv=err_model(cfg.get('err','ideal'),N,rng,M)
    dth=.05*rng.normal(size=M) if cfg.get('err')=='all' else np.zeros(M)
    Y=spec(OM,th0+TH+dth,d0,0.,1.,w*dw,v*dv,**e);pk=Y.max();sg=10**(-snr/20);Y=Y/pk+rng.normal(0,sg,Y.shape);y=Y.ravel();n=y.size
    lo=np.r_[-.3,.3,[-3.5]*N];hi=np.r_[1.5,2.,[3.5]*N]
    # calibration: closing link switched off (s->0, loss compensated), one spectrum; gives delta and g0
    gc=np.zeros(N);gc[0]=gc[-1]=1.;Y0=spec(OM,[0.],d0,gc,1e-6,w*dw,v*dv);Y0=(Y0/Y0.max()+rng.normal(0,sg,Y0.shape)).ravel()
    def r0(q):
        m=spec(OM[::2],[0.],q[1:],gc+q[0],1e-6,w,v).ravel();return q[-1]*0+m*(m@Y0[::2])/(m@m)-Y0[::2]
    cal=[]
    for k in range(4):
        q=np.r_[rng.uniform(0,1),rng.normal(size=N)*[.5,1.,1.5,2.][k],0.]
        try: a=least_squares(r0,q,bounds=(np.r_[-.3,[-3.5]*N,-1],np.r_[1.5,[3.5]*N,1]),x_scale=1.,max_nfev=25);cal.append((a.cost,a.x))
        except Exception: pass
    cal.sort(key=lambda t:t[0]);dc=np.sort(cal[0][1][1:-1]) if cal else np.zeros(N)
    def model(p,th,om): 
        return spec(om,p[0]+th,p[3:],p[1],p[2],w,v)
    def res(p,th,om,Yd):
        m=model(p,th,om).ravel();return m*(m@Yd)/(m@m)-Yd   # scale from a linear equation
    Yc=Y[::3,::2].ravel();ths,oms=TH[::3],OM[::2];cand=[]
    S=[]
    for k in range(4):
        # mode order unknown after a blind calibration: random assignment
        S.append(np.r_[2*np.pi*k/4,cal[0][1][0] if cal else .3,1.,rng.permutation(dc)])
    for k in range(cfg.get('nst',8)):
        S.append(np.r_[rng.uniform(0,2*np.pi),rng.uniform(0,1),np.exp(rng.uniform(np.log(.5),np.log(1.6))),rng.normal(size=N)*[.5,1.,1.5,2.][k%4]])
    LO=np.r_[-np.inf,lo];HI=np.r_[np.inf,hi]
    for q in S:
        try: a=least_squares(res,np.clip(q,LO+1e-6,HI-1e-6),args=(ths,oms,Yc),bounds=(LO,HI),x_scale='jac',max_nfev=20);cand.append((a.cost,a.x))
        except Exception: pass
    cand.sort(key=lambda t:t[0]);best=None
    for c_,q in cand[:3]:
        a=least_squares(res,q,args=(TH,OM,y),bounds=(LO,HI),x_scale='jac',max_nfev=40)
        if best is None or a.cost<best.cost: best=a
    p=best.x;tt,oks,_=slow_type(p[3:],p[0],p[1],p[2]);ttrue=cfg['type']
    el=label_sigma(p[3:],p[0],p[1],p[2])==tuple(cfg['sigma']) if oks else False
    chi=2*best.cost/(n-len(p)-1)/sg**2
    C=np.linalg.pinv(best.jac.T@best.jac,rcond=1e-12)*sg**2;C=(C+C.T)/2;ev,U=np.linalg.eigh(C);Lc=U*np.sqrt(np.clip(ev,0,None));same=0;NS=cfg.get('ns',24)
    for _ in range(NS):
        x=np.clip(p+Lc@rng.normal(size=len(p)),LO,HI)
        same+=int(slow_type(x[3:],x[0],x[1],x[2],n=240)[0]==tt)
    atb=any(p[1:3]<lo[:2]+.01*(hi[:2]-lo[:2]))or any(p[1:3]>hi[:2]-.01*(hi[:2]-lo[:2]))or any(np.abs(p[3:])>3.45)
    ct=res(np.r_[th0,0.,1.,d0],TH,OM,y);ct=.5*ct@ct
    # analysis I: the same data, ideal model
    typeI=chiI=np.nan
    if cfg.get('ideal',True):
        rI,JI=fit_ideal(OM,TH,N,y,WIN['do5'],rng,nst=12,w=w,v=v);pI=JI.full(rI.x);typeI=float(ctype(mono_ideal(canon(p2al(pI,N)),pI[2],th0=pI[0])[0])==ttrue);chiI=2*rI.cost/(n-len(rI.x))/sg**2
    stab=max(np.linalg.eigvals(Mfull(p[3:],p[0]+t,p[1],p[2])).real.max() for t in np.linspace(0,2*np.pi,24,endpoint=False))<0
    return [float(tt==ttrue),float(el),chi,same/NS,float(atb),float(stab),float(np.angle(np.exp(1j*(p[0]-th0)))),p[2],p[1],best.cost,ct,typeI,chiI,n,float(oks)]
def devices():
    R=[r for r in csv.DictReader(open(CSV)) if r['ok']=='1' and 3<=int(r['N'])<=5];D=[]
    for N,types in((3,('2+1','3')),(4,('2+2','3+1','4')),(5,('3+2','2+2+1','5','1+1+1+1+1'))):
        for t in types:
            x=min((r for r in R if int(r['N'])==N and r['type']==t),key=lambda r:float(r['g_loop']))
            ty=tuple(sorted(map(int,t.split('+')),reverse=True));D.append(dict(name=f"N{N}_{t}",d=list(map(float,x['delta_over_eps'].split())),th0=np.pi/2*(1-N)+float(x['base_phase_offset_rad']),type=ty,sigma=tuple(map(int,x['sigma'].split())),gap=float(x['g_loop'])))
    return D
def F(A,n): return A[:,FIELDS.index(n)]
def verdict(A):
    n=F(A,'n');rej=F(A,'chi2L')>1+3*np.sqrt(2/n);und=(~rej)&((F(A,'atbL')>0)|(F(A,'fL')<.95));dec=(~rej)&(~und);return rej,und,dec&(F(A,'typeL')>0),dec&(F(A,'typeL')==0)
def summ(A):
    r,u,o,b=verdict(A);nI=F(A,'n');return f"L_type={F(A,'typeL').mean():.3f} L_elem={F(A,'elemL').mean():.3f} cert_ok={o.mean():.3f} cert_false={b.mean():.3f} undecided={u.mean():.3f} rejected={r.mean():.3f} chi2={np.median(F(A,'chi2L')):.3f} rms_th={np.sqrt((F(A,'thL')**2).mean()):.3f} I_type={np.nanmean(F(A,'typeI')):.3f} chi2I={np.nanmedian(F(A,'chi2I')):.3f} rejI={(F(A,'chi2I')>1+3*np.sqrt(2/nI)).mean():.3f} glob={np.mean(F(A,'costL')<=F(A,'costT')+1e-9):.3f}"
def test():
    for D in devices():
        t=slow_type(D['d'],D['th0']);s=label_sigma(D['d'],D['th0'])
        print(D['name'],'gap',D['gap'],'type',t[0],'ok',t[0]==D['type'],'sigma ok',s==D['sigma'],'min gap 2N %.3f'%t[2]);assert t[0]==D['type'] and s==D['sigma'],D['name']
    print('TEST OK')
def main():
    n=int(sys.argv[1]) if len(sys.argv)>1 else 100;np_=int(sys.argv[2]) if len(sys.argv)>2 else 8;stage=sys.argv[3] if len(sys.argv)>3 else 'A';C=[]
    SN=(20,26,32,40)
    for D in devices():
        for er in('ideal','all'):
            for s in SN: C.append(dict(key=f"{stage}__{D['name']}__{er}__{s}".replace('+','p'),base=dict(D,snr=s,err=er)))
    from multiprocessing import Pool
    os.makedirs(work('link_blind'),exist_ok=True);todo=[c for c in C if not os.path.exists(work(f"link_blind/{c['key']}.npz"))];print(f"total {len(C)} to compute {len(todo)}",flush=True)
    with Pool(np_) as P:
        for i,c in enumerate(todo):
            t0=time.time();s0=zlib.crc32(c['key'].encode())%2**31
            A=np.array(P.map(trial,[dict(c['base'],seed=s0+s) for s in range(n)],chunksize=1),float)
            tmp=work(f"link_blind/.{c['key']}.tmp.npz");np.savez_compressed(tmp,A=A,fields=np.array(FIELDS));os.replace(tmp,work(f"link_blind/{c['key']}.npz"))
            print(f"[{i+1}/{len(todo)}] {c['key']} {summ(A)} {time.time()-t0:.0f}s",flush=True)
if __name__=='__main__':
    if '--test' in sys.argv: test()
    else: main()
