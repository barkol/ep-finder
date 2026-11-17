"""Read-out library (no side effects on import). Units eps=1.
Model: M = diag(alpha - Gamma) + cascade eps_k at (k,k+1) + c at (N,1), c=|c| exp(i(theta0+theta_j)).
Signal: y = s |v^T (-i om - M)^{-1} w|^2, equations (8) and (24) of the paper."""
import os
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v,"1")
import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment

def amp(om,phi,G,cm,al,w,v,eps=None,grad=False):
    """amplitude A(phi_j,om_i), shape (len(phi),len(om)); grad=True: also x_k, l_k."""
    al=np.asarray(al,complex);N=len(al);eps=np.ones(N-1) if eps is None else np.asarray(eps,float)
    z=-1j*np.asarray(om)[None,:];c=(cm*np.exp(1j*np.asarray(phi)))[:,None]
    zt=[z+G-al[k] for k in range(N)]
    A=np.ones_like(z);B=np.zeros_like(z);As=[A];Bs=[B]
    for k in range(N-1):
        A,B=zt[k]*A/eps[k],(zt[k]*B-w[k])/eps[k];As.append(A);Bs.append(B)
    x1=(w[N-1]-zt[N-1]*B)/(zt[N-1]*A-c)
    x=[As[k]*x1+Bs[k] for k in range(N)]
    a=sum(v[k]*x[k] for k in range(N))
    if not grad: return a
    a_=np.ones_like(z);b_=np.zeros_like(z);aa=[None]*N;bb=[None]*N;aa[N-1]=a_;bb[N-1]=b_
    for k in range(N-1,0,-1):
        a_,b_=a_*zt[k]/eps[k-1],(b_*zt[k]-v[k])/eps[k-1];aa[k-1]=a_;bb[k-1]=b_
    lN=(v[0]-b_*zt[0])/(a_*zt[0]-c)
    l=[aa[k]*lN+bb[k] for k in range(N)]
    return a,x,l,c

def unif(N): return np.ones(N)/np.sqrt(N)
def port(N,j=0):
    e=np.zeros(N);e[j]=1.;return e

# ---------- parametrization: p=[th0,G,cm,s, Re a_1..a_{N-1}, Im a_1..a_N] (Re a_N=-sum; Im of the sum = free frame shift)
def npar(N): return 4+(N-1)+N
def p2al(p,N):
    re=np.r_[p[4:4+N-1],0.];re[-1]=-re[:-1].sum()
    return re+1j*p[4+N-1:4+2*N-1]

class Joint:
    """full model: M spectra, free alpha. thetas = relative settings of the swept phase."""
    def __init__(s,om,thetas,N,y,w=None,v=None,fixal=False):
        s.om,s.th,s.N,s.y=om,np.asarray(thetas,float),N,y.ravel();s.w=unif(N) if w is None else w;s.v=unif(N) if v is None else v;s.fixal=fixal
    def full(s,q):
        if s.fixal: return np.r_[q[:4],np.zeros(npar(s.N)-4)]
        return q
    def res(s,q):
        p=s.full(q);a=amp(s.om,s.th+p[0],p[1],p[2],p2al(p,s.N),s.w,s.v)
        return p[3]*(np.abs(a)**2).ravel()-s.y
    def jac(s,q):
        p=s.full(q);N=s.N;a,x,l,c=amp(s.om,s.th+p[0],p[1],p[2],p2al(p,N),s.w,s.v,grad=True)
        ac=np.conj(a);D=lambda dA:(2*p[3]*(ac*dA).real).ravel()
        lx=[l[k]*x[k] for k in range(N)];cc=l[N-1]*x[0]
        cols=[D(1j*c*cc),D(-sum(lx)),D(c/p[2]*cc),(np.abs(a)**2).ravel()]
        if not s.fixal:
            cols+=[D(lx[k]-lx[N-1]) for k in range(N-1)]
            cols+=[D(1j*lx[k]) for k in range(N)]
        return np.stack(cols,1)

WIN={'V4':((.9,3.),(.3,2.)),'ciasne':((1.1,1.8),(.7,1.3)),'do5':((.3,5.),(.05,5.)),'do10':((.3,10.),(.05,10.)),'dwukr':((.7,2.8),(.5,2.))}

def bounds_for(N,win,fixal=False,Are=2.5,Aim=5.):
    (g0,g1),(c0,c1)=win
    lo=[-np.inf,g0,c0,1e-9];hi=[np.inf,g1,c1,1e9]
    if not fixal:
        lo+=[-Are]*(N-1)+[-Aim]*N;hi+=[Are]*(N-1)+[Aim]*N
    return np.array(lo,float),np.array(hi,float)

def scale_ls(J,q):
    """linear scale s=<y,m>/<m,m>, never a fixed value (a fixed scale of 1 fails when the true scale is 0.16)"""
    qq=np.array(q,float);qq[3]=1.;m=J.res(qq)+J.y
    return max(float(m@J.y/(m@m)),1e-8)

def stable(p,N,nth=48):
    al=p2al(p,N);al=al-1j*al.imag.mean()
    return max(np.linalg.eigvals(ringM(al-p[1],p[2]*np.exp(1j*t))).real.max() for t in np.linspace(0,2*np.pi,nth,endpoint=False))<0

def starts(N,win,rng,n,fixal=False):
    """Blind starts: nothing about the device. Phase from a grid, Gamma,|c| from the window, alpha: 0 and random at scales {0.3,1,2}."""
    (g0,g1),(c0,c1)=win;out=[]
    for k in range(n):
        th=2*np.pi*k/max(n,1) if fixal else 2*np.pi*rng.integers(4)/4
        G=rng.uniform(g0,min(g1,g0+2.));cm=np.exp(rng.uniform(np.log(c0),np.log(min(c1,3.))))
        q=[th,G,cm,1.]
        if not fixal:
            sc=[0.05,.5,1.,2.][k%4]
            are=rng.normal(size=N-1)*sc*(.3 if k%2 else 0.);aim=rng.normal(size=N)*sc
            q+=list(are)+list(aim)
        out.append(q)
    return out

def fit(om,thetas,N,y,win,rng,nst=16,fixal=False,w=None,v=None,pre=(3,2),nfev0=25,nfev=120,ntop=3,x0=None):
    """Bounded multistart: stage 1 on thinned data, stage 2 on the full data."""
    M=len(thetas);lo,hi=bounds_for(N,win,fixal)
    J=Joint(om,thetas,N,y,w,v,fixal)
    Y=y.reshape(M,len(om));dm=pre[0] if M>=6 else 1;Jc=Joint(om[::pre[1]],thetas[::dm],N,Y[::dm,::pre[1]],w,v,fixal)
    cand=[]
    S=starts(N,win,rng,nst,fixal)
    if x0 is not None: S=list(x0)+S
    for q in S:
        q=np.array(q,float);q[3]=scale_ls(Jc,q);q=np.clip(q,lo+1e-7*(hi-lo).clip(0,1e3),hi-1e-7*(hi-lo).clip(0,1e3))
        try: r=least_squares(Jc.res,q,jac=Jc.jac,bounds=(lo,hi),method='trf',x_scale='jac',max_nfev=nfev0)
        except Exception: continue
        cand.append((r.cost,r.x))
    cand.sort(key=lambda t:t[0]);best=None;res=[]
    for cst,q in cand[:ntop]:
        r=least_squares(J.res,np.clip(q,lo+1e-9,hi-1e-9),jac=J.jac,bounds=(lo,hi),method='trf',x_scale='jac',max_nfev=nfev)
        res.append(r)
    res.sort(key=lambda r:r.cost)
    for r in res:
        p=J.full(r.x)
        if stable(p,N): best=r;break
    if best is None: best=res[0]
    return best,J

# ---------- monodromy ----------
def ringM(diag,c,eps=None):
    N=len(diag);M=np.diag(np.asarray(diag,complex));e=np.ones(N-1) if eps is None else eps
    M[np.arange(N-1),np.arange(1,N)]=e;M[N-1,0]=c;return M
def _match(a,b): return linear_sum_assignment(np.abs(a[:,None]-b[None,:]))[1]
def _gap(e):
    d=np.abs(e[:,None]-e[None,:]);np.fill_diagonal(d,np.inf);return d.min()
def monodromy(al,cm,th0=0.,n=300,maxref=22):
    """adaptive tracking over the orbit. Returns perm (i->p[i], labels at the base point sorted by Im, Re), min gap, number of exhausted bisections"""
    F=lambda t:np.linalg.eigvals(ringM(al,cm*np.exp(1j*(th0+t))))
    lam=F(0.);lam=lam[np.lexsort((lam.real,lam.imag))];st=lam.copy();t=0.;T=2*np.pi;h0=T/n;h=h0;hmin=h0/2**maxref;mg=np.inf;bad=0
    while t<T-1e-13:
        h=min(h,T-t);e=F(t+h);e=e[_match(lam,e)];g=min(_gap(lam),_gap(e));mv=np.abs(e-lam).max()
        if mv>.2*g:
            if h>hmin: h/=2;continue
            bad+=1
        lam=e;t+=h;mg=min(mg,g);h=min(1.5*h,h0)
    return tuple(_match(lam,st)),mg,bad,st
def ctype(p):
    seen=set();t=[]
    for i in range(len(p)):
        if i in seen: continue
        n=0;j=i
        while j not in seen: seen.add(j);j=p[j];n+=1
        t.append(n)
    return tuple(sorted(t,reverse=True))
def critvals(al):
    co=np.poly(al);mu=np.roots(np.polyder(co));return np.polyval(co,mu)
def W_of(al,cm): return int((np.abs(critvals(al))<cm).sum())
def margin(al,cm): return float(np.min(np.abs(np.abs(critvals(al))-cm)))
def canon(al):
    al=np.asarray(al,complex);return al-al.mean()
def lrt_boundary(r,J,sig,win,rng,K=None,nst=2,nfev=30):
    """Likelihood-ratio test against the discriminant boundary: Delta chi2 = min chi2(p: margin(p)=0) - chi2(p^).
    Every device with a different monodromy lies beyond the boundary, so Delta chi2 bounds from below the support for any other sigma."""
    N=J.N;lo,hi=bounds_for(N,win,False);K=K or 30./sig
    m=lambda q:margin(canon(p2al(q,N)),q[2])
    def res(q): return np.r_[J.res(q)/sig,K*m(q)]
    def jac(q):
        Ja=J.jac(q)/sig;g=np.zeros(len(q));m0=m(q)
        for i in range(len(q)):
            h=np.zeros(len(q));h[i]=1e-6;g[i]=(m(q+h)-m0)/1e-6
        return np.vstack([Ja,K*g])
    chi0=2*r.cost/sig**2;best=np.inf;x0=r.x
    for k in range(nst):
        q=x0.copy()
        if k>0: q[4:]+=rng.normal(size=len(q)-4)*[0,.3,.8,.5,1.2][k%5]
        q=np.clip(q,lo+1e-7,hi-1e-7)
        try: a=least_squares(res,q,jac=jac,bounds=(lo,hi),x_scale='jac',max_nfev=nfev)
        except Exception: continue
        if m(a.x)<3e-3:
            best=min(best,float(np.sum(J.res(a.x)**2))/sig**2-chi0)
    return best
def maxim(s): return int(np.sum((s[1:-1]>s[:-2])&(s[1:-1]>s[2:])))

def simulate(om,thetas,al,G,cm,rng,snr,th0=0.,w=None,v=None,eps=None,dth=None,dc=None,norm=None):
    """data: Gaussian noise with sigma = 10^(-SNR/20) * maximum over the orbit (convention of the paper). dth,dc: setting errors per spectrum."""
    N=len(al);w=unif(N) if w is None else w;v=unif(N) if v is None else v
    th=np.asarray(thetas)+th0+(0 if dth is None else dth)
    if dc is None: y=np.abs(amp(om,th,G,cm,al,w,v,eps))**2
    else: y=np.vstack([np.abs(amp(om,th[[i]],G,cm*(1+dc[i]),al,w,v,eps))**2 for i in range(len(th))])
    pk=y.max() if norm is None else norm
    sig=10**(-snr/20)
    return (y/pk+rng.normal(0,sig,y.shape)).ravel(),sig,pk

def laplace_stab(r,J,sig,nS=200,rng=None):
    """Stability of the verdict: sample from N(p^,sig^2 (J^T J)^-1); fraction of samples with the same W, distance to the boundary in sigmas."""
    N=J.N;p=J.full(r.x);Ja=r.jac if hasattr(r,'jac') else J.jac(r.x)
    C=np.linalg.pinv(Ja.T@Ja,rcond=1e-12)*sig**2
    C=(C+C.T)/2;w_,U=np.linalg.eigh(C);w_=np.clip(w_,0,None);Lc=U*np.sqrt(w_)
    W0=W_of(p2al(p,N),p[2]);rng=rng or np.random.default_rng(0);same=0;ms=[]
    for _ in range(nS):
        q=r.x+Lc@rng.normal(size=len(w_));pp=J.full(q)
        a=p2al(pp,N);v=np.abs(critvals(a))-abs(pp[2]);same+=int((v<0).sum()==W0);ms.append(v)
    ms=np.array(ms);v0=np.abs(critvals(p2al(p,N)))-p[2]
    return same/nS,float(np.min(np.abs(v0)/np.maximum(ms.std(0),1e-12)) if v0.size==ms.shape[1] else 0.)
