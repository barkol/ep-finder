"""High-precision one-orbit propagator of the ring (mpmath, Taylor series per step). No side effects on import."""
import mpmath as mp
def U_of(N,T,alpha=None,th0=0.,ramp=False,c=1,order=None,h=0.5,dps=None):
    """One-orbit propagator of a_k'=a_{k+1}+alpha_k a_k, a_N'=c e^{i th(t)} a_1+alpha_N a_N.
    th=th0+Om t, or th0+Om t - sin(Om t) with ramp=True. Taylor series on steps of length h."""
    mp.mp.dps=dps or int(60+0.75*T*(abs(c)**(1/N))+30)
    T=mp.mpf(T); Om=2*mp.pi/T; order=order or 45
    alpha=[mp.mpf(0)]*N if alpha is None else [mp.mpmathify(x) for x in alpha]
    nst=int(mp.ceil(T/mp.mpf(h))); dt=T/nst
    U=mp.eye(N); th0=mp.mpf(th0)
    for s in range(nst):
        t0=s*dt
        if not ramp:
            g=[mp.mpmathify(c)*mp.exp(1j*(th0+Om*t0))]
            for n in range(order): g.append(g[-1]*1j*Om/(n+1))
        else:
            cs=[0]*(order+1); co,si=mp.cos(Om*t0),mp.sin(Om*t0); f=mp.mpf(1)
            for n in range(order+1):
                cs[n]=[co,-si,-co,si][n%4]*Om**n/f; f*=(n+1)
            tp=[-Om*x for x in cs]; tp[0]+=Om
            g=[mp.mpmathify(c)*mp.exp(1j*(th0+Om*t0-mp.sin(Om*t0)))]
            for n in range(order):
                g.append(1j*sum(tp[m]*g[n-m] for m in range(n+1))/(n+1))
        newU=mp.zeros(N)
        for col in range(N):
            y=[[U[r,col]] for r in range(N)]
            for n in range(order):
                for r in range(N):
                    if r<N-1: v=y[r+1][n]+alpha[r]*y[r][n]
                    else: v=alpha[r]*y[r][n]+sum(g[m]*y[0][n-m] for m in range(n+1))
                    y[r].append(v/(n+1))
            for r in range(N):
                v=mp.mpf(0)
                for n in range(order,-1,-1): v=v*dt+y[r][n]
                newU[r,col]=v
        U=newU
    return U
def R(N,th,k):
    l=mp.exp(1j*(mp.mpf(th)+2*mp.pi*k)/N)
    return mp.matrix([l**j/mp.sqrt(N) for j in range(N)])
def vn(v): return mp.sqrt(sum(abs(x)**2 for x in v))
def dom_index(N,th):
    return max(range(N),key=lambda k:mp.cos((mp.mpf(th)+2*mp.pi*k)/N))
def Poff(N,T,ramp=False,th0=None):
    """population outside the target branch; start on the branch that dominates during the whole orbit (th0=-pi)"""
    th0=-mp.pi if th0 is None else th0
    # branch dominating at mid-orbit: th=0 -> k=0, lambda=1; start th0=-pi, end +pi
    U=U_of(N,T,th0=th0,ramp=ramp)
    v=U*R(N,th0,0); v=v/vn(v)
    # projection on R(N,th0+2pi,k)
    amps=[abs(sum(mp.conj(a)*b for a,b in zip(R(N,th0+2*mp.pi,k),v)))**2 for k in range(N)]
    return float(1-amps[0]),[float(x) for x in amps]

def Fall(N,T,th0,ramp=False,alpha=None,h=0.5):
    U=U_of(N,T,th0=th0,ramp=ramp,alpha=alpha,h=h); Fs=[]
    for k in range(N):
        v=U*R(N,th0,k); v=v/vn(v)
        tgt=R(N,th0+2*mp.pi,k); Fs.append(float(abs(sum(mp.conj(a)*b for a,b in zip(tgt,v)))**2))
    return Fs
