#!/usr/bin/env python3
"""Figure 1 data: eigenvalue branches over one orbit of the swept phase -> data/fig01_branches.npz.

Panels: (a) undetuned ring, (b) detuned ring, (c) four-mode network at phi = 1, (d) the same at phi = pi.
The roots of chi(lambda) are tracked on a dense grid by optimal assignment between neighbouring steps."""
import numpy as np
from scipy.optimize import linear_sum_assignment as lsa
from _paths import data


def track(coeffs):
    """coeffs(th): coefficients of chi; tracking by assignment on a dense grid."""
    th=np.linspace(0,2*np.pi,2401); l=np.roots(coeffs(0.)); W=[l]
    for t in th[1:]:
        x=np.roots(coeffs(t)); a,b=lsa(abs(l[:,None]-x[None,:])); l=x[b]; W.append(l)
    return th,np.array(W)


def ring(de):
    a=-1j*(np.array(de,float)-np.mean(de))
    def coeffs(t):
        c=np.poly(a).astype(complex); c[-1]-=np.exp(1j*t); return c
    return coeffs


def diamond(phi,w=.5):
    b=1+np.exp(1j*phi); d=w*np.exp(1j*phi)
    return lambda t:np.array([1,0,0,-np.exp(1j*t)*b,-np.exp(1j*t)*d])


CASES={'a':ring([0,0,0]),'b':ring([.5,-.5,-2.5]),'c':diamond(1.0),'d':diamond(np.pi)}
if __name__=='__main__':
    out={}
    for k,f in CASES.items():
        th,W=track(f); p=lsa(abs(W[-1][:,None]-W[0][None,:]))[1]
        d=abs(W[:,:,None]-W[:,None,:]); d[:,range(W.shape[1]),range(W.shape[1])]=np.inf
        out['W_'+k]=W; out['p_'+k]=p; out['g_'+k]=d.min(); print(k,'permutation',p,'gap',round(d.min(),3))
    out['th']=th; np.savez(data('fig01_branches.npz'),**out); print('wrote data/fig01_branches.npz')
