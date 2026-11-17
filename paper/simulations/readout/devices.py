"""Catalogue of ring devices (explicit alpha, units eps=1). Cycle types are not entered here; the scripts obtain them by tracking.
Device names are keys of the result files and are kept as they are."""
import numpy as np
w3=np.exp(2j*np.pi*np.arange(3)/3)
COLL=np.array([0.2042-0.9785j,0.7461+0.6662j,-0.9502+0.3123j]);COLL=COLL-COLL.mean();COLL=COLL/np.abs(COLL).mean()  # near-collision device: three points of a fixed collision configuration
D3={'nominal':0*w3,'0.5pierw':.5*w3,'a0.8i':np.array([.8j,0,-.8j]),'a0.8':np.array([.8,0,-.8],complex),
    '1.5i':np.array([1.5j,0,-1.5j]),'a2.0i':np.array([2j,0,-2j]),'2.5pierw':2.5*w3,'d(-1.6,-.2,1.8)i':1j*np.array([-1.6,-.2,1.8]),
    'tr(-1.05,-.75,1.8)i':1j*np.array([-1.05,-.75,1.8]),'tr(.9,.7,-1.6)i':1j*np.array([.9,.7,-1.6]),
    'kol0.98':.98*COLL,'kol1.02':1.02*COLL}
D4={'nominal4':np.zeros(4,complex),'4cyk(.9,.3,-.3,-.9)i':1j*np.array([.9,.3,-.3,-.9]),
    '3cyk(-1,-.6,-.2,2.4)i':1j*np.array([-1.,-.6,-.2,2.4]),
    '22(-1.5,-1.3,1.3,1.5)i':1j*np.array([-1.5,-1.3,1.3,1.5]),
    '2(-2.2,-.2,.3,2.4)i':1j*np.array([-2.2,-.2,.3,2.4]),
    'id(-2.7,-.9,.9,2.7)i':1j*np.array([-2.7,-.9,.9,2.7])}
def Gamma_of(al,G0=1.4):
    """the device must be stable: Gamma > max Re lambda over the orbit; 1.4 by default, otherwise max Re + 0.4"""
    from readout_lib import ringM,canon
    a=canon(al);mr=max(np.linalg.eigvals(ringM(a,np.exp(1j*t))).real.max() for t in np.linspace(0,2*np.pi,181))
    return G0 if mr<=G0-.3 else float(np.round(mr+.4,3))
