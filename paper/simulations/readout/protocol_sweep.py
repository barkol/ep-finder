"""Main sweep of the read-out protocol: joint fit of 24 spectra for the ring devices with N=3,4 against SNR, blind bounded multistart.
Output: results/readout/proto/<stage>__<device>__<snr>.npz, one file per configuration (resumable).
python protocol_sweep.py [trials=200] [processes=10] [stage=A|B|E|P]; the cached results used 104 trials per configuration."""
from _paths import work
import sys,numpy as np
from protocol import *
from devices import *
def main():
    n=int(sys.argv[1]) if len(sys.argv)>1 else 200;np_=int(sys.argv[2]) if len(sys.argv)>2 else 10;stage=sys.argv[3] if len(sys.argv)>3 else 'A'
    C=[];SN=[14,17,20,23,26,30,40]
    def add(tag,name,al,snr,**kw):
        cm=kw.pop('cm',1.);G=kw.pop('G',None)
        if G is None: G=Gamma_of(al*1.0)
        C.append(dict(key=f"{tag}__{name}__{snr}".replace('/','_'),n=kw.pop('nn',n),base=dict(al=al,G=G,cm=cm,snr=snr,M=kw.pop('M',24),**kw)))
    if stage=='A':
        for D in(D3,D4):
            for k,a in D.items():
                for s in SN: add('A',k,a,s)
    K3=('nominal','1.5i','a0.8i','tr(-1.05,-.75,1.8)i','kol0.98','kol1.02')
    if stage=='B':  # windows: devices off the window centre and outside the window, other windows, calibration with the pump off
        for g,cm in[(.95,.35),(1.4,.35),(2.9,.35),(2.9,1.),(1.6,1.9),(2.9,1.9),(3.5,1.),(1.4,.2),(2.2,2.5)]:
            for k in('nominal','1.5i'):
                for s in(14,20,26): add(f'Bg{g}c{cm}',k,D3[k],s,G=g,cm=cm)
        for wn in('ciasne','do5','do10'):
            for k in('nominal','1.5i','tr(-1.05,-.75,1.8)i','kol1.02'):
                for s in(14,20,26,40): add(f'Bw{wn}',k,D3[k],s,win=WIN[wn])
        for k in K3:
            for s in(14,20,26,40): add('Bcal',k,D3[k],s,cal=True)
        for k in('nominal4','22(-1.5,-1.3,1.3,1.5)i','3cyk(-1,-.6,-.2,2.4)i'):
            for s in(20,26,40): add('Bcal',k,D4[k],s,cal=True)
    if stage=='E':  # calibration errors not known to the fit model
        for er in('casc5','probe5','pump','all'):
            for k in K3:
                for s in(20,26,40): add(f'E{er}',k,D3[k],s,err=er)
            for k in('nominal4','22(-1.5,-1.3,1.3,1.5)i','3cyk(-1,-.6,-.2,2.4)i'):
                for s in(26,40): add(f'E{er}',k,D4[k],s,err=er)
    if stage=='P':  # one-port probe and number of spectra
        for D in(D3,D4):
            for k,a in D.items():
                for s in(20,26,40): add('Ps11',k,a,s,probe='s11')
        for M in(1,3,6,12):
            for k in('nominal','1.5i','tr(-1.05,-.75,1.8)i'):
                for s in(20,26): add(f'PM{M}',k,D3[k],s,M=M)
    run_configs(C,work('proto'),nproc=np_)
if __name__=='__main__': main()
