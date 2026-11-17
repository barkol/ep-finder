#!/usr/bin/env python3
"""Figure 6 (read-out of rings in simulation), drawn as one GLE file: spectra and rates as GLE graphs,
certification maps as 3D bars. Data: data/fig06_readout.npz, data/readout_certification.npz."""
from paths import enter_build, data, deliver
import numpy as np, subprocess, os
import gle_bars as g
enter_build()
D=np.load(data('fig06_readout.npz'),allow_pickle=False); O=np.load(data('readout_certification.npz'),allow_pickle=False)
TH,OM,Sg,Wk,G=D['TH'],D['OM'],D['Sg'],D['W'],D['G']; th=TH/np.pi
om=OM if OM.ndim==1 else OM[0] if np.ptp(OM[0])>0 else OM[:,0]; t1=th if th.ndim==1 else th[:,0] if np.ptp(th[:,0])>0 else th[0]
Z=(Sg**.5); Z=Z if Z.shape==(len(t1),len(om)) else Z.T
with open('gle/f2a.z','w') as f:
    f.write(f'! nx {len(om)} ny {len(t1)} xmin {om[0]} xmax {om[-1]} ymin {t1[0]} ymax {t1[-1]}\n')
    for r in Z: f.write(' '.join('%.4g'%x for x in r)+'\n')
np.savetxt('gle/f2a.dat',np.column_stack([t1,Wk[:,0],Wk[:,1],Wk[:,2]]) if Wk.shape[0]==len(t1) else np.column_stack([t1,*Wk]),fmt='%.5g')
np.savetxt('gle/f2b.dat',np.column_stack([t1,float(D['dmin'])*np.ones_like(t1),G.mean(1),D['gsharp']]),fmt='%.5g')
np.savetxt('gle/f2c1.dat',np.column_stack([D['snr'],D['rms_blind_U']]),fmt='%.5g'); np.savetxt('gle/f2c2.dat',np.column_stack([D['snr_proto'],D['rmsj_0']]),fmt='%.5g')
BL,OR,GR,GY='rgb255(0,114,178)','rgb255(213,94,0)','rgb255(0,158,115)','gray50'
W,H=15.75,7.85; fs=.25
o=[f'size {W} {H}','set font texcmr','set hei %.3f'%fs,'set lwidth 0.012','set cap round']
gh=2.25; gy=H-gh-.42
o2=[]
def line_text(x,y,t,col='black',j='lc'): o.extend([f'set color {col}',f'set just {j}',f'amove {x} {y}',f'write "{t}"','set color black'])
# (a) spectrogram: axes in GLE, tracks drawn in graph coordinates
x0,w0=1.0,3.35
o.extend([f'amove {x0} {gy}','begin graph',f'  size {w0} {gh}','  fullsize','  xaxis min -2.4 max 2.4 dticks 2 hei %.3f'%fs,'  yaxis min 0 max 2 dticks 1 hei %.3f'%fs,'  xtitle "\\omega/\\varepsilon" hei .26 dist .1','  ytitle "\\theta/\\pi" hei .26 dist .1','  colormap "gle/f2a.z" 240 180 invert','end graph'])
sx=lambda w:x0+(w+2.4)/4.8*w0; sy=lambda t:gy+t/2*gh
for k,(c,lw,dash) in enumerate(((BL,.035,'1'),(OR,.035,'4'),(GR,.04,'2'))):
    pts=[(sx(a),sy(b)) for a,b in zip(Wk[:,k],t1) if -2.4<=a<=2.4][::6]
    o.extend([f'set color {c}',f'set lwidth {lw}',f'set lstyle {dash}','amove %.3f %.3f'%pts[0]]+['aline %.3f %.3f'%q for q in pts[1:]]+['set lstyle 1','set color black','set lwidth 0.012'])
o.extend(['set just bl',f'amove {x0} {gy+gh+.07}','set hei .26','write "(a)"','set hei %.3f'%fs])
# (b)
x1,w1=5.3,3.0
o.extend([f'amove {x1} {gy}','begin graph',f'  size {w1} {gh}','  fullsize','  xaxis min 0 max 2 dticks 1 hei %.3f'%fs,'  yaxis min 0 max 4 dticks 1 hei %.3f'%fs,'  xtitle "\\theta/\\pi" hei .26 dist .1','  ytitle "rate /\\varepsilon" hei .26 dist .1','  data "gle/f2b.dat" d1=c1,c2 d2=c1,c3 d3=c1,c4',f'  d1 line color {GR} lwidth .04',f'  d2 line color {GY} lstyle 4 lwidth .03',f'  d3 line color {OR} lstyle 2 lwidth .035','end graph'])
bx=lambda t:x1+t/2*w1; by=lambda v:gy+v/4*gh
line_text(bx(.1),by(float(D['dmin']))+.13,'\\delta_{min}',GR); line_text(bx(.1),by(G.mean(1)[0])+.13,'mean width',GY); line_text(bx(1.05),by(.45),'narrowest',OR,'cc')
o.extend(['set just bl',f'amove {x1} {gy+gh+.07}','set hei .26','write "(b)"','set hei %.3f'%fs])
# (c)
x2,w2=9.25,3.0
o.extend([f'amove {x2} {gy}','begin graph',f'  size {w2} {gh}','  fullsize','  xaxis min 6 max 35 dticks 10 hei %.3f'%fs,'  yaxis log min 0.008 max 3 hei %.3f'%fs,'  xtitle "SNR (dB)" hei .26 dist .1','  ytitle "phase error (rad)" hei .26 dist .1','  data "gle/f2c1.dat" d1=c1,c2','  data "gle/f2c2.dat" d2=c1,c2',f'  d1 line marker fcircle msize .11 color {BL} lwidth .025',f'  d2 line marker fsquare msize .11 color {GR} lstyle 4 lwidth .025','end graph'])
cx=lambda s:x2+(s-6)/29*w2; cy=lambda v:gy+(np.log10(v)-np.log10(.008))/(np.log10(3)-np.log10(.008))*gh
o.extend(['set lwidth .01','set color gray50','set lstyle 4',f'amove {x2} {cy(np.pi/3):.3f}',f'aline {x2+w2} {cy(np.pi/3):.3f}','set lstyle 1','set color black','set lwidth .012'])
line_text(x2+w2-.05,cy(np.pi/3)+.13,'\\pi/N',GY,'rc'); line_text(cx(19),cy(.42),'one spectrum',BL,'lc'); line_text(cx(7.5),cy(.03),'joint fit',GR,'lc')
o.extend(['set just bl',f'amove {x2} {gy+gh+.07}','set hei .26','write "(c)"','set hei %.3f'%fs])
# key at the right of the top row
class SharedCanvas(g.P):
    def __init__(s,x0_,y0_,cu,cv,a,b,zh,fs_,gc=(),gr=(),title_=(0,0)):
        g.P.__init__(s,1,1,x0_,y0_,cu,cv,a,b,zh,fs_,gc,gr); s.o=o; s.title_=title_
    def put_text(s,x,y,t,j='cc',hei=None): g.P.put_text(s,x,y,t,j,hei)
kx=12.75; ky=gy+1.75; k=SharedCanvas(0,0,1,1,20,30,1,fs)
for i,(kc,t) in enumerate(((g.N,'certified'),(g.O,'false certificates'),(None,'most rejected ('+'\\chi^2)'))):
    y=ky-i*.42
    k.polygon([(kx,y),(kx+.3,y),(kx+.3,y+.26),(kx,y+.26)],(kc or g.N)['t'])
    if kc is None:
        for f in (.25,.5,.75): k.seg_line((kx+f*.3,y),(kx+f*.3,y+.26),.012)
    k.put_text(kx+.38,y+.13,t,'lc',.235)
k.put_text(kx,ky-1.0,'height: per cent','lc',.235)
# (d): errors first (hard cases in front)
kcol=[7,6,0,1,2,3,4,5]; D4=O['a'][:,kcol,:]
p=SharedCanvas(1.72,1.66,.47,.40,20,36,.72,fs,gc=(2,)); p.grid(D4); 
def axes_labels(p,R,C,row_labels,col_labels,groups,panel_label,title_,tx,ty):
    d=.12
    for r,t in enumerate(row_labels):
        x,y=p.p(0,r+.5); p.put_text(x-d,y-d*.4,t,'rc',.235)
    for c,t in enumerate(col_labels):
        x,y=p.p(c+.5,R); p.put_text(x+.05,y-.13,str(t),'tc',.215)
    for a,b,t in groups:
        A=p.p(a+.08,R); B=p.p(b-.08,R); e=.5
        p.seg_line((A[0],A[1]-e),(B[0],B[1]-e)); x,y=p.p((a+b)/2,R); p.put_text(x,y-e-.09,t,'tc',.235)
    A=p.p(0,0)
    p.seg_line((A[0],A[1]),(A[0],A[1]+p.zh),.012)
    for v in (0,50,100):
        y=A[1]+v/100*p.zh; p.seg_line((A[0]-.05,y),(A[0],y),.012)
        if v: p.put_text(A[0]-.09,y,str(v),'rc',.215)
    p.put_text(tx,ty,'('+panel_label+') '+title_,'bl',.25)
axes_labels(p,4,8,['(3), \\delta=0','(4), \\delta=0','(2,1,1)','(2,2)'],[40,26,14,17,20,26,30,40],[(0,2,'5%'),(2,8,'no error')],'d','drift-matrix fit',0.15,4.22)
# (e): same shape, types along the axis, N=5 in front
ord_=[8,7,6,5,4,3,2,1,0]; E5=O['b'][ord_,:,:].transpose(1,0,2)
p=SharedCanvas(9.25,1.66,.43,.40,20,36,.72,fs,gc=(4,7),gr=(2,)); p.grid(E5)
axes_labels(p,4,9,['none, 26','none, 32','1%, 26','1%, 32'],['(1^5)','(5)','(2,2,1)','(3,2)','(4)','(3,1)','(2,2)','(3)','(2,1)'],[(0,4,'N=5'),(4,7,'N=4'),(7,9,'N=3')],'e',"calibrated rings, \\kappa'=20\\varepsilon",7.85,4.22)
o.append('')
open('fig_resolvability.gle','w').write('\n'.join(x for x in o if x is not None and x!=''))
r=subprocess.run([g.GLE,'-d','pdf','fig_resolvability.gle'],capture_output=True,text=True)
t=r.stdout+r.stderr
if '>>' in t: raise SystemExit('GLE error: ' + t[-500:])
deliver('fig_resolvability.pdf', 'fig06_readout.pdf')
