#!/usr/bin/env python3
"""Figure 5 (interference-selected permutation), drawn as one GLE file: schematic, threshold and gap graphs,
ring comparison and 3D certification bars. Data: data/fig05_interference.npz, data/readout_certification.npz."""
from paths import enter_build, data, deliver
import numpy as np
from gle_canvas import *
import gle_bars as g
enter_build()
d=np.load(data('fig05_interference.npz')); o=np.load(data('readout_certification.npz'),allow_pickle=False)
W,H=15.75,4.35; r=R(W,H); gy,gh=.85,3.0
# (a) schematic
sx,sy,s=1.1,2.85,.78; P={3:(-1.15,0),1:(0,1),2:(0,-1),0:(1.15,0)}; Q={k:(sx+s*v[0],sy+s*v[1]) for k,v in P.items()}; rn=.2
def str_(a,b,col='black',st=1,lw=.016):
    A,B=np.array(Q[a]),np.array(Q[b]); u=(B-A)/np.linalg.norm(B-A); r.l(A+u*rn,B-u*(rn+.02),lw,col,st,True)
for a,b in ((3,1),(3,2),(1,0)): str_(a,b)
str_(2,0,BL,1,.03); str_(1,2,'black',2)
A,B=np.array(Q[0]),np.array(Q[3]); C=(A+B)/2+np.array([0,-1.72]); tt=np.linspace(.06,.93,30); kr=[(1-t)**2*A+2*t*(1-t)*C+t**2*B for t in tt]; r.pl(kr[:-1],.03,OR); r.l(kr[-2],kr[-1]+(kr[-1]-kr[-2])*.3,.03,OR,1,True)
for k,(x,y) in Q.items():
    r.o+=['set lwidth .014','amove %.3f %.3f'%(x,y),f'circle {rn} fill white','set lwidth .012']; r.t(x,y,str(k),'cc',.235)
r.t(sx+.1,sy-.42,'e^{i\\varphi}','lc',.215,BL); r.t(sx+.09,sy+.1,'w','lc',.235); r.t(sx,sy-1.15,'c','cc',.26,OR); r.panel_label(.05,H-.3,'a')
# (b)
f=np.linspace(.02,np.pi-1e-3,500); b=np.abs(1+np.exp(1j*f))
np.savetxt('gle/f4b.dat',np.column_stack([f,np.minimum(256/27/b**4,1e4),np.minimum(256/27*.125/b**4,1e4)]),fmt='%.6g')
np.savetxt('gle/f4bm1.dat',np.column_stack([d['fc1'],np.ones(5)]),fmt='%.5g'); np.savetxt('gle/f4bm2.dat',np.column_stack([d['fc05'],np.ones(5)]),fmt='%.5g')
np.savetxt('gle/f4c.dat',np.column_stack([d['fi'],d['gi'],d['gl']]),fmt='%.5g')
PI=np.pi; xb,wb=3.45,2.75
r.o+=[f'amove {xb} {gy}','begin graph',f'  size {wb} {gh}','  fullsize',f'  xaxis min 0 max {PI:.6f} hei .235','  xnames "0" "\\pi/2" "\\pi"',f'  xplaces 0 {PI/2:.6f} {PI:.6f}','  xsubticks off','  yaxis log min 0.1 max 100 hei .235','  xtitle "\\varphi" hei .26 dist .08','  ytitle "\\tex{$|c|/\\varepsilon$}" hei .26 dist .08','  data "gle/f4b.dat" d1=c1,c2 d2=c1,c3','  data "gle/f4bm1.dat" d3=c1,c2','  data "gle/f4bm2.dat" d4=c1,c2',f'  d1 line color {BL} lwidth .03',f'  d2 line color {OR} lstyle 4 lwidth .03',f'  d3 marker wcircle msize .11 color {BL}',f'  d4 marker wsquare msize .11 color {OR}','end graph']
yb=lambda v:gy+(np.log10(v)+1)/3*gh; xB=lambda p:xb+p/PI*wb
r.l((xb,yb(1)),(xb+wb,yb(1)),.008,GY,2); r.t(xB(.2),yb(18),'(3,1)','lc',.235); r.t(xB(2.05),yb(.17),'4-cycle','cc',.235)
r.t(xB(1.0),yb(.55),'w=\\varepsilon','lc',.215,BL) ; r.t(xB(2.3),yb(30),'w=0.5\\varepsilon','rc',.215,OR); r.panel_label(xb,gy+gh+.07,'b')
# (c)
xc,wc=7.1,2.45
r.o+=[f'amove {xc} {gy}','begin graph',f'  size {wc} {gh}','  fullsize',f'  xaxis min 0 max {PI:.6f} hei .235','  xnames "0" "\\pi/2" "\\pi"',f'  xplaces 0 {PI/2:.6f} {PI:.6f}','  xsubticks off','  yaxis min 0 max 1.6 dticks 0.5 hei .235','  xtitle "\\varphi" hei .26 dist .08','  ytitle "min_{\\theta} gap / \\varepsilon" hei .26 dist .08','  data "gle/f4c.dat" d1=c1,c2 d2=c1,c3',f'  d1 line color black lwidth .03',f'  d2 line color {GR} lstyle 4 lwidth .03','end graph']
yc=lambda v:gy+v/1.6*gh; xC=lambda p:xc+p/PI*wc
r.t(xC(.15),yc(.72),'ideal','lc',.215); r.t(xC(.1),yc(1.43),"\\kappa'=40\\varepsilon",'lc',.215,GR); r.panel_label(xc,gy+gh+.07,'c')
# (d)
xd,wd=9.7,1.45
r.o+=['amove %.3f %.3f'%(xd,gy),f'box {wd} {gh}']
Xd=lambda v:xd+(v+.55)/2.1*wd
for v,t_ in ((0,'(3,1)'),(1,'(4)')):
    r.l((Xd(v),gy),(Xd(v),gy+.06)); r.t(Xd(v),gy-.08,t_,'tc',.215)
for v in (0,.5,1,1.5): r.l((xd,yc(v)),(xd+.06,yc(v)))
r.t(xd+wd/2,gy-.53,'type','cc',.26)
for dx,a,b,mf,mw in ((-.27,o['siec_i'],o['siec_k'],'fcircle','wcircle'),(.05,o['dr_i'],o['dr_k'],'fsquare','wsquare'),(.32,o['bez'],None,'ftriangle',None)):
    for i in (0,1):
        if b is not None: r.m(Xd(i+dx),yc(b[i]),mw,.16,GR)
        r.m(Xd(i+dx),yc(a[i]),mf,.13)
for k,(mk,t_) in enumerate((('fcircle','net'),('fsquare','lab'),('ftriangle','unl'))):
    y=yc(.5)-k*.25; r.m(xd+.2,y,mk,.13); r.t(xd+.33,y,t_,'lc',.2)
r.panel_label(xd,gy+gh+.07,'d')
# (e)
class SharedCanvas(g.P):
    def __init__(s,*a,**k):
        g.P.__init__(s,1,1,*a,**k); s.o=r.o
p=SharedCanvas(12.1,1.72,.32,.28,22,38,.62,.235,gc=(2,4),gr=(3,)); D=o['c']; p.grid(D)
for q,t_ in enumerate(['0','1.0*','1.7','2.6','\\pi*']):
    x,y=p.p(0,q+.5); p.put_text(x-.1,y-.04,t_,'rc',.215)
for c,t_ in enumerate([26,32,26,32,26,32]):
    x,y=p.p(c+.5,5); p.put_text(x+.03,y-.12,str(t_),'tc',.19)
for a,b_,t_ in ((0,2,'none'),(2,4,'1%'),(4,6,'2%')):
    A_,B_=p.p(a+.08,5),p.p(b_-.08,5); e=.42; p.seg_line((A_[0],A_[1]-e),(B_[0],B_[1]-e)); x,y=p.p((a+b_)/2,5); p.put_text(x,y-e-.07,t_,'tc',.215)
A_=p.p(0,0); p.seg_line(A_,(A_[0],A_[1]+p.zh),.012)
for v in (50,100): p.put_text(A_[0]-.08,A_[1]+v/100*p.zh,str(v),'rc',.195)
p.put_text(11.05,4.05,"(e) network, \\kappa'=40\\varepsilon",'bl',.25)
r.finish('fig_interference')
deliver('fig_interference.pdf', 'fig05_interference.pdf')
