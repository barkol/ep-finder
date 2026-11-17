#!/usr/bin/env python3
"""Figure 3 (programming by static detunings), drawn as one GLE file; regions are vector polygons from
contours of data/fig03_programming.npz."""
from paths import enter_build, data, deliver
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from gle_canvas import *
enter_build()
D=np.load(data('fig03_programming.npz'),allow_pickle=False); u,S,e1,e2=D['u'],D['S'],D['e1'],D['e2']
W,H=15.75,5.15; r=R(W,H)
x0,y0,L=1.1,1.0,3.75; sx=lambda v:x0+(v-u[0])/(u[-1]-u[0])*L; sy=lambda v:y0+(v-u[0])/(u[-1]-u[0])*L
cls=np.where(S<0,0.,np.where(S==0,0.,np.where(S<=3,1.,2.)))
fig,ax=plt.subplots()
du=u[1]-u[0]
for level_,kcol in ((.5,'rgb255(200,200,200)'),(1.5,'rgb255(122,122,122)')):
    for iy,row in enumerate(cls):
        on=np.flatnonzero(np.diff(np.r_[0,(row>level_).astype(int),0]))
        for a,b in zip(on[::2],on[1::2]):
            x1,x2=sx(max(u[a]-du/2,u[0])),sx(min(u[b-1]+du/2,u[-1])); y1,y2=sy(max(u[iy]-du/2,u[0])),min(sy(u[iy]+du/2)+.012,y0+L)
            r.polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)],kcol)
cl=ax.contour(u,u,cls,levels=[.5,1.5])
for lv in cl.allsegs:
    for seg in lv:
        if len(seg)>1: r.pl([(sx(p[0]),sy(p[1])) for p in seg[::2]],.014)
plt.close(fig)
for (i,j) in ((0,1),(0,2),(1,2)):
    n=np.array([e1[i]-e1[j],e2[i]-e2[j]]); t=np.array([-n[1],n[0]])/np.linalg.norm(n); m_=u[-1]
    A=-t*9; B=t*9; ts=[]
    for k in (0,1):
        if abs(t[k])>1e-9: ts+=[m_/abs(t[k])]
    q=min(ts); r.l((sx(-t[0]*q),sy(-t[1]*q)),(sx(t[0]*q),sy(t[1]*q)),.012,'black',2)
tex={1:'(0 1)',2:'(0 2)',3:'(1 2)',4:'(0 1 2)',5:'(0 2 1)'}
for k in tex:
    lab,n=ndimage.label(S==k)
    for m_ in range(1,n+1):
        if (lab==m_).sum()<400: continue
        dt=ndimage.distance_transform_edt(lab==m_); p=np.unravel_index(dt.argmax(),dt.shape)
        r.t(sx(u[p[1]]),sy(u[p[0]]),tex[k],'cc',.215,'white' if k>=4 else 'black')
r.o+=['amove %.3f %.3f'%(x0,y0),f'box {L} {L}']
for v in (-2,0,2):
    r.l((sx(v),y0),(sx(v),y0+.07)); r.t(sx(v),y0-.07,str(v),'tc',.235); r.l((x0,sy(v)),(x0+.07,sy(v))); r.t(x0-.08,sy(v),str(v),'rc',.235)
r.t(x0+L/2,.32,'(\\delta_0-\\delta_1)/2^{1/2}\\varepsilon','cc',.26)
r.o+=['amove .22 %.3f'%(y0+L/2),'begin rotate 90','set just cc','set hei .26','write "(\\delta_0+\\delta_1-2\\delta_2)/6^{1/2}\\varepsilon"','end rotate','set hei .25']
r.panel_label(x0,y0+L+.07,'a')
# (b)
Ns,gi,g20,nm=D['Ns'],D['gi'],D['g20'],D['names']; n=len(Ns)
bx,by,bw,bh=6.0,1.5,9.65,3.25; X=lambda i:bx+(i+.8)/(n+.6)*bw; Y=lambda v:by+v/2.25*bh
r.o+=['amove %.3f %.3f'%(bx,by),f'box {bw} {bh}']
for v in (0,.5,1,1.5,2):
    r.l((bx,Y(v)),(bx+.07,Y(v))); r.t(bx-.08,Y(v),('%.1f'%v),'rc',.235)
r.o+=['amove %.3f %.3f'%(bx-.75,by+bh/2),'begin rotate 90','set just cc','set hei .26','write "min_{\\theta} gap / \\varepsilon"','end rotate','set hei .25']
mk={3:('fcircle','wcircle'),4:('fsquare','wsquare'),5:('ftriangle','wtriangle'),6:('fdiamond','wdiamond')}
for N in (3,4,5,6):
    id_=np.where(Ns==N)[0]; A,B=X(id_[0]-.45),X(id_[-1]+.45)
    r.l((A,Y(2*np.sin(np.pi/N))),(B,Y(2*np.sin(np.pi/N))),.012,GY,2)
    r.t((A+B)/2,Y(2*np.sin(np.pi/N))+.15,f'N={N}','cc',.24)
    r.pl([(X(i),Y(gi[i])) for i in id_],.016)
    if N<6: r.pl([(X(i),Y(g20[i])) for i in id_],.012,GR,4)
    for i in id_:
        if N<6: r.m(X(i),Y(g20[i]),mk[N][1],.13,GR)
        r.m(X(i),Y(gi[i]),mk[N][0],.1)
    if N>3: r.l((A-.02,by),(A-.02,by+bh),.006,GY)
for i in range(n):
    r.l((X(i),by),(X(i),by+.06)); r.o+=['amove %.3f %.3f'%(X(i)+.0,by-.07),'begin rotate 90','set just rc','set hei .2',f'write "{nm[i]}"','end rotate','set hei .25']
lx=X(19); ly=by+.2
r.m(lx,ly,'fcircle',.1); r.t(lx+.14,ly,'ideal','lc',.225); r.m(lx+1.0,ly,'wcircle',.13,GR); r.t(lx+1.14,ly,"\\kappa'=20\\varepsilon",'lc',.225)
r.panel_label(bx,by+bh+.07,'b')
r.finish('fig_programming')
deliver('fig_programming.pdf', 'fig03_programming.pdf')
