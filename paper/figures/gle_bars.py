#!/usr/bin/env python3
"""3D bar panels written as GLE commands; GLE renders them as vector graphics. Bar height: certified per cent;
orange cap: false certificates; hatched top: most trials rejected by chi^2; missing cell: not run."""
import numpy as np, math, subprocess, os
from paths import GLE
N=dict(t=(178,210,238),l=(46,84,140),r=(104,146,196)); O=dict(t=(240,170,40),l=(165,95,0),r=(205,130,10))
kcol=lambda c:'rgb255(%d,%d,%d)'%c
class P:
    def __init__(s,w,h,x0,y0,cu,cv,a,b,zh,fs=.235,gc=(),gr=()): s.w,s.h,s.x0,s.y0,s.zh,s.fs=w,h,x0,y0,zh,fs; s.u=(cu*math.cos(math.radians(a)),cu*math.sin(math.radians(a))); s.v=(cv*math.cos(math.radians(b)),-cv*math.sin(math.radians(b))); s.gc,s.gr=gc,gr; s.o=[f'size {w} {h}','set font texcmr','set hei %.3f'%fs,'set lwidth 0.006','set cap round','set join round']
    def p(s,c,r,z=0):
        c=c+.5*sum(q<=c for q in s.gc); r=r+.5*sum(q<=r for q in s.gr)
        return s.x0+c*s.u[0]+r*s.v[0], s.y0+c*s.u[1]+r*s.v[1]+z/100*s.zh
    def polygon(s,pts,f):
        s.o+=[f'begin path stroke fill {kcol(f)}','amove %.4f %.4f'%pts[0]]+['aline %.4f %.4f'%q for q in pts[1:]]+['closepath','end path']
    def seg_line(s,A,B,lw=.006,col='black'): s.o+=[f'set lwidth {lw}',f'set color {col}','amove %.4f %.4f'%A,'aline %.4f %.4f'%B,'set color black','set lwidth 0.006']
    def put_text(s,x,y,t,j='cc',hei=None):
        s.o+=['set just '+j,'set hei %.3f'%(hei or s.fs),'amove %.4f %.4f'%(x,y),'write "%s"'%t.replace('$','').replace(r'\rm ','').replace(r'{\rm loop}','{loop}')]
    def bar(s,c,r,z0,z1,kc,hatch=False,g=.84):
        d=(1-g)/2; c0,c1,r0,r1=c+d,c+1-d,r+d,r+1-d
        if z1>z0:
            s.polygon([s.p(c0,r0,z0),s.p(c0,r1,z0),s.p(c0,r1,z1),s.p(c0,r0,z1)],kc['l'])
            s.polygon([s.p(c0,r1,z0),s.p(c1,r1,z0),s.p(c1,r1,z1),s.p(c0,r1,z1)],kc['r'])
        if kc is O: s.o.append('set lwidth 0.022')
        s.polygon([s.p(c0,r0,z1),s.p(c0,r1,z1),s.p(c1,r1,z1),s.p(c1,r0,z1)],kc['t'])
        if kc is O: s.o.append('set lwidth 0.006')
        if hatch:
            for f in (.2,.4,.6,.8): s.seg_line(s.p(c0+f*(c1-c0),r0,z1),s.p(c0+f*(c1-c0),r1,z1),.012)
    def grid(s,D):
        R,C=D.shape[:2]
        s.polygon([s.p(0,0),s.p(0,R),s.p(C,R),s.p(C,0)],(238,238,238))
        for c in range(C-1,-1,-1):
            for r in range(R):
                ok,false_,rej_,_=D[r,c]
                if np.isnan(ok): continue
                rej=rej_>=50
                if false_<.5: s.bar(c,r,0,ok,N,rej)
                else:
                    s.bar(c,r,0,ok,N); s.bar(c,r,ok,ok+false_,O)
    def axes_labels(s,R,C,row_labels,col_labels,groups,title_,lines_r=(),lines_c=(),z_axis=True):
        d=.13
        for r,t in enumerate(row_labels):
            x,y=s.p(0,r+.5); s.put_text(x-d,y-d*.35,t,'rc')
        for c,t in enumerate(col_labels):
            x,y=s.p(c+.5,R); s.put_text(x+d*.6,y-d*.75,str(t),'lc',s.fs*.92)
        for a,b,t in groups:
            A=s.p(a+.08,R); B=s.p(b-.08,R); e=.46
            s.seg_line((A[0]+e*.75,A[1]-e),(B[0]+e*.75,B[1]-e)); x,y=s.p((a+b)/2,R); s.put_text(x+e*.75+.12,y-e-.17,t,'lc')
        if z_axis:
            A=s.p(0,0); x=A[0]-.0
            s.seg_line((x,A[1]),(x,A[1]+s.zh),.012)
            for v in (0,50,100):
                y=A[1]+v/100*s.zh; s.seg_line((x-.05,y),(x,y),.012)
                if v: s.put_text(x-.09,y,str(v),'rc',s.fs*.9)
        ym=max(s.p(C,0,100)[1],s.p(0,0,100)[1]); s.put_text(.05,ym+.14,title_,'bl',s.fs*1.03)
    def key_box(s,x,y,dy=.42):
        s.polygon([(x,y),(x+.3,y),(x+.3,y+.26),(x,y+.26)],N['t']); s.put_text(x+.42,y+.13,'certified','lc')
        y-=dy; s.polygon([(x,y),(x+.3,y),(x+.3,y+.26),(x,y+.26)],O['t']); s.put_text(x+.42,y+.13,'false certificates','lc')
        y-=dy; s.polygon([(x,y),(x+.3,y),(x+.3,y+.26),(x,y+.26)],N['t'])
        for f in (.25,.5,.75): s.seg_line((x+f*.3,y),(x+f*.3,y+.26),.012)
        s.put_text(x+.42,y+.13,'most rejected ($\\chi^2$)','lc')
    def save(s,stem):
        s.o.append('')
        open(f'gle/{stem}.gle','w').write('\n'.join(s.o))
        r=subprocess.run([GLE,'-d','pdf','-o',f'{stem}.pdf',f'{stem}.gle'],cwd='gle',capture_output=True,text=True)
        subprocess.run(['pdfcrop','--margins','1',f'{stem}.pdf',f'{stem}_c.pdf'],cwd='gle',capture_output=True)
        if not os.path.exists(f'gle/{stem}_c.pdf') or 'nrecognised' in r.stdout+r.stderr or '>>' in r.stdout+r.stderr: raise SystemExit(r.stdout[-600:]+r.stderr[-300:])
