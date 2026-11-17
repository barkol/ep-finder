"""Shared GLE helpers: a canvas that collects GLE commands and renders them to PDF."""
import numpy as np, subprocess
from paths import GLE
BL,OR,GR,GY,INK='rgb255(0,114,178)','rgb255(213,94,0)','rgb255(0,158,115)','gray50','black'
class R:
    def __init__(s,W,H,fs=.25): s.fs=fs; s.o=[f'size {W} {H}','set font texcmr','set hei %.3f'%fs,'set lwidth 0.012','set cap round','set arrowstyle filled','set arrowsize 0.14','set arrowangle 20']
    def t(s,x,y,txt,j='cc',hei=None,col='black'): s.o+=[f'set color {col}',f'set just {j}','set hei %.3f'%(hei or s.fs),'amove %.3f %.3f'%(x,y),f'write "{txt}"','set color black']
    def panel_label(s,x,y,l): s.t(x,y,f'({l})','bl',.26)
    def l(s,A,B,lw=.012,col='black',st=1,arrow=False): s.o+=[f'set lwidth {lw}',f'set color {col}',f'set lstyle {st}','amove %.3f %.3f'%tuple(A),'aline %.3f %.3f'%tuple(B)+(' arrow end' if arrow else ''),'set lstyle 1','set color black','set lwidth 0.012']
    def pl(s,P,lw=.012,col='black',st=1): s.o+=[f'set lwidth {lw}',f'set color {col}',f'set lstyle {st}','amove %.3f %.3f'%tuple(P[0])]+['aline %.3f %.3f'%tuple(q) for q in P[1:]]+['set lstyle 1','set color black','set lwidth 0.012']
    def polygon(s,P,fill,stroke=False): s.o+=[f'begin path fill {fill}'+(' stroke' if stroke else ''),'amove %.3f %.3f'%tuple(P[0])]+['aline %.3f %.3f'%tuple(q) for q in P[1:]]+['closepath','end path']
    def m(s,x,y,mk,ms=.12,col='black'): s.o+=[f'set color {col}','set hei %.3f'%(ms*1.25),'amove %.3f %.3f'%(x,y),f'marker {mk}','set color black','set hei %.3f'%s.fs]
    def finish(s,stem):
        open(stem+'.gle','w').write('\n'.join(s.o)+'\n'); r=subprocess.run([GLE,'-d','pdf',stem+'.gle'],capture_output=True,text=True); t=r.stdout+r.stderr
        if '>>' in t: raise SystemExit(t[-600:])
        print(stem,'ok')
