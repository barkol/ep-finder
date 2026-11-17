#!/usr/bin/env python3
"""Vector composition of cropped GLE panels into one PDF page (pdflatex)."""
import subprocess,shutil
def compose_page(out_name,w,h,base,slots):
    t=[r'\documentclass{article}\usepackage{graphicx}\pagestyle{empty}\pdfoutput=1\setlength\parindent{0pt}',r'\paperwidth=%.2fbp\paperheight=%.2fbp\topskip=0pt\hoffset=-1in\voffset=-1in\oddsidemargin=0pt\topmargin=0pt\headheight=0pt\headsep=0pt\textwidth=%.2fbp\textheight=%.2fbp'%(w,h,w,h),r'\begin{document}\setlength\unitlength{1bp}\begin{picture}(%.2f,%.2f)'%(w,h)]
    if base: t.append(r'\put(0,0){\includegraphics[width=%.2fbp,height=%.2fbp]{%s}}'%(w,h,base))
    for f,x,y,sw,sh in slots: t.append(r'\put(%.2f,%.2f){\includegraphics[width=%.2fbp,height=%.2fbp,keepaspectratio]{gle/%s_c.pdf}}'%(x,y,sw,sh,f))
    t+=[r'\end{picture}\end{document}']
    open(out_name+'_sk.tex','w').write('\n'.join(t))
    r=subprocess.run(['pdflatex','-interaction=nonstopmode',out_name+'_sk.tex'],capture_output=True,text=True)
    assert r.returncode==0,r.stdout[-500:]
    shutil.move(out_name+'_sk.pdf',out_name+'.pdf')
