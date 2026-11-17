#!/usr/bin/env python3
"""Figure N1 of the technical notes: read-out in simulation as 3D bar maps (GLE panels composed by pdflatex).
Data: data/figN1_readout_maps.npz -> output/figN1_readout_maps.pdf"""
from paths import enter_build, data, deliver
import numpy as np
from gle_bars import P
from compose import compose_page
enter_build()
si=np.load(data('figN1_readout_maps.npz'),allow_pickle=False)
# (a) calibrated rings
p=P(20,14,1.55,3.0,.40,.33,24,36,.6,.22,gr=(2,5)); ty=[str(t) for t in si['typy']]; g=[str(x) for x in si['g']]; wr=[('$(%s)$ %s'%(t.replace('+',','),q) if t!='1+1+1+1+1' else '$(1^5)$ %s'%q) for t,q in zip(ty,g)]
p.grid(si['D']); p.axes_labels(9,8,wr,[26,32,40,26,32,'e1','e2','e5'],[(0,2,'none'),(2,3,r'$\sigma$'),(3,5,r'$e_1$'),(5,8,'kept')],r'(a) calibrated rings, type and $g_{loop}/\varepsilon$'); p.save('pS3a')
# (b) four-mode network
p=P(20,14,.75,3.1,.40,.42,16,32,.6,.22,gr=(3,)); p.grid(si['C']); p.axes_labels(5,10,[r'$0$',r'$1.0$',r'$1.7$',r'$2.6$',r'$\pi$'],[26,32]*5,[(0,2,'A'),(2,4,'B'),(4,6,'C'),(6,8,'D'),(8,10,'E')],r'(b) network, rows $\varphi$'); p.key_box(.1,1.0,.38); p.save('pS3b')
compose_page('figN1_readout_maps',446.4,178,None,[('pS3a',0,2,222,174),('pS3b',226,2,220,174)])
deliver('figN1_readout_maps.pdf', 'figN1_readout_maps.pdf')
