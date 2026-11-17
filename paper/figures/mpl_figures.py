"""Matplotlib figures of the paper: figure 1, figure 2, figure 4 and figure N2 of the technical notes.
Only reads cached data from ../data. Width = text width of the two-column layout (446.4 pt), 8 pt font."""
import numpy as np
from paths import data, output
import figstyle as fs, matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Circle, Polygon, Rectangle
fs.apply(); W=446.4/72.27
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7,'axes.labelpad':1.5,'xtick.major.pad':2,'ytick.major.pad':2,'lines.linewidth':1.1,'legend.labelspacing':.2,'legend.handletextpad':.4,'legend.borderpad':.1,'legend.handlelength':1.8})
PAL=[fs.BLUE,fs.ORANGE,fs.GREEN,fs.PINK]; MK=['o','s','^','D']
def arrow(a,p,q,c=fs.INK,ls='-',rad=0,lw=.8,z=3,s=5,sh=4):
    a.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=s,lw=lw,color=c,ls=ls,connectionstyle=f'arc3,rad={rad}',shrinkA=sh,shrinkB=sh,zorder=z))
def icon(a,kind,desc,title_text):
    a.set_aspect('equal'); a.axis('off'); a.set_xlim(-.95,2.9); a.set_ylim(-1.12,.9)
    if kind=='ring':
        P=[(np.cos(t)*.68,np.sin(t)*.68-.08) for t in (np.pi/2,np.pi/2+2.094,np.pi/2-2.094)]
        arrow(a,P[1],P[2]); arrow(a,P[2],P[0]); arrow(a,P[0],P[1],fs.ORANGE,lw=1.2)
    else:
        P={3:(-.72,0),1:(0,.55),2:(0,-.55),0:(.72,0)}
        for p,q in ((3,1),(3,2),(1,0)): arrow(a,P[p],P[q])
        arrow(a,P[2],P[0],fs.BLUE,lw=1.2); arrow(a,P[1],P[2],fs.INK,(0,(2,1.5)))
        arrow(a,P[0],P[3],fs.ORANGE,rad=-.95,lw=1.2); P=list(P.values())
    for x,y in P: a.add_patch(Circle((x,y),.13,fc='white',ec=fs.INK,lw=.6,zorder=4))
    a.text(1.0,.28,desc,ha='left',va='center',fontsize=8); a.text(1.0,-.42,title_text,ha='left',va='center',fontsize=8)
# ---------- figure 1
def fig01():
    """Blender renders (data_scripts/fig01_render_blender.py) plus network icons and labels; one row of four panels."""
    from PIL import Image
    title_={'a':'3-cycle','b':r'type $(2,1)$','c':r'type $(3,1)$','d':'4-cycle'}
    icons={'a':('ring',r'$\delta_k=0$'),'b':('ring',r'$\delta_k\neq0$'),'c':('net',r'$\varphi=1$'),'d':('net',r'$\varphi=\pi$')}
    fig=plt.figure(figsize=(W,2.05))
    gs=fig.add_gridspec(2,4,height_ratios=[.27,1],wspace=.0,hspace=.0,left=.035,right=1.,top=.995,bottom=.0)
    for i,k in enumerate('abcd'):
        icon(fig.add_subplot(gs[0,i]),*icons[k],title_[k])
        im=Image.open(data(f'fig01_render_{k}.png')).convert('RGBA'); t=Image.new('RGBA',im.size,(255,255,255,255)); t.alpha_composite(im); bb=im.getchannel('A').getbbox(); t=t.crop(bb).convert('RGB')
        a=fig.add_subplot(gs[1,i]); a.imshow(np.asarray(t),interpolation='none'); a.axis('off')
        fig.text(gs[0,i].get_position(fig).x0+.002,.985,f'({k})',va='top',ha='left',fontsize=8)
    ax=fig.add_axes([0,0,.05,1]); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.add_patch(FancyArrowPatch((.62,.1),(.62,.68),arrowstyle='-|>',mutation_scale=7,lw=.8,color=fs.INK)); ax.text(.22,.39,r'$\theta:0\to2\pi$',rotation=90,ha='center',va='center',fontsize=7.5)
    fig.savefig(output('fig01_permutations.pdf'),dpi=400)
WS=446.4/72.27

def _lit(a,l,x=0): a.text(x,1.04,f'({l})',transform=a.transAxes,va='bottom',fontsize=7.5)
# ---------- figure 2
def fig02():
    d=np.load(data('ring_dynamics.npz')); fig=plt.figure(figsize=(WS,2.5)); G=fig.add_gridspec(2,4,width_ratios=[.86,1,1,1],wspace=.40,hspace=.62,left=.05,right=.993,top=.945,bottom=.125); ax=[fig.add_subplot(G[0,i]) for i in range(4)]; bx=[fig.add_subplot(G[1,i]) for i in range(4)]
    for q in ax: q.tick_params(labelsize=6.5,length=2); q.xaxis.label.set_size(7); q.yaxis.label.set_size(7)
    a=ax[0]; t=np.linspace(0,2*np.pi,300); a.plot(np.cos(t),np.sin(t),color=fs.GREEN,lw=1.1)
    a.add_patch(FancyArrowPatch((np.cos(.75),np.sin(.75)),(np.cos(1.0),np.sin(1.0)),arrowstyle='-|>',mutation_scale=6,color=fs.GREEN,lw=1.1))
    a.plot([0],[0],marker='*',ms=6,color=fs.INK,ls='none'); a.text(.1,.08,r'EP$_3$',fontsize=6.5)
    for k,(c_,m_) in enumerate(zip((fs.ORANGE,fs.BLUE,fs.GREEN),'os^')):
        z=np.exp(2j*np.pi*k/3); a.plot(z.real,z.imag,m_,color=c_,ms=3.6,mec='w',mew=.3,zorder=4); a.text(1.3*z.real,1.3*z.imag,rf'$\lambda_{k}$',ha='center',va='center',fontsize=6.5)
    a.set_xlim(-1.55,1.55); a.set_ylim(-1.55,1.55); a.set_aspect('equal'); a.set_xticks([-1,0,1]); a.set_yticks([-1,0,1]); a.set_xlabel(r'Re $c$, Re $\lambda$'); a.set_ylabel(r'Im $c$, Im $\lambda$')
    b=ax[1]; b.plot(d['th']/np.pi,2*np.sin(np.pi/3)*np.ones_like(d['th']),color=fs.GREEN,lw=1.3,label='phase'); b.plot(d['th']/np.pi,d['dre'],color=fs.GREY,ls=(0,(3,1.5)),lw=.9,label=r'real $(a,c)$')
    b.set_xlim(0,2); b.set_ylim(0,2.75); b.set_xlabel(r'$\theta/\pi$'); b.set_ylabel(r'$\delta_{\min}/\varepsilon$'); b.legend(loc='upper center',fontsize=6,ncol=2,columnspacing=.7,handlelength=1.2,handletextpad=.3,borderaxespad=0)
    c=ax[2]; f,P=d['fs'],d['P']; c.axvline(.1,color=fs.INK,lw=.5,ls=(0,(3,2))); c.plot(f,P[0,:,0],'o-',color=fs.BLUE,ms=1.8,lw=.9,label='diagonal'); c.plot(f,P[1,:,0],'s--',color=fs.ORANGE,ms=1.8,lw=.9,label='spurious')
    c.set_xscale('log'); c.set_xlim(.02,30); c.set_ylim(0,1.05); c.set_xticks([.1,1,10]); c.set_xlabel(r'$\sigma/\delta_{\min}$'); c.set_ylabel(r'$P$(3-cycle)'); c.legend(loc='upper right',fontsize=6,handlelength=1.5,handletextpad=.3,borderaxespad=0,labelspacing=.15)
    e=ax[3]; p=P[0]; v=e.stackplot(f,p[:,0],p[:,2],p[:,3],colors=[fs.GREEN,fs.PINK,fs.FAINT],ec=fs.INK,lw=.3)
    v[1].set_hatch('////'); e.axvline(.1,color=fs.INK,lw=.5,ls=(0,(3,2))); e.set_xscale('log'); e.set_xlim(.02,30); e.set_ylim(0,1); e.set_xticks([.1,1,10]); e.set_xlabel(r'$\sigma/\delta_{\min}$'); e.set_ylabel('fraction')
    e.text(.06,.5,'3-cycle',rotation=90,ha='center',va='center',fontsize=6.3,color='w'); e.text(10,.8,'identity',ha='center',va='center',fontsize=6.3); e.text(4.5,.11,'transp.',ha='center',va='center',fontsize=6,bbox=dict(fc='w',ec='none',pad=.4))
    for q,l in zip(ax,'abcd'): _lit(q,l)
    T=np.load(data('tables.npz'),allow_pickle=False); k=bx[0]; k.plot(T['rN'],T['rs'],'o-',color=fs.GREEN,ms=2,lw=.9,label=r'$\sin(\pi/N)$'); k.plot(T['rN'],T['rc'],'s--',color=fs.ORANGE,ms=2,lw=.9,label='collision'); k.set_xlabel(r'$N$'); k.set_ylabel(r'radius$/|c|^{1/N}$'); k.set_xticks([3,6,9,12]); k.set_ylim(0,1.45); k.legend(loc='upper center',fontsize=6,ncol=1,borderaxespad=0,handlelength=1.5,handletextpad=.3,labelspacing=.1)
    k=bx[1]; k.loglog(T['tN'],T['ta'],'o-',color=fs.BLUE,ms=2,lw=.9,label=r'$\varepsilon T_{\rm adiab}$'); k.loglog(T['tN'],T['to'],'s--',color=fs.ORANGE,ms=2,lw=.9,label=r'$\varepsilon T_{\rm orbit}$'); k.loglog(T['tN'],T['tg'],'^:',color=fs.GREEN,ms=2,lw=1,label=r'$\overline{\Delta\Gamma}/\varepsilon$'); k.set_xticks([3,5,10,14]); k.set_xticklabels(['3','5','10','14']); k.minorticks_off(); k.set_xlabel(r'$N$'); k.set_ylim(.03,300); k.legend(loc='upper left',fontsize=5.8,ncol=2,columnspacing=.5,borderaxespad=0,handlelength=1.4,handletextpad=.25,labelspacing=.1)
    k=bx[2]; k.fill_between(T['kk'],T['kg0'],T['kg1'],color=fs.FAINT,ec=fs.INK,lw=.5); k.axhline(np.sqrt(3),color=fs.GREEN,lw=.9); k.set_xscale('log'); k.set_xlim(20,300); k.set_xticks([20,50,100,300]); k.set_xticklabels(['20','50','100','300']); k.minorticks_off(); k.set_ylim(1.6,2.3); k.set_xlabel(r"$\kappa'/\varepsilon$"); k.set_ylabel(r'slow gap$/\varepsilon$')
    k=bx[3]; k.loglog(T['kk'],T['kr'],'o-',color=fs.BLUE,ms=2,lw=.9,label='rev./forw.'); k.loglog(T['kk'],T['kl'],'s--',color=fs.ORANGE,ms=2,lw=.9,label=r'$|\Delta\lambda|/\varepsilon$'); k.loglog(T['kk'],T['kS']/100,'^:',color=fs.GREEN,ms=2,lw=1,label=r'$\Delta|S_{21}|^2$'); k.set_xticks([20,50,100,300]); k.set_xticklabels(['20','50','100','300']); k.minorticks_off(); k.set_xlabel(r"$\kappa'/\varepsilon$"); k.set_ylim(1e-3,8); k.legend(loc='upper right',fontsize=5.8,ncol=2,columnspacing=.5,borderaxespad=0,handlelength=1.4,handletextpad=.25,labelspacing=.1)
    for q,l in zip(bx,'efgh'): q.tick_params(labelsize=6.5,length=2); q.xaxis.label.set_size(7); q.yaxis.label.set_size(7); _lit(q,l)
    fig.savefig(output('fig02_undetuned_ring.pdf'))
# ---------- figure 4
def fig04():
    d=np.load(data('ring_dynamics.npz')); fig=plt.figure(figsize=(WS,1.9)); G=fig.add_gridspec(1,3,left=.075,right=.992,top=.91,bottom=.53,wspace=.42,width_ratios=[1,1.05,1])
    ax=[fig.add_subplot(G[0,i]) for i in range(3)]; C3=(fs.ORANGE,fs.BLUE,fs.GREEN)
    for q in ax: q.tick_params(labelsize=6.5,length=2); q.xaxis.label.set_size(7); q.yaxis.label.set_size(7)
    a=ax[0]; TH,Wk=d['TH'],d['W']
    for k,(ls,m_) in enumerate(zip(('-','--',(0,(1,1))),'os^')):
        a.plot(TH/np.pi,Wk[:,k],color=C3[k],ls=ls,lw=1.1); a.plot(TH[0]/np.pi,Wk[0,k],'o',color=C3[k],ms=3,mec='w',mew=.3,zorder=4); a.plot(TH[-1]/np.pi,Wk[-1,k],'s',color=C3[k],ms=3,mec='w',mew=.3,zorder=4)
    a.set_xlim(-.05,2.05); a.set_ylim(-1.25,1.25); a.set_xticks([0,1,2]); a.set_xlabel(r'$\theta/\pi$'); a.set_ylabel(r'$\omega_k/\varepsilon$'); a.text(.04,.04,r'$\bullet$ start\ \ $\blacksquare$ end',transform=a.transAxes,fontsize=6)
    b=ax[1]
    for k,lab,c_,ls,m_ in (('T20_e0p0',r'$20,0$',C3[1],'-','o'),('T20_e1em08',r'$20,10^{-8}$',C3[0],'--','s'),('T20_e0p0001',r'$20,10^{-4}$',C3[2],':','^'),('T110_e0p0001',r'$110,10^{-4}$',fs.PINK,'-.','D')):
        nper=240 if k.startswith('T110') else 120; y=np.clip(d[k+'_inf'],1e-18,1); i_=nper*np.arange(1,7); x=np.arange(len(y))/nper
        b.semilogy(x[i_],y[i_],color=c_,ls=ls,marker=m_,ms=2.3,mfc='w',mew=.6,lw=.9,label=lab)
    b.set_xticks(range(1,7)); b.set_ylim(1e-22,8); b.set_yticks([1e-20,1e-10,1]); b.set_xlabel('orbit'); b.set_ylabel(r'largest $1-F$'); b.legend(loc='lower left',fontsize=5.8,ncol=2,columnspacing=.5,handlelength=1.5,handletextpad=.25,borderaxespad=.1,labelspacing=.12)
    c=ax[2]
    for i,(k,lab,c_,ls,lw) in enumerate((('T110_e0p0',r'$0$',C3[1],'-',1.2),('T110_e1em16',r'$10^{-16}$',C3[0],'--',1.),('T110_e0p0001',r'$10^{-4}$',C3[2],':',1.1))):
        y=d[k+'_lab'][:,0]+(i-1)*.09; c.plot(np.arange(len(y))/240,y,color=c_,ls=ls,lw=lw,drawstyle='steps-post',label=lab)
    c.set_xlim(0,6); c.set_xticks(range(7)); c.set_yticks([0,1,2]); c.set_ylim(-.35,3.1); c.set_xlabel('orbit'); c.set_ylabel('branch'); c.legend(loc='upper center',fontsize=6,ncol=3,columnspacing=.6,handlelength=1.4,handletextpad=.25,borderaxespad=0,title=None)
    for q,l in zip(ax,'abc'): _lit(q,l)
    # (d) elements of S_3 as permutations of the three modes
    E=[({0:0,1:1,2:2},r'$e$'),({0:1,1:0,2:2},r'$(0\,1)$'),({0:2,1:1,2:0},r'$(0\,2)$'),({0:0,1:2,2:1},r'$(1\,2)$'),({0:1,1:2,2:0},r'$(0\,1\,2)$'),({0:2,1:0,2:1},r'$(0\,2\,1)$')]
    S=fig.add_axes([.0,0,1,.29]); S.set_xlim(0,WS*72.27); S.set_ylim(0,1.9*72.27*.29); S.axis('off'); pos=np.array([[0,10.5],[-11,-7],[11,-7]]); Y=19; X=[48,122,192,262,334,398]
    for (sig,lab),x0 in zip(E,X):
        for i in range(3):
            S.add_patch(Circle(pos[i]+(x0,Y),4.6,fc='w',ec=fs.INK,lw=.5,zorder=3)); S.text(*(pos[i]+(x0,Y)),str(i),ha='center',va='center',fontsize=5.6,zorder=4)
            j=sig[i]
            if j!=i: S.add_patch(FancyArrowPatch(tuple(pos[i]+(x0,Y)),tuple(pos[j]+(x0,Y)),shrinkA=4.6*1.0,shrinkB=4.6*1.0,arrowstyle='-|>',mutation_scale=4.5,color=C3[i],lw=.8,connectionstyle='arc3,rad=0.25',zorder=2))
        S.text(x0+13,Y+13,lab,ha='left',va='center',fontsize=6.5)
    fig.text(.008,.325,'(d)',fontsize=7.5,va='center')
    fig.text(55/(WS*72.27),.325,'no ring',ha='center',va='center',fontsize=6.5); fig.text(198/(WS*72.27),.325,'one 2-ring and one fixed mode',ha='center',va='center',fontsize=6.5); fig.text(372/(WS*72.27),.325,'one 3-ring',ha='center',va='center',fontsize=6.5)
    fig.savefig(output('fig04_spectrum_state_group.pdf'))
# ---------- figure N2 (technical notes)
def figN2():
    T=np.load(data('tables.npz'),allow_pickle=False); n=len(T['pN']); x=np.arange(n); fig,ax=plt.subplots(4,1,figsize=(WS,2.6),sharex=True,gridspec_kw=dict(left=.056,right=.995,top=.99,bottom=.215,hspace=.14,height_ratios=[1.3,1,1,1]))
    a=ax[0]; a.plot(x,T['pm'],ls='none',marker='_',color=fs.GREY,ms=5,label=r'$g_{\max}$'); a.plot(x,T['pg'],'o',color=fs.INK,ms=2.6,label=r'$g_{\rm loop}$ ideal'); a.plot(x,T['vg'],'s',mfc='none',mec=fs.GREEN,ms=3,label=r"$g_{\rm loop}$, $\kappa'=20\varepsilon$"); a.set_ylabel(r'gap$/\varepsilon$'); a.set_ylim(0,3.2); a.legend(loc='upper right',bbox_to_anchor=(1,.74),ncol=3,fontsize=6,borderaxespad=0,handletextpad=.2,columnspacing=.8)
    b=ax[1]; b.semilogy(x,T['pT'],'o',color=fs.INK,ms=2.6); b.semilogy(x,T['vT'],'s',mfc='none',mec=fs.GREEN,ms=3); b.set_ylabel(r'$T_{\rm ad}$ ($\mu$s)'); b.set_ylim(.25,40)
    c=ax[2]; cm=np.where(np.isinf(T['c1']),3.4,T['c1'])
    for i in x:
        c.plot([i,i],[T['c0'][i],cm[i]],color=fs.BLUE,lw=1.5,solid_capstyle='butt')
        if np.isinf(T['c1'][i]): c.plot(i,3.4,marker='^',ms=2.8,color=fs.BLUE,clip_on=False)
    c.axhline(1,color=fs.INK,lw=.5,ls=(0,(3,2))); c.set_ylim(0,3.5); c.set_ylabel(r'$|c|/\varepsilon$')
    d=ax[3]; d.semilogy(x-.13,T['xd'],'o',color=fs.INK,ms=2.4,label='detuning'); d.semilogy(x+.13,T['xc'],'s',mfc='none',mec=fs.ORANGE,ms=2.8,label='complex'); d.set_ylim(.006,.4); d.set_ylabel(r'$x_{99}$'); d.legend(loc='upper right',ncol=2,fontsize=6,borderaxespad=0,handletextpad=.1,columnspacing=.8)
    for q,l in zip(ax,'abcd'):
        q.tick_params(labelsize=6.3,length=2); q.yaxis.label.set_size(6.8); q.set_xlim(-1.5,n-.4); q.text(.003,.5,f'({l})',transform=q.transAxes,va='center',fontsize=7.2)
        for N in (4,5,6): q.axvline(x[T['pN']==N][0]-.5,color=fs.GREY,lw=.5)
    d.set_xticks(x); d.set_xticklabels([rf'\texttt{{{t}}}' for t in T['pt']],rotation=90,fontsize=5.8); d.tick_params(axis='x',pad=1)
    for N in (3,4,5,6): ax[0].text(x[T['pN']==N].mean(),2.85,rf'$N={N}$',ha='center',va='center',fontsize=6.5) if N<6 else ax[0].text(x[T['pN']==N].mean(),2.85,rf'$N={N}$',ha='center',va='center',fontsize=6.5)
    fig.align_ylabels(ax); fig.savefig(output('figN2_ladders.pdf'))
