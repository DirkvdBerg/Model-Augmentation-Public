"""Concept sketches only. All traces and state values are schematic, not results."""
from pathlib import Path
import math

OUT = Path(__file__).resolve().parent / 'sketches'
OUT.mkdir(exist_ok=True)
HEADER = r'''% v1: three explanatory concepts; schematic signals, no inferred latent geometry.
\documentclass[tikz,border=0pt]{standalone}
\usepackage{amsmath}
\usetikzlibrary{arrows.meta,calc}
\definecolor{hl}{HTML}{0092B5}
\tikzset{>=Latex,line width=.7pt, lab/.style={font=\small},
 fixed/.style={black!55},learn/.style={hl},
 blk/.style={draw,rounded corners=1pt,minimum width=13mm,minimum height=7mm}}
\begin{document}
'''

def node(x,y,s,opts=''):
    return rf'\node[{opts}] at ({x},{y}) {{{s}}};'+'\n'

def line(x,y,a,b,opts=''):
    return rf'\draw[{opts}] ({x},{y}) -- ({a},{b});'+'\n'

def curve(x,y,w,h,phase=0,opts='',dots=False,fn=None,n=30):
    f = fn or (lambda t: math.sin(2*math.pi*1.4*t+phase))
    pts=[(x+w*i/(n-1),y+h*f(i/(n-1))) for i in range(n)]
    s='\\draw['+opts+'] plot[smooth] coordinates {'+' '.join(f'({a:.4f},{b:.4f})' for a,b in pts)+'};\n'
    if dots:
        for a,b in pts:
            s+=rf'\fill[{opts}] ({a:.4f},{b:.4f}) circle (.025);'+'\n'
    return s

def start(title,footer):
    return r'\begin{tikzpicture}\useasboundingbox (0,0) rectangle (16,9);\fill[white] (0,0) rectangle (16,9);\fontsize{12}{14}\selectfont'+'\n'+node(.6,8.4,title,r'anchor=west,font=\large\bfseries')+node(.6,.38,footer,r'anchor=west,font=\scriptsize,text width=14.8cm,align=left')

END='\\end{tikzpicture}\n'
L1=r'$\displaystyle\min_{\theta_{\rm p},\theta^a_{\rm enc}}\sum_{\tau\in\mathcal B}\|p(\psi_a(\chi_\tau))-e_\tau\|_2^2$'
L2=r'$\displaystyle\min_{\theta_{\rm aug}}\sum_{\tau\in\mathcal B_{\mathcal P}}w_{\mathcal B_{\mathcal P}}\|g_{\rm aug}(\hat x_{\tau|\tau},u_\tau)-\psi_a(\chi_{\tau+1})\|_2^2$'

def a_s1(build=8):
    s=start('1 / S1: give the added states a prediction task','Schematic, normalised; 30 samples per window. State superscripts n suppressed. S1 alternates with V updates; its target is fixed within each S1 step.')
    # Elements have fixed positions across builds. Blue always marks the newest element.
    c=lambda k:'hl' if build==k else 'black'
    if build>=1:
        s+=curve(1,6.5,11.8,.35,opts=c(1),dots=True,fn=lambda t:.65*math.sin(9*t)+.28*math.sin(22*t),n=60)
        s+=line(.8,5.7,13.5,5.7,'->')+line(7,5.8,7,7.5,'dotted')+node(7,5.4,r'$\tau$')+node(13.8,6.65,r'$y^{\rm n}_k$')
        s+=line(.9,7.6,6.8,7.6)+node(3.7,7.85,r'$\chi_\tau:\ (u,y)_{\tau-30:\tau-1}$',r'font=\small')
    if build>=2:
        # Current model path is a separate construction, not force replay.
        s+=curve(7,6.5,5.8,.35,opts='black!55' if build!=2 else c(2),fn=lambda t:.65*math.sin(9*(30+29*t)/59)+.28*math.sin(22*(30+29*t)/59)-.8*math.sin(2*math.pi*1.4*t+.35))
        s+=node(13.8,6.1,r'$\hat y^{\rm n}_{k|\tau}$',c(2))
        s+=node(3.5,4.7,r'$\hat x_{\tau|\tau}=\psi(\chi_\tau)$',c(2))
        s+=line(5.1,4.7,8.6,4.7,c(2)+',->')+node(7,5.03,'current model + controller',r'font=\scriptsize,'+c(2))
        s+=node(10.9,4.7,r'$y^{\rm n}_{\tau:\tau+H-1}-\hat y^{\rm n}_{\tau:\tau+H-1|\tau}$',c(2))
    if build>=3:
        for j in range(0,30,3):
            t=j/29
            xx=7+5.8*t
            yy=6.5+.35*(.65*math.sin(9*(30+29*t)/59)+.28*math.sin(22*(30+29*t)/59))
            yh=yy-.28*math.sin(2*math.pi*1.4*t+.35)
            s+=line(round(xx,4),round(yy,4),round(xx,4),round(yh,4),c(3))
        s+=line(7.8,3.4,13.4,3.4,'black!55')
        s+=curve(7.8,3.4,5.5,.52,phase=.35,opts=c(3),dots=True)
        s+=node(14,3.7,r'$e_\tau$','anchor=west,'+c(3))+line(10.9,4.4,10.9,4.05,'->,'+c(3))
        s+=node(11,2.6,r'$e_\tau=(y^{\rm n}-\hat y^{\rm n})_{\tau:\tau+H-1|\tau}/\sigma_e$',c(3)+r',font=\small')
    if build>=4:
        s+=node(2,3.4,r'$\bar x_{\tau|\tau}=\psi_a(\chi_\tau)$',c(4))
    if build>=5:
        s+=line(3.7,3.4,5.35,3.4,'->,'+c(5))+node(4.6,3.8,r'$p$',r'font=\large,'+c(5))
        s+=node(4.6,3,'temporary',r'font=\scriptsize,'+c(5))
    if build>=6:
        s+=curve(7.8,3.4,5.5,.46,phase=.49,opts=c(6)+',dashed')
        s+=line(5.4,3.4,7.6,3.4,'->,'+c(6))
        s+=node(14,3.05,r'$p(\bar x_{\tau|\tau})$','anchor=west,'+c(6)+r',font=\small')
    if build>=7:
        s+=node(8,1.75,L1,c(7))
        s+=node(8,1.05,'Residual target held fixed; loss trains the decoder and added encoder rows.',r'font=\small,'+c(7))
    if build>=8:
        s+=r'\draw[->,hl,dashed] (6.4,2.2) -- (4.6,2.55) -- (4.6,2.9);\draw[->,hl,dashed] (4.2,2.65) -- (2,3.05);'+'\n'
        s+=node(2.9,2.2,r'$\nabla_{\theta_{\rm p},\theta^a_{\rm enc}}$','hl')
    if build<7:
        eqs=[r'$\chi_\tau=(u,y)_{\tau-30:\tau-1}$',r'$\hat x_{\tau|\tau}=\psi(\chi_\tau)$',r'$e_\tau=\operatorname{vec}(y^{\rm n}-\hat y^{\rm n})_{\tau:\tau+H-1|\tau}/\sigma_e$',r'$\bar x_{\tau|\tau}=\psi_a(\chi_\tau)$',r'$p:\mathbf R^{n_{\bar x}}\to\mathbf R^{H n_y}$',r'$\hat e_\tau=p(\bar x_{\tau|\tau})$']
        s+=node(8,1.7,eqs[build-1],c(build))
    return s+END

def a_s2(build=8):
    s=start('1 / S2: predict the next encoded state','Schematic; only the last few data samples shown. Grey encoder routes and targets are fixed throughout S2.')
    c=lambda k:'hl' if build==k else 'black'
    if build>=1:
        for i in range(9):
            s+=rf'\fill ({1+i*.65},{6.7+.25*math.sin(i):.3f}) circle (.045);'+'\n'
        s+=node(5.45,7.15,r'$\tau-1$')+node(6.35,7.15,r'$\tau$')
        s+=line(1,6.1,5.55,6.1,c(1))+node(3,5.8,r'$\chi_\tau$')
    if build>=2:
        s+=line(1.65,5.3,6.2,5.3,c(2))+node(3.5,5,r'$\chi_{\tau+1}$',c(2))
        s+=node(3.8,7.85,r'$\chi_{\tau+1}:\ \text{drop }(u,y)_{\tau-30},\ \text{add }(u,y)_\tau$',c(2)+r',font=\small')
    if build>=3:
        tc='hl' if build==3 else 'black!55'
        s+=line(6.3,5.3,10.2,5.3,'->,'+tc)+node(8.2,5.65,r'$\psi_a$ fixed',tc)
        s+=node(12.3,5.3,r'$\bar x_{\tau+1|\tau+1}$',tc)
        s+=node(12.3,5.85,'target',r'font=\small,'+tc)
    if build>=4:
        xc='hl' if build==4 else 'black!55'
        s+=node(3,3.9,r'$\hat x_{\tau|\tau}=\psi(\chi_\tau)$',xc)
        s+=node(3,3.3,r'$\hat x=[\tilde x^\top\ \bar x^\top]^\top$',r'font=\small,'+xc)
    if build>=5:
        s+=node(6.8,4.7,r'$u_\tau$ recorded',c(5)) + line(6.8,4.4,6.8,4.05,'->,'+c(5))
    if build>=6:
        s+=node(6.8,3.75,r'$g_{\rm aug}$','blk,'+c(6))
        s+=line(4.9,3.75,6.1,3.75,'->,'+c(6))+line(7.5,3.75,10.4,3.75,'->,'+c(6))
        s+=node(12.3,3.75,r'$\bar x_{\tau+1|\tau}^{\rm pred}$',c(6))
        s+=node(9,3.3,'one sample ahead',r'font=\small,'+c(6))
    if build>=7:
        s+=line(12.3,4.15,12.3,4.9,'<->,'+c(7))+node(13.6,4.5,'compare',r'font=\small,'+c(7))
        s+=node(8,2.15,L2,c(7)+r',font=\small')
    if build>=8:
        s+=node(8,1.3,r'$\theta_{\rm enc}\ \text{fixed};\quad \theta_{\rm aug}\ \text{trained};\quad V\ \text{paused}$',c(8))
        s+=r'\draw[->,hl,dashed] (8,2.5) to[bend right=20] (7.2,3.3);'+'\n'
    if build<7:
        eqs=[r'$\chi_\tau=(u,y)_{\tau-30:\tau-1}$',r'$\chi_{\tau+1}=(u,y)_{\tau-29:\tau}$',r'$\bar x_{\tau+1|\tau+1}=\psi_a(\chi_{\tau+1})$',r'$\hat x_{\tau|\tau}=\psi(\chi_\tau)$',r'$u_\tau=u_\tau^{\rm recorded}$',r'$\bar x^{\rm pred}_{\tau+1|\tau}=g_{\rm aug}(\hat x_{\tau|\tau},u_\tau)$']
        s+=node(8,1.75,eqs[build-1],c(build))
    return s+END

def b_s1():
    s=start('2 / S1: each window makes one regression example','Schematic examples A, B, C, not consecutive windows. One row = one training example; targets fixed within one S1 update.')
    for x,t in [(2,'past window'),(6,'added state'),(11,'future residual vector')]:s+=node(x,7.6,t,r'font=\small')
    for i,p in enumerate([0,1.8,3.5]):
        y=6.6-i*1.65
        s+=node(.7,y,'ABC'[i]) + curve(1.2,y,2.4,.3,p,dots=True)
        s+=line(3.8,y,5.2,y,'->')+node(4.5,y+.35,r'$\psi_a$',r'font=\small')+node(6,y,rf'$\bar x_{{{ "ABC"[i] }}}$')
        s+=line(6.8,y,8.1,y,'->')+node(7.5,y+.35,r'$p$',r'font=\small')
        s+=line(8.3,y,13.3,y,'black!55')+curve(8.3,y,5,.35,p,dots=True)+curve(8.3,y,5,.31,p+.15,'dashed,hl')
        s+=node(14,y,r'$e,\ p(\bar x)$',r'font=\small')
    s+=node(8,1.55,L1)+node(8,.85,'One shared encoder and one shared decoder across all rows.',r'font=\small')
    return s+END

def b_s2():
    s=start('2 / S2: the next row supplies the target','Schematic consecutive rows; all encoded states are cached after S1. No decoder and no rollout in the S2 loss.')
    for x,t in [(2,r'$\hat x_{k|k}$ fixed'),(5,r'$u_k$'),(9,'prediction'),(13,'target')]:s+=node(x,7.5,t,r'font=\small')
    for i,k in enumerate([r'\tau',r'\tau+1',r'\tau+2']):
        y=6.4-i*1.6
        kn=[r'\tau+1',r'\tau+2',r'\tau+3'][i]
        s+=node(.7,y,rf'${k}$',r'font=\small')+node(2.4,y,rf'$[\tilde x_{{{k}}},\bar x_{{{k}}}]$','black!55')+node(5,y,rf'$u_{{{k}}}$')
        s+=line(3.7,y+.5,5.7,y+.5,'black!55')+line(5.7,y+.5,5.7,y,'black!55')
        s+=line(5.6,y,7.5,y,'->,hl')+node(6.5,y+.35,r'$g_{\rm aug}$',r'hl,font=\small')
        s+=node(8.9,y,rf'$\bar x^{{\rm pred}}_{{{kn}}}$','hl')+node(10.8,y,r'$\longleftrightarrow$')+node(13,y,rf'$\bar x_{{{kn}}}$','black!55')
        if i<2:
            s+=rf'\draw[->,black!55,dashed] (3.5,{y-1.6+.1}) -- (4,{y-1.6+.75}) -- (12.5,{y-.3});'+'\n'
    s+=node(8,1.4,L2,r'font=\small')+node(8,.8,r'$\chi_{k+1}=(u,y)_{k-29:k};\qquad\bar x_{k+1}=\psi_a(\chi_{k+1})$',r'font=\small')
    return s+END

def c_s1():
    s=start('3 / S1: distinguish histories with different predictable futures','Schematic counterexample. Residuals stand for different conditional means, not two noisy realisations. No physical state coordinates claimed.')
    for i,(p,lab) in enumerate([(0,'A'),(math.pi,'B')]):
        y=6.6-i*2.2
        s+=node(.7,y,lab)+curve(1.2,y,3,.35,p,dots=True)+node(2.7,y-.65,rf'$\chi_{lab}$',r'font=\small')
        s+=line(4.5,y,6.1,y,'->')+node(5.2,y+.35,r'$\psi_a$')+node(6.8,y,rf'$\bar x_{lab}$')
        s+=line(7.6,y,9,y,'->')+node(8.2,y+.35,r'$p$')+curve(9.4,y,4.5,.5,p,dots=True)
        s+=node(14.6,y,rf'$\mu_{lab}$',r'font=\small')
    s+=node(8,3,r'$\mu_A\ne\mu_B,\quad\bar x_A=\bar x_B\ \Longrightarrow\ p(\bar x_A)=p(\bar x_B)$','hl')
    s+=node(8,2.4,'The same state cannot predict both conditional means.',r'font=\small')
    s+=node(8,1.4,L1)
    return s+END

def c_s2():
    s=start('3 / S2: inferred states still need a learned transition','Schematic example at two adjacent windows. The target is inferred from recorded data, not a measured physical added state.')
    s+=node(3,7.2,r'$\chi_\tau=(u,y)_{\tau-30:\tau-1}$')+node(12,7.2,r'$\chi_{\tau+1}=(u,y)_{\tau-29:\tau}$')
    s+=line(3,6.8,3,5.9,'->,black!55')+node(3,6.4,r'$\psi$','black!55,anchor=west')
    s+=line(12,6.8,12,5.9,'->,black!55')+node(12,6.4,r'$\psi_a$ fixed','black!55,anchor=west')
    s+=node(3,5.5,r'$\hat x_{\tau|\tau}$','black!55')+node(12,5.5,r'$\bar x_{\tau+1|\tau+1}$','black!55')
    s+=node(7,5.5,'No transition was fitted by the S1 loss.',r'font=\small')
    s+=line(3,5.1,3,4,'black!55')+line(3,4,6.3,4,'->,hl')
    s+=node(7,4,r'$g_{\rm aug}$','blk,hl')+node(7,4.95,r'$u_\tau$','hl')+line(7,4.6,7,4.4,'->,hl')
    s+=line(7.7,4,10.2,4,'->,hl')+node(12,4,r'$\bar x_{\tau+1|\tau}^{\rm pred}$','hl')
    s+=line(12,4.4,12,5.1,'<->')+node(13.8,4.8,'fit this step',r'font=\small')
    s+=node(8,2.2,L2,r'font=\small')+node(8,1.3,r'$\theta_{\rm enc}$ fixed; $V$ paused; train $\theta_{\rm aug}$.')
    return s+END

finals=[a_s1(),a_s2(),b_s1(),b_s2(),c_s1(),c_s2()]
(OUT/'ps-concepts.tex').write_text(HEADER+''.join(finals)+r'\end{document}',encoding='utf-8')
(OUT/'concept-1-builds.tex').write_text(HEADER+''.join(a_s1(i) for i in range(1,9))+''.join(a_s2(i) for i in range(1,9))+r'\end{document}',encoding='utf-8')
print(OUT)
