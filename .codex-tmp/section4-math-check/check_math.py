import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from model_augmentation.fit_systems.blocks import Reduced_Gantry_State_Block as Block
from model_augmentation.systems import gantry_ss as gss

torch.set_default_dtype(torch.float64)
torch.manual_seed(4)
nom=torch.stack([getattr(gss,n) for n in Block.PARAM_NAMES]).double()
combo=Block.combos_of(nom,gss.Lb)
sx=np.array([.13,.012,.2,.35,.04,.3]).reshape(6,1)
su=np.array([30.,40.,20.]).reshape(3,1)
mx=np.array([.01,.003,.08,.02,-.01,.03]).reshape(6,1)
mu=np.array([1.,-2.,3.]).reshape(3,1)
blk=Block(params_init=nom,combo_init=combo,Y_op=None,std_x=sx,std_u=su,
          x_mean=mx,u_mean=mu,Ts=1/4000,up_sample=1,flag_loss_reg=False).double()
v=torch.randn(10)*.06
z=torch.randn(4,9,1)*.2
direction=torch.randn(10)

def mck(v,q):
 c=blk.combinations_from_free(v)
 kb,cg1,cg2,cy,cb,mh,ms,md,je,d=c
 lb=blk.Lb.double(); zero=c.new_zeros(())
 M=torch.stack([torch.stack([ms+mh,lb*md/2-mh*q[2],zero]),
                torch.stack([lb*md/2-mh*q[2],je+mh*(d*d+q[2]*q[2]),-mh*d]),
                torch.stack([zero,-mh*d,mh])])
 C=torch.stack([torch.stack([cg1+cg2,lb*(cg1-cg2)/2,zero]),
                torch.stack([lb*(cg1-cg2)/2,cb+lb*lb*(cg1+cg2)/4,zero]),
                torch.stack([zero,zero,cy])])
 K=torch.diag(torch.stack([zero,kb,zero]))
 return M,C,K

S=torch.tensor(sx).flatten(); Xmean=torch.tensor(mx).flatten()
Ustd=torch.tensor(su).flatten(); Umean=torch.tensor(mu).flatten()
P=blk.P_mat.double()
def ct(v,xn,un):
 x=xn*S+Xmean; u=un*Ustd+Umean
 M,C,K=mck(v,x[:3])
 a=torch.linalg.solve(M,P@u-C@x[3:]-K@x[:3])
 return torch.cat([x[3:],a])/S
def rk(v,z):
 x=z[:6,0]; u=z[6:,0]; h=1/4000
 k1=ct(v,x,u); k2=ct(v,x+h*k1/2,u)
 k3=ct(v,x+h*k2/2,u); k4=ct(v,x+h*k3,u)
 return (x+h*(k1+2*k2+2*k3+k4)/6).unsqueeze(-1)
manual=torch.stack([rk(v,zi) for zi in z])
actual=blk.transition_from_free(v,z)
rel=lambda a,b: float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b))
print('RK4 from explicit M,C,K versus code, relative error:',rel(manual,actual),flush=True)
Jmanual=torch.func.jacfwd(lambda vv: torch.stack([rk(vv,zi) for zi in z]))(v)
Jcode=torch.func.jacfwd(lambda vv: blk.transition_from_free(vv,z))(v)
print('NORMALISED RK4 Jacobian versus code, relative error:',rel(Jmanual,Jcode),flush=True)
_,tangent=blk.transition_and_tangent_from_free(v,direction,z)
_,oracle=torch.func.jvp(lambda vv: blk.transition_from_free(vv,z),(v,),(direction,))
print('Hand-written tangent versus JVP, relative error:',rel(tangent,oracle),flush=True)

# Check the displayed fixed-state acceleration derivative independently.
x=z[0,:6,0]*S+Xmean; u=z[0,6:,0]*Ustd+Umean
M,C,K=mck(v,x[:3]); a=torch.linalg.solve(M,P@u-C@x[3:]-K@x[:3])
dM,dC,dK=torch.func.jacfwd(lambda vv: mck(vv,x[:3]))(v)
rhs=torch.stack([-torch.linalg.solve(M,dM[:,:,j]@a+dC[:,:,j]@x[3:]+dK[:,:,j]@x[:3]) for j in range(10)],dim=1)
ctJ=torch.func.jacfwd(lambda vv: ct(vv,z[0,:6,0],z[0,6:,0]))(v)
print('CT acceleration formula with T_x, relative error:',rel(rhs/S[3:,None],ctJ[3:]),flush=True)

# Changing the origin while keeping physical tuples fixed leaves the Jacobian unchanged.
shift=torch.tensor([.008,-.001,.03,.002,.004,-.02]).reshape(1,6,1)
oldmean=blk.x_mean.clone(); oldz=z.clone()
Jold=Jcode.clone(); f_old=actual.clone()
blk.x_mean.add_(shift[0]); newz=z.clone(); newz[:,:6]-=shift/blk.std_x
Jnew=torch.func.jacfwd(lambda vv: blk.transition_from_free(vv,newz))(v)
f_new=blk.transition_from_free(v,newz)
print('Origin-shift Jacobian invariance, relative error:',rel(Jnew,Jold),flush=True)
print('Origin-shift offset identity max residual:',float((f_new-f_old+shift/blk.std_x).abs().max()),flush=True)
blk.x_mean.copy_(oldmean)

# Determinant, including the bound dependence on the mass and inertia coordinates.
maxerr=0.
for scale in [0.,.1,1.,3.]:
 vv=torch.randn(10)*scale
 cc=blk.combinations_from_free(vv)
 JJ=torch.func.jacfwd(blk.combinations_from_free)(vv)
 arg=blk.m_diff_eta0+blk.m_diff_scale*vv[7]
 direct=blk._m_diff_bound(cc)*blk.m_diff_scale/(torch.cosh(arg)**2)
 predicted=cc[torch.arange(10)!=7].prod()*direct
 maxerr=max(maxerr,float(abs(torch.linalg.det(JJ)-predicted)/abs(predicted)))
print('Coordinate determinant formula max relative error:',maxerr,flush=True)

# Projection, gradient and recovery on a common stack.
Phi=Jcode.reshape(-1,10).detach(); plus=torch.linalg.pinv(Phi)
F=torch.randn(Phi.shape[0]); eps=torch.randn_like(F)*.01
H=F-Phi@(plus@F)
print('Projection orthogonality relative residual:',float(torch.linalg.vector_norm(Phi.T@H)/(torch.linalg.vector_norm(Phi)*torch.linalg.vector_norm(F))),flush=True)
betastar=torch.randn(10)*.01
target=Phi@betastar+eps
beta_hat=torch.linalg.lstsq(Phi,target-H).solution
print('Common-stack corrected recovery max error:',float(abs(beta_hat-(betastar+plus@eps)).max()),flush=True)
beta_plain=torch.linalg.lstsq(Phi,target-F).solution
print('Uncorrected recovery max error:',float(abs(beta_plain-(betastar+plus@(eps-F))).max()),flush=True)
Ar=torch.randn(60,10); Fr=torch.randn(60); er=torch.randn(60)*.01
pr=torch.linalg.pinv(Ar); hr=Fr-Ar@(pr@Fr)
br=torch.linalg.lstsq(Ar,Ar@betastar+er-hr).solution
print('Well-conditioned common-stack recovery max error:',float(abs(br-(betastar+pr@er)).max()),flush=True)

# Taylor remainder behind the nonlinear first-order recovery statement.
for t in [1e-1,1e-2,1e-3]:
 vb=v+t*direction
 fb=blk.transition_from_free(vb,z)
 remainder=fb-actual-t*oracle
 print('RK4 Taylor remainder at scale',t,':',float(remainder.norm()),flush=True)

gamma=actual.reshape(-1)-Phi@v
print('Offset span: remainder of Gamma-fbar outside range(Phi):',float(torch.linalg.vector_norm((gamma-actual.reshape(-1))-Phi@(plus@(gamma-actual.reshape(-1))))),flush=True)

w=torch.tensor([1.,-8.,0.,8.,-1.])/12
nodes=torch.arange(-2,3,dtype=torch.float64)
print('Fourth-order stencil moments 0 through 4:',[float(w@(nodes**k)) for k in range(5)],flush=True)

# Transparent exact counterexample for changing the coefficient stack.
A=torch.tensor([[1.],[2.],[3.]])
f=torch.tensor([1.,0.,0.]); A_sub=A[:2]; f_sub=f[:2]
a_all=(torch.linalg.pinv(A)@f).item()
r_sub=f_sub-A_sub[:,0]*a_all
offset=-(torch.linalg.pinv(A_sub)@r_sub).item()
print('Subset example: theta_aux_I=1/14 =',a_all,'; beta_bias with eps=0 =',offset,'(expected -9/70)',flush=True)
print('Subset example: Phi_sub^T corrected_sub =',(A_sub.T@r_sub).item(),flush=True)
rank_all=torch.linalg.matrix_rank(torch.tensor([[1.,0.],[0.,1.]]))
rank_sub=torch.linalg.matrix_rank(torch.tensor([[1.,0.]]))
print('Rank implication counterexample:',int(rank_all),'full,',int(rank_sub),'subset',flush=True)

# Corrected learning-field derivative at fixed tuples is I-P, not I.
Q=torch.eye(Phi.shape[0])-Phi@plus
dF=torch.randn(Phi.shape[0],5)
print('Differentiated coefficient: Phi^T dH residual:',float(torch.linalg.vector_norm(Phi.T@Q@dF)),flush=True)
print('Detached coefficient: Phi^T dF norm:',float(torch.linalg.vector_norm(Phi.T@dF)),flush=True)

# The evaluated correction still depends on states when its expansion point is detached.
xr=z[:,:6].clone().requires_grad_(); ur=z[:,6:].clone().requires_grad_()
_,tr=blk.transition_and_tangent_from_free(v.detach(),direction.detach(),torch.cat([xr,ur],1))
gx,gu=torch.autograd.grad(tr.sum(),(xr,ur))
print('Correction gradients through x,u with detached expansion point:',float(gx.norm()),float(gu.norm()),flush=True)
