"""Read-only mathematical checks for the restructured Section IV."""
import ast
from pathlib import Path
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(15)

def left_inverse(A):
    return torch.linalg.solve(A.T @ A, A.T)

def cos(a, b):
    return torch.dot(a, b) / (a.norm() * b.norm())

A = torch.randn(36, 10)
L = left_inverse(A)
P = A @ L
d = torch.randn(36)
F = torch.randn(36)
H = F - P @ F
rho = (P @ d).norm() / d.norm()
print('projector symmetry max error:',float((P-P.T).abs().max()),flush=True)
print('projector idempotence max error:',float((P@P-P).abs().max()),flush=True)
print('normal-equation orthogonality norm:',float((A.T@H).norm()),flush=True)
print('max identity at v=Pi d error:',float(abs(cos(P@d,d)-rho)),flush=True)
V = torch.randn(2000,10) @ A.T
sample_cos = (V@d)/(V.norm(dim=1)*d.norm())
print('all sampled cosines <= rho:',bool((sample_cos <= rho+1e-13).all()),flush=True)
col=A[:,0]
print('signed cosine anti-aligned:',float(cos(col,-col)),'rho anti-aligned:',float((P@(-col)).norm()/col.norm()),flush=True)
perp=d-P@d
print('rho perpendicular:',float((P@perp).norm()/perp.norm()),flush=True)
print('rho in range:',float((P@col).norm()/col.norm()),flush=True)

# Execute only the actual standalone overlap function, without loading runs.
path=Path('Thesis-writeup/Code/Evaluation/eval_overlap.py')
tree=ast.parse(path.read_text(encoding='utf-8'))
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='overlap')
ns={'torch':torch}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
print('actual eval_overlap.overlap at zero:',ns['overlap'](A,torch.zeros(36)),flush=True)
print('actual evaluator versus explicit Gram projector:',abs(ns['overlap'](A,d)-float(rho)),flush=True)

# Same-stack first-order recovery, with and without projection.
beta_star=torch.randn(10)
eps=torch.randn(36)*.01
y=A@beta_star+eps
bc=L@(y-H)
bu=L@(y-F)
print('corrected recovery max error:',float((bc-beta_star-L@eps).abs().max()),flush=True)
print('uncorrected recovery max error:',float((bu-beta_star-L@(eps-F)).abs().max()),flush=True)

# Stacked orthogonality does not imply orthogonality at each tuple.
Ap=torch.tensor([[1.],[1.]])
Hp=torch.tensor([1.,-1.])
print('stacked-vs-pointwise example:',(Ap.T@Hp).tolist(),(Ap[:,0]*Hp).tolist(),flush=True)

# Exact nonlinear one-step objective, same tangent and zero true discrepancy.
# b(beta)=(beta,beta^2), tangent at 0=(1,0), H=(0,-1).
for beta in [0.,-2**-.5,2**-.5]:
    loss=beta**2+(beta**2-1)**2
    print('nonlinear corrected objective beta/loss:',beta,loss,flush=True)

# Independence of a network parameter needs constant projected contribution,
# not necessarily a zero contribution: Phi=(1,0), H(eta)=(1,eta).
print('nonorthogonal H(eta) example: beta_min=-1 for every eta, Phi^T H=1',flush=True)

# A distinct auxiliary set does not transfer orthogonality to the fitted set.
# phi(x)=1, F_eta(x)=eta*x; auxiliary x=0, training x=1, training y=1.
for eta in [0.,.25,1.]:
    theta_aux=eta*0.
    theta_b=1.-eta
    train_prediction=theta_b+eta-theta_aux
    print('auxiliary-set example eta/beta/train prediction:',eta,theta_b,train_prediction,flush=True)

# The extra protected direction is the residual of the baseline vector.
Ag=torch.tensor([[1.],[0.],[0.]])
bg=torch.tensor([1.,1.,0.])
Pg=Ag@left_inverse(Ag)
w=(torch.eye(3)-Pg)@bg
Ag_ext=torch.column_stack((Ag,bg))
Pg_ext=Ag_ext@left_inverse(Ag_ext)
fg=torch.tensor([0.,1.,1.])
print('offset: baseline=',bg.tolist(),'new orthogonal direction=',w.tolist(),flush=True)
print('offset: a nonuniform field changes from/to:',((torch.eye(3)-Pg)@fg).tolist(),((torch.eye(3)-Pg_ext)@fg).tolist(),flush=True)
print('offset: physical estimate change in fixed linear surrogate:',float((left_inverse(Ag)@(Pg_ext-Pg)@fg).norm()),flush=True)

# Global identifiability plus an invertible coordinate does not imply a regular tangent.
# f_theta(x)=(theta-1)^3*x, theta>0, at theta=1.
print('identifiable cubic family at theta=1: derivative is zero for every x',flush=True)
