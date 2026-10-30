"""Independent thesis formulas on synthetic float64 CPU inputs. No data files or training.

Run: conda run -n GraduationProject python -B <this file>
Each registered name is a forward-map row. Failures remain failures, including known mismatches.
"""
import os
import sys
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'Clean'))
import json
from types import SimpleNamespace as NS
import numpy as np
import torch
from torch import nn
from model_augmentation.systems import gantry_ss as gs
from model_augmentation.fit_systems.blocks import IdentifiableGantryStateBlock as Physics
from model_augmentation.fit_systems.blocks import Linear_Output_Block, Static_ANN_Block
from model_augmentation.fit_systems.pre_encoder import AugmentedEncoder
from model_augmentation.fit_systems.closed_loop import ControllerBank, ClosedLoopSimulator, closed_loop_step, closed_loop_rollout
from model_augmentation.fit_systems.interconnect import AugmentedModel, Interconnect
from model_augmentation.utils.utils import expansion_matrix, selection_matrix
from model_augmentation.fit_systems.obc import OBCBasis, build_stacked_sensitivity
from model_augmentation.fit_systems.ps2_opening import PS2Opening, PS2Spec, _plateaued, calibration_split
from model_augmentation.utils.deepSI_corrections import SystemDataNormalizer, WindowedSystemData
from model_augmentation.utils.torch_nets import AugmentationNetwork
from gantry_dynamic.obc_gantry import fourth_order_central_difference, GantryOBCCorrection, make_reference_field
from gantry_dynamic.data import compute_normalization, servo_sigma
from gantry_dynamic.model import least_squares_start
from gantry_dynamic import controller as ctl

torch.set_num_threads(1)
DT = torch.float64
TESTS = {}
ERRORS = {}
def case(name):
    def register(fn):
        TESTS[name] = fn
        return fn
    return register
def rand(*shape):
    return torch.randn(*shape, dtype=DT)
def close(a, b, rtol=1e-12, atol=1e-12):
    complex_input = torch.as_tensor(a).is_complex() or torch.as_tensor(b).is_complex()
    dtype = torch.complex128 if complex_input else DT
    a, b = torch.as_tensor(a, dtype=dtype), torch.as_tensor(b, dtype=dtype)
    ERRORS[CURRENT] = max(ERRORS.get(CURRENT, 0.0), float((a-b).abs().max()))
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol, check_dtype=False)
def block():
    raw = torch.stack([getattr(gs, n) for n in Physics.PARAM_NAMES]).double()
    return Physics(params_init=raw, combo_init=Physics.combos_of(raw, gs.Lb),
                   Y_op=None, std_x=torch.linspace(.8,1.3,6,dtype=DT),
                   std_u=torch.linspace(.7,1.2,3,dtype=DT), x_mean=rand(6)*.02,
                   u_mean=rand(3)*.02, Ts=.00025, up_sample=1).double()
def matrices(b, free=None):
    c = b.combinations_from_free(b.free_params if free is None else free)
    k,g1,g2,cy,cb,h,m,md,j,d = c
    L=b.Lb
    Y=torch.linspace(-.13,.17,4,dtype=DT)
    z=torch.zeros_like(Y);s=h*0
    M=torch.stack((torch.stack((z+m+h,L/2*md-h*Y,z),-1),
                   torch.stack((L/2*md-h*Y,j+h*(d*d+Y*Y),z-h*d),-1),
                   torch.stack((z,z-h*d,z+h),-1)),-2)
    C=torch.stack((torch.stack((g1+g2,L/2*(g1-g2),s)),
                   torch.stack((L/2*(g1-g2),cb+(g1+g2)*L*L/4,s)),
                   torch.stack((s,s,cy))))
    K=torch.diag(torch.stack((k*0,k,k*0)))
    return c,Y,M,K,C
def physical_rhs(b,x,u,free=None):
    c,_,_,K,C=matrices(b,free)
    k,g1,g2,cy,cb,h,m,md,j,d=c; L=b.Lb
    xp=x.squeeze(-1)*b.std_x_1d+b.x_mean.flatten()
    up=u.squeeze(-1)*b.std_u.flatten()+b.u_mean.flatten()
    Y=xp[:,2]
    z=torch.zeros_like(Y)
    M=torch.stack((torch.stack((z+m+h,L/2*md-h*Y,z),-1),
                   torch.stack((L/2*md-h*Y,j+h*(d*d+Y*Y),z-h*d),-1),
                   torch.stack((z,z-h*d,z+h),-1)),-2)
    a=torch.linalg.solve(M,(up@b.P_mat.T-xp[:,:3]@K.T-xp[:,3:]@C.T).unsqueeze(-1)).squeeze(-1)
    return (torch.cat((xp[:,3:],a),-1)/b.std_x_1d).unsqueeze(-1)
def rk(b,x,u,free=None):
    h=b.Ts/b.up_sample
    for _ in range(b.up_sample):
        k1=physical_rhs(b,x,u,free); k2=physical_rhs(b,x+h*k1/2,u,free)
        k3=physical_rhs(b,x+h*k2/2,u,free); k4=physical_rhs(b,x+h*k3,u,free)
        x=x+h*(k1+2*k2+2*k3+k4)/6
    return x

@case('B_transform')
def test_B_transform():
    L=float(gs.Lb); P=torch.tensor([[1,1,0],[L/2,-L/2,0],[0,0,1]],dtype=DT)
    close(gs.P.double(),P); y=rand(4,3); q=torch.linalg.solve(P.T,y.T).T
    out=Linear_Output_Block(torch.cat((P.T,torch.zeros(3,3,dtype=DT)),1),torch.zeros(3,3,dtype=DT))
    close(out(torch.cat((q,rand(4,6)),1).unsqueeze(-1)).squeeze(-1),y)

@case('B_mass')
def test_B_mass():
    b=block(); _,Y,M,_,_=matrices(b)
    for y,expected in zip(Y,M): close(ctl.mass_matrix_at(float(y)),expected,rtol=2e-7,atol=2e-7)

@case('B_damping')
def test_B_damping():
    b=block(); _,_,_,K,C=matrices(b); km,cm=b._build_KC(b.params_init)
    close(cm,C); close(km,K)

@case('B_stiffness')
def test_B_stiffness(): test_B_damping()

@case('B_mass_split')
def test_B_mass_split():
    b=block(); c,Y,M,_,_=matrices(b); h,m,md,j,d=c[5:]; L=b.Lb
    M0=torch.stack((torch.stack((m+h,L/2*md,h*0)),torch.stack((L/2*md,j+h*d*d,-h*d)),torch.stack((h*0,-h*d,h))))
    M1,M2=b._build_M1M2(h)
    expected1=torch.zeros(3,3,dtype=DT); expected1[0,1]=expected1[1,0]=-h
    expected2=torch.zeros(3,3,dtype=DT); expected2[1,1]=h
    close(M1,expected1);close(M2,expected2)
    close(M0+Y[:,None,None]*M1+Y[:,None,None]**2*M2,M)

@case('B_vector_field')
def test_B_vector_field():
    b=block(); x=rand(4,6,1)*.03;u=rand(4,3,1)
    close(b._deriv_with(x,u,b.pass_mats()),physical_rhs(b,x,u))

@case('B_combinations')
def test_B_combinations():
    b=block();p=b.params_init; L=b.Lb
    expected=torch.stack((p[0]+p[1],p[2],p[3],p[4],p[5]+p[6],p[7],p[8]+p[9]+p[10],p[8]-p[9],p[11]+p[12]+(p[8]+p[9])*L**2/4,p[13]))
    close(Physics.combos_of(p,L),expected)
    close(Physics.combos_of(b.gauge_section(expected,p,L),L),expected)

@case('B_admissibility')
def test_B_admissibility():
    b=block();f=rand(4,10)*.2;c=b.combinations_from_free(f)
    assert bool((c[:,6]*c[:,8]>(b.Lb*c[:,7]/2)**2).all())

@case('B_mdelta')
def test_B_mdelta():
    b=block(); f=rand(4,10)*.2;c0=b.combo_init
    c=c0*torch.exp(f); bound=.99*2/b.Lb*torch.sqrt(c[:,6]*c[:,8])
    b0=.99*2/b.Lb*torch.sqrt(c0[6]*c0[8]);r=c0[7]/b0
    expected=c.clone(); expected[:,7]=bound*torch.tanh(torch.atanh(r)+c0[7]/(b0*(1-r*r))*f[:,7])
    close(b.combinations_from_free(f),expected);close(b.free_from_combinations(expected),f)

@case('A_rk4')
def test_A_rk4():
    b=block();x=rand(4,6,1)*.03;u=rand(4,3,1)
    close(b.nonlinear_function(torch.cat((x,u),1)),rk(b,x,u))

@case('A_normalization')
def test_A_normalization():
    u=rand(40,3).numpy();y=rand(40,3).numpy();sd=WindowedSystemData(u=u,y=y,dt=.00025)
    n=SystemDataNormalizer(u0=np.array([.1,.2,.3]),ustd=np.array([2.,3.,4.]),y0=np.array([.3,.2,.1]),ystd=np.array([4.,3.,2.]))
    got=n.transform(sd);close(got.u,(u-n.u0)/n.ustd);close(got.y,(y-n.y0)/n.ystd)
    cfg=NS(dtype_np=np.float64,nx_phys=6,nu=3)
    norm=compute_normalization(cfg,NS(train_list=[sd]))
    P=gs.P.double().numpy(); q=y@np.linalg.inv(P); v=np.diff(q,axis=0)/sd.dt;v=np.vstack((v[:1],v))
    X=np.hstack((q,v));close(norm.x_mean.ravel(),X.mean(0));close(norm.std_x.ravel(),X.std(0)+1e-8)
    C=np.hstack((P.T,np.zeros((3,3))))
    close(norm.Cd_norm,C*norm.std_x.ravel()/norm.ystd[:,None])

@case('A_norm_transition')
def test_A_norm_transition(): test_A_rk4()

@case('A_norm_output')
def test_A_norm_output():
    u=rand(40,3).numpy();y=rand(40,3).numpy();sd=WindowedSystemData(u=u,y=y,dt=.00025)
    norm=compute_normalization(NS(dtype_np=np.float64,nx_phys=6,nu=3),NS(train_list=[sd]))
    C=np.hstack((gs.P.double().numpy().T,np.zeros((3,3))))
    close(norm.Cd_norm,C*norm.std_x.ravel()/norm.ystd[:,None])
    x=rand(4,6);yp=x@torch.as_tensor(norm.Cd_norm).T
    expected=(x*torch.as_tensor(norm.std_x.ravel())+torch.as_tensor(norm.x_mean.ravel()))@torch.as_tensor(C).T-torch.as_tensor(norm.y0)
    # Mean offset cancellation is numerical when P^-T was computed in float32.
    close(yp,expected/torch.as_tensor(norm.ystd))

def interconnect():
    b=block();ic=Interconnect(8,3,3)
    C=rand(3,6);out=Linear_Output_Block(C,torch.zeros(3,3,dtype=DT))
    ann=Static_ANN_Block(11,8,AugmentationNetwork,5,2,nn.Tanh,{'nx_phys':6}).double()
    # Exercise a nonzero physical correction, beyond the special zero initialisation.
    with torch.no_grad():ann.net.net[-1].weight.copy_(rand(8,5)*.1)
    for item in (b,out,ann):ic.add_block(item)
    ic.connect_block_signals(ann,['x','u'],[])
    ic.connect_signals(ann,'xp','additive',expansion_matrix(range(8),8))
    ic.connect_signals('x',b,'concat',selection_matrix(range(6),8));ic.connect_block_signals(b,['u'],[])
    ic.connect_signals(b,'xp','additive',expansion_matrix(range(6),8))
    ic.connect_signals('x',out,'concat',selection_matrix(range(6),8));ic.connect_block_signals(out,['u'],['y'])
    ic.double();return ic,b,ann,C

@case('A_physical')
def test_A_physical():
    ic,b,ann,C=interconnect();x=rand(4,8)*.02;u=rand(4,3);y,xp=ic(x,u)
    f=ann.net(torch.cat((x,u),1));expected=torch.cat((rk(b,x[:,:6,None],u[:,:,None]).squeeze(-1),torch.zeros(4,2,dtype=DT)),1)+f
    close(xp,expected);close(y,x[:,:6]@C.T)

@case('A_added')
def test_A_added():test_A_physical()

@case('A_network')
def test_A_network():
    ic,b,ann,C=interconnect();z=rand(4,11);v=z
    for layer in ann.net.net:
        v=v@layer.weight.T+layer.bias if isinstance(layer,nn.Linear) else torch.tanh(v)
    close(ann(z[:,:,None]).squeeze(-1),v)

def bank_data():
    A=rand(2,2,2);B=rand(2,2,3);C=rand(2,3,2);D=rand(2,3,3)
    sy=torch.tensor([.8,1.2,1.7],dtype=DT);su=torch.tensor([1.3,.7,1.1],dtype=DT)
    bank=ControllerBank(A,B,C,D,sy,su,dtype=DT)
    return bank,A,B,C,D,sy,su

@case('A_controller_state')
def test_A_controller_state():
    bank,A,B,C,D,sy,su=bank_data();ix=torch.tensor([0,1,0,1]);x=rand(4,2);e=rand(4,3)
    out,xn=bank.step(x,e,(bank.M_state[ix],bank.M_error[ix]))
    close(xn,torch.einsum('bij,bj->bi',A[ix],x)+torch.einsum('bij,bj->bi',B[ix],e*sy))
    close(out,(torch.einsum('bij,bj->bi',C[ix],x)+torch.einsum('bij,bj->bi',D[ix],e*sy))/su)

@case('A_controller_force')
def test_A_controller_force(): test_A_controller_state()

@case('A_cl_step')
def test_A_cl_step():
    bank,A,B,C,D,sy,su=bank_data();x=rand(4,3);xc=rand(4,2);u=rand(4,3);y=rand(4,3);ix=torch.tensor([0,1,0,1])
    class Plant:
        def output_only(self,x):return 2*x
        def __call__(self,x,u,*args): return 2*x,x+u
    got=closed_loop_step(Plant(),bank)(x,xc,u,y,(bank.M_state[ix],bank.M_error[ix]),None,None)
    ep=(y-2*x)*sy
    force=(torch.einsum('bij,bj->bi',C[ix],xc)+torch.einsum('bij,bj->bi',D[ix],ep))/su
    close(got[0],2*x);close(got[1],x+u+force)
    close(got[2],torch.einsum('bij,bj->bi',A[ix],xc)+torch.einsum('bij,bj->bi',B[ix],ep))
    seen=[]
    def observe(x,xc,u,y,ctrl,*args):
        seen.append(xc.clone())
        return x,x,xc+1
    closed_loop_rollout(observe,rand(4,2,3),rand(4,2,3),x,bank,ix)
    close(seen[0],torch.zeros(4,2,dtype=DT));close(seen[1],torch.ones(4,2,dtype=DT))

@case('A_loss')
def test_A_loss():
    target=rand(4,5,3); pred=rand(4,5,3);w=rand(4,3)
    fake=NS(encoder=lambda u,y:rand(4,8),obc=None,objective_matrices=lambda t:{},simulate=lambda *a,**kw:(pred,None))
    got=AugmentedModel.loss(fake,rand(4,2,3),rand(4,2,3),rand(4,5,3),target,loss_w=w)
    expected=sum(((target[i,k,j]-pred[i,k,j])*w[i,j])**2 for i in range(4) for k in range(5) for j in range(3))/(4*5*3)
    close(got,expected)

def encoder(): return AugmentedEncoder(6,3,3,2,2,2,5,2,DT).double()
@case('A_encoder_history')
def test_A_encoder_history():
    sd=WindowedSystemData(u=rand(16,3).numpy(),y=rand(16,3).numpy())
    uh,yh,uf,yf=sd.to_hist_future_data(na=3,nb=3,nf=4,stride=2)
    for i,t in enumerate(range(3,13,2)):
        close(uh[i],sd.u[t-3:t]);close(yh[i],sd.y[t-3:t]);close(uf[i],sd.u[t:t+4]);close(yf[i],sd.y[t:t+4])

@case('A_encoder_linear')
def test_A_encoder_linear():
    e=encoder();u=rand(4,3,3);y=rand(4,3,3)
    chi=torch.cat((y.flatten(1),u.flatten(1)),1)
    W=torch.cat((torch.cat((e.Wb_psi_y,e.Wb_psi_u),1),torch.cat((e.Wa_psi_y,e.Wa_psi_u),1)),0)
    z=torch.cat((u.flatten(1),y.flatten(1)),1)
    for layer in e.net:
        z=z@layer.weight.T+layer.bias if isinstance(layer,nn.Linear) else torch.tanh(z)
    close(e(u,y),chi@W.T+z)

@case('A_encoder_fit')
def test_A_encoder_fit():
    e=encoder();u=rand(32,3,3);y=rand(32,3,3);Phi=torch.cat((y.flatten(1),u.flatten(1),torch.ones(32,1,dtype=DT)),1)
    coef=rand(19,6);target=Phi@coef
    W,b=least_squares_start(e,6,u,y,target)
    close(W,coef[:-1]);close(b,coef[-1]);close(e(u,y)[:,:6],target)

@case('A_encoder_sim')
def test_A_encoder_sim():
    # Exercise target/history assembly with an observable synthetic linear baseline.
    # The separate A_rk4 test verifies the nonlinear baseline transition itself.
    from unittest.mock import patch
    from gantry_dynamic import model as gm
    enc=encoder();I=torch.eye(3,dtype=DT);Z=torch.zeros(3,3,dtype=DT)
    A=torch.cat((torch.cat((I,.1*I),1),torch.cat((Z,.9*I),1)),0)
    B=torch.cat((Z,I),0)
    class H(nn.Module):
        def output_only(self,x):return x[:,:3]
        def forward(self,x,u,*args):
            return x[:,:3],torch.cat((x[:,:6]@A.T+u@B.T,torch.zeros(len(x),2,dtype=DT)),1)
    bank=ControllerBank(np.zeros((1,1,1)),np.zeros((1,1,3)),np.zeros((1,3,1)),np.zeros((1,3,3)),np.ones(3),np.ones(3),DT)
    fs=NS(encoder=enc,hfn=H(),na=3,nb=3,norm=NS(u0=np.zeros(3),ustd=np.ones(3),y0=np.zeros(3),ystd=np.ones(3)),
          simulator=NS(bank=bank,train_ctrl_rows=[0]),_sync_obc_coefficient=lambda:None,objective_matrices=lambda:{})
    sd=WindowedSystemData(u=rand(40,3).numpy(),y=rand(40,3).numpy())
    captured=[]
    def inspect_fit(e,nx,u,y,target):
        # With held inputs: q_k=q_{k-1}+.1*v_{k-1},
        # v_{k-1}=(q_{k-1}-q_{k-2})/.1*.9+u_{k-2}.
        vprev=(y[:,-1]-y[:,-2])/.1*.9+u[:,-2]
        expected=torch.cat((y[:,-1]+.1*vprev,.9*vprev+u[:,-1]),1)
        close(target,expected);captured.append(len(target))
        least_squares_start(e,nx,u,y,target)
    cfg=NS(device='cpu',nx_phys=6,nx_ann=2,stride=1)
    norm=NS(x_mean=np.zeros(6),std_x=np.ones(6))
    with patch.object(gm,'least_squares_start',side_effect=inspect_fit):
        gm.prefit_physical_encoder(fs,cfg,NS(train_data=NS(sdl=[sd])),norm)
    assert captured==[37]

@case('A_init_beta')
def test_A_init_beta():
    b=block();close(b.free_params,torch.zeros(10,dtype=DT));close(b.recover_combinations(),b.combo_init)

@case('A_init_network')
def test_A_init_network():
    n=AugmentationNetwork(11,8,5,2,nn.Tanh,6).double();close(n(rand(4,11))[:,:6],torch.zeros(4,6,dtype=DT))
    assert bool((n.net[-1].weight[6:].abs()<=1/np.sqrt(5)).all());close(n.net[-1].bias,torch.zeros(8,dtype=DT))

@case('A_init_encoder')
def test_A_init_encoder():
    e=encoder();bound=np.sqrt(6/(2+18));assert bool((torch.cat((e.Wa_psi_y,e.Wa_psi_u),1).abs()<=bound).all())
    close(e.net[-1].weight[6:],torch.zeros(2,5,dtype=DT));close(e.net[-1].bias[6:],torch.zeros(2,dtype=DT))

@case('P_residual')
def test_P_residual():
    p=object.__new__(PS2Opening);p._mats={};p.n_xb=6;p.n_a=2
    yf=rand(4,3,3);yp=rand(4,3,3);b={'uf':rand(4,3,3),'yf':yf,'ctrl_ix':torch.zeros(4,dtype=torch.long)}
    got=p._residual(NS(simulate=lambda *a,**k:(yp,None)),b,rand(4,8))
    err=yf-yp;close(got,err.flatten(1)/torch.sqrt(torch.sum(err**2)/err.numel()))

@case('P_s1_loss')
def test_P_s1_loss():
    p=object.__new__(PS2Opening);p.n_xb=6;p.n_a=2;p._cal={};p._decoder=nn.Linear(2,9).double()
    x=rand(4,8);target=rand(4,9);p._residual=lambda *a:target
    fs=NS(encoder=lambda *a:x);p._cal={'uh':rand(4,3,3),'yh':rand(4,3,3)}
    close(p._cal_unexplained(fs),torch.sum((p._decoder(x[:,6:])-target)**2)/torch.sum(target**2))
    # Capture the gradients of the actual S1 loss, without an optimizer update.
    p.fit_ix=[0];p.spec=PS2Spec();p._rng=np.random.default_rng(284)
    p._sample=lambda *args,**kwargs:p._cal
    p._aux=list(p._decoder.parameters())
    expected=torch.autograd.grad(((p._decoder(x[:,6:])-target)**2).sum()/target.numel(),p._aux)
    def inspect_step():
        for param,want in zip(p._aux,expected):close(param.grad,want)
    p._opt=NS(step=inspect_step)
    p._s1_step(fs)

@case('P_pairs')
def test_P_pairs():
    p=object.__new__(PS2Opening);p._M=3;p._dev='cpu';p._dt=DT;p.ctrl_rows=[0]
    u=rand(30,3).numpy();y=rand(30,3).numpy();p.records=[(u,y)]
    b=p._sample(NS(na=3,nb=3),[0],4,np.random.default_rng(20),pair=True)
    for i in range(4):
        k=np.where((u==b['u0'][i].numpy()).all(1))[0][0]
        close(b['uh'][i],u[k-3:k]);close(b['yh'][i],y[k-3:k]);close(b['uh1'][i],u[k-2:k+1]);close(b['yh1'][i],y[k-2:k+1])

@case('P_s2_cost')
def test_P_s2_cost():
    # Zero update budget: evaluates the real nested state_map_loss, without any training.
    p=object.__new__(PS2Opening);p.spec=PS2Spec(s2_cap=0,n_snap=8,n_snap_cal=4);p.fit_ix=[0];p.cal_ix=[1]
    p.n_xb=6;p.n_a=2;p._added_cols=[6,7]
    ann=nn.Linear(11,8).double()
    class ANN(nn.Module):
        def __init__(self):super().__init__();self.net=ann
        def forward(self,z):return self.net(z.squeeze(-1)).unsqueeze(-1)
    fs=NS(hfn=NS(connected_blocks=[ANN()]),_ann_ix=0,encoder=lambda u,y:u)
    X,U,T=rand(4,8),rand(4,3),rand(4,2)
    p._sample=lambda *a,**kw:dict(uh=X,yh=X,u0=U,uh1=torch.cat((rand(4,6),T),1),yh1=X)
    out=p._fit_added_state_transition(fs)
    # Merged map MM3: the code divides by the sample variance, so its cost is the thesis quotient times (B - 1) / B.
    expected=((ann(torch.cat((X,U),1))[:,6:]-T)**2).sum()/((T-T.mean(0))**2).sum()*(len(T)-1)/len(T)
    close(out['residual_start'],expected)

@case('O_sensitivity')
def test_O_sensitivity():
    b=block();v=rand(10)*.01;z=rand(4,9,1)*.03
    actual=build_stacked_sensitivity(b.transition_from_free,v,z,range(6),append_value=True)
    direct=torch.func.jacfwd(lambda f:rk(b,z[:,:6],z[:,6:],f))(v)[:,:,0,:].reshape(-1,10)
    close(actual[:,:10],direct);close(actual[:,10],rk(b,z[:,:6],z[:,6:],v).flatten())

@case('O_offset')
def test_O_offset():
    Phi=rand(24,10);f=rand(24);beta=rand(10);gamma=f-Phi@beta;F=rand(24)
    J=torch.cat((Phi,f[:,None]),1);G=torch.cat((Phi,gamma[:,None]),1)
    a=OBCBasis(J,beta).coefficient(F); expected=torch.linalg.solve(G.T@G,G.T@F)
    close(torch.cat((a[:-1]+beta*a[-1],a[-1:])),expected);close(J@a,G@expected)

@case('O_difference')
def test_O_difference():
    q=rand(15,3).numpy();ts=.03
    expected=np.stack([(-q[i+2]+8*q[i+1]-8*q[i-1]+q[i-2])/(12*ts) for i in range(2,13)])
    close(fourth_order_central_difference(q,ts),expected)

@case('O_coefficient')
def test_O_coefficient():
    J=rand(24,11);F=rand(24);theta=OBCBasis(J,rand(10)).coefficient(F)
    close(theta,torch.linalg.solve(J.T@J,J.T@F));close(J.T@(F-J@theta),torch.zeros(11,dtype=DT))

@case('O_correction')
def test_O_correction():
    b=block();corr=GantryOBCCorrection(b,8,6,range(8),DT);v=rand(10)*.01;theta=rand(11)*.01
    corr.set_frozen(theta,v);x=rand(4,8,1)*.02;u=rand(4,3,1)
    f=rk(b,x[:,:6],u,v);J=torch.func.jacfwd(lambda t:rk(b,x[:,:6],u,t))(v)
    expected=torch.cat((torch.einsum('bikp,p->bik',J,theta[:-1])+theta[-1]*f,torch.zeros(4,2,1,dtype=DT)),1)
    close(corr(x,u),expected)

@case('O_field')
def test_O_field():
    ic,b,ann,C=interconnect();z=rand(4,11,1);z[:,6:8]=0
    got=make_reference_field(NS(Z_field=z),range(6))(ann)
    expected=ann.net(z.squeeze(-1))[:,:6].reshape(-1)
    close(got,expected)

@case('O_tuple_position')
def test_O_tuple_position():
    # Supply synthetic records in memory, without opening any data file.
    from unittest.mock import patch
    from gantry_dynamic import obc_gantry as og
    y=rand(15,3).numpy();u=rand(15,3).numpy()
    norm=NS(x_mean=rand(6).numpy(),std_x=np.linspace(.8,1.3,6),u_mean=rand(3).numpy(),std_u=np.linspace(.7,1.2,3))
    cfg=NS(ts_new=.03,dtype_pt=DT)
    with patch.object(og,'loadmat',return_value={}),patch.object(og,'traj_dir',return_value='synthetic'),patch.object(og,'_record_at_fs_new',return_value=(u,y)):
        ref=og.build_reference_set(cfg,norm,2,['memory'],verbose=False)
    P=gs.P.double().numpy();q=np.linalg.solve(P.T,y.T).T
    v=np.stack([(-q[i+2]+8*q[i+1]-8*q[i-1]+q[i-2])/(12*.03) for i in range(2,13)])
    k=np.arange(2,12);x=np.hstack((q[k],v[k-2]));xn=np.hstack((q[k+1],v[k-1]))
    expected=np.hstack(((x-norm.x_mean)/norm.std_x,(u[k]-norm.u_mean)/norm.std_u))
    close(ref.Z_step.squeeze(-1),expected);close(ref.x_phys_next_rec,xn)
    close(ref.Z_field[:,6:8],torch.zeros(10,2,1,dtype=DT))

@case('O_physical_step')
def test_O_physical_step():
    ic,b,ann,C=interconnect();corr=GantryOBCCorrection(b,8,6,range(8),DT)
    v=rand(10)*.01;theta=rand(11)*.01;corr.set_frozen(theta,v);ic.obc_correction=corr
    x=rand(4,8)*.02;u=rand(4,3)
    f=rk(b,x[:,:6,None],u[:,:,None],v)
    J=torch.func.jacfwd(lambda t:rk(b,x[:,:6,None],u[:,:,None],t))(v)
    removed=torch.einsum('bikp,p->bi',J,theta[:-1])+theta[-1]*f.squeeze(-1)
    raw=ann.net(torch.cat((x,u),1))
    expected=rk(b,x[:,:6,None],u[:,:,None]).squeeze(-1)+raw[:,:6]-removed
    y,xp=ic(x,u)
    close(xp[:,:6],expected);close(xp[:,6:],raw[:,6:])

@case('L_blocks')
def test_L_blocks():
    b=block();m=b.pass_mats();K,C,A,h,alpha,beta,gamma,N0,N1,N2=m
    R=torch.linalg.inv(torch.tensor([[float(alpha),float(beta),0],[float(beta),float(gamma+h*b.params_init[13]**2),float(-h*b.params_init[13])],[0,float(-h*b.params_init[13]),float(h)]],dtype=DT))
    M1,M2=b._build_M1M2(h);Z=torch.zeros(3,3,dtype=DT);I=torch.eye(3,dtype=DT)
    expected=torch.cat((torch.cat((Z,I,Z,Z,Z),1),torch.cat((-R@K,-R@C,-R@M1,-R@M2,R),1)),0)
    close(A,expected)

@case('L_chain')
def test_L_chain(): test_B_vector_field()

@case('L_det')
def test_L_det():
    b=block();c,Y,M,_,_=matrices(b);h,m,md,j,d=c[5:];a=b.Lb*md/2
    expected=h*(m*h*(Y+a/m)**2+(m+h)*(j-a*a/m))
    close(torch.linalg.det(M),expected)
    _,_,_,mh,alpha,beta,gamma,*_=b.pass_mats()
    close(mh*(alpha*gamma-beta**2+2*beta*mh*Y+mh*(alpha-mh)*Y**2),expected)

@case('L_eta')
def test_L_eta(): test_B_mdelta()

@case('L_entries')
def test_L_entries(): test_B_mass_split();test_B_damping();test_B_combinations()

@case('S_metric')
def test_S_metric():
    from model_augmentation.fit_systems.closed_loop import closed_loop_free_run_rms_batch
    # Real metric callable, short recurrence and a synthetic encoder, no optimiser.
    class H(nn.Module):
        def __init__(self):super().__init__();self.p=nn.Parameter(torch.zeros(1,dtype=DT))
        def output_only(self,x):return x[:,:3]
        def forward(self,x,u,*args):return self.output_only(x),x
    bank=ControllerBank(np.zeros((1,1,1)),np.zeros((1,1,3)),np.zeros((1,3,1)),np.zeros((1,3,3)),np.ones(3),np.ones(3),DT)
    y=rand(12,3).numpy();sd=WindowedSystemData(u=rand(12,3).numpy(),y=y)
    fs=NS(norm=NS(u0=np.zeros(3),ustd=np.ones(3),y0=np.zeros(3),ystd=np.ones(3)),na=3,nb=3,hfn=H(),encoder=lambda u,y:torch.zeros(len(u),6,dtype=DT),objective_matrices=lambda:{})
    got,agg=closed_loop_free_run_rms_batch(fs,[sd],bank,[0])
    close(got[0],np.sqrt((y[3:]**2).sum(0)/9));close(servo_sigma([y],3)[0],np.std(y[3:],axis=0))
    sigma=np.std(y[3:],axis=0)
    sim=ClosedLoopSimulator(bank,[],[0],None,[],sigma[None])
    close(sim.validation_error(fs,NS(sdl=[sd])),np.mean(np.sqrt((y[3:]**2).mean(0))/sigma))

@case('S_controller')
def test_S_controller():
    num,den=ctl.normalized_controller_coefficients();s=rand(4).numpy()*1j*100
    expected=10*ctl.W*(s+ctl.W/6)*(s+ctl.W/3)/(s*(s+3*ctl.W)*(s+10*ctl.W))
    close(np.polyval(num,s)/np.polyval(den,s),expected)
    Y=.07;ts=.00025
    G=ctl.P.T@np.linalg.solve((1j*ctl.W)**2*ctl.mass_matrix_at(Y)+1j*ctl.W*ctl.C_DAMP+ctl.K_STIFF,ctl.P)
    close(ctl.stage_frequency_response(Y,1j*ctl.W),G)
    A,B,C,D=ctl.controller_ss(Y,ts)
    z=np.exp(1j*.17);sb=2/ts*(z-1)/(z+1)
    shape=lambda s:10*ctl.W*(s+ctl.W/6)*(s+ctl.W/3)/(s*(s+3*ctl.W)*(s+10*ctl.W))
    gains=1/np.abs(np.diag(G)*shape(1j*ctl.W))
    # Expanded discrete polynomial coefficients followed by tf2ss are less
    # well conditioned than the factored continuous formula at mapped s.
    close(C@np.linalg.solve(z*np.eye(len(A))-A,B)+D,np.diag(gains*shape(sb)),rtol=2e-9,atol=1e-11)

@case('P_stops')
def test_P_stops():
    assert _plateaued([1,.7,.69,.69,.69],3,.02)
    assert calibration_split(['A1','A2','A3','B1','B2'])==([0,1,3,4],[2])

# Distinct map rows with identical formulas get distinct, traceable test names.
ALIASES = {
    'B_state':'B_vector_field', 'A_output':'B_transform',
    'A_norm_inverse':'B_vector_field', 'A_coordinates':'B_mdelta',
    'A_model_loop_state':'A_controller_state', 'A_model_loop_force':'A_cl_step',
    'A_residual_state':'A_cl_step', 'A_residual_force':'A_cl_step',
    'A_cl_output':'A_cl_step', 'A_cl_input':'A_cl_step',
    'A_loss_physical':'A_physical', 'A_loss_added':'A_added',
    'A_loss_initial':'A_encoder_linear',
    'O_basis':'O_sensitivity', 'O_projected':'O_physical_step',
    'O_orthogonality':'O_coefficient', 'O_added_step':'O_physical_step',
    'L_matrices':'B_mass_split', 'L_factorization':'B_vector_field',
    'L_constant':'B_vector_field', 'L_delta':'L_chain',
    'L_G_state':'L_blocks', 'L_G_latent':'L_chain', 'L_G_output':'B_transform',
    'L_reduced_mass':'B_mass_split',
}
for alias, target in ALIASES.items():
    def test_alias(fn=TESTS[target]):
        fn()
    test_alias.__name__='test_'+alias
    globals()[test_alias.__name__]=test_alias
    TESTS[alias]=test_alias

if __name__ == '__main__':
    results=[]
    for CURRENT,fn in TESTS.items():
        torch.manual_seed(284);np.random.seed(284)
        try:
            fn();status='pass';detail=''
        except Exception as exc:
            status='fail';detail=type(exc).__name__+': '+str(exc).splitlines()[0]
        results.append(dict(row=CURRENT,status=status,max_abs=ERRORS.get(CURRENT),detail=detail))
    print('EQUATION_RESULTS='+json.dumps(results))
    print('%d passed, %d failed' % (sum(r['status']=='pass' for r in results),sum(r['status']=='fail' for r in results)))
    sys.exit(any(r['status']=='fail' for r in results))
