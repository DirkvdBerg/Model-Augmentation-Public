"""Exact algebra checks and rollout derivative checks for gaps.md Section 9.

Category: structural verification on abstract examples; NOT gantry identification.
The all-dimensional proofs live in gaps.md. Finite examples cross-check them,
not prove their quantifiers. MATLAB independently implements the same examples.
Run with the GraduationProject environment. No training data or truth required.
"""
import json
from pathlib import Path

import numpy as np
import sympy as s
import torch

HERE = Path(__file__).resolve().parent
CHECKS = {}


def check(name, value):
    value = bool(value)
    CHECKS[name] = value
    assert value, name


def zero(x):
    return all(s.simplify(v) == 0 for v in x)


def projector(x):
    return x * x.pinv()


def algebra():
    a, b = s.symbols('a b', real=True)
    eye = s.eye(5)
    L = s.diag(2, 3, 4, 5, 6)
    E = L.inv() * eye[:, 4]
    S = L.inv() * s.Matrix.hstack(eye[:, 0] + eye[:, 4], eye[:, 1])
    R = L.inv() * s.Matrix.hstack(a*eye[:, 0]+eye[:, 2]+eye[:, 4],
                                 b*eye[:, 1]+eye[:, 3])
    M = eye - projector(L*E)
    Sb, Rb = M*L*S, M*L*R
    PR = Rb*(Rb.T*Rb).inv()*Rb.T
    PN = projector(L*E) + PR
    H = s.simplify(Sb.T*(eye-PR)*Sb)
    expected = s.diag(1/(1+a*a), 1/(1+b*b))
    check('weighted_profiled_curvature', zero(H-expected))
    N = L*s.Matrix.hstack(E, R)
    directPN = N*(N.T*N).inv()*N.T
    check('sequential_equals_joint_projection', zero(PN-directPN))
    check('projector_symmetric_idempotent', zero(PN-PN.T) and zero(PN*PN-PN))
    check('nonorthogonal_can_be_identifiable', H.subs({a:1,b:2}).det()>0
          and not zero(Sb.T*Rb.subs({a:1,b:2})))
    check('orthogonal_limit_preserves_curvature', zero(H.subs({a:0,b:0})-Sb.T*Sb))
    redundant = s.Matrix.hstack(Rb.subs({a:1,b:2}), 2*Rb.subs({a:1,b:2}))
    check('redundant_neural_columns_allowed', zero(projector(redundant)-PR.subs({a:1,b:2})))
    D = s.Matrix(s.symbols('d0:5', real=True))
    residual = s.Matrix(s.symbols('e0:5', real=True))
    Db = M*L*D
    before = (Sb.T*Sb).inv()*Sb.T*M*L*residual
    after = (Sb.T*Sb).inv()*Sb.T*M*L*(residual-D)
    check('fixed_contribution_displacement', zero(after-before+Sb.pinv()*Db))
    check('displacement_normal_equation', zero(Sb.T*Sb*(after-before)+Sb.T*Db))
    # Encoder freedom closes a cancellation route absent from S and R separately.
    e1, e2 = s.eye(3)[:,0], s.eye(3)[:,1]
    e = e1+e2
    me = s.eye(3)-projector(e)
    check('encoder_creates_cancellation', s.Matrix.hstack(e1,e2).rank()==2
          and zero(me*(e1+e2)) and s.Matrix.hstack(e,e2,e1).rank()==2)
    t = s.symbols('t', real=True)
    d = t*e1
    check('zero_value_not_zero_tangent', zero(d.subs(t,0)) and e1.dot(d.diff(t))==1)
    # Zero physical/learned subspace intersection alone needs physical full rank.
    rankdef = s.Matrix.hstack(e1,e1)
    check('physical_rank_is_required', rankdef.rank()<rankdef.cols
          and zero(rankdef*s.Matrix([1,-1])))
    beta = s.symbols('beta', positive=True)
    sp = s.Matrix([1,0]); rp = s.Matrix([1,s.sqrt(beta)])
    hb = s.simplify((sp.T*(s.eye(2)-projector(rp))*sp)[0])
    check('soft_penalty_restores_local_curvature',s.simplify(hb-beta/(1+beta))==0)
    check('blind_penalty_cannot_restore_curvature',
          zero(sp.T*(s.eye(2)-projector(sp))*sp))
    hnum=H.subs({a:1,b:2})
    overlap=(Sb.T*PR*Sb).subs({a:1,b:2})
    rho2=max(overlap.eigenvals())
    check('principal_angle_bound_attained',rho2==s.Rational(4,5)
          and min(hnum.eigenvals())==1-rho2)
    return {'curvature': [str(H[0,0]),str(H[1,1])],
            'rank_increment_nonorthogonal': int(s.Matrix.hstack(L*E,L*R,L*S).subs({a:1,b:2}).rank()
                -s.Matrix.hstack(L*E,L*R).subs({a:1,b:2}).rank())}


def symbolic_dynamics():
    # One physical state (also scheduling variable), one learned state, one output.
    x,z,u = s.symbols('x z u', real=True)
    th,k,g,c = s.symbols('th k g c', real=True)
    q = s.Matrix([x,z]); p = s.Matrix([th,k,g,c])
    F = s.Matrix([th*x + x*x/10 + k*z + u, g*z + x/5 + u/7])
    J,B = F.jacobian(q), F.jacobian(p)
    q0 = s.Matrix([c,2*c])
    state, sens = q0, q0.jacobian(p)
    ys, ss = [], []
    for uk in [s.Rational(1,5),s.Rational(-1,10),s.Rational(3,10)]:
        sub = {x:state[0],z:state[1],u:uk}
        sens = J.subs(sub, simultaneous=True)*sens+B.subs(sub, simultaneous=True)
        state = F.subs(sub, simultaneous=True)
        ys.append(state[0]); ss.append(sens[0,:])
    stacked = s.Matrix(ys)
    recurrence = s.Matrix.vstack(*ss)
    check('symbolic_rollout_chain_rule', zero(stacked.jacobian(p)-recurrence))
    check('state_scheduler_derivative_included', J[0,0]==th+x/5)
    check('latent_K_path_present', s.simplify(recurrence[0,1]-2*c)==0)
    check('encoder_path_present',s.simplify(recurrence[0,3]-(th+2*k+c/5))==0)


def numerical_dynamics():
    # Shared encoder coefficient, multiple windows; all outputs after each update.
    torch.set_default_dtype(torch.float64)
    sequences = np.array([[.2,-.1,.3,.15],[-.2,.4,.1,-.1],[.1,.2,-.3,.2]])
    features = np.array([.6,-.4,.8])
    par = np.array([.65,.22,.45,.3])  # physical th, learned k/g, encoder c

    def rollout(p):
        th,k,g,c = p.unbind()
        out = []
        for feature, seq in zip(features,sequences):
            x,z = c*feature,2*c*feature
            for uk in seq:
                x,z = th*x+x*x/10+k*z+float(uk),g*z+x/5+float(uk)/7
                out.append(x)
        return torch.stack(out)

    def recurrence(p, omit_scheduler=False):
        th,k,g,c = p
        rows=[]
        for feature,seq in zip(features,sequences):
            x,z=c*feature,2*c*feature
            T=np.zeros((2,4)); T[:,3]=[feature,2*feature]
            for uk in seq:
                J=np.array([[th if omit_scheduler else th+x/5,k],[.2,g]])
                B=np.array([[x,z,0,0],[0,0,z,0]])
                T=J@T+B
                x,z=th*x+x*x/10+k*z+uk,g*z+x/5+uk/7
                rows.append(T[0].copy())
        return np.array(rows)

    pt=torch.tensor(par,requires_grad=True)
    autograd=torch.autograd.functional.jacobian(rollout,pt).detach().numpy()
    analytic=recurrence(par)
    fd=np.zeros_like(analytic)
    for j in range(len(par)):
        direction=np.eye(len(par))[j]*1e-6
        fd[:,j]=(rollout(torch.tensor(par+direction)).detach().numpy()
                 -rollout(torch.tensor(par-direction)).detach().numpy())/(2e-6)
    ad_error=float(np.max(np.abs(analytic-autograd)))
    fd_error=float(np.max(np.abs(analytic-fd)))
    wrong_error=float(np.max(np.abs(recurrence(par,True)-fd)))
    check('recurrence_matches_autograd',ad_error<1e-12)
    check('recurrence_matches_finite_difference',fd_error<1e-8)
    check('omitted_scheduler_is_detected',wrong_error>1e-3)
    direction=np.array([.3,-.2,.4,.1]); direction/=np.linalg.norm(direction)
    base=rollout(pt).detach().numpy()
    errors=[]
    for step in [1e-2,5e-3,2.5e-3]:
        actual=rollout(torch.tensor(par+step*direction)).detach().numpy()
        errors.append(float(np.linalg.norm(actual-base-step*analytic@direction)))
    ratios=[errors[i]/errors[i+1] for i in range(2)]
    check('quadratic_remainder_observed',all(3.8<r<4.2 for r in ratios))
    return {'category':'structural verification example, not gantry data',
            'max_autograd_error':ad_error,'max_finite_difference_error':fd_error,
            'omitted_scheduler_error':wrong_error,'remainder_halving_ratios':ratios}


if __name__=='__main__':
    algebra_result=algebra()
    symbolic_dynamics()
    numerical=numerical_dynamics()
    result={'category':'structural verification; generic examples only',
            'engine':'sympy '+s.__version__, 'checks':CHECKS,
            'algebra':algebra_result,'numerical':numerical}
    (HERE/'finite_horizon_separation_sympy.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
