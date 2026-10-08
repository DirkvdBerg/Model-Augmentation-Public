function sysT = tdg_truth_lin(Y, cfg, g)
% TDG_TRUTH_LIN  T_AF truth linearised at rest at payload position Y, friction off.
%   sysT = TDG_TRUTH_LIN(Y, cfg) returns the continuous 8-state model of
%   vendor/gantrySystemExtendedCoulomb.m with cc = 0 at x0 = [0; 0; Y; 0; 0; 0; 0; 0],
%   input stage force [F_X1; F_X2; F_Y], output stage position [X1; X2; Y].
%   This is the loop NOISE-INJECTION.md section 3 calibrates on ("the simulation's own
%   truth, linearised at rest, friction off").
%
%   Exactness: with cc = 0 the ODE is dxdt = A(M(x)) x + B(M(x)) u, and M depends on
%   Y and delta_a only. At rest (velocities 0, Theta = delta_a = 0, u = 0) the
%   generalised force -K4 x - C4 v + u is zero, so the derivative of M^-1 multiplies
%   zero and the Jacobian is A(M(Y, 0)), B(M(Y, 0)) exactly. check_noise_calibration.m
%   compares it with a finite-difference Jacobian of the ODE itself.
%   The chart is called with mh = mh_rigid (gtd_run_simulation swaps it), so the same
%   is done here.
%   sysT = TDG_TRUTH_LIN(Y, cfg, g) adds the tanh friction's slope at rest (D-225
%   amendment): d/dv [cc_i tanh(g v_i)] = cc_i g at v = 0, a stage-rail damper, i.e.
%   P diag(cc g) P' added to the logical damping block. Exact at rest for the small
%   velocities of standstill noise (|v| << 1/g). g omitted or 0: friction off, as before.
    m1=cfg.m1; m2=cfg.m2; mb=cfg.mb; mh=cfg.mh_rigid; ma=cfg.ma; Lb=cfg.Lb; d=cfg.d;
    Jb=cfg.Jb; Jh=cfg.Jh; L0=cfg.L0; P=cfg.P;
    da = 0;
    M = [ m1+m2+mb+mh+ma,  (m1-m2)*Lb/2 - (mh+ma)*Y - ma*L0 - ma*da,  0,  0;
          (m1-m2)*Lb/2 - (mh+ma)*Y - ma*L0 - ma*da, ...
              Jb+Jh+(m1+m2)*Lb^2/4 + (mh+ma)*d^2 + mh*Y^2 + ma*(Y+L0+da)^2,  -(mh+ma)*d,  -ma*d;
          0,  -(mh+ma)*d,  mh+ma,  ma;
          0,  -ma*d,       ma,     ma];
    C4 = [cfg.cg1+cfg.cg2,             (cfg.cg1-cfg.cg2)*Lb/2,                    0,      0;
          (cfg.cg1-cfg.cg2)*Lb/2,      cfg.cb1+cfg.cb2+(cfg.cg1+cfg.cg2)*Lb^2/4,  0,      0;
          0,                           0,                                         cfg.cy, 0;
          0,                           0,                                         0,      cfg.ca];
    K4 = [0, 0, 0, 0; 0, cfg.kb1+cfg.kb2, 0, 0; 0, 0, 0, 0; 0, 0, 0, cfg.ka];
    if nargin > 2 && g > 0
        C4(1:3, 1:3) = C4(1:3, 1:3) + P * diag(g * [cfg.cc1; cfg.cc2; cfg.ccy]) * P.';
    end
    A = [zeros(4), eye(4); -M\K4, -M\C4];
    B = [zeros(4,3); M \ [eye(3); zeros(1,3)]];
    sysT = ss(A, B*P, [P.', zeros(3,5)], zeros(3));
end
