% GTD_CHECK_TRANSFORM  Value-correctness gate for the logical->stage force map.
%   Run from repo root:  >> gtd_check_transform
%   Verifies F_stage = P^{-1} F_logical five independent ways (spec 1.6 / 7.5,
%   lessons rule: value-correctness, not shape-only).

clear; clc;
cfg = gtd_config('joint', true, 0.50);
Lb  = cfg.Lb;
tol = 1e-9;

% (1) P^{-1} equals the analytic inverse
Pinv_expected = [0.5,  1/Lb, 0;
                 0.5, -1/Lb, 0;
                 0,    0,    1];
assert(norm(inv(cfg.P) - Pinv_expected, 'fro') < tol, 'P^{-1} mismatch');
fprintf('(1) P^{-1} matches analytic form.\n');

% (2) Pure symmetric logical force -> equal rail forces, no yaw, no Y
Fs = gtd_logical_to_stage([1, 0, 0], cfg);
assert(abs(Fs(1)-Fs(2)) < tol && abs(Fs(3)) < tol, 'sym force not symmetric');
fprintf('(2) f_sym -> [%.4g %.4g %.4g]  (equal rails).\n', Fs);

% (3) Pure anti-symmetric logical torque -> opposite rails, zero net translation
Fa = gtd_logical_to_stage([0, 1, 0], cfg);
assert(abs(Fa(1)+Fa(2)) < tol && abs(Fa(3)) < tol, 'anti force not a pure couple');
assert(abs(Fa(1) - 1/Lb) < tol, 'anti scale not 1/Lb');
fprintf('(3) f_anti -> [%.4g %.4g %.4g]  (opposite rails, 1/Lb).\n', Fa);

% (4) Virtual-work invariance for random logical force/position
Fl = randn(3,1);  ql = randn(3,1);
Fst = gtd_logical_to_stage(Fl.', cfg).';    % P^{-1} Fl
qst = cfg.P.' * ql;                          % position convention q_stage = P' q_l
assert(abs(Fst.'*qst - Fl.'*ql) < 1e-9, 'virtual work not invariant');
fprintf('(4) virtual work invariant: F_s.q_s = F_l.q_l = %.6g.\n', Fl.'*ql);

% (5) Consistency with the actual plant: injecting F_stage into the stage plant
%     equals P' times injecting F_logical into the logical plant. Evaluated at a
%     finite frequency -- the open-loop plant has rigid-body modes (K singular:
%     no stiffness on translation/Y), so its DC gain is infinite and unusable.
m1=cfg.m1; m2=cfg.m2; mb=cfg.mb; mh=cfg.mh; Jb=cfg.Jb; Jh=cfg.Jh; d=cfg.d;
M0 = [m1+m2+mb+mh,      (m1-m2)*Lb/2,                 0;
      (m1-m2)*Lb/2,     Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2, -mh*d;
      0,                -mh*d,                         mh];
Gphys    = getss(cfg.n, M0, cfg.C_damp, cfg.K);   % logical open-loop plant
sys_cont = cfg.P.' * Gphys * cfg.P;               % stage open-loop plant (as built)
w        = 2*pi*37;                                % finite, non-resonant [rad/s]
resp_stage   = evalfr(sys_cont, 1j*w) * gtd_logical_to_stage(Fl.', cfg).';
resp_logical = cfg.P.' * (evalfr(Gphys, 1j*w) * Fl);
rel = norm(resp_stage - resp_logical) / norm(resp_logical);
assert(rel < 1e-6, 'plant-consistency mismatch (rel %.2e)', rel);
fprintf('(5) plant consistency @37Hz: rel err %.2e.\n', rel);

fprintf('\nP-transform gate OK\n');
