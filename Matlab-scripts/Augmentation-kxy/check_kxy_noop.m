function check_kxy_noop()
% CHECK_KXY_NOOP  Class A gate: gantrySystemExtendedKxy(k_xy=0) == gantrySystemExtended.
%
% Proves the copied ODE is faithful before anything downstream trusts it. Runs
% both functions on the same random states and inputs and requires BIT-IDENTICAL
% derivatives at k_xy = 0, then requires a DIFFERENT derivative at k_xy > 0
% (otherwise the new argument is being ignored and every later result is void).
%
% This runs the FUNCTIONS directly, outside Simulink. It says nothing about
% whether the Simulink model picks the new function up; that is a separate
% question, answered by check_kxy_reaches_plant.m.
%
% Run from the repo root:
%   addpath('Matlab-scripts/Augmentation'); addpath('Matlab-scripts/Augmentation-kxy');
%   check_kxy_noop

    p = local_params();
    rng(42);
    N = 200;
    max_abs_diff_0 = 0;
    min_abs_diff_k = inf;
    K_TEST = 1000;

    for i = 1:N
        x = randn(8,1) .* [1e-3; 1e-4; 3e-1; 1e-5; 1e-2; 1e-3; 1e-2; 1e-2];
        u = randn(3,1) * 10;

        d_orig = gantrySystemExtended(u, x, p.m1, p.m2, p.mb, p.mh, p.Lb, p.Jb, ...
                    p.Jh, p.d, p.cg1, p.cg2, p.cb1, p.cb2, p.cy, p.kb1, p.kb2, ...
                    p.ma, p.ka, p.ca, p.L0);

        d_zero = gantrySystemExtendedKxy(u, x, p.m1, p.m2, p.mb, p.mh, p.Lb, p.Jb, ...
                    p.Jh, p.d, p.cg1, p.cg2, p.cb1, p.cb2, p.cy, p.kb1, p.kb2, ...
                    p.ma, p.ka, p.ca, p.L0, 0);

        d_kxy  = gantrySystemExtendedKxy(u, x, p.m1, p.m2, p.mb, p.mh, p.Lb, p.Jb, ...
                    p.Jh, p.d, p.cg1, p.cg2, p.cb1, p.cb2, p.cy, p.kb1, p.kb2, ...
                    p.ma, p.ka, p.ca, p.L0, K_TEST);

        max_abs_diff_0 = max(max_abs_diff_0, max(abs(d_zero - d_orig)));
        min_abs_diff_k = min(min_abs_diff_k, max(abs(d_kxy  - d_orig)));
    end

    fprintf('\nClass A gate: gantrySystemExtendedKxy versus gantrySystemExtended\n');
    fprintf('  %d random (x, u) samples\n\n', N);

    a1 = (max_abs_diff_0 == 0);
    a2 = (min_abs_diff_k > 0);

    fprintf('  A1 k_xy = 0 reproduces the original EXACTLY : %s  (max abs diff %.3e)\n', ...
            tf2s(a1), max_abs_diff_0);
    fprintf('  A2 k_xy = %g changes the derivative          : %s  (min abs diff %.3e)\n', ...
            K_TEST, tf2s(a2), min_abs_diff_k);

    if ~a2
        fprintf(['\n  A2 FAILED: the k_xy argument is being ignored. Every T4 result ' ...
                 'would be void.\n']);
    end
    fprintf('\nRESULT: %s\n\n', tf2s(a1 && a2));
end

function s = tf2s(b)
    if b, s = 'PASS'; else, s = 'FAIL'; end
end

function p = local_params()
% Nominal parameters, matching Matlab-scripts/Augmentation/main_augmentation.m.
% ma/ka/ca follow the ma_frac = 0.10 absorber convention used for the
% augmentation dataset; only their presence matters for this check, not the values.
    p.mb  = 22.8;  p.mh  = 10.1;  p.m1 = 10.2;  p.m2 = 10.7;
    p.Jb  = 1.0;   p.Jh  = 0.05;
    p.cg1 = 14.5;  p.cg2 = 20.3;  p.cy = 10.0;
    p.cb1 = 9.0;   p.cb2 = 9.0;
    p.kb1 = 1987.5; p.kb2 = 1987.5;
    p.Lb  = 0.725; p.d   = 0.1;
    ma_frac = 0.10;
    p.ma  = ma_frac * p.mh;
    p.mh  = p.mh - p.ma;          % mh_rigid: caller must pass the rigid mass
    fa    = 400;
    p.ka  = p.ma * (2*pi*fa)^2;
    zeta_a = 0.02;
    p.ca  = 2 * zeta_a * sqrt(p.ka * p.ma);
    p.L0  = 0.0;
end
