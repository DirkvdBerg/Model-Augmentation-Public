function check_cubic_noop()
% CHECK_CUBIC_NOOP  Class A gate: gantrySystemCubic(k3=0) == gantrySystem.
%
% Proves the copied ODE is faithful before anything downstream trusts it. Runs
% both functions on the same random states and inputs and requires BIT-IDENTICAL
% derivatives at k3 = 0, then three further checks that the added term is real,
% physical and correctly placed.
%
% This gate is load-bearing here in a way it was not for T4. T4's change was a
% matrix entry; this one is a structural addition to the equation, so "the new
% argument is silently ignored" and "the force is on the wrong row" are both
% live failure modes.
%
% This runs the FUNCTIONS directly, outside Simulink. It says nothing about
% whether the Simulink model picks the new function up; that is a separate
% question, answered by check_cubic_reaches_plant.m.
%
% Run from the repo root:
%   matlab -batch "addpath(genpath('kamtin-fp-model/03 Simulink gantry')); addpath('Matlab-scripts/Augmentation-cubic'); check_cubic_noop"

    THIS_DIR  = fileparts(mfilename('fullpath'));
    REPO_ROOT = fileparts(fileparts(THIS_DIR));
    addpath(genpath(fullfile(REPO_ROOT, 'kamtin-fp-model', '03 Simulink gantry')));
    addpath(THIS_DIR);
    assert(exist('gantrySystem', 'file') == 2, ...
           'gantrySystem not found after path bootstrap. Expected it under kamtin-fp-model/03 Simulink gantry/functions.');

    p = local_params();
    rng(42);
    N = 200;
    K_TEST = 1000;          % deliberately large: this gate is about presence, not sizing

    max_abs_diff_0   = 0;
    min_abs_diff_k   = inf;
    max_abs_diff_org = 0;   % at the origin in X and Y, the cubic force must vanish
    sign_ok          = true;

    for i = 1:N
        x = randn(6,1) .* [1e-1; 1e-4; 3e-1; 1e-2; 1e-3; 1e-2];
        u = randn(3,1) * 10;

        d_orig = call_orig(p, u, x);
        d_zero = call_cubic(p, u, x, 0);
        d_k3   = call_cubic(p, u, x, K_TEST);

        max_abs_diff_0 = max(max_abs_diff_0, max(abs(d_zero - d_orig)));
        min_abs_diff_k = min(min_abs_diff_k, max(abs(d_k3 - d_orig)));

        % A3: X = Y = 0 with arbitrary Theta and velocities -> force is exactly zero
        x0 = x;  x0(1) = 0;  x0(3) = 0;
        d_org_0 = call_orig(p, u, x0);
        d_k3_0  = call_cubic(p, u, x0, K_TEST);
        max_abs_diff_org = max(max_abs_diff_org, max(abs(d_k3_0 - d_org_0)));

        % A4: with X > 0 and Y = 0 the added X acceleration must be NEGATIVE
        % (restoring). M is symmetric positive definite, so inv(M)(1,1) > 0.
        xs = x;  xs(1) = abs(x(1)) + 1e-3;  xs(3) = 0;
        dd = call_cubic(p, u, xs, K_TEST) - call_orig(p, u, xs);
        if dd(4) >= 0
            sign_ok = false;
        end
    end

    fprintf('\nClass A gate: gantrySystemCubic versus gantrySystem\n');
    fprintf('  %d random (x, u) samples\n\n', N);

    a1 = (max_abs_diff_0 == 0);
    a2 = (min_abs_diff_k > 0);
    a3 = (max_abs_diff_org == 0);
    a4 = sign_ok;

    fprintf('  A1 k3 = 0 reproduces the original EXACTLY   : %s  (max abs diff %.3e)\n', tf2s(a1), max_abs_diff_0);
    fprintf('  A2 k3 = %g changes the derivative           : %s  (min abs diff %.3e)\n', K_TEST, tf2s(a2), min_abs_diff_k);
    fprintf('  A3 force vanishes at X = Y = 0              : %s  (max abs diff %.3e)\n', tf2s(a3), max_abs_diff_org);
    fprintf('  A4 force is restoring for X > 0             : %s\n', tf2s(a4));

    if ~a2
        fprintf(['\n  A2 FAILED: the k3 argument is being ignored. Every result ' ...
                 'downstream would be void.\n']);
    end
    if ~a3
        fprintf(['\n  A3 FAILED: a cubic spring to ground must produce exactly zero ' ...
                 'force at the origin. The term is misplaced.\n']);
    end
    if ~a4
        fprintf('\n  A4 FAILED: the sign is wrong, this is a NEGATIVE stiffness.\n');
    end

    fprintf('\nRESULT: %s\n\n', tf2s(a1 && a2 && a3 && a4));
end

% -- helpers -----------------------------------------------------------------

function d = call_orig(p, u, x)
    d = gantrySystem(u, x, p.m1, p.m2, p.mb, p.mh, p.Lb, p.Jb, p.Jh, p.d, ...
                     p.cg1, p.cg2, p.cb1, p.cb2, p.cy, p.kb1, p.kb2);
end

function d = call_cubic(p, u, x, k3)
    d = gantrySystemCubic(u, x, p.m1, p.m2, p.mb, p.mh, p.Lb, p.Jb, p.Jh, p.d, ...
                          p.cg1, p.cg2, p.cb1, p.cb2, p.cy, p.kb1, p.kb2, k3);
end

function s = tf2s(b)
    if b, s = 'PASS'; else, s = 'FAIL'; end
end

function p = local_params()
% Nominal parameters, matching gtd_config.m. No absorber: this is the 6-state
% baseline plant, so ma / ka / ca / L0 do not appear.
    p.mb  = 22.8;  p.mh  = 10.1;  p.m1 = 10.2;  p.m2 = 10.7;
    p.Jb  = 1.0;   p.Jh  = 0.05;
    p.cg1 = 14.5;  p.cg2 = 20.3;  p.cy = 10.0;
    p.cb1 = 9.0;   p.cb2 = 9.0;
    p.kb1 = 1987.5; p.kb2 = 1987.5;
    p.Lb  = 0.725; p.d   = 0.1;
end
