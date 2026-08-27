function check_coulomb_noop()
% CHECK_COULOMB_NOOP  Class A gate: gantrySystemExtendedCoulomb(cc=0) == gantrySystemExtended.
%
% Proves the copied ODE is faithful before anything downstream trusts it. Runs
% both functions on the same random states and inputs and requires BIT-IDENTICAL
% derivatives at cc = 0, then requires a DIFFERENT derivative at cc > 0
% (otherwise the new arguments are being ignored and every later result is void).
%
% Two extra checks that are specific to friction and cheap to run here:
%   A3 dissipation : the friction force must never do positive work, i.e.
%                    v_stage' * Fc_stage >= 0 for every sample (Fc opposes motion).
%   A4 frame       : building the friction in the LOGICAL frame instead of the
%                    stage frame must give a DIFFERENT answer. If it does not,
%                    the projection is not doing anything and the "wrong frame"
%                    trap would be invisible.
%
% This runs the FUNCTIONS directly, outside Simulink. It says nothing about
% whether the Simulink model picks the new function up; that is a separate
% question, answered by check_coulomb_reaches_plant.m.
%
% Run from the repo root:
%   matlab -batch "addpath('Matlab-scripts/Augmentation'); addpath('Matlab-scripts/Augmentation-coulomb'); check_coulomb_noop"

    p = local_params();
    rng(42);
    N = 200;
    max_abs_diff_0 = 0;
    min_abs_diff_c = inf;
    min_dissip     = inf;
    min_frame_diff = inf;

    % THEORY: garcia2013 Table -- identified Coulomb friction of the H-gantry
    CC = [16.8; 18.35; 11.6];
    P  = [1, 1, 0; p.Lb/2, -p.Lb/2, 0; 0, 0, 1];
    TS = 5e-5;      % generator step (1/cfg.fs); sizes the Karnopp band V_EPS

    for i = 1:N
        x = randn(8,1) .* [1e-3; 1e-4; 3e-1; 1e-5; 1e-2; 1e-3; 1e-2; 1e-2];
        u = randn(3,1) * 10;

        d_orig = gantrySystemExtended(u, x, p.m1, p.m2, p.mb, p.mh, p.Lb, p.Jb, ...
                    p.Jh, p.d, p.cg1, p.cg2, p.cb1, p.cb2, p.cy, p.kb1, p.kb2, ...
                    p.ma, p.ka, p.ca, p.L0);

        d_zero = gantrySystemExtendedCoulomb(u, x, p.m1, p.m2, p.mb, p.mh, p.Lb, p.Jb, ...
                    p.Jh, p.d, p.cg1, p.cg2, p.cb1, p.cb2, p.cy, p.kb1, p.kb2, ...
                    p.ma, p.ka, p.ca, p.L0, 0, 0, 0, TS);

        d_cc   = gantrySystemExtendedCoulomb(u, x, p.m1, p.m2, p.mb, p.mh, p.Lb, p.Jb, ...
                    p.Jh, p.d, p.cg1, p.cg2, p.cb1, p.cb2, p.cy, p.kb1, p.kb2, ...
                    p.ma, p.ka, p.ca, p.L0, CC(1), CC(2), CC(3), TS);

        max_abs_diff_0 = max(max_abs_diff_0, max(abs(d_zero - d_orig)));
        min_abs_diff_c = min(min_abs_diff_c, max(abs(d_cc   - d_orig)));

        % A3: dissipation. Fc opposes v in the STAGE frame, per rail.
        v_stage  = P.' * x(5:7);
        Fc_stage = CC .* sign(v_stage);
        min_dissip = min(min_dissip, v_stage.' * Fc_stage);

        % A4: the wrong-frame trap must be detectable.
        Fc_right = P * (CC .* sign(P.' * x(5:7)));   % correct: stage frame
        Fc_wrong =      CC .* sign(     x(5:7));     % wrong: logical frame
        min_frame_diff = min(min_frame_diff, max(abs(Fc_right - Fc_wrong)));
    end

    fprintf('\nClass A gate: gantrySystemExtendedCoulomb versus gantrySystemExtended\n');
    fprintf('  %d random (x, u) samples,  cc = [%.2f %.2f %.2f] N\n\n', N, CC);

    a1 = (max_abs_diff_0 == 0);
    a2 = (min_abs_diff_c > 0);
    a3 = (min_dissip >= 0);
    a4 = (min_frame_diff > 0);

    fprintf('  A1 cc = 0 reproduces the original EXACTLY   : %s  (max abs diff %.3e)\n', ...
            tf2s(a1), max_abs_diff_0);
    fprintf('  A2 cc = Garcia changes the derivative       : %s  (min abs diff %.3e)\n', ...
            tf2s(a2), min_abs_diff_c);
    fprintf('  A3 friction never does positive work        : %s  (min v''*Fc  %.3e)\n', ...
            tf2s(a3), min_dissip);
    fprintf('  A4 stage frame differs from logical frame   : %s  (min abs diff %.3e)\n', ...
            tf2s(a4), min_frame_diff);

    if ~a2
        fprintf(['\n  A2 FAILED: the cc arguments are being ignored. Every result ' ...
                 'downstream would be void.\n']);
    end
    if ~a3
        fprintf('\n  A3 FAILED: the friction sign is inverted; it is INJECTING energy.\n');
    end
    % ---- A5/A6: the stick state itself ------------------------------------
    % The slip branch is all A2-A4 exercise. These two test the behaviour that
    % was ADDED: a stage at rest must HOLD under a sub-breakaway force and must
    % MOVE once the force exceeds it. The old hard-sign version fails A5 by
    % construction, since sign(0) = 0 leaves no friction to hold anything.
    x_rest = zeros(8,1);
    x_rest(3) = 0.10;                  % parked at Y = 0.10 m, all velocities zero
    % Stage-frame forces well inside and well outside the friction cone, mapped
    % to the logical frame the way the input enters.
    u_hold  = P * [0.5*CC(1); 0.5*CC(2); 0.5*CC(3)];
    u_break = P * [3.0*CC(1); 3.0*CC(2); 3.0*CC(3)];
    d_hold  = gantrySystemExtendedCoulomb(u_hold, x_rest, p.m1, p.m2, p.mb, p.mh, ...
                p.Lb, p.Jb, p.Jh, p.d, p.cg1, p.cg2, p.cb1, p.cb2, p.cy, ...
                p.kb1, p.kb2, p.ma, p.ka, p.ca, p.L0, CC(1), CC(2), CC(3), TS);
    d_break = gantrySystemExtendedCoulomb(u_break, x_rest, p.m1, p.m2, p.mb, p.mh, ...
                p.Lb, p.Jb, p.Jh, p.d, p.cg1, p.cg2, p.cb1, p.cb2, p.cy, ...
                p.kb1, p.kb2, p.ma, p.ka, p.ca, p.L0, CC(1), CC(2), CC(3), TS);
    % Frictionless response to the SAME breakaway force, as the reference A7
    % needs. Without it, "it accelerates" is satisfied by friction VANISHING
    % just as well as by friction saturating, and the first version of this gate
    % passed on exactly that error.
    d_free  = gantrySystemExtendedCoulomb(u_break, x_rest, p.m1, p.m2, p.mb, p.mh, ...
                p.Lb, p.Jb, p.Jh, p.d, p.cg1, p.cg2, p.cb1, p.cb2, p.cy, ...
                p.kb1, p.kb2, p.ma, p.ka, p.ca, p.L0, 0, 0, 0, TS);
    % Rail accelerations: the stuck condition is on the STAGE frame, per rail.
    a_hold  = P.' * d_hold(5:7);
    a_break = P.' * d_break(5:7);
    a_free_ = P.' * d_free(5:7);
    a5 = max(abs(a_hold)) < 1e-9;
    a6 = min(abs(a_break)) > 1e-3;
    % A7: breaking away must still be RESISTED. Applied 3*cc against a saturated
    % cc leaves 2*cc, so the rail must accelerate strictly LESS than it does with
    % no friction at all. If the breakaway branch drops the force to zero (the
    % sign(0) trap) these are EQUAL and A7 fails while A6 still passes.
    ratio = min(abs(a_break) ./ max(abs(a_free_), eps));
    a7 = ratio < 0.95;

    fprintf('\n  A5 at rest, |F| = 0.5*cc: the stage HOLDS       : %s  (max |a_rail| %.3e m/s^2)\n', ...
            tf2s(a5), max(abs(a_hold)));
    fprintf('  A6 at rest, |F| = 3.0*cc: it BREAKS AWAY        : %s  (min |a_rail| %.3e m/s^2)\n', ...
            tf2s(a6), min(abs(a_break)));
    fprintf('  A7 breakaway is still RESISTED (< frictionless) : %s  (a_break/a_free %.3f)\n', ...
            tf2s(a7), ratio);
    if ~a5
        fprintf(['\n  A5 FAILED: the stick state is not holding. With sign(0) = 0 this ' ...
                 'fails by\n  construction, so a failure here means the Karnopp solve ' ...
                 'is not active.\n']);
    end
    if ~a7
        fprintf(['\n  A7 FAILED: on breakaway the friction is going to ZERO instead of ' ...
                 'saturating\n  at cc. Check that the slip force uses sign(F_required) ' ...
                 'and not sign(v),\n  which is arbitrary for a rail that was stuck.\n']);
    end

    fprintf('\nRESULT: %s\n\n', tf2s(a1 && a2 && a3 && a4 && a5 && a6 && a7));
end

function s = tf2s(b)
    if b, s = 'PASS'; else, s = 'FAIL'; end
end

function p = local_params()
% Nominal parameters, matching gtd_config.m (the values the dataset is generated
% with), including its ma_frac = 0.10 absorber convention. Only their presence
% matters for a no-op check, not the values.
    p.mb  = 22.8;  p.mh  = 10.1;  p.m1 = 10.2;  p.m2 = 10.7;
    p.Jb  = 1.0;   p.Jh  = 0.05;
    p.cg1 = 14.5;  p.cg2 = 20.3;  p.cy = 10.0;
    p.cb1 = 9.0;   p.cb2 = 9.0;
    p.kb1 = 1987.5; p.kb2 = 1987.5;
    p.Lb  = 0.725; p.d   = 0.1;
    ma_frac = 0.10;
    p.ma  = ma_frac * p.mh;
    p.mh  = p.mh - p.ma;          % mh_rigid: caller must pass the rigid mass
    fa    = 150;
    p.ka  = p.ma * (2*pi*fa)^2;
    zeta_a = 0.05;
    p.ca  = 2 * zeta_a * sqrt(p.ka * p.ma);
    p.L0  = 0.10;
end
