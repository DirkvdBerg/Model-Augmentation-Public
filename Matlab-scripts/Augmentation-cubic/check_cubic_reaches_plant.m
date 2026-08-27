function check_cubic_reaches_plant()
% CHECK_CUBIC_REACHES_PLANT  Class B gate: k3 actually reaches the integrated plant.
%
% check_cubic_noop proves the FUNCTION is faithful. It says nothing about
% whether the Simulink model picks that function up, or whether the k3 in the
% base workspace is the k3 the ODE sees. This gate answers that, end to end,
% through the real generator path.
%
% CONTROLLER IS FROZEN THROUGHOUT, and here it is frozen for free: cfg.K cannot
% express a cubic term at all, so gtd_build_plant produces an identical Cfb, G,
% reference and limit scaling in every arm. The ONLY thing that differs between
% the runs below is the plant. That is the change-one-thing design.
%
% THE TRAP THIS AVOIDS. A plain "the data changed" test is ambiguous, because
% for a LINEAR spring cfg.K feeds both the controller design and the ODE, so a
% naive difference test can report success on a physically incoherent dataset
% (a controller designed for a sprung plant driving an unsprung one). That trap
% is structurally absent for the cubic, but B3 below is included anyway because
% it is decisive rather than merely suggestive: the output difference must scale
% LINEARLY in k3 for small k3, which no plumbing accident reproduces.
%
% Run from the repo root (about 4 Simulink runs of 12 s each, so give it time):
%   matlab -batch "addpath('Matlab-scripts/Augmentation-cubic'); check_cubic_reaches_plant"

    THIS_DIR  = fileparts(mfilename('fullpath'));
    REPO_ROOT = fileparts(fileparts(THIS_DIR));
    addpath(genpath(fullfile(REPO_ROOT, 'Matlab-scripts', 'Augmentation')));
    addpath(genpath(fullfile(REPO_ROOT, 'kamtin-fp-model', '03 Simulink gantry')));
    addpath(THIS_DIR);

    CUBIC_MDL = 'gantry_2025a_cubic';
    if exist([CUBIC_MDL '.slx'], 'file') ~= 2 && exist(CUBIC_MDL, 'file') ~= 4
        error('check_cubic_reaches_plant:noModel', ...
              'Model %s not found. Run make_cubic_model first.', CUBIC_MDL);
    end

    K3_TEST = 50;     % deliberately far above the production value: this gate is
                      % about presence and scaling, not sizing. derive_k3 sets the
                      % production number.

    cfg     = gtd_config('augmentation', false, 0);
    records = gtd_build_records(cfg);

    % Pick the record with the LARGEST |Y_op|: the cubic force goes as |Y|^3, so
    % a record parked near Y = 0 would show nothing however well the wiring works.
    [~, idx] = max(arrayfun(@(r) abs(r.Y_op), records));
    rec = records(idx);
    fprintf('\nClass B gate: does k3 reach the integrated plant\n');
    fprintf('  record   : %s  (|Y_op| = %.3f m, the largest available)\n', rec.id, abs(rec.Y_op));
    fprintf('  k3 test  : %g N/m^3\n\n', K3_TEST);

    % Build the excitation ONCE and reuse it in every arm.
    plant         = gtd_build_plant(rec.Y_op, cfg);
    [r, t]        = gtd_make_reference(rec, cfg);
    ms            = gtd_make_multisine(rec, plant, cfg);
    [f_safe, ~]   = gtd_enforce_limits(plant, r, ms.f_stage, cfg);

    q_orig  = run_arm(rec, r, t, f_safe, plant, cfg, 'gantry_2025a', 0);
    q_zero  = run_arm(rec, r, t, f_safe, plant, cfg, CUBIC_MDL,      0);
    q_k1    = run_arm(rec, r, t, f_safe, plant, cfg, CUBIC_MDL,      K3_TEST);
    q_k2    = run_arm(rec, r, t, f_safe, plant, cfg, CUBIC_MDL,  2 * K3_TEST);

    d_zero = max(abs(q_zero(:) - q_orig(:)));
    d_k1   = norm(q_k1(:) - q_zero(:));
    d_k2   = norm(q_k2(:) - q_zero(:));
    ratio  = d_k2 / max(d_k1, eps);

    b1 = (d_zero == 0);
    b2 = (d_k1 > 0);
    b3 = (ratio > 1.7) && (ratio < 2.3);

    fprintf('  B1 k3 = 0 is bit-identical to the original model : %s  (max abs diff %.6e)\n', tf2s(b1), d_zero);
    fprintf('  B2 k3 > 0 changes the trajectory                 : %s  (norm diff %.6e)\n', tf2s(b2), d_k1);
    fprintf('  B3 difference scales linearly in k3              : %s  (ratio %.4f, want ~2)\n', tf2s(b3), ratio);

    if ~b1
        fprintf(['\n  B1 FAILED: the copied model does not reproduce the original at k3 = 0.\n' ...
                 '     Something other than the spring differs between the two models.\n']);
    end
    if ~b2
        fprintf(['\n  B2 FAILED: k3 never reaches the ODE. Check that the chart parameter\n' ...
                 '     was created with scope Parameter and that assignin targeted base.\n']);
    end
    if ~b3
        fprintf(['\n  B3 FAILED: the response does not scale linearly in k3. Either k3 is\n' ...
                 '     large enough here to be outside the small-perturbation regime, or it\n' ...
                 '     is reaching the plant through the wrong path. Re-run with a smaller\n' ...
                 '     K3_TEST before concluding it is wiring.\n']);
    end

    fprintf('\nRESULT: %s\n\n', tf2s(b1 && b2 && b3));
end

% -- helpers -----------------------------------------------------------------

function q = run_arm(rec, r, t, f_safe, plant, cfg, mdl, k3)
    cfg.mdl = mdl;
    % Re-assert immediately before the run: push_params must not shadow it, and
    % this is the ordering generate_trajectory_data_kxy.m had to adopt.
    assignin('base', 'k3', k3);
    out = gtd_run_simulation(rec, r, t, f_safe, plant, cfg);
    q   = out.q_with;
    fprintf('    ran %-22s k3 = %-8g -> rms(q) = [%.4e %.4e %.4e]\n', ...
            mdl, k3, rms(q(:,1)), rms(q(:,2)), rms(q(:,3)));
end

function s = tf2s(b)
    if b, s = 'PASS'; else, s = 'FAIL'; end
end
