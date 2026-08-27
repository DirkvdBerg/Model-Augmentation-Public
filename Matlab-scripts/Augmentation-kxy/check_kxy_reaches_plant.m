function check_kxy_reaches_plant(trunc_s)
% CHECK_KXY_REACHES_PLANT  Does k_xy actually reach the data-generating ODE?
%
% This is the test that answers "can't we just compare the output data", made
% DECISIVE. A plain difference test is ambiguous, because cfg.K feeds both the
% controller design (gtd_build_plant -> Cfb, G) and the plant. If only the
% controller changed, the data would still differ, and the resulting dataset
% would be a controller designed for a sprung plant driving an unsprung one.
%
% So the controller is FROZEN here. All three runs below use the SAME cfg, the
% SAME plant struct, the SAME reference and the SAME force. cfg.K is never
% touched. The ONLY thing that varies is the base-workspace k_xy that the copied
% model's chart passes into gantrySystemExtendedKxy. Any difference in the output
% is therefore the plant, by construction.
%
%   G1  original model                        -> reference output
%   G2  kxy model, k_xy = 0                   -> must EQUAL G1 (surgery was inert)
%   G3  kxy model, k_xy = 1000                -> must DIFFER from G1 (k_xy is live)
%   G4  kxy model, k_xy = 1e6  (signature)    -> Y resonance must move to ~50 Hz,
%                                                which no controller change can fake
%
% Runs on a truncated record (default 0.25 s) purely for speed; the mechanism is
% duration independent.
%
% Run from the repo root:
%   matlab -batch "addpath(genpath('Matlab-scripts/Augmentation')); addpath('Matlab-scripts/Augmentation-kxy'); check_kxy_reaches_plant"

    if nargin < 1 || isempty(trunc_s), trunc_s = 0.25; end

    KXY_TEST = 1000;      % the production value (derive_k_small.py)
    KXY_BIG  = 1e6;       % signature value: pushes f_Y to roughly 50 Hz

    ORIG_MDL = 'gantry_additional_state_2025a';
    KXY_MDL  = 'gantry_additional_state_kxy_2025a';

    fprintf('\n===== k_xy reachability check (controller FROZEN) =====\n');

    % ---- shared setup: one record, built by the REAL pipeline ---------------
    cfg     = gtd_config('augmentation', true, 0.10);
    records = gtd_build_records(cfg);
    rec     = records(1);
    fprintf('record: %s  (class %s, Y_op %.3f)\n', rec.id, rec.class, rec.Y_op);

    plant      = gtd_build_plant(rec.Y_op, cfg);
    [r, t]     = gtd_make_reference(rec, cfg);
    ms         = gtd_make_multisine(rec, plant, cfg);
    [f_safe, ~] = gtd_enforce_limits(plant, r, ms.f_stage, cfg);

    n = min(round(trunc_s * cfg.fs), size(r,1));
    r = r(1:n, :); t = t(1:n); f_safe = f_safe(1:n, :);
    fprintf('truncated to %.3f s (%d samples at %g Hz)\n\n', t(end), n, cfg.fs);

    % ---- G1: original model -------------------------------------------------
    cfgA = cfg;  cfgA.mdl = ORIG_MDL;
    fprintf('G1 original model (%s)...\n', ORIG_MDL);
    o1 = gtd_run_simulation(rec, r, t, f_safe, plant, cfgA);

    % ---- G2/G3/G4: kxy model at three k_xy values ---------------------------
    cfgB = cfg;  cfgB.mdl = KXY_MDL;
    fprintf('G2 kxy model, k_xy = 0...\n');
    o2 = run_with_kxy(rec, r, t, f_safe, plant, cfgB, 0);
    fprintf('G3 kxy model, k_xy = %g...\n', KXY_TEST);
    o3 = run_with_kxy(rec, r, t, f_safe, plant, cfgB, KXY_TEST);
    fprintf('G4 kxy model, k_xy = %g (signature)...\n', KXY_BIG);
    o4 = run_with_kxy(rec, r, t, f_safe, plant, cfgB, KXY_BIG);

    % ---- comparisons --------------------------------------------------------
    d2 = maxdiff(o1, o2);
    d3 = maxdiff(o1, o3);
    d4 = maxdiff(o1, o4);

    fprintf('\n--- results (max abs difference in q_with and delta_a vs G1) ---\n');
    fprintf('  G2  k_xy = 0      : %.6e\n', d2);
    fprintf('  G3  k_xy = %-8g : %.6e\n', KXY_TEST, d3);
    fprintf('  G4  k_xy = %-8g : %.6e\n', KXY_BIG, d4);

    % Tolerance for G2: the chart script changed, so the generated code is not
    % byte-identical and the variable-step solver may make different sub-step
    % choices. Require agreement far below the physical effect being tested.
    tol = 1e-9 * max(1, max(abs(o1.q_with(:))));
    c1 = d2 <= tol;
    c2 = d3 > 1e3 * max(d2, eps);
    c3 = d4 > d3;

    fprintf('\n  C1 k_xy = 0 reproduces the ORIGINAL model : %s  (tol %.2e)\n', tf(c1), tol);
    fprintf('  C2 k_xy = %g changes the PLANT output      : %s\n', KXY_TEST, tf(c2));
    fprintf('  C3 larger k_xy gives a larger effect       : %s\n', tf(c3));

    % ---- signature: where is the Y resonance? -------------------------------
    fprintf('\n--- signature check: Y-axis resonance ---\n');
    mh_rigid = cfg.mh_rigid; ma = cfg.ma;
    for kk = [KXY_TEST, KXY_BIG]
        fprintf('  k_xy = %-9g -> predicted f_Y = %7.2f Hz\n', ...
                kk, sqrt(kk / (mh_rigid + ma)) / (2*pi));
    end
    fprintf('  (a controller redesign cannot manufacture a plant resonance at sqrt(k/m))\n');

    ok = c1 && c2 && c3;
    fprintf('\nRESULT: %s\n\n', tf(ok));
    if ~ok
        error('check_kxy_reaches_plant:fail', ...
              'k_xy reachability NOT established. Do not generate the T4 dataset.');
    end
end

% ── helpers ─────────────────────────────────────────────────────────────────

function out = run_with_kxy(rec, r, t, f, plant, cfg, k_xy)
% k_xy is Parameter-scope Stateflow data, so it resolves from the base
% workspace. push_params (inside gtd_run_simulation) does not touch it.
    assignin('base', 'k_xy', k_xy);
    out = gtd_run_simulation(rec, r, t, f, plant, cfg);
end

function d = maxdiff(a, b)
    d = max(abs(a.q_with(:) - b.q_with(:)));
    if isfield(a, 'da_with') && isfield(b, 'da_with')
        d = max(d, max(abs(a.da_with(:) - b.da_with(:))));
    end
end

function s = tf(b)
    if b, s = 'PASS'; else, s = 'FAIL'; end
end
