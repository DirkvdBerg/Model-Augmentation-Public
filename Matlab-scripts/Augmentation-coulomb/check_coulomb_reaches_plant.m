function check_coulomb_reaches_plant(TEND)
% CHECK_COULOMB_REACHES_PLANT  Class B gate: the cc actually reach the integrator.
%
% check_coulomb_noop proves the FUNCTION is right. This proves the MODEL calls it
% and that cc1/cc2/ccy resolve from the base workspace, which is the part that
% silently fails (a mis-scoped chart datum would leave friction at zero and every
% later number would be a no-friction number wearing a friction label).
%
% Three runs of the SAME record through the SAME pipeline:
%
%   B1  original model, cc irrelevant        reference
%   B2  coulomb model, cc = 0                must be BIT-IDENTICAL to B1
%   B3  coulomb model, cc = Garcia           must DIFFER
%
% B1 is run with the original model's solver FORCED to fixed-step ode4 IN MEMORY
% ONLY, so that B1 and B2 differ in nothing but the chart script. The original
% .slx is closed without saving and is never written. Comparing against the
% stock variable-step ode45 instead would confound the chart edit with a solver
% change and B2 could never be bit-identical.
%
% TEND (optional) truncates the record; default 1.0 s, enough to cover the 0.5 s
% opening hold plus the start of the active phase, and it keeps the gate quick.
% Pass [] for the full 12 s record.
%
% Run from the repo root:
%   matlab -batch "addpath(genpath('Matlab-scripts/Augmentation')); addpath('Matlab-scripts/Augmentation-coulomb'); check_coulomb_reaches_plant"

    if nargin < 1, TEND = 1.0; end

    REC_ID = 'V1_standstill_Yp10';
    % THEORY: garcia2013 -- identified Coulomb friction of the H-gantry
    CC = struct('cc1', 16.8, 'cc2', 18.35, 'ccy', 11.6);

    cfg = gtd_config('augmentation', true, 0.10);
    rec = pick_record(cfg, REC_ID);

    plant         = gtd_build_plant(rec.Y_op, cfg);
    [r, t]        = gtd_make_reference(rec, cfg);
    ms            = gtd_make_multisine(rec, plant, cfg);
    [f_safe, ~]   = gtd_enforce_limits(plant, r, ms.f_stage, cfg);

    if ~isempty(TEND)
        n = min(numel(t), round(TEND / cfg.ts));
        t = t(1:n);  r = r(1:n, :);  f_safe = f_safe(1:n, :);
    end

    fprintf('\nClass B gate: do cc1/cc2/ccy reach the integrator?\n');
    fprintf('  record %s,  %.3f s,  fixed-step ode4 @ %.1e s\n\n', ...
            rec.id, t(end), cfg.ts);

    % The cc reach the model ONLY through cfg: gtd_run_simulation's push_params
    % assigns cfg.cc1/cc2/ccy into the base workspace on every call, so anything
    % assigned there beforehand is overwritten. Set them on cfg, never on base.

    % ---- B1: original model, solver forced fixed-step IN MEMORY -------------
    cfg1 = set_cc(cfg, 0, 0, 0);  cfg1.mdl = 'gantry_additional_state_2025a';
    load_system(cfg1.mdl);
    set_param(cfg1.mdl, 'SolverType', 'Fixed-step');
    set_param(cfg1.mdl, 'Solver',     'ode4');
    set_param(cfg1.mdl, 'FixedStep',  sprintf('%.12g', cfg.ts));
    o1 = gtd_run_simulation(rec, r, t, f_safe, plant, cfg1);
    close_system(cfg1.mdl, 0);      % DISCARD: the original .slx is not written
    fprintf('  B1 original model            : done (%d samples)\n', size(o1.q_with,1));

    % ---- B2: coulomb model, cc = 0 ------------------------------------------
    cfg2 = set_cc(cfg, 0, 0, 0);
    cfg2.mdl = 'gantry_additional_state_coulomb_2025a';
    o2 = gtd_run_simulation(rec, r, t, f_safe, plant, cfg2);
    fprintf('  B2 coulomb model, cc = 0     : done (%d samples)\n', size(o2.q_with,1));

    % ---- B3: coulomb model, cc = Garcia -------------------------------------
    cfg3 = set_cc(cfg, CC.cc1, CC.cc2, CC.ccy);
    cfg3.mdl = 'gantry_additional_state_coulomb_2025a';
    o3 = gtd_run_simulation(rec, r, t, f_safe, plant, cfg3);
    fprintf('  B3 coulomb model, cc = Garcia: done (%d samples)\n\n', size(o3.q_with,1));

    d12 = max(abs(o2.q_with(:) - o1.q_with(:)));
    d13 = max(abs(o3.q_with(:) - o1.q_with(:)));

    b1 = (d12 == 0);
    b2 = (d13 > 0);

    fprintf('  B1 cc = 0 is bit-identical to the original : %s  (max abs diff %.6e m)\n', ...
            tf2s(b1), d12);
    fprintf('  B2 cc = Garcia changes the trajectory      : %s  (max abs diff %.6e m)\n', ...
            tf2s(b2), d13);

    if ~b2
        fprintf(['\n  B2 FAILED: cc are NOT reaching the integrator. Most likely the ' ...
                 'chart data scope, or a stale copy of the .slx. Rebuild with ' ...
                 'make_coulomb_model.\n']);
    end
    if ~b1
        fprintf(['\n  B1 FAILED: cc = 0 is not a no-op through the model even though ' ...
                 'the FUNCTION gate passed. Something other than friction changed.\n']);
    end

    % Per-axis effect, for context rather than as a gate.
    lbl = {'X', 'Theta', 'Y'};
    fprintf('\n  per-axis max |B3 - B1| :');
    for k = 1:3
        fprintf('  %s %.3e', lbl{k}, max(abs(o3.q_with(:,k) - o1.q_with(:,k))));
    end
    fprintf('\n');

    fprintf('\nRESULT: %s\n\n', tf2s(b1 && b2));
end

% ── helpers ─────────────────────────────────────────────────────────────────

function cfg = set_cc(cfg, c1, c2, cy_)
% The ONLY correct place to set the Coulomb magnitudes. gtd_run_simulation's
% push_params copies cfg.cc1/cc2/ccy into the base workspace on every call, so a
% direct assignin to base would be silently overwritten by the cfg value.
    cfg.cc1 = c1;
    cfg.cc2 = c2;
    cfg.ccy = cy_;
end

function rec = pick_record(cfg, id)
    records = gtd_build_records(cfg);
    idx = find(strcmp({records.id}, id), 1);
    assert(~isempty(idx), 'record id %s not found', id);
    rec = records(idx);
end

function s = tf2s(b)
    if b, s = 'PASS'; else, s = 'FAIL'; end
end
