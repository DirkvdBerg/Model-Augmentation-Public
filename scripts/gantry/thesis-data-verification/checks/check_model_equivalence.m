function check_model_equivalence()
% CHECK_MODEL_EQUIVALENCE  C1 and C12 of IMPLEMENTATION.md.
%
% Stated before the run:
%   C1  the generator model (model/gantry_tdg_2025a.slx: three unrecorded loops removed,
%       noise block added) with d = 0 reproduces the vendored original model
%       (vendor/original/gantry_additional_state_coulomb_2025a.slx) BITWISE on the recorded
%       signals q_aug and delta_a_ode, for one record with motion, multisine and friction.
%       PASS iff max |difference| == 0 on both.
%   C12 the force logged at the plant input equals the saved u_total + d:
%       PASS iff max |u_plant - (u_total + d)| <= 1e-6 N, with d = 0 and with a nonzero d.
%       REVISED after the first run (2026-09-26, logs/C1_model_equivalence.log): the
%       criterion stated first, 1e-9 N, FAILED at 1.2e-8 / 1.5e-8 N. Cause: rounding
%       accumulated in the controller states between lsim (the saved u_fb) and the
%       Simulink LTI block, 1.7e-11 of the 700 N peak force. A plumbing error shows the
%       test force itself (1 N rms, about 4 N peak) and a one-sample shift about 40 N, so
%       1e-6 N keeps six orders of margin to both.
%   Also reported: wall time per run of both models.
    here = fileparts(fileparts(mfilename('fullpath')));                % scripts/gantry/thesis-data-verification
    addpath(fullfile(fileparts(fileparts(fileparts(here))), 'Thesis-writeup', 'Code', 'Data'));   % the generator
    addpath(fullfile(here, 'vendor'));     % gantrySystem.m, gantrySystemCoriolisCentripetal.m: the original model's other loops
    cfg = tdg_init(true);
    rng(12345);                                   % test signals only, not a record seed

    % test record: Y_op 0.15, three S-curve moves on X and Y at 0.75 max, A_prod multisine
    Y_op = 0.15;
    plant = gtd_build_plant(Y_op, cfg, 1);
    N = cfg.N_record;  t = (0:N-1)' * cfg.ts;
    r = repmat([0, 0, Y_op], N, 1);
    k0 = cfg.n_hold + 1;
    tg = [0.10 0.25; -0.12 0.02; 0.05 -0.20];      % [X_sym, Y] targets
    cur = [0, Y_op];
    for m = 1:size(tg, 1)
        px = sprof(tg(m,1) - cur(1), 1.5, 22.5, 0.012, cfg.ts);
        py = sprof(tg(m,2) - cur(2), 1.5, 37.5, 0.012, cfg.ts);
        K = max(numel(px), numel(py));
        px = [px; px(end)*ones(K-numel(px),1)]; py = [py; py(end)*ones(K-numel(py),1)]; %#ok<AGROW>
        idx = k0 + (0:K-1);
        r(idx, :) = [cur(1)+px, cur(1)+px, cur(2)+py];
        r(idx(end)+1:end, :) = repmat(r(idx(end), :), N - idx(end), 1);
        cur = tg(m, :);
        k0 = idx(end) + 1 + round(0.3/cfg.ts);
    end
    df = 1/cfg.t_record;  bins = round((cfg.f_low:cfg.f_step:cfg.f_high)/df) + 1;
    flog = zeros(N, 3);
    A = [cfg.A_sym, cfg.A_anti, cfg.A_Y];
    for ch = 1:3
        X = zeros(N, 1);  X(bins) = exp(1j*2*pi*rand(numel(bins), 1));
        x = ifft(X, 'symmetric');  flog(:, ch) = A(ch) * x / rms(x);
    end
    f = gtd_logical_to_stage(flog, cfg);
    rec = struct('id', 'eqcheck');

    c_orig = cfg;  c_orig.mdl = cfg.mdl_orig;
    o1 = gtd_run_simulation(rec, r, t, f, plant, c_orig, zeros(N, 3));
    o2 = gtd_run_simulation(rec, r, t, f, plant, cfg, zeros(N, 3));
    dq = max(abs(o1.q_with - o2.q_with), [], 'all');
    dd = max(abs(o1.da_with - o2.da_with), [], 'all');
    fprintf('model run time: original %.1f s, generator model %.1f s\n', o1.sim_s, o2.sim_s);
    fprintf('record: peak |q - r| %.3g m, rails stuck (|v|<1e-3) %s %%\n', max(abs(o2.q_with - r), [], 'all'), ...
            mat2str(round(100*mean(abs(diff(o2.q_with)*cfg.fs) < 1e-3), 1)));
    fprintf('C1  max|dq| = %.3g m, max|d delta_a| = %.3g m  -> %s\n', dq, dd, pf(dq == 0 && dd == 0));

    e0 = max(abs(o2.u_plant - (o2.u_total + o2.d)), [], 'all');
    dn = randn(N, 3);                              % 1 N rms test force, not the calibrated noise
    o3 = gtd_run_simulation(rec, r, t, f, plant, cfg, dn);
    e1 = max(abs(o3.u_plant - (o3.u_total + o3.d)), [], 'all');
    fprintf('C12 max|u_plant - (u_total + d)|: d = 0 %.3g N, d ~ 1 N white %.3g N  -> %s\n', e0, e1, pf(e0 <= 1e-6 && e1 <= 1e-6));
    fprintf('    effect of the 1 N test force on q: rms %s m\n', mat2str(rms(o3.q_with - o2.q_with), 3));
end

function p = sprof(D, vmax, amax, T3, ts)
    if D == 0, p = 0; return; end
    m = thirdOrderSetpointETEL(abs(D), vmax, amax, amax/T3, Inf, ts);
    p = sign(D) * m(:, 1);
end

function s = pf(ok)
    if ok, s = 'PASS'; else, s = 'FAIL'; end
end
