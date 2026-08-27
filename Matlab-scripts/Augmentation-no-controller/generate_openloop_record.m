function generate_openloop_record()
% GENERATE_OPENLOOP_RECORD  One open-loop record: no controller, no reference.
%
% Integrates gantrySystemExtended.m directly. Simulink is not used, so there is no
% feedback path and u_total IS the applied force rather than a reconstruction of what a
% controller did (gtd_run_simulation.m rebuilds u_fb with lsim after the fact).
%
% v2, 2026-08-10: the input is now evaluated at the RK4 STAGE TIMES rather than held
% constant across the step. check_openloop_drift.m measured the ZOH version 2 % away from
% ode45 on the rectified drift, while the stage-sampled version matches ode45 to five
% significant figures at full step size. The input is a multisine, so it is known
% analytically and can be evaluated at t, t+ts/2 and t+ts exactly; no interpolation.
%
% Excitation is a zero-mean multisine in [130,180] Hz. NOTE, contrary to what v1 of this
% file claimed: a zero-mean force does NOT leave the position where it started. The plant
% is nonlinear (M depends on Y and delta_a), so oscillation rectifies into a DC response
% and Y slides about 1.8 mm before settling. That drift is COMMON MODE, the 6-state
% baseline reproduces it to 2.6e-11 m, so it cancels in any model comparison, but the
% record does not sit at its nominal operating point and that matters when building a set
% at specified Y values.
%
% Saves in the gtd_save_record schema so scripts/gantry/msd-offset/plant.py can load it,
% in DOUBLE precision, plus a single-precision copy. The pair isolates float32 storage
% from solver error. u_half is saved as well so a replay can reproduce the stage-sampled
% integration exactly; a plain ZOH replay of u_total will differ by the 2 % noted above,
% which is a sampling artefact rather than a model error.

    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(here));
    addpath(here); addpath(fullfile(here, 'data'));

    cfg = gtd_config('augmentation', true, 0.10);  % MA_FRAC 0.10 = what the records use
    ts  = cfg.ts;  fs = cfg.fs;                    % 5e-5 s, 20 kHz
    T   = 12.0;    N  = round(T / ts);
    t   = (0:N-1)' * ts;
    Y0  = 0.10;                                    % nominal operating point

    % ---- zero-mean multisine, analytic in t, stage coordinates [X1, X2, Y] --
    rng(0);
    fr = (cfg.f_low : 1 : cfg.f_high)';            % 130..180 Hz, 1 Hz spacing
    ph = 2*pi*rand(numel(fr), 3);
    A_rms = [20, 20, 15];                          % [N] HEURISTIC: half the closed-loop A_sym/A_Y

    raw = zeros(N, 3);
    for c = 1:3, raw(:, c) = sum(cos(2*pi*t*fr' + ph(:, c)'), 2); end
    sc = A_rms ./ std(raw, 0, 1);                  % fix the scale once, on the record grid
    ustage = @(tq) arrayfun(@(c) sc(c) * sum(cos(2*pi*tq*fr + ph(:, c))), 1:3);
    ulog   = @(tq) (cfg.P * ustage(tq)')';

    u_stage = raw .* sc;                           % on-grid values, for the record
    u_half  = zeros(N, 3);                         % midpoint values, for exact replay
    for k = 1:N, u_half(k, :) = ustage(t(k) + 0.5*ts); end

    fprintf('multisine: %d lines %g-%g Hz, stage RMS [%.1f %.1f %.1f] N, DC [%.2e %.2e %.2e]\n', ...
        numel(fr), cfg.f_low, cfg.f_high, std(u_stage), mean(u_stage));

    % ---- fixed-step RK4, input at the RK stage times ------------------------
    p = {cfg.m1, cfg.m2, cfg.mb, cfg.mh_rigid, cfg.Lb, cfg.Jb, cfg.Jh, cfg.d, ...
         cfg.cg1, cfg.cg2, cfg.cb1, cfg.cb2, cfg.cy, cfg.kb1, cfg.kb2, ...
         cfg.ma, cfg.ka, cfg.ca, cfg.L0};
    f8 = @(x, u) gantrySystemExtended(u, x, p{:});

    x = [0; 0; Y0; 0; 0; 0; 0; 0];                 % from rest at the nominal point
    q_logical = zeros(N, 3);  qd_logical = zeros(N, 3);
    da = zeros(N, 1);  vda = zeros(N, 1);
    u_log_grid = (cfg.P * u_stage')';
    u_log_half = (cfg.P * u_half')';
    tic
    for k = 1:N
        q_logical(k, :)  = x(1:3)';
        qd_logical(k, :) = x(5:7)';                % exact velocities, NOT differentiated
        da(k) = x(4);  vda(k) = x(8);
        u1 = u_log_grid(k, :)';                    % t_k
        u2 = u_log_half(k, :)';                    % t_k + ts/2, used by stages 2 and 3
        if k < N, u4 = u_log_grid(k+1, :)'; else, u4 = ulog(t(k) + ts)'; end
        k1 = f8(x, u1);             k2 = f8(x + 0.5*ts*k1, u2);
        k3 = f8(x + 0.5*ts*k2, u2); k4 = f8(x + ts*k3, u4);
        x  = x + (ts/6)*(k1 + 2*k2 + 2*k3 + k4);
    end
    fprintf('RK4 %d steps (stage-sampled input) in %.1f s\n', N, toc);

    q_stage = (cfg.P' * q_logical')';

    % ---- accuracy reference: ode45 on the first 0.5 s -----------------------
    nref = round(0.5 / ts);
    opt  = odeset('RelTol', 1e-12, 'AbsTol', 1e-14);
    [~, Xr] = ode45(@(tq, xx) f8(xx, ulog(tq)'), t(1:nref), [0;0;Y0;0;0;0;0;0], opt);
    ref_err = max(abs(Xr(:, 1:3) - q_logical(1:nref, :)), [], 1);
    fprintf('RK4 vs ode45(1e-12) over 0.5 s, max |dq| logical: [%.3e %.3e %.3e] m\n', ref_err);

    % ---- save --------------------------------------------------------------
    outdir = fullfile(root, 'data', 'gantry', 'matlab', 'trajectory', 'openloop');
    if ~exist(outdir, 'dir'), mkdir(outdir); end

    S = struct();
    S.u_total   = u_stage;      S.u_fb = zeros(N, 3);   S.f_sim = u_stage;
    S.u_half    = u_half;                                   % midpoint input, for exact replay
    S.y         = q_stage;      S.x_logical = [q_logical, qd_logical];
    S.r_sim     = zeros(N, 3);  S.Y_trajectory = q_stage(:, 3);
    S.t_sim     = t;            S.fs = fs;              S.dt = ts;
    S.split     = 'openloop';   S.amp_rms = A_rms;      S.seed = 0;
    S.track     = 'openloop-augmentation-band';
    S.delta_a   = da;           S.vdelta_a = vda;       S.x_aug = [da, vda];
    S.Y_op      = Y0;           S.rk4_vs_ode45_max = ref_err;
    S.input_sampling = 'rk4-stage';
    save(fullfile(outdir, 'OL1_multisine_Yp10.mat'), '-struct', 'S');

    fn = fieldnames(S);
    for i = 1:numel(fn)
        v = S.(fn{i});
        if isnumeric(v) && ~isscalar(v), S.(fn{i}) = single(v); end
    end
    save(fullfile(outdir, 'OL1_multisine_Yp10_single.mat'), '-struct', 'S');

    fprintf('wrote %s\n', fullfile(outdir, 'OL1_multisine_Yp10.mat'));
    fprintf('y stage std [%.4e %.4e %.4e] m\n', std(q_stage));
    fprintf('Y drift over record %.4e m, so the record ends at Y = %.6f not %.6f\n', ...
        q_stage(end,3) - Y0, q_stage(end,3), Y0);
end
