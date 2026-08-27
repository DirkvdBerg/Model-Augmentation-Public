function out = gtd_run_simulation(record, ms, cfg)
% GTD_RUN_SIMULATION  Open-loop simulation of one record (the only impure fn).
%   out = GTD_RUN_SIMULATION(record, ms, cfg) integrates the 8-state gantry EOM
%   with the excitation ms as the ONLY input. No reference, no controller, no
%   feedforward, so u_total == the applied stage force and nothing has to be
%   reconstructed after the fact.
%
%   CHANGED (open-loop variant of Matlab-scripts/Augmentation/data). The
%   closed-loop version called Simulink through the base workspace and then
%   rebuilt u_fb with lsim(Cfb, r - q). Both are gone:
%     - Simulink carries the feedback path inside the model, which is exactly
%       what this folder exists to remove;
%     - the lsim reconstruction is meaningless without a controller.
%   Replaced by the fixed-step RK4 of generate_openloop_record.m, which is the
%   integrator already gated on this plant: with the input evaluated at the RK4
%   STAGE times it matches ode45(RelTol 1e-12) to five significant figures on
%   the rectified drift, where a zero-order hold on the same step is 2 % off
%   (check_openloop_drift.m).
%
%   States are integrated in LOGICAL coordinates, x = [q_l(3); delta_a; qd_l(3);
%   vdelta_a], and the record's y is the stage-coordinate position q_stage =
%   P' * q_logical. Velocities come out of the integrator EXACTLY and are passed
%   through in out.qd_logical, so gtd_save_record does not have to differentiate.
%
%   Returned struct: t_sim, r_sim (zeros), f_ms, f_half, q_with, q_logical,
%   qd_logical, u_fb (zeros), u_total, da_with, vda_with, q_without, da_without,
%   rk4_vs_ode45.

    assert(isfield(cfg,'openloop') && cfg.openloop, 'gtd_run_simulation:mode', ...
        'this is the open-loop variant; cfg.openloop must be true');
    assert(cfg.use_msd, 'gtd_run_simulation:msd', ...
        ['open-loop generation is implemented for the 8-state augmented plant only ' ...
         '(USE_MSD = true). The 6-state baseline has no MATLAB open-loop path here; ' ...
         'it is simulated in Python from the same u_total.']);

    ts = cfg.ts;
    f_stage = ms.f_stage;  f_half = ms.f_half;
    N = size(f_stage, 1);                    % SIMULATED length (record + discarded periods)
    t = (0:N-1)' * ts;
    nskip = ms.n_skip;                       % leading samples thrown away after integration

    u_log_grid = (cfg.P * f_stage')';        % stage -> logical forces (F_l = P * F_s)
    u_log_half = (cfg.P * f_half')';

    p  = {cfg.m1, cfg.m2, cfg.mb, cfg.mh_rigid, cfg.Lb, cfg.Jb, cfg.Jh, cfg.d, ...
          cfg.cg1, cfg.cg2, cfg.cb1, cfg.cb2, cfg.cy, cfg.kb1, cfg.kb2, ...
          cfg.ma, cfg.ka, cfg.ca, cfg.L0};
    f8 = @(x, u) gantrySystemExtended(u, x, p{:});

    x0 = [0; 0; record.Y_op; 0; 0; 0; 0; 0];             % from rest at the operating point

    % ── fixed-step RK4, input at the RK stage times ─────────────────────────
    x = x0;
    q_logical = zeros(N, 3);  qd_logical = zeros(N, 3);
    da = zeros(N, 1);  vda = zeros(N, 1);
    tic
    for k = 1:N
        q_logical(k, :)  = x(1:3)';
        qd_logical(k, :) = x(5:7)';
        da(k) = x(4);  vda(k) = x(8);
        u1 = u_log_grid(k, :)';                          % t_k
        u2 = u_log_half(k, :)';                          % t_k + ts/2, stages 2 and 3
        if k < N, u4 = u_log_grid(k+1, :)'; else, u4 = cfg.P * ms.fun(t(k) + ts)'; end
        k1 = f8(x, u1);             k2 = f8(x + 0.5*ts*k1, u2);
        k3 = f8(x + 0.5*ts*k2, u2); k4 = f8(x + ts*k3, u4);
        x  = x + (ts/6)*(k1 + 2*k2 + 2*k3 + k4);
    end
    fprintf('  RK4 %d steps (stage-sampled input) in %.1f s\n', N, toc);

    q_stage = (cfg.P' * q_logical')';

    % ── fidelity reference: ode45 at tight tolerance over the first window ──
    rk4_vs_ode45 = [NaN NaN NaN];
    if cfg.ol_ode45_tol > 0
        nref = round(cfg.ol_ode45_tol / ts);
        opt  = odeset('RelTol', 1e-12, 'AbsTol', 1e-14);
        ulog = @(tq) (cfg.P * ms.fun(tq)')';
        [~, Xr] = ode45(@(tq, xx) f8(xx, ulog(tq)'), t(1:nref), x0, opt);
        rk4_vs_ode45 = max(abs(Xr(:, 1:3) - q_logical(1:nref, :)), [], 1);
        fprintf('  RK4 vs ode45(1e-12) over %.2f s, max |dq| logical [%.3e %.3e %.3e] m\n', ...
                cfg.ol_ode45_tol, rk4_vs_ode45);
    end

    % ── WITHOUT excitation (informativeness baseline) ───────────────────────
    % Open loop from rest with zero force there is nothing to integrate PROVIDED
    % x0 is an equilibrium; check it rather than assume it, and only pay for a
    % second RK4 pass if it is not.
    dx0 = f8(x0, zeros(3,1));
    if max(abs(dx0)) < 1e-12
        q_without  = repmat((cfg.P' * x0(1:3))', N, 1);
        da_without = zeros(N, 1);
    else
        warning('gtd_run_simulation:notEquilibrium', ...
            'x0 is not an equilibrium (max |dx| = %.3e); running the zero-force pass', ...
            max(abs(dx0)));
        [q_without, da_without] = rk4_zero(f8, x0, N, ts, cfg.P);
    end

    % ── discard the leading periods ─────────────────────────────────────────
    % Everything above was integrated from rest at Y_op. The saved record starts
    % where the rectified drift has settled, so it no longer starts from rest and
    % no longer starts at Y_op: x0_logical below is what a replay must be seeded
    % with, and Y_op survives only as the NOMINAL label of the operating point.
    if nskip > 0
        d = q_stage(nskip+1, :) - q_stage(1, :);
        fprintf('  discarded %.2f s (%d samples), drift inside it [%+.3e %+.3e %+.3e] m\n', ...
                nskip*ts, nskip, d);
        keep = (nskip+1) : N;
        q_stage = q_stage(keep, :);  q_logical = q_logical(keep, :);
        qd_logical = qd_logical(keep, :);  da = da(keep);  vda = vda(keep);
        f_stage = f_stage(keep, :);  f_half = f_half(keep, :);
        q_without = q_without(keep, :);  da_without = da_without(keep);
        N = numel(keep);  t = (0:N-1)' * ts;          % saved record restarts at t = 0
    end
    x0_logical = [q_logical(1, :), da(1), qd_logical(1, :), vda(1)];

    out = struct('t_sim',t, 'r_sim',zeros(N,3), 'f_ms',f_stage, 'f_half',f_half, ...
                 'q_with',q_stage, 'q_logical',q_logical, 'qd_logical',qd_logical, ...
                 'u_fb',zeros(N,3), 'u_total',f_stage, ...
                 'da_with',da, 'vda_with',vda, ...
                 'q_without',q_without, 'da_without',da_without, ...
                 'rk4_vs_ode45',rk4_vs_ode45, 'Y_op',record.Y_op, ...
                 'x0_logical',x0_logical, 'n_skip',nskip, 't_skip',nskip*ts);
end

% ── helpers ─────────────────────────────────────────────────────────────────

function [q_stage, da] = rk4_zero(f8, x0, N, ts, P)
    x = x0;  q = zeros(N,3);  da = zeros(N,1);  z = zeros(3,1);
    for k = 1:N
        q(k,:) = x(1:3)';  da(k) = x(4);
        k1 = f8(x, z);             k2 = f8(x + 0.5*ts*k1, z);
        k3 = f8(x + 0.5*ts*k2, z); k4 = f8(x + ts*k3, z);
        x  = x + (ts/6)*(k1 + 2*k2 + 2*k3 + k4);
    end
    q_stage = (P' * q')';
end
