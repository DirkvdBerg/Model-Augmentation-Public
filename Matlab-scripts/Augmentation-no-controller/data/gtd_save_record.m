function gtd_save_record(out, record, cfg)
% GTD_SAVE_RECORD  Write one record's signals in the spec 1.12 schema.
%   Saves u_total, u_fb, f_sim, y, x_logical, x_aug=[delta_a, vdelta_a] (MSD),
%   r_sim, Y_trajectory, t_sim, fs, dt, split, amp_rms, plus provenance
%   (multisine seed, track id).
%
%   Full augmented ground-truth state = [x_logical (6), x_aug (2)] = 8 states,
%   for encoder pre-training. Baseline states are in LOGICAL coordinates (project
%   convention); the MSD states delta_a, vdelta_a are scalar relative coordinates
%   along Y (frame-agnostic). Velocities (qdot_logical, vdelta_a) are obtained by
%   differentiation, consistent with the reference generators.
%
%   amp_rms = [A_sym, A_anti, A_Y] has MIXED units [N, N*m, N]: A_anti is a yaw
%   TORQUE (logical coordinate 2 is the tilt angle), not a force (see D-080).

%   CHANGED (open-loop variant): open loop there is no reference and no
%   controller, so r_sim and u_fb are zeros and u_total = f_ms exactly. Two
%   further differences from the closed-loop schema, both forced by the record:
%     - DOUBLE precision, not single. Positions here sit around 0.1 to 0.4 m, so
%       float32 quantizes them at ~3e-8 m, thirty times coarser than the 1e-9 m
%       replay tolerance this data has to be checked against. A single copy is
%       written alongside so the storage effect can still be isolated, which is
%       the pairing generate_openloop_record.m introduced.
%     - Velocities are the integrator's own states, not gradient() of position.
%     - u_half (the input at t + ts/2) is stored so a replay can reproduce the
%       stage-sampled RK4 exactly; a plain ZOH replay differs by ~2 %, which is
%       a sampling artefact and not a model error (check_openloop_drift.m).

    if isfield(cfg, 'openloop') && cfg.openloop
        save_openloop(out, record, cfg);
        return
    end

    P = cfg.P;  ts = cfg.ts;
    q = double(out.q_with);

    q_logical = ((P') \ q')';                      % stage -> logical positions
    qdot_logical = zeros(size(q_logical));
    for j = 1:3
        qdot_logical(:,j) = gradient(q_logical(:,j), ts);
    end

    S.u_total      = single(out.u_total);
    S.u_fb         = single(out.u_fb);
    S.f_sim        = single(out.f_ms);
    S.y            = single(q);
    S.x_logical    = single([q_logical, qdot_logical]);
    S.r_sim        = single(out.r_sim);
    S.Y_trajectory = single(q(:,3));
    S.t_sim        = single(out.t_sim);
    S.fs           = cfg.fs;
    S.dt           = single(ts);
    S.split        = record.split;
    S.amp_rms      = out.amp;                       % applied [A_sym, A_anti, A_Y] = [N, N*m, N]
    S.seed         = record.seed;
    S.track        = cfg.track;
    if cfg.use_msd
        da    = double(out.da_with);
        vda   = gradient(da, ts);                   % MSD velocity (differentiated, consistent w/ qdot)
        S.delta_a  = single(da);
        S.vdelta_a = single(vda);
        S.x_aug    = single([da, vda]);             % [delta_a, vdelta_a] augmentation states
    end

    if ~exist(cfg.out_dir, 'dir'), mkdir(cfg.out_dir); end
    save(fullfile(cfg.out_dir, [record.id '.mat']), '-struct', 'S');
end

% ── open-loop writer ────────────────────────────────────────────────────────

function save_openloop(out, record, cfg)
    N = size(out.u_total, 1);

    S.u_total      = double(out.u_total);          % == the applied stage force
    S.u_half       = double(out.f_half);           % same input at t + ts/2
    S.u_fb         = zeros(N, 3);                  % no controller
    S.f_sim        = double(out.f_ms);
    S.y            = double(out.q_with);
    S.x_logical    = double([out.q_logical, out.qd_logical]);   % exact velocities
    S.r_sim        = zeros(N, 3);                  % no reference
    S.Y_trajectory = double(out.q_with(:, 3));
    S.t_sim        = double(out.t_sim);
    S.fs           = cfg.fs;
    S.dt           = double(cfg.ts);
    S.split        = record.split;
    S.amp_rms      = out.amp;                      % applied stage RMS [F_X1, F_X2, F_Y] = N
    S.seed         = record.seed;
    S.track        = cfg.track;
    S.loop         = 'open';
    S.Y_op         = record.Y_op;                  % NOMINAL label of the operating point
    S.Y_start      = double(out.q_with(1, 3));     % where the record actually starts, after
                                                   % the discarded periods rectified it away
                                                   % from Y_op
    S.x0_logical   = double(out.x0_logical);       % 8-state IC of the SAVED record:
                                                   % [X Th Y da dX dTh dY vda]. A replay must
                                                   % seed with this, not with rest at Y_op.
    S.n_skip       = out.n_skip;
    S.t_skip       = out.t_skip;
    S.n_periods    = cfg.ol_n_periods;
    S.n_discard    = cfg.ol_n_discard;
    S.input_sampling = 'rk4-stage';
    S.rk4_vs_ode45_max = out.rk4_vs_ode45;
    if cfg.use_msd
        S.delta_a  = double(out.da_with);
        S.vdelta_a = double(out.vda_with);         % integrator state, not differentiated
        S.x_aug    = double([out.da_with, out.vda_with]);
    end

    if ~exist(cfg.out_dir, 'dir'), mkdir(cfg.out_dir); end
    save(fullfile(cfg.out_dir, [record.id '.mat']), '-struct', 'S');   % v7, so scipy.io.loadmat can read it

    fn = fieldnames(S);
    for i = 1:numel(fn)
        v = S.(fn{i});
        if isnumeric(v) && ~isscalar(v), S.(fn{i}) = single(v); end
    end
    save(fullfile(cfg.out_dir, [record.id '_single.mat']), '-struct', 'S');   % v7, so scipy.io.loadmat can read it
end
