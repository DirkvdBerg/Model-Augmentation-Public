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

    P = cfg.P;  ts = cfg.ts;
    q = double(out.q_with);
    % CLOSED-LOOP-NOISE: (CN-009) q is the TRUE position; the recorded y is the MEASURED one, q + v,
    % the signal that fed the controller. cfg.save_double stores the signals in double: float32 at
    % |q| = 0.1-0.3 m has a 7-18 nm spacing, the size of the noise. Default (noise off) = production.
    if isfield(cfg, 'save_double') && cfg.save_double, prec = @double; else, prec = @single; end
    v = zeros(size(q));
    if isfield(out, 'v_enc'), v = double(out.v_enc); end
    ym = q + v;

    q_logical = ((P') \ q')';                      % stage -> logical positions
    qdot_logical = zeros(size(q_logical));
    for j = 1:3
        qdot_logical(:,j) = gradient(q_logical(:,j), ts);
    end

    S.u_total      = prec(out.u_total);
    S.u_fb         = prec(out.u_fb);
    S.f_sim        = prec(out.f_ms);
    S.y            = prec(ym);                        % CLOSED-LOOP-NOISE: measured
    S.x_logical    = prec([q_logical, qdot_logical]);
    S.r_sim        = prec(out.r_sim);
    S.Y_trajectory = prec(ym(:,3));                  % CLOSED-LOOP-NOISE: = y(:,3)
    S.t_sim        = prec(out.t_sim);
    S.fs           = cfg.fs;
    S.dt           = single(ts);
    S.split        = record.split;
    S.amp_rms      = out.amp;                       % applied [A_sym, A_anti, A_Y] = [N, N*m, N]
    S.seed         = record.seed;
    S.track        = cfg.track;
    if cfg.use_msd
        da    = double(out.da_with);
        vda   = gradient(da, ts);                   % MSD velocity (differentiated, consistent w/ qdot)
        S.delta_a  = prec(da);
        S.vdelta_a = prec(vda);
        S.x_aug    = prec([da, vda]);             % [delta_a, vdelta_a] augmentation states
    end

    if isfield(cfg, 'save_double') && cfg.save_double       % CLOSED-LOOP-NOISE: (CN-009)
        S.q_true = q;  S.v_enc = v;
        S.noise_seed = cfg.noise_seed;  S.noise_model = cfg.noise_model;
    end

    if ~exist(cfg.out_dir, 'dir'), mkdir(cfg.out_dir); end
    save(fullfile(cfg.out_dir, [record.id '.mat']), '-struct', 'S');
end
