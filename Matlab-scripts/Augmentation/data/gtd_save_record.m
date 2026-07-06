function gtd_save_record(out, record, cfg)
% GTD_SAVE_RECORD  Write one record's signals in the spec 1.12 schema.
%   Saves u_total, u_fb, f_sim, y, x_logical, delta_a (MSD), r_sim, Y_trajectory,
%   t_sim, fs, dt, split, amp_rms, plus provenance (multisine seed, track id).

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
    S.amp_rms      = out.amp;                       % applied [A_sym, A_anti, A_Y]
    S.seed         = record.seed;
    S.track        = cfg.track;
    if cfg.use_msd
        S.delta_a  = single(out.da_with);
    end

    if ~exist(cfg.out_dir, 'dir'), mkdir(cfg.out_dir); end
    save(fullfile(cfg.out_dir, [record.id '.mat']), '-struct', 'S');
end
