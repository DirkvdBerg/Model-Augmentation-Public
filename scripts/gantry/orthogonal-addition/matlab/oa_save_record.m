function oa_save_record(out, record, cfg)
% OA_SAVE_RECORD  Copy of Matlab-scripts/Augmentation/data/gtd_save_record.m for the OA datasets.
%   Identical schema, plus delta_a / vdelta_a / x_aug ALWAYS present and ZERO: the q1 loop does not
%   log absorber states, and the Python loaders (_load_record_float64, load_mat_aug) read these
%   fields. Zero is recorded in PROVENANCE.txt so no one mistakes it for a measured state.

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
    S.amp_rms      = out.amp;
    S.seed         = record.seed;
    S.track        = cfg.track;
    z = zeros(size(q, 1), 1);
    S.delta_a  = single(z);
    S.vdelta_a = single(z);
    S.x_aug    = single([z, z]);

    if ~exist(cfg.out_dir, 'dir'), mkdir(cfg.out_dir); end
    save(fullfile(cfg.out_dir, [record.id '.mat']), '-struct', 'S');
end
