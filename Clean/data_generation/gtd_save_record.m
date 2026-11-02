function info = gtd_save_record(out, record, cfg, fpath, meta)
% GTD_SAVE_RECORD  Write one record version to fpath (MAT v7, compressed, scipy-readable).
%   info = GTD_SAVE_RECORD(out, record, cfg, fpath, meta) saves, in double:
%     u_total, u_fb, f_sim (the injected multisine), y (measured = true q + encoder noise v, the
%     signal the controller saw), y_true (q), x_logical, r_sim, Y_trajectory, t_sim, fs, dt,
%     split, amp_rms [A_sym N, A_anti N m, A_Y N], seed (= seed_ms), track, delta_a, vdelta_a,
%     x_aug, v_enc (v, zeros in the noise-free version; diagnostics only), seed_ms, seed_sp,
%     noise_seed (NaN in the noise-free version), version, pair (the file of the other
%     version) and meta (built by generate_thesis_data).
%   Positions m, velocities m/s, forces N, time s. The write goes to <file>_part.mat first, so a
%   killed job never leaves a truncated file under the final name. Returns the file size.
    P = cfg.P;  ts = cfg.ts;
    q = out.q_with;
    q_generalised = ((P') \ q')';                      % stage -> generalised positions
    qdot_generalised = zeros(size(q_generalised));
    for j = 1:3
        qdot_generalised(:,j) = gradient(q_generalised(:,j), ts);
    end

    S.u_total      = out.u_total;
    S.u_fb         = out.u_fb;
    S.f_sim        = out.f_ms;
    S.y            = out.y_meas;                  % measured = q + v
    S.y_true       = q;
    S.x_logical    = [q_generalised, qdot_generalised];
    S.r_sim        = out.r_sim;
    S.Y_trajectory = q(:,3);
    S.t_sim        = out.t_sim;
    S.fs           = cfg.fs;
    S.dt           = ts;
    S.split        = record.split;
    S.amp_rms      = meta.amp_applied;
    S.seed         = record.seed_ms;
    S.track        = 'thesis_TAF';
    da             = out.da_with;
    vda            = gradient(da, ts);
    S.delta_a      = da;
    S.vdelta_a     = vda;
    S.x_aug        = [da, vda];
    S.v_enc        = out.v;
    S.seed_ms      = record.seed_ms;
    S.seed_sp      = record.seed_sp;
    S.noise_seed   = meta.noise_seed;
    S.version      = meta.version;
    S.pair         = meta.pair;
    S.meta         = meta;

    d = fileparts(fpath);
    if ~exist(d, 'dir'), mkdir(d); end
    assert(~isfile(fpath), 'gtd_save_record: refusing to overwrite %s', fpath);
    tmp = [fpath(1:end-4) '_part.mat'];
    save(tmp, '-struct', 'S', '-v7');
    movefile(tmp, fpath);
    fi = dir(fpath);
    info = struct('bytes', fi.bytes);
end
