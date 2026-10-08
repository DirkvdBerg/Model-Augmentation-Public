function info = gtd_save_record(out, record, cfg, fpath, meta)
% GTD_SAVE_RECORD  Write one record's signals in the spec 1.12 schema, in double.
%   info = GTD_SAVE_RECORD(out, record, cfg, fpath, meta) saves the copied schema
%   (u_total, u_fb, f_sim, y, x_logical, r_sim, Y_trajectory, t_sim, fs, dt, split,
%   amp_rms, seed, track, delta_a, vdelta_a, x_aug) to fpath (MAT v7, compressed,
%   readable by scipy.io.loadmat) and returns the file size.
%
%   Units: positions m, velocities m/s, forces N, time s; amp_rms = [A_sym (N),
%   A_anti (N m, a yaw torque), A_Y (N)]. Since D-225 y is the MEASURED stage position
%   [X1 X2 Y] = true q + encoder noise v (the signal the controller saw, as on the
%   machine); y_true is q; x_logical, Y_trajectory and delta_a are ground truth from q.
%
%   THESIS-DATA changes against scripts/gantry/thesis-data-verification/vendor/original/gtd_save_record.m:
%     - every signal in double (NOISE-INJECTION section 5: float32 rounds Y by up to
%       8.6 nm, the size of the noise); dt double
%     - v_enc: the encoder noise [X1 X2 Y] (m), diagnostics only, never a model input;
%       zeros in the noise-free version (D-225; replaces D-218's d_in force). y_true: q.
%     - seeds: seed (= seed_ms, the multisine seed, as before), seed_ms, seed_sp
%       (setpoint draws), noise_seed (NaN in the noise-free version)
%     - version ('noisy' | 'noise-free'), pair (the twin's file name)
%     - meta: the record's settings, the truth and controller parameters, band, line
%       step, realised multisine scale, post-simulation check, noise model, generator
%       provenance (see SPEC.md; built by the driver)
%     - explicit output path (the version and split folders are the driver's)

    P = cfg.P;  ts = cfg.ts;
    q = double(out.q_with);
    q_logical = ((P') \ q')';                      % stage -> logical positions
    qdot_logical = zeros(size(q_logical));
    for j = 1:3
        qdot_logical(:,j) = gradient(q_logical(:,j), ts);
    end

    S.u_total      = double(out.u_total);
    S.u_fb         = double(out.u_fb);
    S.f_sim        = double(out.f_ms);
    S.y            = double(out.y_meas);          % D-225: measured = q + v
    S.y_true       = q;
    S.x_logical    = [q_logical, qdot_logical];
    S.r_sim        = double(out.r_sim);
    S.Y_trajectory = q(:,3);
    S.t_sim        = double(out.t_sim);
    S.fs           = cfg.fs;
    S.dt           = ts;
    S.split        = record.split;
    S.amp_rms      = meta.amp_applied;
    S.seed         = record.seed_ms;
    S.track        = 'thesis_TAF';
    da             = double(out.da_with);
    vda            = gradient(da, ts);
    S.delta_a      = da;
    S.vdelta_a     = vda;
    S.x_aug        = [da, vda];
    % THESIS-DATA additions
    S.v_enc        = double(out.v);
    S.seed_ms      = record.seed_ms;
    S.seed_sp      = record.seed_sp;
    S.noise_seed   = meta.noise_seed;
    S.version      = meta.version;
    S.pair         = meta.pair;
    S.meta         = meta;

    d = fileparts(fpath);
    if ~exist(d, 'dir'), mkdir(d); end
    assert(~isfile(fpath), 'gtd_save_record: refusing to overwrite %s', fpath);
    % THESIS-DATA: atomic write. A job killed during save (RAM watchdog) must not leave a
    % truncated file under the final name, which a resumed run would skip as done.
    tmp = [fpath(1:end-4) '_part.mat'];
    save(tmp, '-struct', 'S', '-v7');
    movefile(tmp, fpath);
    fi = dir(fpath);
    info = struct('bytes', fi.bytes);
end
