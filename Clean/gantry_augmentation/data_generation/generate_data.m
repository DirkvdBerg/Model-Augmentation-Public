% GENERATE_DATA  Generates the thesis data sets: every record simulated in closed loop with and without encoder noise.
%   Run from MATLAB in this folder. RECORDS and OUT_DIR below can be set before the call to override them.

if ~exist('RECORDS', 'var'), RECORDS = {}; end        % record names to generate, e.g. {'TR-P3', 'E6'}; empty generates all 42
if ~exist('OUT_DIR', 'var')                            % data folder; the default is the one the training code reads
    OUT_DIR = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
end

cfg = data_config();
addpath(cfg.here, fullfile(cfg.here, 'system'), fullfile(cfg.here, 'controller'), fullfile(cfg.here, 'inputs'));
dirs = struct('noisy', fullfile(OUT_DIR, cfg.name_noisy), 'free', fullfile(OUT_DIR, cfg.name_free));

% the truth model; Simulink writes its build files to a short temporary folder, since long paths exceed
% the 260-character limit of Windows
if ~isfile(fullfile(cfg.here, 'system', 'model', [cfg.mdl '.slx'])), build_model(); end
if ~exist(cfg.cache_dir, 'dir'), mkdir(cfg.cache_dir); end
Simulink.fileGenControl('set', 'CacheFolder', cfg.cache_dir, 'CodeGenFolder', cfg.cache_dir);
load_system(fullfile(cfg.here, 'system', 'model', [cfg.mdl '.slx']));

recs = record_table(cfg);
if ~isempty(RECORDS)
    assert(all(ismember(RECORDS, {recs.rec})), 'unknown record name in RECORDS');
    recs = recs(ismember({recs.rec}, RECORDS));
end
for k = 1:numel(recs)
    fprintf('%d/%d  %s  %s\n', k, numel(recs), recs(k).file, recs(k).desc);
    generate_record(recs(k), cfg, dirs);
end
write_manifest(record_table(cfg), cfg, dirs);

function generate_record(e, cfg, dirs)
% Simulate one record with and without encoder noise, check both against the limits and save them
    sd = cfg.split_dir.(e.set);
    file_noisy = fullfile(dirs.noisy, sd, [e.file '.mat']);
    file_free  = fullfile(dirs.free,  sd, [e.file '.mat']);
    assert(~isfile(file_noisy) && ~isfile(file_free), '%s exists; data files are never overwritten', e.file);

    loop = build_loop(e.Y_op, cfg);
    [r, t] = make_reference(e, cfg);
    ms = make_multisine(e, loop, cfg);
    s = precheck_limits(loop, r, ms.f_stage, cfg);
    % I3M, I5M and I6M use the same multisine scale as their E partners, so the only difference within
    % a pair is the motion; generate the partners first. With two partners (I3M), the lower scale is used
    fixed = ~isempty(e.scale_from);
    if fixed
        s = Inf;
        for j = 1:numel(e.scale_from)
            partner = fullfile(dirs.noisy, sd, [e.scale_from{j} '.mat']);
            assert(isfile(partner), '%s: generate its partner %s first', e.file, e.scale_from{j});
            P = load(partner, 'meta');
            s = min(s, P.meta.scale_realised);
        end
    end
    v = encoder_noise(cfg, e.seed_axes);

    % both versions must pass the limit check on the simulated truth; otherwise the multisine is
    % scaled down in steps of 0.1 and both are simulated again, so the versions differ only in the noise
    while true
        f = s * ms.f_stage;
        noisy = run_simulation(r, t, f, v, loop, cfg);
        free  = run_simulation(r, t, f, zeros(cfg.N_record, 3), loop, cfg);
        failed = unique([check_simulation(noisy, cfg), check_simulation(free, cfg)]);
        if isempty(failed), break; end
        fprintf('  limit check failed at scale %.2f: %s\n', s, strjoin(failed, ', '));
        assert(~fixed, '%s fails the limit check at the scale of its partners', e.file);
        assert(any(ms.A) && s > 0.05, '%s fails the limit check and has no multisine to scale down', e.file);
        s = round(10*s - 1) / 10;
    end

    % stored with each record; the training code builds the controller of a record from meta.controller
    m = struct('file', e.file, 'rec', e.rec, 'set', e.set, 'block', e.block, 'class', e.class, ...
               'Y_op', e.Y_op, 'desc', e.desc, 'p', e.p, ...
               'controller', struct('name', 'K1', 'design_Y', cfg.K1_design_Y, 'gain', 1, 'fbw_Hz', cfg.fbw), ...
               'truth', truth_params(cfg), 'band_Hz', [cfg.f_low cfg.f_high], ...
               'amp_applied', s * ms.A, 'scale_realised', s, 'seed_ms', e.seed_ms, 'seed_sp', e.seed_sp);
    m.version = 'noisy';
    m.noise_seed = e.seed_noise;
    m.noise = struct('axis_seeds', e.seed_axes, 'fc_Hz', cfg.noise_fc, 'g_m_per_rtHz', cfg.noise_g, ...
                     'floor_m_per_rtHz', cfg.noise_floor);
    save_record(file_noisy, t, r, f, noisy, m);
    m.version = 'noise-free';
    m.noise_seed = NaN;
    m.noise = struct();
    save_record(file_free, t, r, f, free, m);
end

function failed = check_simulation(out, cfg)
% Names of the limits the simulated truth violates: position, yaw, speed, peak and rms plant force
    lim = cfg.lim;  q = out.q;  vel = diff(q) * cfg.fs;
    ok = struct('pos',   all(max(abs(q)) <= [lim.pos_X, lim.pos_X, lim.pos_Y]), ...
                'yaw',   max(abs(q(:,1) - q(:,2))) <= lim.diff, ...
                'vel',   all(max(abs(vel)) <= lim.vel), ...
                'fpeak', all(max(abs(out.u)) <= lim.force_peak), ...
                'frms',  all(rms(out.u) <= lim.force_rms));
    names = fieldnames(ok);
    failed = names(~cell2mat(struct2cell(ok)))';
end

function save_record(path, t, r, f, out, meta)
% One record file: time, reference, injected force, plant force, measured and true position [s, m, N]
    S = struct('t_sim', t, 'r_sim', r, 'f_sim', f, 'u_total', out.u, 'y', out.y, 'y_true', out.q, 'meta', meta);
    if ~exist(fileparts(path), 'dir'), mkdir(fileparts(path)); end
    save(path, '-struct', 'S', '-v7');
end

function write_manifest(recs, cfg, dirs)
% MANIFEST.csv per data set folder, one row per record file on disk; the training code takes its
% splits from the block column. realisation, twin and twin_of are constant and keep the file format
    vers = {'noisy', dirs.noisy, cfg.name_free;  'noise-free', dirs.free, cfg.name_noisy};
    for k = 1:2
        root = vers{k, 2};
        if ~exist(root, 'dir'), continue; end
        fid = fopen(fullfile(root, 'MANIFEST.csv'), 'w');
        fprintf(fid, ['file,version,folder,record,set,block,realisation,twin,twin_of,class,Y_op,controller,' ...
                      'excitation,A_sym_N,A_anti_Nm,A_Y_N,scale,seed_ms,seed_sp,noise_seed,pair_file,MB,description\n']);
        for j = 1:numel(recs)
            e = recs(j);
            sd = cfg.split_dir.(e.set);
            path = fullfile(root, sd, [e.file '.mat']);
            if ~isfile(path), continue; end
            S = load(path, 'meta');  m = S.meta;
            info = dir(path);
            fprintf(fid, '%s,%s,%s,%s,%s,%s,1,0,,%s,%.4f,%s,%s,%.4f,%.4f,%.4f,%.2f,%.0f,%.0f,%.0f,%s,%.1f,"%s"\n', ...
                    e.file, vers{k, 1}, sd, e.rec, e.set, e.block, e.class, e.Y_op, m.controller.name, ...
                    e.p.excitation, m.amp_applied, m.scale_realised, e.seed_ms, e.seed_sp, m.noise_seed, ...
                    [vers{k, 3} '/' sd '/' e.file '.mat'], info.bytes/1e6, e.desc);
        end
        fclose(fid);
    end
end

function t = truth_params(cfg)
% The parameters of the simulated truth, stored in every record
    names = {'mb','mh','m1','m2','Jb','Jh','cg1','cg2','cy','cb1','cb2','kb1','kb2','Lb','d', ...
             'cc1','cc2','ccy','friction_gain','ma_frac','ma','mh_rigid','L0','fa','ka','zeta_a','ca','f_pole'};
    for k = 1:numel(names), t.(names{k}) = cfg.(names{k}); end
    t.friction = 'Coulomb friction cc_i tanh(g v_i) per actuator X1, X2, Y';
    t.solver = sprintf('ode4 fixed step %.3g s', cfg.ts);
end
