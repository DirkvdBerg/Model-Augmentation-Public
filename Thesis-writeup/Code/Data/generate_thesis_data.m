function generate_thesis_data(sel, opts)
% GENERATE_THESIS_DATA  Generate T_AF records in both versions (with and without noise).
%
%   generate_thesis_data()                    every record of tdg_manifest (48 records, each noisy + noise-free)
%   generate_thesis_data({'TR-P3_r1','E6_r1'}) the named files only
%   generate_thesis_data('training')           one block: training | validation | test-I | test-E | frf
%   generate_thesis_data(sel, struct('versions', {{'noisy'}}))   one version only
%   generate_thesis_data(sel, struct('simulator', 'replica'))    step the loop outside Simulink
%       (tdg_replica.m, D-231; about 1 GB of RAM instead of Simulink's 2.85 GB)
%   generate_thesis_data(sel, struct('out_dir', D))   write both versions and the run log under
%       folder D instead of Thesis-writeup/Data (reproduction test: regenerate a saved record
%       and compare it with the saved file)
%
%   Per record, in one pass: inputs (reference, multisine, linear pre-check) are built
%   once and shared; the noisy version runs with v = the record's own encoder-noise draw
%   (D-225), the noise-free version with v = 0; both must pass the post-simulation limit check on the
%   simulated truth, else the multisine is scaled down in steps of 0.1 and both are
%   re-simulated (same scale for both, so they stay identical except for the noise).
%   A record without multisine that fails only the velocity limit gets its velocity cap
%   lowered in 0.05 m/s steps (DATA-DESIGN section 10: "lowered, not forced").
%   When the scale is already fixed by a saved file (the other version of the record, or the
%   record of a noise-only twin), it is read from that file and may not change.
%   A record that fails is logged (generation_failures.csv in cfg.log_dir) and the stage continues.
%   Existing files are never overwritten (a record is skipped when its files exist).
%   Before each record the free space on C: is checked: if writing it would leave less
%   than cfg.disk_reserve_GB, the stage stops and reports (handoff section 13).
%   Output: Thesis-writeup/Data/<cfg.name_noisy | cfg.name_free>/{Training,Validation,Test}/ (D-230)
%   <file>.mat and MANIFEST.csv per version; a line per record in generation_log.csv in
%   cfg.log_dir (scripts/gantry/thesis-data-verification/logs/).
%   Run in MATLAB from this folder: tdg_init_paths; generate_thesis_data(...). One MATLAB job
%   at a time; the watchdog launcher is scripts/gantry/thesis-data-verification/tools/run_matlab.sh.
    if nargin < 1, sel = []; end
    if nargin < 2, opts = struct(); end
    if ~isfield(opts, 'versions'), opts.versions = {'noisy', 'noise-free'}; end
    if ~isfield(opts, 'est_MB'), opts.est_MB = 60; end        % per file, updated from the pilot
    here = fileparts(mfilename('fullpath'));
    addpath(here);
    if isfield(opts, 'simulator') && strcmp(opts.simulator, 'replica')
        cfg = tdg_config();                   % D-231: no Simulink loaded at all
        cfg.simulator = 'replica';
    else
        cfg = tdg_init(false);
    end
    if isfield(opts, 'out_dir')        % reproduction test: nothing is written into the data set
        cfg.data_root = opts.out_dir;
        cfg.out_noisy = fullfile(opts.out_dir, cfg.name_noisy);
        cfg.out_free  = fullfile(opts.out_dir, cfg.name_free);
        cfg.log_dir   = opts.out_dir;
    end
    man = tdg_manifest(cfg);
    man_all = man;                    % the full table: noise twins look up their record in it
    man = select(man, sel);
    fprintf('generate_thesis_data: %d records selected, versions %s\n', numel(man), strjoin(opts.versions, ' + '));
    % a job killed during save leaves <file>_part.mat; remove them (never a final file)
    for root = {cfg.out_noisy, cfg.out_free}
        parts = dir(fullfile(root{1}, '*', '*_part.mat'));
        for k = 1:numel(parts)
            fprintf('  removing interrupted write %s\n', fullfile(parts(k).folder, parts(k).name));
            delete(fullfile(parts(k).folder, parts(k).name));
        end
    end

    K1 = gtd_build_plant(0, cfg, 1).Cfb;
    gen = provenance(cfg);
    truth = truth_params(cfg);
    nzmap = containers.Map('KeyType', 'double', 'ValueType', 'any');
    if ~exist(cfg.log_dir, 'dir'), mkdir(cfg.log_dir); end
    logf = fullfile(cfg.log_dir, 'generation_log.csv');
    if ~isfile(logf)
        fid = fopen(logf, 'w');
        fprintf(fid, 'time,file,version,split,Y_op,K_gain,excitation,scale_pre,scale_post,sim_s,MB,noise_added_nm_X1,noise_added_nm_X2,noise_added_nm_Y,stick_X1,stick_X2,stick_Y,Fpeak_frac_max,Frms_frac_max,plumbing_N\n');
        fclose(fid);
    end

    failf = fullfile(cfg.log_dir, 'generation_failures.csv');
    nfail = 0;
    for k = 1:numel(man)
        e = man(k);
        sd = cfg.split_dir.(e.set);
        fn = fullfile(cfg.out_noisy, sd, [e.file '.mat']);
        ff = fullfile(cfg.out_free,  sd, [e.file '.mat']);
        want_n = any(strcmp(opts.versions, 'noisy'));
        want_f = any(strcmp(opts.versions, 'noise-free')) && e.free;
        need_n = want_n && ~isfile(fn);
        need_f = want_f && ~isfile(ff);
        fprintf('\n=== %d/%d  %s  [%s, %s] ===\n', k, numel(man), e.file, e.block, e.desc);
        if ~need_n && ~need_f
            fprintf('  skip: already on disk\n');  continue
        end
        freeGB = free_disk_GB(cfg.data_root);
        needGB = (need_n + need_f) * opts.est_MB / 1e3;
        if freeGB - needGB < cfg.disk_reserve_GB
            fprintf('STAGE STOP: disk: %.2f GB free, record needs ~%.2f GB, reserve %.1f GB. Stopped before %s.\n', ...
                    freeGB, needGB, cfg.disk_reserve_GB, e.file);
            break
        end
        % one record; a failure is logged and the stage continues with the next record
        try
            gen_one(e, sd, fn, ff, need_n, need_f, cfg, K1, nzmap, gen, truth, logf, man_all);
        catch err
            nfail = nfail + 1;
            fprintf('RECORD FAIL %s: %s\n', e.file, err.message);
            if ~isfile(failf)
                fid = fopen(failf, 'w');  fprintf(fid, 'time,file,message\n');  fclose(fid);
            end
            fid = fopen(failf, 'a');
            fprintf(fid, '%s,%s,"%s"\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')), e.file, strrep(err.message, '"', char(39)));
            fclose(fid);
        end
    end
    if nfail > 0, fprintf('\n%d record(s) FAILED, see %s\n', nfail, failf); end
    tdg_write_manifest(cfg, tdg_manifest(cfg), opts.versions);   % D-238: only the versions run
    fprintf('\ngenerate_thesis_data: done\n');
end

function gen_one(e, sd, fn, ff, need_n, need_f, cfg, K1, nzmap, gen, truth, logf, man)
% Generate one manifest record in the requested versions (see the header of
% generate_thesis_data). Errors propagate to the caller, which logs them.
    t0 = tic;
    in = tdg_build_inputs(e, cfg);
    has_ms = ~strcmp(e.p.excitation, 'none');
    % D-225 amendment: per-axis encoder noise, shape-corrected with K1 at the record's Y_op
    % (the loop that shapes it there); cached per Y_op in the handle object
    if ~isKey(nzmap, e.Y_op)
        nzmap(e.Y_op) = tdg_noise_psd(e.Y_op, cfg, K1);          %#ok<NASGU>
    end
    nz = nzmap(e.Y_op);
    [v, vseeds] = tdg_noise_draw(nz, e.seed_noise);
    N = cfg.N_record;

    % The multisine scale must be the same in both versions of a record and in a noise-only
    % twin and its record (REVIEW.md finding 3). When it is already fixed by a saved file, it
    % is read from there and may not be lowered; otherwise the noisy run decides it.
    s_req = [];
    if e.twin > 0
        b = man(strcmp({man.rec}, e.rec) & [man.twin] == 0 & [man.real] == e.real);
        bf = fullfile(cfg.out_noisy, sd, [b.file '.mat']);
        assert(isfile(bf), 'noise twin %s: generate its record %s first', e.file, b.file);
        B = load(bf, 'meta');  s_req = B.meta.scale_realised;
    elseif ~need_n && isfile(fn)
        A = load(fn, 'meta');  s_req = A.meta.scale_realised;
    elseif ~need_f && e.free && isfile(ff)
        A = load(ff, 'meta');  s_req = A.meta.scale_realised;
    end
    s = in.chk.scale;
    if ~isempty(s_req), s = s_req; end
    vcap = [];                                                    % velocity cap history [from ... to]
    while true
        f = s * in.ms.f_stage;
        out_n = []; out_f = []; pc_n = struct('pass', true); pc_f = struct('pass', true);
        if need_n
            out_n = gtd_run_simulation(e, in.r, in.t, f, in.plant, cfg, v);
            pc_n = tdg_post_check(out_n, cfg);
        end
        if need_f
            out_f = gtd_run_simulation(e, in.r, in.t, f, in.plant, cfg, zeros(N, 3));
            pc_f = tdg_post_check(out_f, cfg);
        end
        if pc_n.pass && pc_f.pass, break; end
        fl = [strsplit(getfield_or(pc_n, 'failed'), ','), strsplit(getfield_or(pc_f, 'failed'), ',')];
        fails = strjoin(unique(fl(~cellfun(@isempty, fl))), ',');
        fprintf('  POST-CHECK FAIL at scale %.2f: %s\n', s, fails);
        assert(isempty(s_req), '%s fails the post-check at the scale fixed by its saved twin (%.2f)', e.file, s);
        if has_ms && s > 0.05
            s = round(10*s - 1) / 10;                             % DATA-DESIGN section 10 item 2
        elseif strcmp(fails, 'vel') && strcmp(e.class, 'route') && e.p.X_vmax - 0.05 >= 1.5
            % DATA-DESIGN section 10: "a record whose realised velocity exceeds it is lowered,
            % not forced"; HEURISTIC: step 0.05 m/s on both axis caps, not below the trained 1.5 m/s
            if isempty(vcap), vcap = e.p.X_vmax; end
            e.p.X_vmax = e.p.X_vmax - 0.05;  e.p.Y_vmax = e.p.Y_vmax - 0.05;
            vcap = [vcap, e.p.X_vmax]; %#ok<AGROW>
            fprintf('  velocity cap lowered to %.2f m/s\n', e.p.X_vmax);
            in = tdg_build_inputs(e, cfg);
        else
            error('%s fails the post-simulation limits (%s) and has no multisine to scale down', e.file, fails);
        end
    end

    % plumbing: the plant received exactly u_total = lsim(Cfb, r - y) + f, i.e. the saved y is
    % the signal the controller saw (C12, per record; D-225)
    plumb = 0;
    for o = {out_n, out_f}
        if ~isempty(o{1}), plumb = max(plumb, max(abs(o{1}.u_plant - o{1}.u_total), [], 'all')); end
    end
    assert(plumb <= 1e-6, '%s: u_plant differs from u_total by %.3g N', e.file, plumb);

    % common meta
    m = struct();
    m.file = e.file;  m.rec = e.rec;  m.set = e.set;  m.block = e.block;  m.real = e.real;  m.twin = e.twin;
    m.class = e.class;  m.Y_op = e.Y_op;  m.desc = e.desc;  m.p = e.p;
    m.vcap_lowered = vcap;
    m.controller = struct('name', ternary(e.K_gain == 1, 'K1', 'K2-g+'), 'design_Y', cfg.K1_design_Y, ...
                          'gain', e.K_gain, 'fbw_Hz', cfg.fbw, 'rule', 'ruleOfThumb per stage axis, rigid nominal plant, tustin');
    m.truth = truth;
    m.band_Hz = [cfg.f_low cfg.f_high];  m.f_step_Hz = cfg.f_step;
    m.n_lines = getfield_or(in.ms.info, 'n_lines');
    m.A_prod = [cfg.A_sym, cfg.A_anti, cfg.A_Y];
    m.amp_applied = s * in.ms.A;                                 % [N, N m, N]
    m.scale_precheck = in.chk.scale;  m.scale_realised = s;
    m.precheck = in.chk;  m.ref_peaks = in.pk;
    m.seed_ms = e.seed_ms;  m.seed_sp = e.seed_sp;
    m.seed_keys = struct('ms', sprintf('TAF|%s|%s|r%d|nt0|ms', e.set, e.rec, e.real), ...
                         'sp', sprintf('TAF|%s|%s|r%d|nt0|sp', e.set, e.rec, e.real), ...
                         'noise', sprintf('TAF|%s|%s|r%d|nt%d|noise', e.set, e.rec, e.real, e.twin));
    m.generator = gen;

    sizes = [0 0];  written = {};
    if need_n
        mn = m;  mn.version = 'noisy';  mn.noise_seed = e.seed_noise;
        mn.pair = ternary(e.free, [cfg.name_free '/' sd '/' e.file '.mat'], '');   % '/' on any OS
        if strcmp(cfg.noise_model, 'lowpass')                    % D-238
            mn.noise = struct('seed', e.seed_noise, 'axis_seeds', vseeds, 'node', 'encoder', ...
                              'model', 'lowpass', 'fc_Hz', nz.fc_Hz, 'g_m_per_rtHz', nz.g, ...
                              'floor_m_per_rtHz', nz.floor, ...
                              'target', 'scripts/gantry/closed-loop-noise/outputs/g1_spectra/g1.npz', ...
                              'target_sha256', nz.target_sha256, 'std_v_m', std(v), 'method', nz.method);
        else
            mn.noise = struct('seed', e.seed_noise, 'axis_seeds', vseeds, 'node', 'encoder', ...
                              'Y_op_calibrated', e.Y_op, 'controller', 'K1', 'corner_Hz', nz.corner, ...
                              'correction', nz.correction, 'friction_gain', nz.friction_gain, ...
                              'hold_below_Hz', nz.hold_below, ...
                              'target', 'scripts/gantry/closed-loop-noise/outputs/g1_spectra/g1.npz', ...
                              'target_sha256', nz.target_sha256, 'std_v_m', std(v), ...
                              'method', 'D-225 amendment: independent per-axis generators, phi_j = Phi_e,jj / |S_jj|^2 above the corner');
        end
        mn.post_check = pc_n;
        info = gtd_save_record(out_n, e, cfg, fn, mn);  sizes(1) = info.bytes;  written{end+1} = 'noisy';
    end
    if need_f
        mf = m;  mf.version = 'noise-free';  mf.noise_seed = NaN;
        mf.pair = [cfg.name_noisy '/' sd '/' e.file '.mat'];
        mf.noise = struct('seed', NaN, 'note', 'v = 0 (noise-free twin)');
        mf.post_check = pc_f;
        info = gtd_save_record(out_f, e, cfg, ff, mf);  sizes(2) = info.bytes;  written{end+1} = 'noise-free';
    end

    % report
    added = [NaN NaN NaN];
    % what the noise adds to the MEASURED servo error, noisy minus noise-free (D-225)
    if ~isempty(out_n) && ~isempty(out_f), added = 1e9 * rms(out_n.y_meas - out_f.y_meas); end
    o = out_n;  if isempty(o), o = out_f; end
    vel = zeros(size(o.q_with));
    for j = 1:3, vel(:, j) = gradient(o.q_with(:, j), cfg.ts); end
    % share of samples below 1 mm/s, the copied driver's reporting threshold; since D-224 the
    % friction has no stick state, so this is a low-velocity share, not a stick fraction
    stick = 100 * mean(abs(vel) < 1e-3, 1);
    pc = ternary(isempty(out_n), pc_f, pc_n);
    el = toc(t0);
    fprintf('  scale pre %.2f post %.2f | noise adds rms %s nm | stick %s %% | F peak %s, rms %s of limit | %.1f + %.1f MB | %.1f s\n', ...
            in.chk.scale, s, mat2str(round(added, 3)), mat2str(round(stick, 1)), ...
            mat2str(round(pc.force_peak_frac, 2)), mat2str(round(pc.force_rms_frac, 2)), sizes/1e6, el);
    fid = fopen(logf, 'a');
    fprintf(fid, '%s,%s,%s,%s,%.4f,%.2f,%s,%.2f,%.2f,%.1f,%.1f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3g\n', ...
            char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')), e.file, strjoin(written, '+'), sd, e.Y_op, e.K_gain, ...
            e.p.excitation, in.chk.scale, s, el, sum(sizes)/1e6, added, stick, ...
            max(pc.force_peak_frac), max(pc.force_rms_frac), plumb);
    fclose(fid);
end

% ── helpers ─────────────────────────────────────────────────────────────────

function man = select(man, sel)
    if isempty(sel), return; end
    if ischar(sel) || isstring(sel)
        keep = strcmp({man.block}, char(sel));
    else
        keep = ismember({man.file}, sel) | ismember({man.rec}, sel);
    end
    assert(any(keep), 'selection matched no record');
    man = man(keep);
end

function v = getfield_or(s, f)
    if isfield(s, f), v = s.(f); else, v = ''; end
end

function v = ternary(c, a, b)
    if c, v = a; else, v = b; end
end

function t = truth_params(cfg)
% The T_AF parameters as simulated (SPEC.md section 3), with the friction law and its gain (D-224).
    names = {'mb','mh','m1','m2','Jb','Jh','cg1','cg2','cy','cb1','cb2','kb1','kb2','Lb','Lh','d', ...
             'cc1','cc2','ccy','friction_gain','ma_frac','ma','mh_rigid','L0','fa','ka','zeta_a','ca','f_pole'};
    for k = 1:numel(names), t.(names{k}) = cfg.(names{k}); end
    t.friction = 'tanh-smoothed Coulomb per stage rail, cc_i tanh(g v_i) (gantrySystemExtendedTanh.m, D-223, D-224)';
    t.solver = sprintf('ode4 fixed step %.3g s', cfg.ts);
end

function g = provenance(cfg)
    g.date = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    g.matlab = version;
    g.model = cfg.mdl;
    g.model_sha256 = sha256file(fullfile(cfg.mdl_dir, [cfg.mdl '.slx']));
    g.simulator = cfg.simulator;                                 % D-231: 'simulink' or 'replica'
    try
        head = strtrim(fileread(fullfile(cfg.root, '.git', 'HEAD')));
        if startsWith(head, 'ref:')
            ref = strtrim(head(5:end));
            g.git_head = strtrim(fileread(fullfile(cfg.root, '.git', strrep(ref, '/', filesep))));
        else
            g.git_head = head;
        end
    catch
        g.git_head = 'unknown';
    end
    files = [dir(fullfile(cfg.here, '*.m')); dir(fullfile(cfg.here, 'vendor', '*.m'))];
    g.code_sha256 = struct();
    for k = 1:numel(files)
        g.code_sha256.(matlab.lang.makeValidName(files(k).name)) = sha256file(fullfile(files(k).folder, files(k).name));
    end
end

function h = sha256file(f)
% SHA-256 through Windows certutil, so MATLAB can run without its JVM (-nojvm).
    [st, out] = system(sprintf('certutil -hashfile "%s" SHA256', f));
    tok = regexp(out, '([0-9a-fA-F]{2}\s?){32}', 'match', 'once');
    if st ~= 0 || isempty(tok), h = 'unavailable'; return; end
    h = lower(regexprep(tok, '\s', ''));
end

function g = free_disk_GB(p)
% Free bytes on the drive of path p, without Java (PowerShell Get-PSDrive).
    [st, out] = system(sprintf('powershell -NoProfile -Command "(Get-PSDrive %s).Free"', p(1)));
    v = str2double(strtrim(out));
    assert(st == 0 && isfinite(v), 'free_disk_GB: cannot read free space of %s:', p(1));
    g = v / 1e9;
end
