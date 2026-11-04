function generate_thesis_data(sel, opts)
% GENERATE_THESIS_DATA  Generate the thesis records, each with and without encoder noise.
%   generate_thesis_data()                     all 48 records of tdg_manifest, both versions
%   generate_thesis_data({'TR-P3_r1','E6_r1'})  the named records only
%   generate_thesis_data('training')            one block: training | validation | test-I | test-E | frf
%   opts fields (all optional):
%     versions   {'noisy'} or {'noise-free'} for one version only (default both)
%     simulator  'replica' steps the loop in plain MATLAB (tdg_replica.m, about 1 GB of RAM)
%                instead of the Simulink model (about 2.85 GB); both give the same records bitwise
%     out_dir    write both version folders and the run log under this folder instead of
%                Thesis-writeup/Data
%     ms_twins   true: the multisine evaluation twins of the motion-only test records (needs out_dir)
%
%   Per record, the inputs (reference, multisine, linear pre-check) are built once and shared by
%   both versions: the noisy version runs with the record's own encoder-noise draw, the noise-free
%   version with zero noise. Both must pass the limit check on the simulated truth; otherwise the
%   multisine is scaled down in steps of 0.1 and both are simulated again, so the two versions
%   differ only in the noise. A scale already fixed by a saved file (the other version of the
%   record, or the E partners of a twin) is read from that file and may not change.
%   Finished files are skipped and never overwritten, so a killed run can be started again.
%   Output: <data root>/<cfg.name_noisy | cfg.name_free>/{Training,Validation,Test}/<file>.mat and
%   MANIFEST.csv per version; one line per record in generation_log.csv in cfg.log_dir.
    if nargin < 1, sel = []; end
    if nargin < 2, opts = struct(); end
    if ~isfield(opts, 'versions'), opts.versions = {'noisy', 'noise-free'}; end
    addpath(fileparts(mfilename('fullpath')));
    if isfield(opts, 'simulator') && strcmp(opts.simulator, 'replica')
        cfg = tdg_config();                                       % no Simulink loaded
        cfg.simulator = 'replica';
    else
        cfg = tdg_init();
    end
    if isfield(opts, 'out_dir')
        cfg.data_root = opts.out_dir;
        cfg.out_noisy = fullfile(opts.out_dir, cfg.name_noisy);
        cfg.out_free  = fullfile(opts.out_dir, cfg.name_free);
        cfg.log_dir   = opts.out_dir;
    end
    if isfield(opts, 'ms_twins') && opts.ms_twins
        assert(isfield(opts, 'out_dir'), 'ms_twins needs opts.out_dir: the twins never go into the data set folders');
        cfg.ms_twins = true;
    end
    man = select(tdg_manifest(cfg), sel);
    fprintf('generate_thesis_data: %d records selected, versions %s\n', numel(man), strjoin(opts.versions, ' + '));
    % a job killed while saving leaves <file>_part.mat, never a final file
    for root = {cfg.out_noisy, cfg.out_free}
        parts = dir(fullfile(root{1}, '*', '*_part.mat'));
        for k = 1:numel(parts)
            fprintf('  removing interrupted write %s\n', fullfile(parts(k).folder, parts(k).name));
            delete(fullfile(parts(k).folder, parts(k).name));
        end
    end

    gen = provenance(cfg);
    truth = truth_params(cfg);
    nz = tdg_noise_psd(cfg);
    if ~exist(cfg.log_dir, 'dir'), mkdir(cfg.log_dir); end
    logf = fullfile(cfg.log_dir, 'generation_log.csv');
    if ~isfile(logf)
        fid = fopen(logf, 'w');
        fprintf(fid, 'time,file,version,split,Y_op,K_gain,excitation,scale_pre,scale_post,sim_s,MB,noise_added_nm_X1,noise_added_nm_X2,noise_added_nm_Y,stick_X1,stick_X2,stick_Y,Fpeak_frac_max,Frms_frac_max,plumbing_N\n');
        fclose(fid);
    end

    for k = 1:numel(man)
        e = man(k);
        sd = cfg.split_dir.(e.set);
        fn = fullfile(cfg.out_noisy, sd, [e.file '.mat']);
        ff = fullfile(cfg.out_free,  sd, [e.file '.mat']);
        need_n = any(strcmp(opts.versions, 'noisy')) && ~isfile(fn);
        need_f = any(strcmp(opts.versions, 'noise-free')) && ~isfile(ff);
        fprintf('\n=== %d/%d  %s  [%s, %s] ===\n', k, numel(man), e.file, e.block, e.desc);
        if ~need_n && ~need_f
            fprintf('  skip: already on disk\n');  continue
        end
        gen_one(e, sd, fn, ff, need_n, need_f, cfg, nz, gen, truth, logf);
    end
    tdg_write_manifest(cfg, tdg_manifest(cfg), opts.versions);    % only the versions that were run
    fprintf('\ngenerate_thesis_data: done\n');
end

function gen_one(e, sd, fn, ff, need_n, need_f, cfg, nz, gen, truth, logf)
% Simulate, check, save and log one record in the requested versions.
    t0 = tic;
    in = tdg_build_inputs(e, cfg);
    has_ms = ~strcmp(e.p.excitation, 'none');
    [v, vseeds] = tdg_noise_draw(nz, e.seed_noise);

    % the multisine scale is the same in both versions of a record; a scale fixed by a saved file
    % is taken from it, otherwise the run below decides it
    s_req = [];
    if ~isempty(e.scale_from)
        % an I twin takes the lowest realised scale of its E partners, so a pair differs in one property only
        sc = zeros(1, numel(e.scale_from));
        for j = 1:numel(e.scale_from)
            pf = fullfile(cfg.out_noisy, sd, [e.scale_from{j} '.mat']);
            assert(isfile(pf), '%s: generate its partner %s first', e.file, e.scale_from{j});
            P = load(pf, 'meta');  sc(j) = P.meta.scale_realised;
        end
        s_req = min(sc);
    elseif ~need_n && isfile(fn)
        A = load(fn, 'meta');  s_req = A.meta.scale_realised;
    elseif ~need_f && isfile(ff)
        A = load(ff, 'meta');  s_req = A.meta.scale_realised;
    end
    s = in.chk.scale;
    if ~isempty(s_req), s = s_req; end
    while true
        f = s * in.ms.f_stage;
        out_n = []; out_f = []; pc_n = struct('pass', true); pc_f = struct('pass', true);
        if need_n
            out_n = gtd_run_simulation(in.r, in.t, f, in.plant, cfg, v);
            pc_n = tdg_post_check(out_n, cfg);
        end
        if need_f
            out_f = gtd_run_simulation(in.r, in.t, f, in.plant, cfg, zeros(cfg.N_record, 3));
            pc_f = tdg_post_check(out_f, cfg);
        end
        if pc_n.pass && pc_f.pass, break; end
        fl = [strsplit(getfield_or(pc_n, 'failed'), ','), strsplit(getfield_or(pc_f, 'failed'), ',')];
        fails = strjoin(unique(fl(~cellfun(@isempty, fl))), ',');
        fprintf('  POST-CHECK FAIL at scale %.2f: %s\n', s, fails);
        assert(isempty(s_req), '%s fails the limit check (%s) at the scale %.2f fixed by a saved file', e.file, fails, s);
        assert(has_ms && s > 0.05, '%s fails the limit check (%s) and has no multisine to scale down', e.file, fails);
        s = round(10*s - 1) / 10;
    end

    % the plant received exactly u_total = lsim(Cfb, r - y) + f, so the saved y is the signal the
    % controller saw
    plumb = 0;
    for o = {out_n, out_f}
        if ~isempty(o{1}), plumb = max(plumb, max(abs(o{1}.u_plant - o{1}.u_total), [], 'all')); end
    end
    assert(plumb <= 1e-6, '%s: the plant force differs from u_total by %.3g N', e.file, plumb);

    m = struct();
    m.file = e.file;  m.rec = e.rec;  m.set = e.set;  m.block = e.block;  m.real = e.real;  m.twin = e.twin;
    m.class = e.class;  m.Y_op = e.Y_op;  m.desc = e.desc;  m.p = e.p;
    m.vcap_lowered = [];                                          % no velocity cap is ever lowered
    m.controller = struct('name', ternary(e.K_gain == 1, 'K1', 'K2-g+'), 'design_Y', cfg.K1_design_Y, ...
                          'gain', e.K_gain, 'fbw_Hz', cfg.fbw, 'rule', 'ruleOfThumb per stage axis, rigid nominal plant, tustin');
    m.truth = truth;
    m.band_Hz = [cfg.f_low cfg.f_high];  m.f_step_Hz = cfg.f_step;
    m.n_lines = getfield_or(in.ms.info, 'n_lines');               % '' without multisine
    m.A_prod = [cfg.A_sym, cfg.A_anti, cfg.A_Y];
    m.amp_applied = s * in.ms.A;                                  % [N, N m, N]
    m.scale_precheck = in.chk.scale;  m.scale_realised = s;
    m.scale_from = e.scale_from;
    m.precheck = in.chk;  m.ref_peaks = in.pk;
    m.seed_ms = e.seed_ms;  m.seed_sp = e.seed_sp;
    m.seed_keys = struct('ms', sprintf('TAF|%s|%s|r%d|nt0|ms', e.set, e.rec, e.real), ...
                         'sp', sprintf('TAF|%s|%s|r%d|nt0|sp', e.set, e.rec, e.real), ...
                         'noise', sprintf('TAF|%s|%s|r%d|nt0|noise', e.set, e.rec, e.real));
    m.generator = gen;

    sizes = [0 0];  written = {};
    if need_n
        mn = m;  mn.version = 'noisy';  mn.noise_seed = e.seed_noise;
        mn.pair = [cfg.name_free '/' sd '/' e.file '.mat'];         % '/' on any OS
        mn.noise = struct('seed', e.seed_noise, 'axis_seeds', vseeds, 'node', 'encoder', ...
                          'model', 'lowpass', 'fc_Hz', nz.fc_Hz, 'g_m_per_rtHz', nz.g, ...
                          'floor_m_per_rtHz', nz.floor, ...
                          'target', 'scripts/gantry/closed-loop-noise/outputs/g1_spectra/g1.npz', ...
                          'target_sha256', nz.target_sha256, 'std_v_m', std(v), 'method', nz.method);
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

    % run log: what the noise adds to the measured servo error, and the share of samples below
    % 1 mm/s (a low-velocity share; the tanh friction has no stick state)
    added = [NaN NaN NaN];
    if ~isempty(out_n) && ~isempty(out_f), added = 1e9 * rms(out_n.y_meas - out_f.y_meas); end
    o = out_n;  if isempty(o), o = out_f; end
    vel = zeros(size(o.q_with));
    for j = 1:3, vel(:, j) = gradient(o.q_with(:, j), cfg.ts); end
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
% Records by block name (char) or by file or record name (cell).
    if isempty(sel), return; end
    if ischar(sel) || isstring(sel)
        keep = strcmp({man.block}, char(sel));
    else
        keep = ismember({man.file}, sel) | ismember({man.rec}, sel);
    end
    assert(any(keep), 'the selection matches no record of tdg_manifest');
    man = man(keep);
end

function v = getfield_or(s, f)
    if isfield(s, f), v = s.(f); else, v = ''; end
end

function v = ternary(c, a, b)
    if c, v = a; else, v = b; end
end

function t = truth_params(cfg)
% The simulated truth parameters, the friction law and the solver, stored in every record.
    names = {'mb','mh','m1','m2','Jb','Jh','cg1','cg2','cy','cb1','cb2','kb1','kb2','Lb','Lh','d', ...
             'cc1','cc2','ccy','friction_gain','ma_frac','ma','mh_rigid','L0','fa','ka','zeta_a','ca','f_pole'};
    for k = 1:numel(names), t.(names{k}) = cfg.(names{k}); end
    t.friction = 'tanh-smoothed Coulomb per stage rail, cc_i tanh(g v_i) (gantrySystemExtendedTanh.m, D-223, D-224)';
    t.solver = sprintf('ode4 fixed step %.3g s', cfg.ts);
end

function g = provenance(cfg)
% Run date, MATLAB version, model and code hashes, git commit.
    g.date = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    g.matlab = version;
    g.model = cfg.mdl;
    g.model_sha256 = sha256file(fullfile(cfg.mdl_dir, [cfg.mdl '.slx']));
    g.simulator = cfg.simulator;
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
    files = [dir(fullfile(cfg.here, '*.m')); dir(fullfile(cfg.vendor_dir, '*.m'))];
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
