function check_records(files, root)
% CHECK_RECORDS  C3, C4b, C8 and C14 of IMPLEMENTATION.md, on SAVED records.
%   check_records()                  every noisy record on disk
%   check_records({'E1_r1'})         named files
%   check_records(files, root)       records under root/<cfg.name_noisy | cfg.name_free>/ instead of
%                                    Thesis-writeup/Data (a pilot folder, generate_thesis_data out_dir)
%
% Stated before the run, per record (schema of D-224 to D-226, 2026-09-28: tanh friction,
% encoder noise; the D-218 criteria on d_in and V_BRK are replaced):
%   C14 schema: the copied fields plus y_true, v_enc, seeds, version, pair, meta are present;
%       every signal is double, N = 240000 rows; t_sim = (0:N-1) ts; post_check.pass true;
%       y == y_true + v_enc exactly (the saved y is the measured signal); noise-free
%       v_enc == 0 and y == y_true; noisy v_enc ~= 0 and equal, bitwise, to a fresh draw
%       from the stored noise_seed at the stored Y_op (tdg_noise_psd + tdg_noise_draw), whose
%       per-axis seeds equal meta.noise.axis_seeds.
%   C3  noise-free twin: r_sim, f_sim, t_sim, amp_rms, seed, seed_ms, seed_sp, fs, dt,
%       split, meta.p, meta.truth, meta.controller, meta.scale_realised are bitwise equal
%       between the two versions; only y, y_true and derived fields, u_fb, u_total, v_enc and
%       the noise / version / pair / post_check meta differ. Noise-only twins: same r_sim and
%       f_sim as their record, different v_enc.
%   C4b delivered excitation from the saved f_sim: 383 lines per logical channel on
%       106:0.5:297 Hz, off-grid energy < 1e-20 of the total, rms = amp_rms within 1e-9
%       relative (multisine records); f_sim == 0 otherwise.
%   C8  truth parameters in meta equal the design: ma 5.05 kg (0.50 mh), zeta_a 0.03,
%       ka = ma (2 pi 150)^2, ca = 2 zeta sqrt(ka ma), pole 212.13 Hz (+-0.01), L0 0.10 m,
%       cc 16.80 / 18.35 / 11.60 N, tanh friction at friction_gain == cfg.friction_gain (D-226);
%       controller K1 (K2-g+ for E5 and its twin).
%   Reported, no pass/fail (NOISE-INJECTION section 6): the rms the noise adds to the measured
%   y (noisy minus noise-free), all frequencies and above the 50 Hz corner, beside the measured
%   Telica standstill level.
    here = fileparts(fileparts(mfilename('fullpath')));                % scripts/gantry/thesis-data-verification
    addpath(fullfile(fileparts(fileparts(fileparts(here))), 'Thesis-writeup', 'Code', 'Data'));   % the generator
    cfg = tdg_config();
    if nargin > 1 && ~isempty(root)
        cfg.out_noisy = fullfile(root, cfg.name_noisy);
        cfg.out_free  = fullfile(root, cfg.name_free);
    end
    man = tdg_manifest(cfg);
    K1 = gtd_build_plant(0, cfg, 1).Cfb;
    T = load(cfg.noise_target);
    if nargin < 1 || isempty(files)
        files = {};
        for k = 1:numel(man)
            if isfile(fullfile(cfg.out_noisy, cfg.split_dir.(man(k).set), [man(k).file '.mat'])), files{end+1} = man(k).file; end %#ok<AGROW>
        end
    end
    sigs = {'u_total','u_fb','f_sim','y','y_true','x_logical','r_sim','Y_trajectory','t_sim','delta_a','vdelta_a','x_aug','v_enc'};
    same = {'r_sim','f_sim','t_sim','amp_rms','seed','seed_ms','seed_sp','fs','dt','split'};
    N = cfg.N_record;  t_ref = (0:N-1)' * cfg.ts;
    bins = round((cfg.f_low:cfg.f_step:cfg.f_high) * cfg.t_record) + 1;
    win = hann(2048, 'periodic');
    okall = true;
    fprintf('%-12s %-5s %-5s %-5s %-5s | noise adds rms all f / above corner [nm]  | measured %s nm | MB noisy/free\n', ...
            'file', 'C14', 'C3', 'C4b', 'C8', mat2str(1e9*T.rms_total_m, 3));
    for i = 1:numel(files)
        e = man(strcmp({man.file}, files{i}));
        sd = cfg.split_dir.(e.set);
        fn = fullfile(cfg.out_noisy, sd, [e.file '.mat']);
        A = load(fn);  B = [];
        ff = fullfile(cfg.out_free, sd, [e.file '.mat']);
        if e.free && isfile(ff), B = load(ff); end

        % C14
        ok14 = true;  why = {};
        for s = sigs
            ok = isfield(A, s{1}) && isa(A.(s{1}), 'double') && size(A.(s{1}), 1) == N;
            if ~ok, why{end+1} = s{1}; end %#ok<AGROW>
            ok14 = ok14 && ok;
        end
        ok14 = ok14 && isequal(A.t_sim, t_ref) && any(A.v_enc(:)) && A.meta.post_check.pass && A.noise_seed == e.seed_noise;
        ok14 = ok14 && isequal(A.y, A.y_true + A.v_enc);
        nz = tdg_noise_psd(e.Y_op, cfg, K1);
        [vre, sre] = tdg_noise_draw(nz, A.noise_seed);
        ok14 = ok14 && isequal(vre, A.v_enc) && isequal(sre, A.meta.noise.axis_seeds);
        if ~isempty(B)
            ok14 = ok14 && all(isfield(B, sigs)) && ~any(B.v_enc(:)) && isequal(B.y, B.y_true) && ...
                   B.meta.post_check.pass && isnan(B.noise_seed);
        end

        % C3
        ok3 = true;
        if ~isempty(B)
            for s = same, ok3 = ok3 && isequal(A.(s{1}), B.(s{1})); end
            for s = {'p','truth','controller','scale_realised','amp_applied','seed_ms','seed_sp'}
                ok3 = ok3 && isequaln(A.meta.(s{1}), B.meta.(s{1}));   % isequaln: p.T3 is NaN for cycloid routes
            end
            ok3 = ok3 && ~isequal(A.y, B.y);
        elseif e.twin > 0
            base = man(strcmp({man.rec}, e.rec) & [man.twin] == 0 & [man.real] == e.real);
            C = load(fullfile(cfg.out_noisy, sd, [base.file '.mat']), 'r_sim', 'f_sim', 'v_enc', 'seed_ms', 'seed_sp');
            ok3 = isequal(A.r_sim, C.r_sim) && isequal(A.f_sim, C.f_sim) && ~isequal(A.v_enc, C.v_enc) && ...
                  A.seed_ms == C.seed_ms && A.seed_sp == C.seed_sp;
        end

        % C4b
        flog = (cfg.P * A.f_sim.').';
        if strcmp(e.p.excitation, 'none')
            ok4 = ~any(A.f_sim(:));
        else
            F = fft(flog);  Pw = abs(F(1:N/2+1, :)).^2;
            on = false(size(Pw, 1), 1);  on(bins) = true;
            off = sum(Pw(~on, :), 1) ./ sum(Pw, 1);
            nl = sum(Pw(on, :) > 1e-6 * max(Pw(on, :)), 1);
            ok4 = all(off < 1e-20) && all(nl == numel(bins)) && all(abs(rms(flog) ./ A.amp_rms - 1) < 1e-9);
        end

        % C8
        tr = A.meta.truth;
        ka = 0.50*10.1*(2*pi*150)^2;
        ok8 = abs(tr.ma - 5.05) < 1e-12 && tr.zeta_a == 0.03 && abs(tr.ka/ka - 1) < 1e-12 && ...
              abs(tr.ca - 2*0.03*sqrt(ka*5.05)) < 1e-9 && abs(tr.f_pole - 212.13) < 0.01 && tr.L0 == 0.10 && ...
              isequal([tr.cc1 tr.cc2 tr.ccy], [16.80 18.35 11.60]) && tr.friction_gain == cfg.friction_gain && ...
              contains(tr.friction, 'tanh') && tr.fa == 150 && ...
              A.meta.controller.gain == e.K_gain && A.meta.controller.design_Y == 0;

        % noise added (report)
        added = [NaN NaN NaN];  addb = added;
        if ~isempty(B)
            dy = A.y - B.y;
            added = 1e9 * rms(dy);
            [Pxx, fw] = pwelch(dy(round(1/cfg.ts)+1:end, :), win, 1024, 2048, cfg.fs);
            band = fw >= cfg.noise_corner;                      % above the shape-correction corner
            addb = 1e9 * sqrt(sum(Pxx(band, :), 1) * (fw(2) - fw(1)));
        end
        fi = dir(fn);  mb = [fi.bytes, NaN] / 1e6;
        if ~isempty(B), fi = dir(ff);  mb(2) = fi.bytes / 1e6; end
        ok = ok14 && ok3 && ok4 && ok8;
        okall = okall && ok;
        fprintf('%-12s %-5s %-5s %-5s %-5s | %-20s / %-20s |                   | %.1f / %.1f %s\n', e.file, pf(ok14), pf(ok3), pf(ok4), pf(ok8), ...
                mat2str(round(added, 2)), mat2str(round(addb, 2)), mb, strjoin(why, ','));
    end
    fprintf('C3 / C4b / C8 / C14 on %d saved records -> %s\n', numel(files), pf(okall));
end

function s = pf(ok)
    if ok, s = 'PASS'; else, s = 'FAIL'; end
end
