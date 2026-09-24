function cn_gen_multisine(mode, select, out_dir, noise_dir, cc_zero)
% CN_GEN_MULTISINE  The multisine half of the production dataset, with encoder noise (CN-009).
%
% Function copy of Matlab-scripts/Augmentation-coulomb/generate_trajectory_data_coulomb.m
% (working tree, 2026-09-24): same knobs (TRACK augmentation, MA_FRAC 0.50, Garcia cc, band
% [140 230], AMP_SCALE 6 with the D-207 three-line amplitude form, zeta_a 0.03), same call sequence
% (gtd_build_plant, gtd_make_reference, gtd_make_multisine, gtd_enforce_limits,
% gtd_run_simulation, gtd_save_record, gtd_plot_record, report_stick). CLOSED-LOOP-NOISE changes:
%   * model gantry_cn_noise_2025a (encoder-noise node, make_noise_model.m) instead of
%     gantry_additional_state_coulomb_2025a; friction ODE = the pre-D-209 copy in this folder;
%   * mode 'off'   : v = 0, production single precision (gate (a));
%     mode 'off64' : v = 0, double precision + q_true (reference for gate (c));
%     mode 'noise' : v from noise_dir/<id>.mat, double precision (the twin);
%   * select (cellstr of id prefixes, {} = all), out_dir (figures in out_dir/figures).
    if nargin < 2 || isempty(select), select = {}; end
    if nargin < 5, cc_zero = false; end
    here = fileparts(mfilename('fullpath'));
    addpath(here);
    cfg = gtd_config('augmentation', true, 0.50);
    cfg.mdl = 'gantry_cn_noise_2025a';
    cfg.cc1 = 16.80;  cfg.cc2 = 18.35;  cfg.ccy = 11.60;   % THEORY: garcia2013, identified values
    if cc_zero                            % CN-009a: friction off, the generator loop is linear
        cfg.cc1 = 0;  cfg.cc2 = 0;  cfg.ccy = 0;
        fprintf('cc = 0 (CN-009a linear-loop check)\n');
    end
    cfg.out_dir = out_dir;
    cfg.fig_dir = fullfile(cfg.out_dir, 'figures');
    % ── D-204 five-knob port (verbatim values) ────────────────────────────────
    BAND_OVERRIDE = [140, 230];  AMP_SCALE = 6;  ZETA_A_OVERRIDE = 0.03;
    fn = sqrt(cfg.ka/cfg.ma)/(2*pi) * sqrt(1 + cfg.ma/(cfg.mh - cfg.ma));
    assert(BAND_OVERRIDE(1) < fn && fn < BAND_OVERRIDE(2), 'band misses the absorber mode');
    cfg.f_low = BAND_OVERRIDE(1);  cfg.f_high = BAND_OVERRIDE(2);
    cfg.zeta_a = ZETA_A_OVERRIDE;
    cfg.ca     = 2*cfg.zeta_a*sqrt(cfg.ka*cfg.ma);
    cfg.A_sym  = AMP_SCALE * cfg.A_sym;
    cfg.A_Y    = AMP_SCALE * cfg.A_Y;
    cfg.A_anti = 0.5 * cfg.A_sym * cfg.Lb;
    cfg.save_double = ~strcmp(mode, 'off');
    fprintf('cn_gen_multisine: mode %s, out %s\n', mode, cfg.out_dir);
    fprintf('Amplitude override x%g: A_sym=%.1f N  A_Y=%.1f N  A_anti=%.1f N*m\n', ...
            AMP_SCALE, cfg.A_sym, cfg.A_Y, cfg.A_anti);
    if ~exist(cfg.out_dir, 'dir'), mkdir(cfg.out_dir); end
    if ~exist(cfg.fig_dir, 'dir'), mkdir(cfg.fig_dir); end
    load_system(cfg.mdl);
    set_param(cfg.mdl, 'SolverType', 'Fixed-step');
    set_param(cfg.mdl, 'Solver', 'ode4');
    set_param(cfg.mdl, 'FixedStep', sprintf('%.12g', cfg.ts));
    records = gtd_build_records(cfg);
    records = filter_records(records, select);
    for k = 1:numel(records)
        rec = records(k);
        fprintf('\n=== %d/%d  %s  [%s] ===\n', k, numel(records), rec.id, rec.split);
        assert(~isfile(fullfile(cfg.out_dir, [rec.id '.mat'])), 'refusing to overwrite %s', rec.id);
        tic;
        plant         = gtd_build_plant(rec.Y_op, cfg);
        [r, t]        = gtd_make_reference(rec, cfg);
        ms            = gtd_make_multisine(rec, plant, cfg);
        [f_safe, chk] = gtd_enforce_limits(plant, r, ms.f_stage, cfg);
        if chk.scale < 1, fprintf('  scaled multisine to %.0f%% to meet limits\n', 100*chk.scale); end
        cfg = attach_noise(cfg, mode, noise_dir, rec.id, numel(t));
        out     = gtd_run_simulation(rec, r, t, f_safe, plant, cfg);
        out.amp = chk.scale * ms.A;
        gtd_save_record(out, rec, cfg);
        gtd_plot_record(out, rec, cfg, false);
        report_stick(out, cfg);
        fprintf('  saved %s.mat (%.0f s)\n', rec.id, toc);
    end
    close_system(cfg.mdl, 0);
    fprintf('\nDone: %d records in %s\n', numel(records), cfg.out_dir);
end

function records = filter_records(records, select)
    if isempty(select), return; end
    keep = false(1, numel(records));
    for k = 1:numel(records), keep(k) = any(startsWith(records(k).id, select)); end
    assert(any(keep), 'select matched no record ids');
    records = records(keep);
end

function report_stick(out, cfg)
% verbatim from the production driver: stick fraction per PHYSICAL rail from the TRUE position
    V_STICK = 1e-3;   % [m/s] HEURISTIC: reporting threshold for "not sliding"
    v_stage = zeros(size(out.q_with));
    for j = 1:3, v_stage(:,j) = gradient(out.q_with(:,j), cfg.ts); end
    frac = mean(abs(v_stage) < V_STICK, 1);
    fprintf('  stick fraction (|v| < %.0e m/s) : X1 %.1f%%  X2 %.1f%%  Y %.1f%%\n', ...
            V_STICK, 100*frac(1), 100*frac(2), 100*frac(3));
end
