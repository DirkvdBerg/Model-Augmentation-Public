function cn_gen_telica(mode, select, out_dir, noise_dir)
% CN_GEN_TELICA  The Telica-profile half of the production dataset, with encoder noise (CN-009).
%
% Function copy of Matlab-scripts/Augmentation-telica-profiles-coulomb/
% generate_trajectory_data_telica_coulomb.m (working tree = HEAD 548e112, which differs from the
% run that wrote TP1 only in OUT_DIR_NAME and RESUME): TRACK augmentation, MA_FRAC 0.50, Garcia
% cc, zeta_a 0.03, no multisine; gtd_build_records_telica / gtd_make_reference_telica. Same
% CLOSED-LOOP-NOISE changes and modes as cn_gen_multisine.m.
    if nargin < 2 || isempty(select), select = {}; end
    here = fileparts(mfilename('fullpath'));
    addpath(here);
    cfg = gtd_config('augmentation', true, 0.50);
    cfg.mdl = 'gantry_cn_noise_2025a';
    cfg.cc1 = 16.80;  cfg.cc2 = 18.35;  cfg.ccy = 11.60;   % THEORY: garcia2013, identified values
    cfg.zeta_a = 0.03;
    cfg.ca     = 2*cfg.zeta_a*sqrt(cfg.ka*cfg.ma);
    cfg.out_dir = out_dir;
    cfg.fig_dir = fullfile(cfg.out_dir, 'figures');
    cfg.save_double = ~strcmp(mode, 'off');
    fprintf('cn_gen_telica: mode %s, out %s\n', mode, cfg.out_dir);
    if ~exist(cfg.out_dir, 'dir'), mkdir(cfg.out_dir); end
    if ~exist(cfg.fig_dir, 'dir'), mkdir(cfg.fig_dir); end
    load_system(cfg.mdl);
    set_param(cfg.mdl, 'SolverType', 'Fixed-step');
    set_param(cfg.mdl, 'Solver', 'ode4');
    set_param(cfg.mdl, 'FixedStep', sprintf('%.12g', cfg.ts));
    records = gtd_build_records_telica(cfg);
    if ~isempty(select)
        keep = false(1, numel(records));
        for k = 1:numel(records), keep(k) = any(startsWith(records(k).id, select)); end
        assert(any(keep), 'select matched no record ids');
        records = records(keep);
    end
    for k = 1:numel(records)
        rec = records(k);
        fprintf('\n=== %d/%d  %s  [%s] ===\n', k, numel(records), rec.id, rec.split);
        assert(~isfile(fullfile(cfg.out_dir, [rec.id '.mat'])), 'refusing to overwrite %s', rec.id);
        tic;
        plant         = gtd_build_plant(rec.Y_op, cfg);
        [r, t]        = gtd_make_reference_telica(rec, cfg);
        ms            = gtd_make_multisine(rec, plant, cfg);   % returns zero force ('none')
        [f_safe, chk] = gtd_enforce_limits(plant, r, ms.f_stage, cfg);
        if chk.scale < 1, fprintf('  scaled multisine to %.0f%% to meet limits\n', 100*chk.scale); end
        cfg = attach_noise(cfg, mode, noise_dir, rec.id, numel(t));
        out     = gtd_run_simulation(rec, r, t, f_safe, plant, cfg);
        out.amp = chk.scale * ms.A;
        gtd_save_record(out, rec, cfg);
        gtd_plot_record(out, rec, cfg, false);
        V_STICK = 1e-3;                     % verbatim from the production driver (HEURISTIC)
        v_stage = zeros(size(out.q_with, 1), 3);
        for j = 1:3, v_stage(:,j) = gradient(out.q_with(:,j), cfg.ts); end
        frac = mean(abs(v_stage) < V_STICK, 1);
        fprintf('  stick fraction (|v| < %.0e m/s) : X1 %.1f%%  X2 %.1f%%  Y %.1f%%\n', ...
                V_STICK, 100*frac(1), 100*frac(2), 100*frac(3));
        fprintf('  saved %s.mat (%.0f s)\n', rec.id, toc);
    end
    close_system(cfg.mdl, 0);
    fprintf('\nDone: %d records in %s\n', numel(records), cfg.out_dir);
end
