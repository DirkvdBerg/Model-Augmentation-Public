% GENERATE_TRAJECTORY_DATA  Driver for the gantry trajectory-data generator.
%   Run from the repo root. Set the three toggles below. Each mode writes to its
%   own top-level folder (data/gantry/matlab/trajectory/<track>/<tag>/).
%
%   Pipeline per record:
%     gtd_build_plant -> gtd_make_reference -> gtd_make_multisine
%     -> gtd_enforce_limits (linear pre-check + scale-down)
%     -> gtd_run_simulation (Simulink) -> gtd_save_record
%
%   Spec: docs/trajectory-generation-spec-draft.md

clear; clc; close all;

TRACK   = 'augmentation';        % 'joint' (broadband [1,200]) | 'joint_lowf' (as joint, from the 1/t_record fundamental, 0.0833 Hz) | 'augmentation' (narrowband [130,180])
USE_MSD = true;           % true = augmented (baseline + hidden MSD), false = baseline
MA_FRAC = 0.50;           % hidden-MSD mass fraction (0.50 = 50/50 split payload / hidden MSD)
PLOT    = true;          % true = save a positions+forces PNG per record to figures/
SHOW    = false;          % true = also display each figure on screen (not only save)
SELECT  = '';             % id prefix filter; '' = all. e.g. 'T9' -> T9*, 'T1' -> T1,T10-T14,
                          % 'E' -> all test, {'T3','E1'} -> those prefixes. Errors if no match.
% Figures are shown on screen when SHOW is true OR a subset is selected; a full
% batch with SHOW=false saves PNGs without opening windows.

% Output folder override. '' = the gtd_config default. Set it whenever a config
% knob (here MA_FRAC) changes, so a new dataset lands in a NEW folder and can
% never overwrite an existing one. fig_dir must be overridden too, otherwise the
% PNGs of the old set are replaced while the .mat files are not.
OUT_DIR_NAME = 'augmentation_ma50';
% RESUME picks what happens when OUT_DIR_NAME already holds records. A full batch
% takes hours, so an interrupted run must be continuable without regenerating the
% records that are already on disk, and without overwriting them.
%   false : hard stop if ANY .mat is present (the guard against a wrong folder)
%   true  : skip the records that already exist, generate only the missing ones
RESUME = true;

cfg     = gtd_config(TRACK, USE_MSD, MA_FRAC);
if ~isempty(OUT_DIR_NAME)
    cfg.out_dir = fullfile(cfg.root, 'data', 'gantry', 'matlab', 'trajectory', OUT_DIR_NAME);
    cfg.fig_dir = fullfile(cfg.out_dir, 'figures');
    % Hard stop rather than a silent overwrite: existing .mat files here mean the
    % folder already holds a dataset and OUT_DIR_NAME was not bumped.
    if ~RESUME && isfolder(cfg.out_dir) && ~isempty(dir(fullfile(cfg.out_dir, '*.mat')))
        error('generate_trajectory_data:outDirNotEmpty', ...
              ['Output folder already contains .mat records:\n  %s\n' ...
               'Pick a new OUT_DIR_NAME or set RESUME = true.'], cfg.out_dir);
    end
end
records = gtd_build_records(cfg);

if ~isempty(SELECT)
    pref = cellstr(SELECT);
    keep = false(1, numel(records));
    for k = 1:numel(records)
        keep(k) = any(startsWith(records(k).id, pref));
    end
    assert(any(keep), 'SELECT ''%s'' matched no record ids', strjoin(pref, ''','''));
    records = records(keep);
end

fprintf('Generating %d records (%s)  ->  %s\n', numel(records), TRACK, cfg.out_dir);

for k = 1:numel(records)
    rec = records(k);
    fprintf('\n=== %d/%d  %s  [%s] ===\n', k, numel(records), rec.id, rec.split);

    if RESUME && isfile(fullfile(cfg.out_dir, [rec.id '.mat']))
        fprintf('  skip (already on disk)\n');
        continue
    end

    plant = gtd_build_plant(rec.Y_op, cfg);
    [r, t] = gtd_make_reference(rec, cfg);
    ms     = gtd_make_multisine(rec, plant, cfg);

    [f_safe, chk] = gtd_enforce_limits(plant, r, ms.f_stage, cfg);
    if chk.scale < 1
        fprintf('  scaled multisine to %.0f%% to meet limits\n', 100*chk.scale);
    end

    out = gtd_run_simulation(rec, r, t, f_safe, plant, cfg);
    out.amp = chk.scale * ms.A;
    gtd_save_record(out, rec, cfg);
    if PLOT, gtd_plot_record(out, rec, cfg, SHOW || ~isempty(SELECT)); end   % show on screen if SHOW or a subset

    if cfg.use_msd
        fprintf('  delta_a rms with/without = %.3e / %.3e (%.1fx) | force peak [%.0f %.0f %.0f] N\n', ...
                rms(out.da_with), rms(out.da_without), ...
                rms(out.da_with)/max(rms(out.da_without),eps), max(abs(out.u_total)));
    end
    fprintf('  saved %s.mat\n', rec.id);
end

fprintf('\nDone: %d records in %s\n', numel(records), cfg.out_dir);
