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

TRACK   = 'augmentation';        % 'joint' (broadband [1,200]) | 'augmentation' (narrowband [130,180])
USE_MSD = true;           % true = augmented (baseline + hidden MSD), false = baseline
MA_FRAC = 0.50;           % hidden-MSD mass fraction
PLOT    = true;          % true = save a positions+forces PNG per record to figures/
SELECT  = 'T9';             % id prefix filter; '' = all. e.g. 'T9' -> T9*, 'T1' -> T1,T10-T14,
                          % 'E' -> all test, {'T3','E1'} -> those prefixes. Errors if no match.

cfg     = gtd_config(TRACK, USE_MSD, MA_FRAC);
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
    if PLOT, gtd_plot_record(out, rec, cfg, ~isempty(SELECT)); end   % show on screen for a selected subset

    if cfg.use_msd
        fprintf('  delta_a rms with/without = %.3e / %.3e (%.1fx) | force peak [%.0f %.0f %.0f] N\n', ...
                rms(out.da_with), rms(out.da_without), ...
                rms(out.da_with)/max(rms(out.da_without),eps), max(abs(out.u_total)));
    end
    fprintf('  saved %s.mat\n', rec.id);
end

fprintf('\nDone: %d records in %s\n', numel(records), cfg.out_dir);
