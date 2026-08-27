% GENERATE_TRAJECTORY_DATA  Driver for the gantry trajectory-data generator.
%   Run from the repo root. Set the three toggles below. Each mode writes to its
%   own top-level folder (data/gantry/matlab/trajectory/<track>/<tag>/).
%
%   CHANGED (open-loop variant): the reference and the controller are gone from
%   the loop. gtd_make_reference and gtd_build_plant are not called at all, and
%   gtd_enforce_limits is replaced by a direct peak-force check, because that
%   function's pre-check is a CLOSED-LOOP response bound (it asks what the
%   controller would do to the reference) and there is no loop here. Open loop
%   the only limit that still applies to the input is the force ceiling itself.
%
%   Pipeline per record:
%     gtd_make_multisine (analytic, stage coords) -> peak-force check
%     -> gtd_run_simulation (fixed-step RK4, no Simulink) -> gtd_save_record
%
%   Spec: docs/trajectory-generation-spec-draft.md

clear; clc; close all;

TRACK   = 'augmentation';        % 'joint' (broadband [1,200]) | 'augmentation' (narrowband [130,180])
USE_MSD = true;           % true = augmented (baseline + hidden MSD), false = baseline
MA_FRAC = 0.10;           % hidden-MSD mass fraction
PLOT    = true;          % true = save a positions+forces PNG per record to figures/
SHOW    = false;         % true = also display each figure on screen (not only save)
SELECT  = '';            % id prefix filter; '' = all nine. 'OT3' = the amplitude probe.
                          % 'E' -> all test, {'T3','E1'} -> those prefixes. Errors if no match.
% Figures are shown on screen when SHOW is true OR a subset is selected; a full
% batch with SHOW=false saves PNGs without opening windows.

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

    ms = gtd_make_multisine(rec, [], cfg);      % no plant: nothing is loop-shaped here

    pk = max(abs(ms.f_stage));
    fprintf('  input stage RMS [%.1f %.1f %.1f] N, peak [%.1f %.1f %.1f] N (limit [%.0f %.0f %.0f])\n', ...
            std(ms.f_stage), pk, cfg.lim.force_peak);
    assert(all(pk <= cfg.lim.force_peak), 'gtd:forcePeak', ...
           'peak stage force exceeds the enforced limit; lower cfg.ol_A_rms');

    out = gtd_run_simulation(rec, ms, cfg);
    out.amp = ms.A;
    gtd_save_record(out, rec, cfg);
    if PLOT, gtd_plot_record(out, rec, cfg, SHOW || ~isempty(SELECT)); end   % show on screen if SHOW or a subset

    if cfg.use_msd
        fprintf('  delta_a rms with/without = %.3e / %.3e (%.1fx) | force peak [%.0f %.0f %.0f] N\n', ...
                rms(out.da_with), rms(out.da_without), ...
                rms(out.da_with)/max(rms(out.da_without),eps), max(abs(out.u_total)));
    end
    % Open-loop records rectify: a zero-mean force does NOT leave the position
    % where it started, because M depends on Y and delta_a. Report the drift and
    % the post-settling AC content, which are the two numbers that decide the
    % amplitude (drift must stay small against the 0.15 m record spacing, AC is
    % the informative part of the output).
    y  = out.q_with;
    dr = y(end, :) - y(1, :);
    ii = out.t_sim >= 0.5 * out.t_sim(end);              % second half = second period
    ac = std(detrend(y(ii, :), 1), 0, 1);
    fprintf('  drift [%+.3e %+.3e %+.3e] m  (Y drift = %.1f%% of the 0.15 m spacing)\n', ...
            dr, 100*abs(dr(3))/0.15);
    fprintf('  AC rms 2nd half [%.3e %.3e %.3e] m\n', ac);
    fprintf('  saved %s.mat\n', rec.id);
end

fprintf('\nDone: %d records in %s\n', numel(records), cfg.out_dir);
