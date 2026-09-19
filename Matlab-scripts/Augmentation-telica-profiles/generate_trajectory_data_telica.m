% GENERATE_TRAJECTORY_DATA_TELICA  Same generator, real Telica motion profiles, NO multisine.
%
% Mirrors Matlab-scripts/Augmentation/data/generate_trajectory_data.m and calls the
% SAME gtd_* functions, unmodified, following the Matlab-scripts/Augmentation-coulomb/
% variant-folder pattern. Nothing under Matlab-scripts/Augmentation/ is modified.
%
% WHAT THIS DATASET IS FOR. Every record in the production set carries a multisine,
% either on top of a motion profile or at standstill. Evaluation records without one
% therefore sit off the training input distribution and the model extrapolates badly
% in the absorber band. These records have excitation = 'none': the reference
% trajectory is the ONLY excitation, and the trajectory is the one the real machine
% runs, so the argument is an ASMPT-operational one rather than a synthetic one.
%
% THE PROFILE IS CYCLOIDAL, NOT JERK-LIMITED. Measured this session from
% kamtin-data/Data Telica/'06 40 mm XL 80 mm YL', all 15 operating points. The
% measured acceleration tracks sin(2*pi*tau) pointwise to within 0.003 of peak with
% no plateau anywhere; the velocity-limited cycloidal model fits to 9.4 um rms on X
% (0.023 % of stroke) against 51.9 um for the next-best candidate (poly5 min-jerk).
% The project record previously described this move as "jerk-limited"; that was an
% inference and it was wrong. See D-206 and gtd_make_reference_telica.m.
%
% FOUR OVERRIDES, and two knobs deliberately DROPPED:
%   cfg.out_dir   -> a SEPARATE data directory; existing datasets untouched
%   cfg.fig_dir   -> MUST be overridden too. gtd_config derives it from out_dir at
%                    config time, so overriding out_dir alone leaves gtd_plot_record
%                    writing PNGs into the baseline folder under identical per-record
%                    filenames. This was missed once on the kxy variant.
%   cfg.zeta_a    -> 0.03, and cfg.ca recomputed from it (ca is DERIVED in gtd_config)
%   MA_FRAC       -> 0.50, passed to gtd_config
%   DROPPED: BAND_OVERRIDE and AMP_SCALE. They shape the multisine and there is no
%   multisine, so carrying them would be carrying dead knobs. This is also why the
%   folder name has no b140-230 or a6 in it: there is no band or amplitude for those
%   to describe. gtd_config's cfg.f_low/f_high (130/180 for this track) are likewise
%   inert here and are left at their defaults rather than overridden to a value that
%   would only be misleading.
%
% THE PLANT IS IDENTICAL to augmentation_ma50_b140-230_a6_z03. MA_FRAC = 0.50 and
% zeta_a = 0.03 are the same, the controller is untouched, and the only difference
% between the two datasets is the reference and the absence of excitation.
%
% OPEN QUESTION THIS DATASET EXISTS TO ANSWER. With no multisine, nothing
% deliberately excites the 212.13 Hz coupled absorber mode; whatever excitation
% exists comes from the acceleration transitions of the real profile. A cycloidal
% profile is the SMOOTHEST common profile class: its acceleration is a single sine
% at 1/T (12.5 Hz on X, about 10 Hz on Y), it rolls off steeply, and it has no jerk
% discontinuity to spread energy upward. The prior expectation is therefore that
% these records carry little absorber excitation. MEASURED on TP1: 1.32 % total
% output contribution (3.09 % on X, 0.12 % on Y), against 5.62 % for the
% jerk-limited multisine-free record E4_multisine_off, i.e. a factor 4.3 down.
% So the absorber IS excited by the real profile, but least of any motion record.
% Re-measure whenever the record set changes:
%   matlab -batch "addpath('Matlab-scripts/Augmentation-coulomb'); measure_absorber_contribution('augmentation_ma50_z03_telica','TP1_telica_y000.mat',0.50,false)"
% If the contribution is negligible, the honest conclusion is that the absorber is
% not identifiable from ASMPT's production profiles. That is a result, not a failure.
%
% Run it any way you like. Press Run in the editor, call it by name from any working
% directory, or from the shell:
%   matlab -batch "addpath('Matlab-scripts/Augmentation-telica-profiles'); generate_trajectory_data_telica"

clear; clc; close all;

% ─────────────────── path bootstrap (mirrors the coulomb variant) ───────────
% The gtd_* helpers live in Matlab-scripts/Augmentation/data/. Adding them here
% rather than relying on the caller means the script works from the editor Run
% button, which is how it actually gets launched.
%
% This ADDS to the path, it does not shadow: the new files are named
% gtd_build_records_telica and gtd_make_reference_telica, not gtd_build_records or
% gtd_make_reference, so the originals stay the ones every other script resolves.
THIS_DIR  = fileparts(mfilename('fullpath'));                 % .../Augmentation-telica-profiles
REPO_ROOT = fileparts(fileparts(THIS_DIR));                   % repo root
addpath(genpath(fullfile(REPO_ROOT, 'Matlab-scripts', 'Augmentation')));
addpath(THIS_DIR);
assert(exist('gtd_config', 'file') == 2, ...
       ['gtd_config still not found after the path bootstrap. Expected it in %s. ' ...
        'Check that the repo layout is intact.'], ...
       fullfile(REPO_ROOT, 'Matlab-scripts', 'Augmentation', 'data'));

% ─────────────────────────── knobs ──────────────────────────────────────────
TRACK           = 'augmentation';   % must match the dataset being compared against
MA_FRAC         = 0.50;             % D-204/D-205: matches augmentation_ma50_b140-230_a6_z03
ZETA_A_OVERRIDE = 0.03;             % JPE floor for jointed metal; target scales as 1/zeta_a
OUT_DIR_NAME    = 'augmentation_ma50_z03_telica';
% {} = all 7 records (TP1-TP4 train, VP1/VP2 val, EP1 test). Probe form:
% {'TP1'} generates one record, the coulomb variant's convention, for measuring
% the absorber contribution before committing to the full set.
SELECT          = {};
PLOT            = true;
SHOW            = false;
RESUME          = false;            % false = hard stop if the folder already holds records
% ─────────────────────────────────────────────────────────────────────────────

% USE_MSD is hardcoded true, NOT read from a toggle, for the same reason the coulomb
% variant hardcodes it: this dataset exists for the augmentation, which carries the
% hidden MSD. The original generate_trajectory_data.m currently sits at USE_MSD =
% false, so a naive diff of the two files flags this line. It is correct as written.
cfg = gtd_config(TRACK, true, MA_FRAC);

% --- overrides ---
cfg.zeta_a = ZETA_A_OVERRIDE;
% ca is DERIVED from zeta_a in gtd_config (c = 2*zeta*sqrt(k*m)); recompute it from
% the formula rather than scaling the old value.
cfg.ca     = 2*cfg.zeta_a*sqrt(cfg.ka*cfg.ma);

cfg.out_dir = fullfile(REPO_ROOT, 'data', 'gantry', 'matlab', 'trajectory', OUT_DIR_NAME);
% fig_dir must be derived from the NEW out_dir. gtd_config bakes it at config time
% from the ORIGINAL out_dir, so overriding out_dir alone would leave gtd_plot_record
% writing PNGs into the baseline folder under identical per-record filenames,
% silently overwriting that dataset's figures.
cfg.fig_dir = fullfile(cfg.out_dir, 'figures');

% Hard stop rather than a silent overwrite: existing .mat files here mean the folder
% already holds a dataset and OUT_DIR_NAME was not bumped.
if ~RESUME && isfolder(cfg.out_dir) && ~isempty(dir(fullfile(cfg.out_dir, '*.mat')))
    error('generate_trajectory_data_telica:outDirNotEmpty', ...
          ['Output folder already contains .mat records:\n  %s\n' ...
           'Pick a new OUT_DIR_NAME or set RESUME = true.'], cfg.out_dir);
end
if ~exist(cfg.out_dir, 'dir'), mkdir(cfg.out_dir); end
if ~exist(cfg.fig_dir, 'dir'), mkdir(cfg.fig_dir); end

fn = cfg.fa * sqrt(1 + cfg.ma/cfg.mh_rigid);
fprintf('Telica motion-profile dataset (no multisine)\n');
fprintf('  profile      : cycloidal (sinusoidal acceleration), velocity-limited\n');
fprintf('  X            : 40.0 mm, 80.0 ms, vmax 1.000 m/s, amax 39.36 m/s^2 (no cruise)\n');
fprintf('  Y            : 80.0 mm, 99.9 ms, vmax 1.501 m/s, amax 50.60 m/s^2 (6.7 ms cruise)\n');
fprintf('  absorber     : coupled mode %.2f Hz, anti-resonance %.0f Hz, zeta_a %.4f\n', ...
        fn, cfg.fa, cfg.zeta_a);
fprintf('  excitation   : NONE on every record (reference trajectory only)\n');
fprintf('  out_dir      : %s\n', cfg.out_dir);

records = gtd_build_records_telica(cfg);

if ~isempty(SELECT)
    keep = false(1, numel(records));
    for k = 1:numel(records)
        keep(k) = any(startsWith(records(k).id, SELECT));
    end
    assert(any(keep), 'SELECT matched no record ids');
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

    plant         = gtd_build_plant(rec.Y_op, cfg);
    % The ONLY call that differs from the production driver: the Telica reference
    % shape. Every other class delegates to the original gtd_make_reference.
    [r, t]        = gtd_make_reference_telica(rec, cfg);
    ms            = gtd_make_multisine(rec, plant, cfg);   % returns zero force ('none')
    [f_safe, chk] = gtd_enforce_limits(plant, r, ms.f_stage, cfg);
    if chk.scale < 1
        fprintf('  scaled multisine to %.0f%% to meet limits\n', 100*chk.scale);
    end

    out     = gtd_run_simulation(rec, r, t, f_safe, plant, cfg);
    out.amp = chk.scale * ms.A;
    gtd_save_record(out, rec, cfg);
    if PLOT, gtd_plot_record(out, rec, cfg, SHOW || ~isempty(SELECT)); end

    % Reference diagnostics: how many moves the record actually holds, and the
    % realised kinematics of the commanded reference. A profile that silently
    % degenerated (stroke clipped at a range bound, dwell swallowing the window)
    % shows up here rather than three steps later in training.
    report_reference(r, cfg);

    if cfg.use_msd
        fprintf('  delta_a rms with/without = %.3e / %.3e (%.1fx) | force peak [%.0f %.0f %.0f] N\n', ...
                rms(out.da_with), rms(out.da_without), ...
                rms(out.da_with)/max(rms(out.da_without),eps), max(abs(out.u_total)));
    end
    fprintf('  saved %s.mat\n', rec.id);
end

fprintf('\nDone: %d records in %s\n', numel(records), cfg.out_dir);
% NOTE the .mat extension: measure_absorber_contribution builds its path as
% fullfile(..., dataset, record) and asserts isfile, so the record argument is a
% FILENAME, not the bare record id.
fprintf(['\nNEXT: measure the absorber contribution before training on this set.\n' ...
         '  addpath(''Matlab-scripts/Augmentation-coulomb'');\n' ...
         '  measure_absorber_contribution(''%s'', ''%s.mat'', %.2f, false)\n'], ...
        OUT_DIR_NAME, records(1).id, MA_FRAC);

function report_reference(r, cfg)
% Moves, dwell share and realised peak kinematics of the COMMANDED reference.
% r is in STAGE coordinates [X1, X2, Y]; X1 == X2 by construction here.
    v = diff(r) * cfg.fs;
    a = diff(v) * cfg.fs;
    moving = any(abs(v) > 1e-9, 2);
    % A move is a maximal run of moving samples; count the rising edges.
    n_moves = sum(diff([false; moving]) == 1);
    fprintf('  reference: %d moves | moving %.1f%% of record | Y span [%.3f %.3f] m\n', ...
            n_moves, 100*mean(moving), min(r(:,3)), max(r(:,3)));
    fprintf('             peak |v| [%.3f %.3f %.3f] m/s (lim %.1f) | peak |a| [%.1f %.1f %.1f] m/s^2\n', ...
            max(abs(v)), cfg.lim.vel, max(abs(a)));
    fprintf('             yaw max |X1-X2| = %.2e m (lim %.1e)\n', ...
            max(abs(r(:,1)-r(:,2))), cfg.lim.diff);
end
