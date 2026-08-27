% GENERATE_TRAJECTORY_DATA_KXY  T4 dataset: same generator, X/Y spring added.
%
% Mirrors Matlab-scripts/Augmentation/data/generate_trajectory_data.m and calls
% the SAME gtd_* functions, unmodified. Four things are overridden:
%
%   cfg.mdl      -> the kxy model copy (chart passes k_xy into the ODE)
%   cfg.out_dir  -> a SEPARATE data directory; the existing dataset is untouched
%   cfg.fig_dir  -> MUST be overridden too. gtd_config derives it from out_dir at
%                   config time, so overriding out_dir alone leaves figures
%                   writing into the baseline folder. This was missed once.
%   base k_xy    -> the spring constant reaching the plant
%
% CONTROLLER IS DELIBERATELY FROZEN. cfg.K is NOT changed, so gtd_build_plant
% produces the identical Cfb, G, reference and limit scaling as the existing
% dataset. The ONLY difference between this dataset and the current one is the
% plant stiffness. That is the change-one-thing design: if the controller also
% changed, a retrained model's improvement could come from either.
%
% Note that cfg.K at k=10 would barely move the controller anyway (added
% resonance 0.158 Hz against a 130 Hz band edge), so freezing it costs nothing
% physically and buys a clean comparison. Set FREEZE_CONTROLLER = false to make
% the controller consistent with the sprung plant instead.
%
% CAVEAT, and it is why the force-peak print at the bottom of the loop matters:
% freezing cfg.K also means gtd_enforce_limits does its linear pre-check on the
% UNSPRUNG plant, so it never sees the spring's station-keeping force and cannot
% scale the multisine to account for it. At k=10 that force is about 3 N against
% a 30 N multisine and the omission is harmless. It is NOT harmless at large k,
% which is how k=1000 slipped through and suppressed the motion.
%
% THE PYTHON SIDE MUST MATCH, AND IT ALREADY DOES. This changes the TRUTH only;
% the model needs the same k_xy or the test measures baseline-versus-truth
% mismatch instead of the marginal poles. That half is DONE, and it is NOT an
% edit to model_augmentation/systems/gantry_ss.py (user decision, 2026-07-29:
% subclass, leave the shared file alone). It is
% scripts/gantry/drift-isolation/t4_xy_stiffness/blocks_t4.py, reached from the
% CLI as --k_xy 10, which run_training.py requires to be paired with
% --mode augmentation_kxy so the data below and the model cannot fall out of step.
% Do not edit gantry_ss.py.
%
% PREREQUISITES, both must have passed:
%   check_kxy_noop            (k_xy = 0 reproduces the original ODE exactly)
%   check_kxy_reaches_plant   (k_xy reaches the plant, controller frozen)
%
% Run it any way you like. Press Run in the editor, call it by name from any
% working directory, or from the shell:
%   matlab -batch "addpath('Matlab-scripts/Augmentation-kxy'); generate_trajectory_data_kxy"
% The script puts the gtd_* helpers on the path itself (see below), so it no
% longer matters what the working directory is.

clear; clc; close all;   % close all: 22 records x PLOT=true piles up figure windows

% ─────────────────── path bootstrap (added 2026-07-29) ──────────────────────
% The gtd_* helpers live in Matlab-scripts/Augmentation/data/, and running this
% file without them on the path fails at the first call with
% "Unrecognized function or variable 'gtd_config'". Adding them here rather than
% relying on the caller means the script works from the editor Run button, which
% is how it actually gets launched. `clear` above does not touch the path, and
% mfilename('fullpath') is independent of the working directory.
%
% This ADDS to the path, it does not shadow: the copied ODE is named
% gantrySystemExtendedKxy, not gantrySystemExtended, so the original function
% stays the one every other script resolves. Nothing under Augmentation/ moves.
THIS_DIR  = fileparts(mfilename('fullpath'));                 % .../Augmentation-kxy
REPO_ROOT = fileparts(fileparts(THIS_DIR));                   % repo root
addpath(genpath(fullfile(REPO_ROOT, 'Matlab-scripts', 'Augmentation')));
addpath(THIS_DIR);
assert(exist('gtd_config', 'file') == 2, ...
       ['gtd_config still not found after the path bootstrap. Expected it in %s. ' ...
        'Check that the repo layout is intact.'], ...
       fullfile(REPO_ROOT, 'Matlab-scripts', 'Augmentation', 'data'));

% ─────────────────────────── knobs ──────────────────────────────────────────
TRACK   = 'augmentation';   % must match the dataset being replaced
MA_FRAC = 0.10;             % hidden absorber mass fraction (unchanged)
% [N/m] X and Y spring. LOWERED 1000 -> 10 on 2026-07-29 after the first
% generation attempt showed visibly suppressed motion. Reason, measured in
% t4_xy_stiffness/derive_k_small.py (criterion P3):
%   Above k = c^2/(4m) (5.63 N/m for X, 2.48 for Y) BOTH modes are underdamped,
%   so the decay rate is -c/(2m) and does not depend on k at all. k=10 and
%   k=1000 give the IDENTICAL stability margin 8.085e-05 and the IDENTICAL
%   3.09 s decay. The extra stiffness buys nothing T4 wants.
%   What it does buy is a bias force: an ABSOLUTE spring pulling to the machine
%   origin costs k*|Y| just to hold station. At k=1000 and Y=-0.30 m that is
%   300 N against a 30 N multisine, i.e. the spring is 10x the excitation. At
%   k=10 it is 3.0 N, about 10%: a perturbation instead of a takeover.
%   gtd_enforce_limits CANNOT catch this, because its pre-check runs on the
%   frozen-cfg.K (unsprung) plant and never sees the spring force.
K_XY    = 10;
FREEZE_CONTROLLER = true;   % see the header; true = change ONLY the plant
SELECT  = {};               % e.g. {'T1','V1'} to generate a subset; {} = all
PLOT    = true;
SHOW    = false;
% ─────────────────────────────────────────────────────────────────────────────

% USE_MSD is hardcoded true, NOT read from a toggle. Deliberate: T4 must mirror
% the dataset the augmentation actually trains on, which carries the hidden MSD.
% The original generate_trajectory_data.m currently sits at USE_MSD = false
% (it was last used for a baseline batch), so a naive diff of the two files
% flags this line. It is correct as written; do not "fix" it to match.
cfg = gtd_config(TRACK, true, MA_FRAC);

% --- the four overrides ---
cfg.mdl     = 'gantry_additional_state_kxy_2025a';
cfg.out_dir = fullfile(REPO_ROOT, 'data', 'gantry', 'matlab', 'trajectory', ...
                       [TRACK '_kxy']);
% FIXED 2026-07-29. This line was MISSING and it mattered: gtd_config.m:121 bakes
%   cfg.fig_dir = fullfile(cfg.out_dir, 'figures')
% at CONFIG time, from the ORIGINAL out_dir. Overriding out_dir afterwards does
% not follow through to fig_dir, so gtd_plot_record.m:87 kept writing PNGs into
% the BASELINE folder, under identical per-record filenames, overwriting the
% baseline dataset's figures with sprung-plant ones. The .mat files were never
% affected (gtd_save_record uses out_dir, which is overridden correctly), so the
% symptom was only "no figures appear in augmentation_kxy" while real damage was
% happening elsewhere. Derive fig_dir from the NEW out_dir, exactly as
% gtd_config would have.
cfg.fig_dir = fullfile(cfg.out_dir, 'figures');
if ~FREEZE_CONTROLLER
    cfg.K = [K_XY, 0, 0; 0, cfg.kb1 + cfg.kb2, 0; 0, 0, K_XY];
end
assignin('base', 'k_xy', K_XY);

if ~exist(cfg.out_dir, 'dir'), mkdir(cfg.out_dir); end

fprintf('T4 dataset generation\n');
fprintf('  model        : %s\n', cfg.mdl);
fprintf('  k_xy         : %g N/m\n', K_XY);
fprintf('  controller   : %s\n', ternary(FREEZE_CONTROLLER, ...
        'FROZEN (cfg.K unchanged, only the plant differs)', ...
        'consistent with the sprung plant (cfg.K updated)'));
fprintf('  out_dir      : %s\n', cfg.out_dir);

records = gtd_build_records(cfg);
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

    plant         = gtd_build_plant(rec.Y_op, cfg);
    [r, t]        = gtd_make_reference(rec, cfg);
    ms            = gtd_make_multisine(rec, plant, cfg);
    [f_safe, chk] = gtd_enforce_limits(plant, r, ms.f_stage, cfg);
    if chk.scale < 1
        fprintf('  scaled multisine to %.0f%% to meet limits\n', 100*chk.scale);
    end

    assignin('base', 'k_xy', K_XY);      % re-assert: push_params must not shadow it
    out = gtd_run_simulation(rec, r, t, f_safe, plant, cfg);
    % RESTORED 2026-07-29: the first copy dropped this line (and captured chk as
    % ~), so gtd_save_record died on "Unrecognized field name amp". It is not
    % optional bookkeeping: amp_rms is the applied multisine amplitude AFTER the
    % limit scaling, i.e. the record's own statement of how hard it was driven.
    out.amp = chk.scale * ms.A;
    gtd_save_record(out, rec, cfg);
    if PLOT, gtd_plot_record(out, rec, cfg, SHOW || ~isempty(SELECT)); end

    % RESTORED 2026-07-29: dropped in the first copy. This is the diagnostic that
    % would have shown the k_xy force problem on record 1 instead of leaving it
    % to be noticed by eye. FORCE PEAK IS THE ONE TO WATCH: with an absolute
    % spring, holding Y at -0.3 m costs k_xy*0.3 N of DC on top of the multisine.
    if cfg.use_msd
        fprintf('  delta_a rms with/without = %.3e / %.3e (%.1fx) | force peak [%.0f %.0f %.0f] N\n', ...
                rms(out.da_with), rms(out.da_without), ...
                rms(out.da_with)/max(rms(out.da_without),eps), max(abs(out.u_total)));
    end
    fprintf('  saved %s.mat\n', rec.id);
end

fprintf('\nDone: %d records in %s\n', numel(records), cfg.out_dir);

function s = ternary(c, a, b)
    if c, s = a; else, s = b; end
end
