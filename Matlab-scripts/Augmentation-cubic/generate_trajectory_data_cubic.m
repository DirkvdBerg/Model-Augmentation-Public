% GENERATE_TRAJECTORY_DATA_CUBIC  D-135 dataset: baseline plant, cubic spring on X and Y.
%
% Mirrors Matlab-scripts/Augmentation/data/generate_trajectory_data.m and calls
% the SAME gtd_* functions, unmodified. Four things are overridden:
%
%   cfg.mdl      -> the cubic model copy (chart passes k3 into the ODE)
%   cfg.out_dir  -> a SEPARATE data directory, named after k3 so the k3 = 0 null
%                   arm and the production arm can never land on each other
%   cfg.fig_dir  -> MUST be overridden too. gtd_config derives it from out_dir at
%                   config time, so overriding out_dir alone leaves figures
%                   writing into ANOTHER dataset's folder. This has already
%                   destroyed two baseline figures once, during T4.
%   base k3      -> the spring constant reaching the plant
%
% USE_MSD = false, deliberately. The absorber is REMOVED for this experiment, so
% the truth is the 6-state baseline gantry plus one static nonlinearity and
% nothing else. That is what makes it an isolating test: at k3 = 0 the Python
% baseline equals the truth EXACTLY, which is a null control no other dataset in
% this project offers.
%
% CONSEQUENCES OF THE NON-MSD BRANCH, all existing behaviour on an exercised path:
%   * gtd_config selects the 1 to 7 Hz multisine band instead of 130 to 180 Hz.
%     KEPT ON PURPOSE. The 130 to 180 band exists only to excite the 150 Hz
%     absorber, which is gone. A cubic force goes as position cubed, and at
%     150 Hz the displacements are tiny for any given force, so a high-frequency
%     band would barely excite the term the ANN is being asked to learn.
%   * gtd_run_simulation reads q1 (the gantrySystem ODE output) rather than
%     q_aug, skips the second without-multisine run, and does not swap mh.
%   * the records carry no meaningful delta_a / vdelta_a.
%
% CONTROLLER IS FROZEN, and for free: cfg.K cannot express a cubic term, so
% gtd_build_plant produces the identical Cfb, G, reference and limit scaling as
% the unsprung dataset. The only difference is the plant.
%
% CAVEAT, and it is why the force-peak print at the bottom of the loop matters:
% gtd_enforce_limits does its linear pre-check on the frozen-cfg.K (unsprung)
% plant, so it never sees the spring force and cannot scale the multisine to
% account for it. At the sizes derive_k3 produces the spring is a fraction of a
% percent of the excitation and the omission is harmless. It would NOT be
% harmless at a P3-sized k3, which is how T4's k = 1000 slipped through and
% suppressed the motion.
%
% THE PYTHON SIDE MUST NOT MATCH. This changes the TRUTH only. The baseline model
% (model_augmentation/systems/gantry_ss.py, Gantry_State_Block) stays unsprung,
% and the mismatch IS the learning target. Do not edit gantry_ss.py.
%
% PREREQUISITES, all three must have passed:
%   check_cubic_noop           (k3 = 0 reproduces the original ODE exactly)
%   make_cubic_model           (the .slx copy exists and its chart calls the copy)
%   check_cubic_reaches_plant  (k3 reaches the plant and scales linearly)
% and derive_k3 must have been run to set K3 below.
%
% Run it any way you like:
%   matlab -batch "addpath('Matlab-scripts/Augmentation-cubic'); generate_trajectory_data_cubic"

function generate_trajectory_data_cubic(varargin)
% Usage:
%   generate_trajectory_data_cubic()                          production arm
%   generate_trajectory_data_cubic('K3', 0)                   null arm
%   generate_trajectory_data_cubic('SELECT', {'T1','V1'})     subset smoke
%   generate_trajectory_data_cubic('PLOT', false)             skip figures
%
% A FUNCTION rather than a script (T4's wrapper is a script) so the knobs can be
% passed in without editing the file. A script starting with `clear` cannot be
% parameterised at all, which forces an edit-run-revert cycle around every
% subset run, and that is how a smoke setting gets left in place by accident.

close all;   % PLOT = true piles up figure windows

% ─────────────────── path bootstrap ─────────────────────────────────────────
% The gtd_* helpers live in Matlab-scripts/Augmentation/data/. Adding them here
% rather than relying on the caller means the script works from the editor Run
% button. This ADDS to the path, it does not shadow: the copied ODE is named
% gantrySystemCubic, not gantrySystem, so the original stays the one every other
% script resolves. Nothing under Augmentation/ or kamtin-fp-model/ moves.
THIS_DIR  = fileparts(mfilename('fullpath'));
REPO_ROOT = fileparts(fileparts(THIS_DIR));
addpath(genpath(fullfile(REPO_ROOT, 'Matlab-scripts', 'Augmentation')));
addpath(genpath(fullfile(REPO_ROOT, 'kamtin-fp-model', '03 Simulink gantry')));
addpath(THIS_DIR);
assert(exist('gtd_config', 'file') == 2, ...
       'gtd_config not found after the path bootstrap. Check that the repo layout is intact.');

% ─────────────────────────── knobs ──────────────────────────────────────────
TRACK   = 'augmentation';   % with USE_MSD = false this affects only track_id;
                            % the band comes from the non-MSD branch regardless
% [N/m^3] cubic spring on X and Y. FROM derive_k3, run 2026-07-30. Do not guess.
% Sized by the free-run degradation it induces, NOT by excitation preservation:
% the spring is truth-only, so the whole spring force is a mismatch force on a
% K = 0 integrating axis, and only its DC part ramps. Per-axis gains 0.173 m/N
% (X) and 0.635 m/N (Y). This value puts the weakest validation record
% (V1_standstill_Yp10) at 10x the 1.66e-4 m untrained baseline; the other three
% val records land at 142x, 38x and 58x. Worst free-run excursion 0.078 m
% against a 0.4 m travel limit, and the spring is 0.235% of the multisine, so
% T4's P3 excitation criterion is satisfied by a factor of 40 and is not the
% binding constraint. Set K3 = 0 to generate the NULL ARM.
K3      = 2.616;
SELECT  = {};               % e.g. {'T1','V1'} to generate a subset; {} = all
PLOT    = true;
SHOW    = false;

% --- optional overrides from the call ---------------------------------------
for a = 1:2:numel(varargin)
    switch upper(varargin{a})
        case 'K3',     K3     = varargin{a+1};
        case 'SELECT', SELECT = varargin{a+1};
        case 'PLOT',   PLOT   = varargin{a+1};
        case 'SHOW',   SHOW   = varargin{a+1};
        otherwise,     error('generate_trajectory_data_cubic:opt', ...
                             'Unknown option "%s"', varargin{a});
    end
end
% ─────────────────────────────────────────────────────────────────────────────

cfg = gtd_config(TRACK, false, 0);   % USE_MSD = false: no absorber

% --- the overrides -----------------------------------------------------------
cfg.mdl = 'gantry_2025a_cubic';

% Name the folder after k3 so the null arm and the production arm CANNOT
% overwrite each other, and so a folder on disk always states what made it.
if K3 == 0
    ds_name = [TRACK '_cubic_k0'];
else
    ds_name = [TRACK '_cubic'];
end
% A SELECT run is a SUBSET and must never land in a production folder. A partial
% dataset that looks complete is the failure mode behind this project's
% record-count lesson: at 2 records the encoder-init Y RMS is about 500x worse
% than at 14, so any number computed from a partial set is silently invalid.
if ~isempty(SELECT)
    ds_name = sprintf('%s_n%d', ds_name, numel(SELECT));
end
cfg.out_dir = fullfile(REPO_ROOT, 'data', 'gantry', 'matlab', 'trajectory', ds_name);
% gtd_config bakes fig_dir from out_dir at CONFIG time, so overriding out_dir
% alone leaves gtd_plot_record writing PNGs into another dataset's figures
% folder under identical per-record filenames. Derive it from the NEW out_dir.
cfg.fig_dir = fullfile(cfg.out_dir, 'figures');

assert(~isfolder(cfg.out_dir) || isempty(dir(fullfile(cfg.out_dir, '*.mat'))), ...
       ['Refusing to write: %s already contains .mat records. Generated datasets ' ...
        'are never overwritten. Move or delete the old one deliberately first.'], cfg.out_dir);

assignin('base', 'k3', K3);

if ~exist(cfg.out_dir, 'dir'), mkdir(cfg.out_dir); end

fprintf('D-135 cubic-spring dataset generation\n');
fprintf('  model        : %s\n', cfg.mdl);
fprintf('  k3           : %g N/m^3  (on X and Y)\n', K3);
fprintf('  absorber     : ABSENT (use_msd = false, 6-state truth)\n');
fprintf('  band         : %g to %g Hz\n', cfg.f_low, cfg.f_high);
fprintf('  controller   : FROZEN (cfg.K cannot express a cubic term)\n');
fprintf('  out_dir      : %s\n', cfg.out_dir);
if K3 == 0
    fprintf('  *** NULL ARM: baseline model equals truth exactly. ***\n');
end

records = gtd_build_records(cfg);
if ~isempty(SELECT)
    % EXACT match on the record tag (the part before the first underscore), NOT
    % startsWith. T4's wrapper used startsWith and it silently over-selects:
    % SELECT = {'T1'} matches T1, T10, T11, T12, T13 and T14, so a "2 record"
    % smoke generates 7. Measured 2026-07-30.
    keep = false(1, numel(records));
    for k = 1:numel(records)
        tag     = strtok(records(k).id, '_');
        keep(k) = any(strcmp(tag, SELECT));
    end
    assert(any(keep), 'SELECT matched no record tags. Use tags like {''T1'',''V2''}.');
    records = records(keep);
end

fprintf('Generating %d records  ->  %s\n', numel(records), cfg.out_dir);

% Provenance sidecar: a dataset folder must state what produced it, so a later
% reader never has to infer k3 from the numbers. Same intent as D-131's sidecar.
fid = fopen(fullfile(cfg.out_dir, 'PROVENANCE.txt'), 'w');
fprintf(fid, 'dataset   : %s\n', ds_name);
fprintf(fid, 'generator : generate_trajectory_data_cubic.m (D-135)\n');
fprintf(fid, 'model     : %s\n', cfg.mdl);
fprintf(fid, 'ode       : gantrySystemCubic.m, f_nl = [-k3*X^3; 0; -k3*Y^3]\n');
fprintf(fid, 'k3        : %.10g N/m^3 (X and Y)\n', K3);
fprintf(fid, 'use_msd   : false (no absorber, 6-state truth)\n');
fprintf(fid, 'band      : %g to %g Hz\n', cfg.f_low, cfg.f_high);
fprintf(fid, 'created   : %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'note      : TRUTH ONLY. The Python baseline stays unsprung; the\n');
fprintf(fid, '            mismatch is the learning target. Do not edit gantry_ss.py.\n');
fclose(fid);

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

    assignin('base', 'k3', K3);      % re-assert: push_params must not shadow it
    out = gtd_run_simulation(rec, r, t, f_safe, plant, cfg);
    out.amp = chk.scale * ms.A;      % applied amplitude AFTER limit scaling
    gtd_save_record(out, rec, cfg);
    if PLOT, gtd_plot_record(out, rec, cfg, SHOW || ~isempty(SELECT)); end

    % FORCE PEAK IS THE ONE TO WATCH. gtd_enforce_limits pre-checks on the
    % unsprung plant and never sees the spring, so this print is the only place
    % a sizing error shows up on record 1 rather than by eye much later.
    fprintf('  |Y_op| = %.3f m -> spring force %.4e N | force peak [%.0f %.0f %.0f] N\n', ...
            abs(rec.Y_op), K3 * abs(rec.Y_op)^3, max(abs(out.u_total)));
    fprintf('  saved %s.mat\n', rec.id);
end

fprintf('\nDone: %d records in %s\n', numel(records), cfg.out_dir);
end
