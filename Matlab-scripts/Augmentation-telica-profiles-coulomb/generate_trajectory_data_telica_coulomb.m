% GENERATE_TRAJECTORY_DATA_TELICA_COULOMB  Telica motion profiles + Garcia friction.
%
% The friction twin of augmentation_ma50_z03_telica. It is to that dataset what
% augmentation_ma50_b140-230_a6_z03_coulomb is to the production dataset: the SAME
% records, the SAME plant, the SAME reference profiles, with dry friction added to
% the simulated TRUTH ONLY. The Python baseline stays frictionless, so the
% augmentation has to capture the friction. See D-204.
%
% ONE NEW FILE, BY DESIGN. The record list and the Telica reference shape are NOT
% copied here. This driver addpath's Matlab-scripts/Augmentation-telica-profiles
% and calls gtd_build_records_telica and gtd_make_reference_telica directly, for
% the same reason Augmentation-coulomb reuses Augmentation's gtd_* helpers
% untouched: two copies of a record builder diverge silently, and the moment they
% do, the friction and frictionless Telica sets stop being a matched pair, which
% is the entire point of generating them.
%
% FOUR OVERRIDES relative to generate_trajectory_data_telica.m, and nothing else:
%   cfg.mdl           -> the coulomb model copy (its chart passes cc into the ODE)
%   cfg.cc1/cc2/ccy   -> Garcia's identified values
%   cfg.out_dir       -> a SEPARATE data directory
%   cfg.fig_dir       -> MUST be overridden too. gtd_config derives it from out_dir
%                        at config time, so overriding out_dir alone leaves
%                        gtd_plot_record writing PNGs into the frictionless Telica
%                        folder under identical per-record filenames. This was
%                        missed once on the kxy variant.
%
% THE cc ARE NOT A KNOB. cc1 = 16.8 N, cc2 = 18.35 N, ccy = 11.6 N are Garcia's
% identified values for this machine (THEORY: garcia2013, Table of identified
% parameters, measured by displacing each axis at constant velocity). No sweep.
%
% FRICTION LAW: Karnopp stick-slip, i.e. Garcia's cc*sign(v) while sliding with the
% v = 0 case filled in as Coulomb's law actually states it (|F| <= cc at rest). The
% implementation is gantrySystemExtendedCoulomb.m in Matlab-scripts/Augmentation-coulomb,
% verified by gates 1-3 (D-204): Garcia's constant-velocity identification recovers
% cc1/cc2/ccy to machine precision, Coulomb's law holds at every visited state, and
% the conditioning matches the frictionless control.
%
% ############################################################################
% # RE-MEASURE THE STICK BAND BEFORE TRUSTING THIS DATASET.                  #
% # V_EPS_MARGIN = 9 in gantrySystemExtendedCoulomb.m was measured on the     #
% # 140-230 Hz MULTISINE excitation, where total acceleration is 9-21 m/s^2.  #
% # These records have NO multisine and a different acceleration profile      #
% # (the Telica moves peak near 39-51 m/s^2 on the reference alone), and the  #
% # margin is EXCITATION-DEPENDENT: the band is sized on the friction         #
% # deceleration 0.653 m/s^2 but Leine's criterion constrains how far an RK4  #
% # collocation point TRAVELS, which the total acceleration sets. Generate    #
% # ONE record first (SELECT = {'TP1'}), then run:                            #
% #   check_leine_collocation('augmentation_ma50_z03_telica_coulomb', ...     #
% #                           'TP1_....mat', 0.50)                            #
% # and set V_EPS_MARGIN from that before committing the full set.            #
% ############################################################################
%
% CONTROLLER IS DELIBERATELY FROZEN. cfg.C_damp and cfg.K are NOT changed, so
% gtd_build_plant produces the identical Cfb, G and reference as the frictionless
% Telica set. Coulomb is not representable in a linear plant model anyway. The ONLY
% difference between the two datasets is the dry friction inside the integrator.
%
% EXPECT THE CONTROLLER TO BEHAVE DIFFERENTLY, and it is not a bug. On the
% production dataset friction cut the realised forces by 30-80% and arrested the Y
% axis hard; on E1_resonance_sweep, whose drive never reached the breakaway
% threshold, the X rails locked solid. Watch for the same on any low-force Telica
% record. That is real behaviour of a frictional stage under a friction-blind
% controller, and it is part of what the dataset is for.
%
% SOLVER. The coulomb model copy is built fixed-step ode4 by make_coulomb_model.
% FIXED_STEP is re-asserted here so the value is visible at the call site and is
% whatever this script says, not whatever was last saved into the .slx.
%
% Run:
%   matlab -batch "addpath('Matlab-scripts/Augmentation-telica-profiles-coulomb'); generate_trajectory_data_telica_coulomb"

clear; clc; close all;

% ─────────────────── path bootstrap ─────────────────────────────────────────
% Three folders are added and none of them shadow each other: the new files are
% named *_telica and *_telica_coulomb, never gtd_build_records or
% gtd_make_reference, so the originals stay the ones every other script resolves.
THIS_DIR   = fileparts(mfilename('fullpath'));            % .../Augmentation-telica-profiles-coulomb
REPO_ROOT  = fileparts(fileparts(THIS_DIR));              % repo root
TELICA_DIR = fullfile(REPO_ROOT, 'Matlab-scripts', 'Augmentation-telica-profiles');
COULOMB_DIR= fullfile(REPO_ROOT, 'Matlab-scripts', 'Augmentation-coulomb');
addpath(genpath(fullfile(REPO_ROOT, 'Matlab-scripts', 'Augmentation')));
addpath(TELICA_DIR);
addpath(COULOMB_DIR);
addpath(THIS_DIR);
assert(exist('gtd_config', 'file') == 2, ...
       'gtd_config not found after the path bootstrap. Check the repo layout.');
assert(exist('gtd_build_records_telica', 'file') == 2, ...
       ['gtd_build_records_telica not found. It lives in %s and is CALLED, not ' ...
        'copied, so the friction and frictionless Telica sets cannot diverge.'], TELICA_DIR);
assert(exist('gtd_make_reference_telica', 'file') == 2, ...
       'gtd_make_reference_telica not found. Expected it in %s.', TELICA_DIR);

% ─────────────────────────── knobs ──────────────────────────────────────────
TRACK           = 'augmentation';   % must match the dataset being compared against
MA_FRAC         = 0.50;             % matches augmentation_ma50_z03_telica
ZETA_A_OVERRIDE = 0.03;             % JPE floor for jointed metal
OUT_DIR_NAME    = 'augmentation_ma50_z03_telica_coulomb';
% THEORY: garcia2013 -- Coulomb friction of actuators X1, X2 and the Y payload,
% identified by displacing each axis at constant velocity. NOT a knob.
CC1 = 16.80;   % [N]
CC2 = 18.35;   % [N]
CCY = 11.60;   % [N]
% FULL SET, all 7 records (TP1-TP4 train, VP1/VP2 val, EP1 test). The probe is
% done: TP1 was generated first and BOTH gates passed on it, with zero collocation
% violations and a perturbation gain ratio of 1.609, so V_EPS_MARGIN = 9 transfers
% to these profiles unchanged and the full set is committed.
% SELECT = {'TP1'} restores the probe form.
SELECT          = {};
PLOT            = true;
SHOW            = false;
% TRUE, deliberately: TP1 is already on disk and verified, and regenerating it
% would discard that. RESUME skips any record whose .mat already exists.
RESUME          = true;
FIXED_STEP      = [];               % [] = cfg.ts (5e-5 s)
% ─────────────────────────────────────────────────────────────────────────────

cfg = gtd_config(TRACK, true, MA_FRAC);

% --- override 1/4: the friction-carrying model copy ---
cfg.mdl = 'gantry_additional_state_coulomb_2025a';
% --- override 2/4: Garcia's identified Coulomb forces ---
cfg.cc1 = CC1;  cfg.cc2 = CC2;  cfg.ccy = CCY;

% Plant knobs, identical to the frictionless Telica set so the pair differs by
% friction alone. ca is DERIVED from zeta_a in gtd_config (c = 2*zeta*sqrt(k*m));
% recompute from the formula rather than scaling the old value.
cfg.zeta_a = ZETA_A_OVERRIDE;
cfg.ca     = 2*cfg.zeta_a*sqrt(cfg.ka*cfg.ma);

% --- override 3/4 and 4/4: a SEPARATE output folder, figures included ---
cfg.out_dir = fullfile(REPO_ROOT, 'data', 'gantry', 'matlab', 'trajectory', OUT_DIR_NAME);
cfg.fig_dir = fullfile(cfg.out_dir, 'figures');

if isempty(FIXED_STEP), FIXED_STEP = cfg.ts; end

% Hard stop rather than a silent overwrite.
if ~RESUME && isfolder(cfg.out_dir) && ~isempty(dir(fullfile(cfg.out_dir, '*.mat')))
    error('generate_trajectory_data_telica_coulomb:outDirNotEmpty', ...
          ['Output folder already contains .mat records:\n  %s\n' ...
           'Pick a new OUT_DIR_NAME or set RESUME = true.'], cfg.out_dir);
end
if ~exist(cfg.out_dir, 'dir'), mkdir(cfg.out_dir); end
if ~exist(cfg.fig_dir, 'dir'), mkdir(cfg.fig_dir); end

% Re-assert the solver at the call site so the value is whatever FIXED_STEP says,
% not whatever was last saved into the .slx.
load_system(cfg.mdl);
set_param(cfg.mdl, 'SolverType', 'Fixed-step');
set_param(cfg.mdl, 'Solver',     'ode4');
set_param(cfg.mdl, 'FixedStep',  sprintf('%.12g', FIXED_STEP));

fn = cfg.fa * sqrt(1 + cfg.ma/cfg.mh_rigid);
fprintf('Telica motion-profile dataset WITH Garcia friction (no multisine)\n');
fprintf('  model        : %s\n', cfg.mdl);
fprintf('  cc (Garcia)  : cc1 %.2f  cc2 %.2f  ccy %.2f  N\n', CC1, CC2, CCY);
fprintf('  solver       : %s / %s, fixed step %.3e s\n', ...
        get_param(cfg.mdl,'SolverType'), get_param(cfg.mdl,'Solver'), FIXED_STEP);
fprintf('  absorber     : coupled mode %.2f Hz, anti-resonance %.0f Hz, zeta_a %.4f\n', ...
        fn, cfg.fa, cfg.zeta_a);
fprintf('  excitation   : NONE on every record (reference trajectory only)\n');
fprintf('  out_dir      : %s\n', cfg.out_dir);
fprintf('  twin of      : augmentation_ma50_z03_telica (frictionless)\n');

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

    report_reference(r, cfg);

    if cfg.use_msd
        fprintf('  delta_a rms with/without = %.3e / %.3e (%.1fx) | force peak [%.0f %.0f %.0f] N\n', ...
                rms(out.da_with), rms(out.da_without), ...
                rms(out.da_with)/max(rms(out.da_without),eps), max(abs(out.u_total)));
    end
    % Stick fraction is the friction-specific diagnostic: a record whose rails never
    % break away carries no information on those axes. E1_resonance_sweep on the
    % production dataset came out 100% stuck on X1 and X2 with a 4e-09 N drive, and
    % that only became visible from this line.
    % out.q_with is already in STAGE coordinates [X1 X2 Y] and holds POSITIONS,
    % so differentiate it; do not project it with P. Identical to the computation
    % in generate_trajectory_data_coulomb.m so the two datasets' numbers compare.
    V_STICK = 1e-3;
    v_stage = zeros(size(out.q_with, 1), 3);
    for j = 1:3
        v_stage(:,j) = gradient(out.q_with(:,j), cfg.ts);
    end
    frac = mean(abs(v_stage) < V_STICK, 1);
    fprintf('  stick fraction (|v| < %.0e m/s) : X1 %.1f%%  X2 %.1f%%  Y %.1f%%\n', ...
            V_STICK, 100*frac(1), 100*frac(2), 100*frac(3));
    fprintf('  saved %s.mat\n', rec.id);
end

fprintf('\nDone: %d records in %s\n', numel(records), cfg.out_dir);
fprintf(['\nNEXT, in this order:\n' ...
         '  1. RE-MEASURE THE STICK BAND (see this file''s banner):\n' ...
         '       addpath(''Matlab-scripts/Augmentation-coulomb'');\n' ...
         '       check_leine_collocation(''%s'', ''%s.mat'', %.2f)\n' ...
         '       check_friction_invariants(''%s'', ''%s.mat'', %.2f)\n' ...
         '  2. Absorber contribution, friction vs frictionless twin:\n' ...
         '       measure_absorber_contribution(''%s'', ''%s.mat'', %.2f, true)\n' ...
         '  3. Only then set SELECT = {} and generate the full set.\n'], ...
        OUT_DIR_NAME, records(1).id, MA_FRAC, ...
        OUT_DIR_NAME, records(1).id, MA_FRAC, ...
        OUT_DIR_NAME, records(1).id, MA_FRAC);

function report_reference(r, cfg)
% Moves, dwell share and realised peak kinematics of the COMMANDED reference.
% r is in STAGE coordinates [X1, X2, Y]; X1 == X2 by construction here. Copied
% from the frictionless driver because MATLAB local functions are file-scoped;
% it reports the COMMAND, which friction does not change, so the two drivers
% printing the same numbers here is the expected result and a useful check.
    v = diff(r) * cfg.fs;
    a = diff(v) * cfg.fs;
    moving = any(abs(v) > 1e-9, 2);
    n_moves = sum(diff([false; moving]) == 1);
    fprintf('  reference: %d moves | moving %.1f%% of record | Y span [%.3f %.3f] m\n', ...
            n_moves, 100*mean(moving), min(r(:,3)), max(r(:,3)));
    fprintf('  reference kinematics: vmax %.3f m/s, amax %.2f m/s^2 (X), %.3f m/s, %.2f m/s^2 (Y)\n', ...
            max(abs(v(:,1))), max(abs(a(:,1))), max(abs(v(:,3))), max(abs(a(:,3))));
end
