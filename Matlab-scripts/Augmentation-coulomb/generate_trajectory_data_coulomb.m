% GENERATE_TRAJECTORY_DATA_COULOMB  Same generator, Garcia dry friction added.
%
% Mirrors Matlab-scripts/Augmentation/data/generate_trajectory_data.m and calls
% the SAME gtd_* functions, unmodified. Four things are overridden:
%
%   cfg.mdl              -> the coulomb model copy (chart passes cc into the ODE)
%   cfg.cc1/cc2/ccy      -> Garcia's identified values, reaching the plant at last
%   cfg.out_dir          -> a SEPARATE data directory; existing datasets untouched
%   cfg.fig_dir          -> MUST be overridden too. gtd_config derives it from
%                           out_dir at config time, so overriding out_dir alone
%                           leaves figures writing into the baseline folder. This
%                           was missed once on the kxy variant.
%
% THE cc ARE NOT A KNOB. cc1 = 16.8 N, cc2 = 18.35 N, ccy = 11.6 N are Garcia's
% identified values for this machine and they are used as given. There is no
% sweep and nothing here is tuned to make the offset behave. If the offset
% survives, that is the answer; if it collapses, that is equally the answer.
%
% FRICTION LAW: Karnopp stick-slip, i.e. Garcia's cc*sign(v) while sliding with
% the v = 0 case filled in as Coulomb's law actually states it (|F| <= cc at
% rest). This REPLACED hard sign on 2026-08-01. Measured reason: sign(0) = 0
% makes the vector field discontinuous, and a 1e-12 m/s perturbation grew by 1e6
% within 3 s, putting a ~1e-6 m floor under any replay. Karnopp brings that gain
% to 1.47, matching the frictionless system's 1.43. No new physical parameter is
% introduced; classical Coulomb friction has one coefficient and is already
% set-valued at rest. See gantrySystemExtendedCoulomb.m and D-138.
%
% CONTROLLER IS DELIBERATELY FROZEN. cfg.C_damp and cfg.K are NOT changed, so
% gtd_build_plant produces the identical Cfb, G, reference and limit scaling as
% the frictionless dataset. Coulomb is not representable in a linear plant model
% anyway, so there is nothing to put there. The ONLY difference between this
% dataset and the existing one is the dry friction inside the integrator. That is
% the change-one-thing design.
%
% EXPECT THE CONTROLLER TO BEHAVE DIFFERENTLY, and it is not a bug. Cfb was
% designed on a frictionless linear plant; with Coulomb in the loop the closed
% loop can hunt or sit in a deadband at standstill. That is real behaviour of a
% frictional stage under a friction-blind controller, and it is part of what the
% record is for. It does mean this dataset is NOT interchangeable with the
% frictionless one for anything other than the friction question.
%
% SOLVER. The model copy is built fixed-step ode4 by make_coulomb_model (the
% original is variable-step ode45, which is wrong for a discontinuous sign; see
% that file's header). FIXED_STEP is re-asserted here so it is visible at the
% call site and so check_step_halving can drive it.
%
% PREREQUISITES, both must have passed:
%   check_coulomb_noop           (cc = 0 reproduces the original ODE exactly)
%   check_coulomb_reaches_plant  (cc reach the integrator; cc = 0 bit-identical)
%
% Run it any way you like. Press Run in the editor, call it by name from any
% working directory, or from the shell:
%   matlab -batch "addpath('Matlab-scripts/Augmentation-coulomb'); generate_trajectory_data_coulomb"

clear; clc; close all;

% ─────────────────── path bootstrap (mirrors the kxy variant) ───────────────
% The gtd_* helpers live in Matlab-scripts/Augmentation/data/. Adding them here
% rather than relying on the caller means the script works from the editor Run
% button, which is how it actually gets launched.
%
% This ADDS to the path, it does not shadow: the copied ODE is named
% gantrySystemExtendedCoulomb, not gantrySystemExtended, so the original function
% stays the one every other script resolves. Nothing under Augmentation/ moves.
THIS_DIR  = fileparts(mfilename('fullpath'));                 % .../Augmentation-coulomb
REPO_ROOT = fileparts(fileparts(THIS_DIR));                   % repo root
addpath(genpath(fullfile(REPO_ROOT, 'Matlab-scripts', 'Augmentation')));
addpath(THIS_DIR);
assert(exist('gtd_config', 'file') == 2, ...
       ['gtd_config still not found after the path bootstrap. Expected it in %s. ' ...
        'Check that the repo layout is intact.'], ...
       fullfile(REPO_ROOT, 'Matlab-scripts', 'Augmentation', 'data'));

% ─────────────────────────── knobs ──────────────────────────────────────────
TRACK   = 'augmentation';   % must match the dataset being compared against
MA_FRAC = 0.50;             % D-204: matches augmentation_ma50_b140-230_a6_z03
% THEORY: garcia2013 (garcia2013_gantry-decoupling-control.pdf, Table of
% identified parameters) -- Coulomb friction of actuators X1, X2 and the Y
% payload, identified by displacing each axis at constant velocity. NOT a knob.
CC1 = 16.80;   % [N]
CC2 = 18.35;   % [N]
CCY = 11.60;   % [N]
% One record by default. V1_standstill_Yp10 is deliberate: it is the record every
% existing offset number is quoted on, so this dataset is directly comparable.
% Use {} for all 22, or e.g. {'T6'} for the sliding-regime record.
SELECT  = {'E'};            % TEST SET ONLY: E1-E4. The 2026-09-19 batch was stopped
                            % during record 19/22, leaving T1-T14 and V1-V4 complete on
                            % disk. This driver has NO RESUME, so re-running the full set
                            % would regenerate all 22 and discard that work; the prefix
                            % filter finishes the job instead. Restore {} for a clean
                            % full regeneration.
% SELECT  = {};             % FULL SET, 22 records (T1-14 train, V1-4 val, E1-4 test).
                            % The D-204 probe is done: T3 was generated first at these
                            % knobs and cleared every prerequisite (gates 1/2/3 pass, the
                            % limiter never fires, the absorber is not demoted), so the
                            % batch is now committed. There is no RESUME in this driver,
                            % so an interrupted run restarts from scratch and T3 is
                            % regenerated along with the rest.
% SELECT  = {'T3'};         % D-204 probe form: ONE record before committing 22. T3 is the
                            % standstill record the band and amplitude were originally
                            % established on, and standstill is the most stick-prone case,
                            % so it is the worst case for V_EPS and directly comparable to
                            % its frictionless counterpart. Use {} for all 22.
PLOT    = true;
SHOW    = false;
FIXED_STEP = [];            % [] = use cfg.ts (5e-5 s). Set explicitly to probe
                            % step dependence; check_step_halving does this.
% ─────────────────────────────────────────────────────────────────────────────

% USE_MSD is hardcoded true, NOT read from a toggle. Deliberate: the friction
% question is about the dataset the augmentation actually trains on, which
% carries the hidden MSD. The original generate_trajectory_data.m currently sits
% at USE_MSD = false, so a naive diff of the two files flags this line. It is
% correct as written; do not "fix" it to match.
cfg = gtd_config(TRACK, true, MA_FRAC);

% --- the four overrides ---
cfg.mdl     = 'gantry_additional_state_coulomb_2025a';
cfg.cc1     = CC1;
cfg.cc2     = CC2;
cfg.ccy     = CCY;
% NEW FOLDER, NOT AN OVERWRITE. `augmentation_coulomb` holds the SUPERSEDED
% hard-sign dataset (D-138 as originally written). That model used sign(0) = 0,
% which amplified round-off by 1e6 and put a ~1e-6 m floor under every replay
% measurement; the Karnopp stick state removes it. The old dataset stays on disk
% so the two laws can be compared directly and so nothing already quoted against
% it silently changes underneath.
cfg.out_dir = fullfile(REPO_ROOT, 'data', 'gantry', 'matlab', 'trajectory', ...
                       'augmentation_ma50_b140-230_a6_z03_coulomb');
% fig_dir must be derived from the NEW out_dir. gtd_config bakes it at config
% time from the ORIGINAL out_dir, so overriding out_dir alone would leave
% gtd_plot_record writing PNGs into the baseline folder under identical
% per-record filenames, silently overwriting that dataset's figures.
cfg.fig_dir = fullfile(cfg.out_dir, 'figures');

% ── D-204 five-knob port: match augmentation_ma50_b140-230_a6_z03 ───────────
% Applied AFTER gtd_config for the same reason the production generator does it:
% the shipped defaults stay intact so every earlier dataset stays reproducible.
% MA_FRAC is passed to gtd_config above; the remaining three are set here.
% NOTE none of these touch the stick band. v_eps = margin*(cc1+cc2)/m_total*ts
% and m_total = m1+m2+mb+mh_rigid+ma = m1+m2+mb+mh is INVARIANT to ma_frac, so
% the band is identical at 0.10 and 0.50. What they DO change is the
% acceleration, and therefore how far an RK4 collocation point travels per step,
% which is what V_EPS actually has to cover. That is why this probe exists.
BAND_OVERRIDE   = [140, 230];   % holds the 212.13 Hz pole AND the 150 Hz anti-resonance
AMP_SCALE       = 6;            % largest value at which NO record is scaled down (frictionless
                                % pre-check; RE-CHECK with friction in the loop, D-204)
ZETA_A_OVERRIDE = 0.03;         % JPE floor for jointed metal; target scales as 1/zeta_a

fn = sqrt(cfg.ka/cfg.ma)/(2*pi) * sqrt(1 + cfg.ma/(cfg.mh - cfg.ma));
assert(BAND_OVERRIDE(1) < fn && fn < BAND_OVERRIDE(2), ...
       'BAND_OVERRIDE [%g %g] does not contain the coupled absorber mode %.2f Hz', ...
       BAND_OVERRIDE(1), BAND_OVERRIDE(2), fn);
cfg.f_low = BAND_OVERRIDE(1);  cfg.f_high = BAND_OVERRIDE(2);
cfg.zeta_a = ZETA_A_OVERRIDE;
cfg.ca     = 2*cfg.zeta_a*sqrt(cfg.ka*cfg.ma);   % ca is DERIVED from zeta_a in gtd_config
cfg.A_sym  = AMP_SCALE * cfg.A_sym;
cfg.A_Y    = AMP_SCALE * cfg.A_Y;
% A_anti is DERIVED from A_sym in gtd_config (A_anti = 0.5*A_sym*Lb, so the anti
% channel contributes the same per-rail force RMS as the symmetric one), exactly
% like ca above. Recompute it from the formula rather than scaling the old value.
%
% THESE TWO LINES WERE MISSING, and that is the defect this block was fixed for.
% Scaling A_sym alone left A_Y and A_anti at their gtd_config defaults, so the
% generated dataset carried amp_rms (240, 14.5, 30) against the frictionless
% (240, 87, 180): the X_anti and Y multisines were 6x weaker on the friction side.
% The pair was therefore NOT a controlled comparison, and a baseline-residual plot
% built on it showed the friction residual coming out SMALLER than the frictionless
% one (Y ratio 0.14, against the amplitude ratio 30/180 = 0.167) purely because the
% excitation differed. See generate_trajectory_data.m:170-180, which has carried
% the correct three-line form and its warning all along.
cfg.A_anti = 0.5 * cfg.A_sym * cfg.Lb;
fprintf('Knob port: band [%g %g] Hz (mode %.2f Hz), amp x%g, zeta_a %.4f, ca %.4f\n', ...
        cfg.f_low, cfg.f_high, fn, AMP_SCALE, cfg.zeta_a, cfg.ca);
% Print the RESULTING amplitudes, not just the scale factor. Logging only
% AMP_SCALE is what let the defect above survive: the log said "amp x6" while two
% of three channels were unscaled.
fprintf('Amplitude override x%g: A_sym=%.1f N  A_Y=%.1f N  A_anti=%.1f N*m\n', ...
        AMP_SCALE, cfg.A_sym, cfg.A_Y, cfg.A_anti);

if isempty(FIXED_STEP), FIXED_STEP = cfg.ts; end

if ~exist(cfg.out_dir, 'dir'), mkdir(cfg.out_dir); end
if ~exist(cfg.fig_dir, 'dir'), mkdir(cfg.fig_dir); end

% Re-assert the solver at the call site so it is visible here and so the value
% is whatever FIXED_STEP says, not whatever was last saved into the .slx.
load_system(cfg.mdl);
set_param(cfg.mdl, 'SolverType', 'Fixed-step');
set_param(cfg.mdl, 'Solver',     'ode4');
set_param(cfg.mdl, 'FixedStep',  sprintf('%.12g', FIXED_STEP));

fprintf('Coulomb dataset generation\n');
fprintf('  model        : %s\n', cfg.mdl);
fprintf('  cc (Garcia)  : cc1 %.2f  cc2 %.2f  ccy %.2f  N\n', CC1, CC2, CCY);
fprintf('  solver       : %s / %s, fixed step %.3e s\n', ...
        get_param(cfg.mdl,'SolverType'), get_param(cfg.mdl,'Solver'), FIXED_STEP);
fprintf('  controller   : FROZEN (cfg.K, cfg.C_damp unchanged; only the plant differs)\n');
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

    out     = gtd_run_simulation(rec, r, t, f_safe, plant, cfg);
    out.amp = chk.scale * ms.A;
    gtd_save_record(out, rec, cfg);
    if PLOT, gtd_plot_record(out, rec, cfg, SHOW || ~isempty(SELECT)); end

    if cfg.use_msd
        fprintf('  delta_a rms with/without = %.3e / %.3e (%.1fx) | force peak [%.0f %.0f %.0f] N\n', ...
                rms(out.da_with), rms(out.da_without), ...
                rms(out.da_with)/max(rms(out.da_without),eps), max(abs(out.u_total)));
    end
    % Friction-specific diagnostic: the share of samples on each PHYSICAL rail
    % whose speed is below 1 mm/s. If a standstill record is stuck for most of
    % its length the record carries little information, and any "the offset went
    % away" reading would be about an uninformative dataset, not about physics.
    report_stick(out, cfg);
    fprintf('  saved %s.mat\n', rec.id);
end

fprintf('\nDone: %d records in %s\n', numel(records), cfg.out_dir);

function report_stick(out, cfg)
% Stick fraction per PHYSICAL rail: the share of samples whose rail speed is
% below V_STICK.
%
% FRAME: out.q_with is ALREADY in STAGE coordinates [X1, X2, Y]. That is not
% obvious from gtd_run_simulation, which never says so; it is settled by
% gtd_save_record.m:19, which maps this exact array stage -> logical with
% ((P')\q')'. So the rail velocities are a plain time derivative of q_with with
% NO projection. Applying P.' here would double-transform and the numbers would
% not be any physical velocity.
%
% V_STICK is a REPORTING threshold only. It does not enter the model (the model
% is hard sign, no smoothing and no stick state), so no simulated quantity
% depends on its value.
    V_STICK = 1e-3;   % [m/s] HEURISTIC: reporting threshold for "not sliding"
    v_stage = zeros(size(out.q_with));
    for j = 1:3
        v_stage(:,j) = gradient(out.q_with(:,j), cfg.ts);   % [dX1 dX2 dY], STAGE
    end
    frac = mean(abs(v_stage) < V_STICK, 1);
    fprintf('  stick fraction (|v| < %.0e m/s) : X1 %.1f%%  X2 %.1f%%  Y %.1f%%\n', ...
            V_STICK, 100*frac(1), 100*frac(2), 100*frac(3));
end
