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

TRACK   = 'augmentation';      % 'joint' (broadband [1,200]) | 'joint_lowf' (as joint, from the 1/t_record fundamental, 0.0833 Hz) | 'augmentation' (narrowband [130,180])
USE_MSD = true;           % true = augmented (baseline + hidden MSD), false = baseline
MA_FRAC = 0.50;           % hidden-MSD mass fraction (0.50 = 50/50 split payload / hidden MSD)
PLOT    = true;          % true = save a positions+forces PNG per record to figures/
SHOW    = false;          % true = also display each figure on screen (not only save)
SELECT  = '';             % all 22 (T1-14, V1-4, E1-4). The T3-only probe that established the
                          % band and amplitude is done; this set is generated whole because a
                          % ratio axis needs every record at the same excitation.
% SELECT  = 'T3';         % probe form: one record. RESUME=true skips the 18 already on
                          % disk, so this generates only the E* test set. Safe to run now that
                          % E1's sweep band follows cfg (gtd_build_records.m) instead of the
                          % literal [130 180], which would have missed the 212 Hz mode.
                          % id prefix filter; '' = all. e.g. 'T9' -> T9*, 'T1' -> T1,T10-T14,
                          % 'E' -> all test, {'T3','E1'} -> those prefixes. Errors if no match.
% Figures are shown on screen when SHOW is true OR a subset is selected; a full
% batch with SHOW=false saves PNGs without opening windows.

% Output folder override. '' = the gtd_config default. Set it whenever a config
% knob (here MA_FRAC) changes, so a new dataset lands in a NEW folder and can
% never overwrite an existing one. fig_dir must be overridden too, otherwise the
% PNGs of the old set are replaced while the .mat files are not.
% NAMING: encode BOTH knobs that move the excitation, ma_frac and the band. The band was left
% out of the earlier names and that is precisely how augmentation_ma50 / augmentation_ma50_a5 /
% joint_lowf_ma50_a5 came to carry a band that misses the absorber without it being visible.
OUT_DIR_NAME = 'augmentation_ma50_b140-230_a6_z03';
% Excitation amplitude override, applied AFTER gtd_config so the shipped defaults stay
% intact and the earlier datasets remain reproducible. [] = keep the gtd_config value.
% WHY 5x: the enforced limits are [2000 2000 1420] N peak / [916 916 656] N rms, and the
% standstill records (the ones that carry the absorber signature) peaked at only
% [177 157 97] N. That headroom is the one lever that raises the signature against a
% FIXED floor: the float32 storage quantum on Y at |Y| = 0.30 m is 2.98e-8 m against a
% 3.8e-6 m signature, i.e. 128 levels on the very channel the absorber acts along.
% gtd_enforce_limits arbitrates per record, so the aggressive records (T11, E3, already
% at ~1000 N) scale themselves back down via chk.scale. HEURISTIC, same basis as the
% GATE-2 defaults it replaces.
% AMENDMENT: the quantisation argument above was measured under the OLD band, which missed the
% absorber. With BAND_OVERRIDE = [165 260] the response at the mode is 1.57e-05 m on T3, about
% 500x the float32 quantum, so quantisation no longer binds and 5x is not needed for that
% reason. It is retained so this dataset differs from augmentation_ma50_a5 by exactly ONE knob,
% the band, which is what makes the two comparable. Measured cost: none. The limiter never
% fired and force peaks sat at 806 to 870 N against a 2000 N limit.
%
% AMENDMENT 2026-09-03, 10x: the augmentation on `b165-260_a5` trains to a MEASURED 0.12%
% contribution from its augmented states (condition-B ablation on job 81088, D-170), and every
% excitation-side explanation for that has been eliminated by measurement. The one physical
% quantity that separates this application from the demonstrated method is the effect-to-motion
% ratio: the gantry moves O(10 cm) and `delta_a` moves O(10 um), a ratio of 2.1e-4, against
% O(1e-1) in Jan's ECC MSD. This raises that ratio as far as the ACTUATOR allows, so the ratio
% can be swept rather than assumed.
% HEADROOM, COMPUTED 2026-09-03 for all 22 records at BAND_OVERRIDE = [140 230]. The limiter is
% a pure linear pre-check (lsim, no Simulink) whose two parts separate: q0/u0_fb depend only on
% the trajectory and q_ms/u_ms scale exactly with amplitude, so the largest multiplier that
% still passes was solved per record by bisection rather than guessed. Result:
%     E3_aprbs_above     6.04   <- BINDING
%     T11_aprbs_100      7.59      T10 7.83   T13 7.93   V2 7.93   T9 8.01   T8 8.13
%     T4 8.12   V1 8.19   T14 8.23   T7 8.29   T1 8.52   T3 8.56   T6 8.56   T2 8.69
%     T5 8.82   V4 8.93   E2 11.02   E1 20.36 (rms-bound)   E4 unbounded (no multisine)
% Every record binds on force_peak(3), the 1420 N Y-axis peak, NOT on the 2000 N X limits and
% not on position or velocity. So 6 is the largest value at which NO record is scaled down and
% the set stays homogeneous, which is the condition for using it as a clean ratio axis.
% 10 was the T3-only probe value and it tripped the limiter (scaled to 80%); do not restore it.
% NOTE the pre-check runs on the 6-state baseline (gtd_build_plant sees no ma/ka/ca), while the
% Simulink run uses the 8-state truth, so realised forces can differ slightly from the numbers
% the pre-check arbitrated on. That is pre-existing behaviour, not introduced here.
AMP_SCALE = 6;
% RESUME picks what happens when OUT_DIR_NAME already holds records. A full batch
% takes hours, so an interrupted run must be continuable without regenerating the
% records that are already on disk, and without overwriting them.
%   false : hard stop if ANY .mat is present (the guard against a wrong folder)
%   true  : skip the records that already exist, generate only the missing ones
RESUME = false;   % PROBE: fresh OUT_DIR_NAME, so the hard stop is the guard we want

cfg     = gtd_config(TRACK, USE_MSD, MA_FRAC);
% Excitation band override, applied AFTER gtd_config for the same reason as AMP_SCALE: the
% shipped bands stay intact so every earlier dataset remains reproducible. [] = keep gtd_config.
%
% WHY THIS IS NEEDED. gtd_config's 'augmentation' band [130,180] is labelled "targets fa=150
% +/- margin", i.e. it is built around the absorber's STANDALONE frequency. The payload is a
% FREE mass that recoils, so the mode that actually exists is the free-free two-mass root
%     fn   = fa*sqrt(1 + ma/mh_rigid)        158.11 Hz at ma_frac 0.10,  212.13 Hz at 0.50
%     zeta = zeta_a*sqrt(1 + ma/mh_rigid)      0.0528 at 0.10,             0.0711 at 0.50
% fa is held fixed by construction (ka = ma*(2*pi*fa)^2), so fa is the ANTI-resonance, not the
% resonance, and the band never followed ma_frac. Verified three ways, agreeing to 0.01 Hz:
% multisine_frequency_range_MSD.m eigendecomposition (157.89 / 211.60 Hz damped),
% scripts/gantry/FRF/frf_mass_fraction.py, and the measured FRF of T3 (peak at 210.9 Hz).
% Measured consequence on the ma50 datasets: input at the absorber is 45 to 57 dB down.
%
% HEURISTIC [165 260] at ma_frac = 0.50. Roughly +/-1.5 half-power bandwidths about the 211.60 Hz
% mode, which is the same RELATIVE coverage [130,180] gave its own mode at ma_frac = 0.10. The
% mode is not only higher at 0.50 but BROADER (BW_hp 30.08 Hz against 16.66), so scaling the old
% band by frequency alone would sit inside the skirts. Wider is not free: a flat multisine
% spreads a fixed force RMS over its lines, so 1141 lines against the original 601 costs 27 % of
% per-line amplitude. The anti-resonance at 150.8 Hz is deliberately EXCLUDED; it is 60 Hz below
% the pole now (it was 8 Hz at 0.10) and reaching it would cost another ~20 % dilution for a
% feature that contributes little to the residual this track exists to excite.
%
% SUPERSEDED 2026-09-03 by [140 230], which INCLUDES the anti-resonance. The exclusion argument
% above rests on "a feature that contributes little to the residual". That is half right and the
% half that is wrong matters. Derived and verified numerically to 1.1e-15 against the closed
% form (mhr + ma held fixed by gtd_config, so the s^2 cancels):
%     G_truth - G_base = ma^2 / ( M [ mhr*ma s^2 + M ca s + M ka ] )
% The augmentation TARGET is a pure two-pole resonance at the coupled mode with NO finite zero,
% so the block never has to realise the anti-resonance. But the target does not vanish there
% either: |Delta| at 150 Hz is 1.093e-07 against 3.941e-07 at the peak, only 3.6x down, because
% at the anti-resonance the truth barely moves while the baseline moves normally and the
% DISCREPANCY is the baseline's own motion. Two reasons to include it:
%   (1) the notch appears in the SUM, G_base + Delta, by cancellation. A trained model that
%       reproduces a notch at 150 Hz which the baseline does not have is direct evidence it
%       captured the absorber rather than fitting locally inside the excitation band. That is a
%       validation feature the [165 260] set cannot provide.
%   (2) MEASURED on b165-260_a5 T3: F_Y at 150.00 Hz is 4.4e-08 N, i.e. 159 dB below the in-band
%       median. The set contains no information there at all.
% Width is NOT the cost it was for [165 260]: 90 Hz against 95, so per-line amplitude is
% marginally HIGHER, not diluted. The cost is upper-skirt coverage, +18 Hz above the 211.60 Hz
% mode (0.6 BW_hp) against +48 Hz before.
BAND_OVERRIDE = [140, 230];
if ~isempty(BAND_OVERRIDE)
    assert(numel(BAND_OVERRIDE) == 2 && BAND_OVERRIDE(1) > 0 ...
           && BAND_OVERRIDE(1) < BAND_OVERRIDE(2) && BAND_OVERRIDE(2) < cfg.fs/2, ...
           'BAND_OVERRIDE must be [lo hi] with 0 < lo < hi < fs/2');
    if USE_MSD    % report the mode the band is supposed to be covering, so a stale band is visible
        fn = cfg.fa * sqrt(1 + cfg.ma/cfg.mh_rigid);
        assert(BAND_OVERRIDE(1) < fn && fn < BAND_OVERRIDE(2), ...
               'BAND_OVERRIDE [%g %g] does not contain the coupled absorber mode %.2f Hz', ...
               BAND_OVERRIDE(1), BAND_OVERRIDE(2), fn);
        fprintf(['Band override [%g %g] -> [%g %g] Hz   (coupled mode %.2f Hz, ' ...
                 'anti-resonance %.0f Hz, ma_frac %.2f)\n'], cfg.f_low, cfg.f_high, ...
                BAND_OVERRIDE(1), BAND_OVERRIDE(2), fn, cfg.fa, cfg.ma_frac);
    end
    cfg.f_low = BAND_OVERRIDE(1);  cfg.f_high = BAND_OVERRIDE(2);
end
% Absorber damping override, applied AFTER gtd_config for the same reason as AMP_SCALE and
% BAND_OVERRIDE: the shipped default stays intact so every earlier dataset remains reproducible.
% [] = keep the gtd_config value (0.05).
%
% WHY. The augmentation target peaks at |Delta| = f*sqrt(1-f) / (2*zeta_a*M*wa^2) with
% f = ma_frac (derived from G_truth - G_base, verified to 0.25% against the numerical FRF), so
% the target scales as 1/zeta_a. This is the only remaining lever on the effect-to-motion ratio
% that the ACTUATOR does not cap, and D-170 records why that ratio is the quantity of interest.
% 0.03 is the BOTTOM of the cited range, not a free choice: JPE Precision Point gives
% zeta = 0.03 - 0.07 for metal structures with joints (see gtd_config.m and docs/references.md),
% so 0.05 -> 0.03 is 1.67x and is all the source allows.
% NOTE it does NOT change the limit pre-check: gtd_build_plant builds the 6-state baseline from
% m1/m2/mb/mh/Jb/Jh and never sees ma, ka or ca, so the homogeneous AMP_SCALE ceiling computed
% at zeta_a = 0.05 still holds here.
% CAUTION a lower zeta_a also makes the mode SHARPER (BW_hp = 2*zeta_n*fn shrinks with it), so
% this raises the size of the target and the sharpness of its pole at the same time.
ZETA_A_OVERRIDE = 0.03;
if ~isempty(ZETA_A_OVERRIDE) && USE_MSD
    cfg.zeta_a = ZETA_A_OVERRIDE;
    % ca is DERIVED from zeta_a in gtd_config (c = 2*zeta*sqrt(k*m)); recompute it from the
    % formula rather than scaling the old value, the same way A_anti is handled below.
    cfg.ca = 2*cfg.zeta_a*sqrt(cfg.ka*cfg.ma);
    fprintf('Damping override: zeta_a=%.4f  ca=%.4f N*s/m  (target scales as 1/zeta_a)\n', ...
            cfg.zeta_a, cfg.ca);
end
if ~isempty(AMP_SCALE) && AMP_SCALE ~= 1
    cfg.A_sym = AMP_SCALE * cfg.A_sym;
    cfg.A_Y   = AMP_SCALE * cfg.A_Y;
    % A_anti is DERIVED from A_sym in gtd_config (A_anti = 0.5*A_sym*Lb, so the anti
    % channel contributes the same per-rail force RMS as the symmetric one). Scaling
    % A_sym alone would silently break that relation, so recompute it from the formula
    % rather than scaling the old value.
    cfg.A_anti = 0.5 * cfg.A_sym * cfg.Lb;
    fprintf('Amplitude override x%g: A_sym=%.1f N  A_Y=%.1f N  A_anti=%.1f N*m\n', ...
            AMP_SCALE, cfg.A_sym, cfg.A_Y, cfg.A_anti);
end
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
