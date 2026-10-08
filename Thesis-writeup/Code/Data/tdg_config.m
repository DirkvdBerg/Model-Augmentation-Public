function cfg = tdg_config()
% TDG_CONFIG  Every constant of the thesis data generator (T_AF truth), one struct.
%   Copy of gtd_config.m with the production overrides of
%   generate_trajectory_data_coulomb.m folded in (both in scripts/gantry/thesis-data-verification/vendor/original/), and the design
%   values of Thesis-writeup/Documentation/DATA-DESIGN.md (SPEC.md lists each
%   number with its source). Pure configuration: no simulation, no file writes.
%
%   THESIS-DATA changes against gtd_config (see IMPLEMENTATION.md):
%     - self-contained paths: only this folder and vendor/ are put on the path
%       (gtd_config added kamtin-fp-model and Matlab-scripts/Augmentation)
%     - one truth, T_AF: MSD on, ma_frac 0.50, zeta_a 0.03, Garcia cc (no TRACK switch)
%     - B_tr 106 to 297 Hz on a 0.5 Hz line grid (D-219), A_prod (6x gtd_config)
%     - controller design point Y = 0 (K1, DATA-DESIGN 5.11), K2-g+ factor 1.2
%     - output folders under Thesis-writeup/Data, one per version

    here = fileparts(mfilename('fullpath'));                     % .../Code/Data
    cfg.here = here;
    cfg.root = fileparts(fileparts(fileparts(here)));            % repo root
    addpath(here);
    addpath(fullfile(here, 'vendor'));                           % verbatim copies, never edited

    % ── Physical parameters (gtd_config, Garcia 2013 identified set) ─────────
    cfg.mb=22.8; cfg.mh=10.1; cfg.m1=10.2; cfg.m2=10.7; cfg.Jb=1.0; cfg.Jh=0.05;
    cfg.cg1=14.5; cfg.cg2=20.3; cfg.cy=10; cfg.cb1=9; cfg.cb2=9;
    cfg.kb1=1987.5; cfg.kb2=1987.5; cfg.Lb=0.725; cfg.Lh=0.25; cfg.d=0.1;
    % THEORY: garcia2013, identified Coulomb friction of X1, X2, Y (constant-velocity
    % identification). NOT a knob (generate_trajectory_data_coulomb.m).
    cfg.cc1=16.80; cfg.cc2=18.35; cfg.ccy=11.60;
    % Friction law (D-224): F_i = cc_i tanh(g v_i) per stage rail, gantrySystemExtendedTanh.m.
    % THEORY: tanh-smoothed Coulomb (ASMPT supervisor 2026-09-28; the D-116 form, v0 = 1/g); for
    % |v| >> 1/g it is Garcia's cc*sign(v). g from the D-223 sweep: the smallest gain whose
    % standstill settles below the measured machine level with no limit cycle and whose sliding
    % response has converged (D-223). Replaces the Karnopp stick of D-209, which drove a
    % micrometre stick-slip limit cycle the machine lacks. D-226: g = 1000 (v0 = 1 mm/s), not
    % D-224's 100, so the law stays Coulomb-like under the production multisine (X rails at
    % 7 to 8 mm/s rms); it still settles at standstill (D-223: 0.06 / 0.1 / 1.7 nm).
    cfg.friction_gain = 1000;                 % [s/m]

    % ── Payload absorber (T_AF, DATA-DESIGN section 0; D-188 knobs) ─────────
    cfg.use_msd  = true;
    cfg.ma_frac  = 0.50;                                  % D-204/D-188 production value
    cfg.ma       = cfg.ma_frac * cfg.mh;
    cfg.mh_rigid = cfg.mh - cfg.ma;
    cfg.L0       = 0.10;                                  % gtd_config: offset in +Y [m]
    cfg.fa       = 150;                                   % THEORY (gtd_config): MSD natural frequency [Hz] = anti-resonance
    cfg.ka       = cfg.ma * (2*pi*cfg.fa)^2;              % THEORY: k = m (2 pi f)^2
    cfg.zeta_a   = 0.03;                                  % ZETA_A_OVERRIDE, JPE floor for jointed metal (gtd_config comment)
    cfg.ca       = 2*cfg.zeta_a*sqrt(cfg.ka*cfg.ma);      % THEORY: c = 2 zeta sqrt(k m)
    % free-free coupled pole, as asserted in generate_trajectory_data_coulomb.m
    cfg.f_pole   = cfg.fa * sqrt(1 + cfg.ma/cfg.mh_rigid);

    % ── Model ────────────────────────────────────────────────────────────────
    cfg.mdl_orig = 'gantry_additional_state_coulomb_2025a';   % vendor/, a verbatim copy, never edited
    cfg.mdl      = 'gantry_tdg_2025a';                        % built by tdg_make_model.m
    cfg.mdl_dir  = fullfile(here, 'model');
    cfg.cache_dir = fullfile(tempdir, 'tdg_slprj');           % short path: Windows 260-char build limit
    % D-231: 'simulink' runs the model; 'replica' steps the same loop in plain MATLAB
    % (tdg_replica.m, bitwise equal in every check, about 1 GB of RAM instead of 2.85 GB)
    cfg.simulator = 'simulink';

    % ── Coordinates, rates, controller (gtd_config) ─────────────────────────
    cfg.n   = 3;
    cfg.P   = [1, 1, 0; cfg.Lb/2, -cfg.Lb/2, 0; 0, 0, 1];
    cfg.fs  = 20e3;  cfg.ts = 1/cfg.fs;  cfg.fbw = 100;
    cfg.C_damp = [cfg.cg1+cfg.cg2,             (cfg.cg1-cfg.cg2)*cfg.Lb/2,                       0;
                  (cfg.cg1-cfg.cg2)*cfg.Lb/2,  cfg.cb1+cfg.cb2+(cfg.cg1+cfg.cg2)*cfg.Lb^2/4,    0;
                  0,                            0,                                                cfg.cy];
    cfg.K = [0,0,0; 0, cfg.kb1+cfg.kb2, 0; 0,0,0];
    cfg.K1_design_Y = 0;          % DATA-DESIGN 5.11: K1 designed once at Y = 0 (user 2026-09-26)
    cfg.K2_gain     = 1.2;        % DATA-DESIGN 5.11 / 7.4: K2-g+ = K1 gains +20 %

    % ── Record timing (gtd_config) ──────────────────────────────────────────
    cfg.t_hold       = 0.5;
    cfg.t_active     = 10;
    cfg.t_record     = 12;
    cfg.n_hold       = round(cfg.t_hold   / cfg.ts);
    cfg.n_hold_short = round(0.1          / cfg.ts);
    cfg.n_active     = round(cfg.t_active / cfg.ts);
    cfg.N_record     = round(cfg.t_record / cfg.ts);

    % ── Limits (gtd_config lim struct; datasheet) ───────────────────────────
    lim.pos_X      = 0.375;
    lim.pos_Y      = 0.400;
    lim.diff       = 6e-3;
    lim.vel        = 2.0;
    lim.acc_X      = 30.0;       % datasheet maximum = DATA-DESIGN "max" (user 2026-09-26)
    lim.acc_Y      = 50.0;
    lim.force_peak = [2000, 2000, 1420];
    lim.force_rms  = [916,  916,  656];
    cfg.lim = lim;
    cfg.acc_margin = 0.01;       % DATA-DESIGN section 10 item 11: 1 % margin at 1.0 max (FIR discretisation)

    % ── Multisine (DATA-DESIGN section 0: B_tr and A_prod) ──────────────────
    AMP_SCALE = 6;                                % A_prod = 6x the gtd_config levels (a6 datasets, AUDIT M1)
    cfg.A_sym  = AMP_SCALE * 40;                  % [N]   240
    cfg.A_Y    = AMP_SCALE * 30;                  % [N]   180
    cfg.A_anti = 0.5 * cfg.A_sym * cfg.Lb;        % [N m] 87, derived as in gtd_config
    cfg.yaw_budget      = 2e-3;
    cfg.n_ms_candidates = 30;
    cfg.f_low  = 106;  cfg.f_high = 297;          % B_tr, D-219
    cfg.f_step = 0.5;                             % line grid [Hz] = every 6th DFT line of 12 s (D-219)
    % the band must hold the absorber anti-resonance and pole (assert of the copied driver; D-219)
    assert(cfg.f_low < cfg.fa && cfg.fa < cfg.f_pole && cfg.f_pole < cfg.f_high, ...
           'B_tr [%g %g] Hz does not hold the anti-resonance %.2f and the pole %.2f Hz', ...
           cfg.f_low, cfg.f_high, cfg.fa, cfg.f_pole);

    % ── Noise (NOISE-INJECTION.md, D-225: at the encoder) ───────────────────
    % The measured Telica standstill error spectrum is injected as encoder noise, full band,
    % uncorrected (tdg_noise_psd.m); D-218's input force and its 300 Hz calibration band,
    % roll-off and tolerance are superseded.
    cfg.noise_target = fullfile(here, 'noise', 'noise_target.mat');   % exported from g1.npz
    cfg.noise_node   = 'encoder';                  % D-225
    % D-225 amendment: one independent generator per axis (no cross-spectra), each axis's
    % measured auto-spectrum divided by its own |S_jj|^2 above the corner, uncorrected below.
    % HEURISTIC: 50 Hz corner and +-10 % shape tolerance, agreed with the user 2026-09-28,
    % to confirm with the supervisor (below the corner S -> 0 and the correction would need a
    % large slow encoder noise that the stage physically follows).
    cfg.noise_corner     = 50;                     % [Hz]
    cfg.noise_correction = 'diag';                 % per axis |S_jj|^2 (the agreed form)
    cfg.noise_shape_tol  = 0.10;                   % flatness of the band ratios (check)
    % D-238 (supersedes the shape mapping above for new data): low-order model per stage axis,
    % phi_j(f) = g_j^2 |H_butter2(f; fc_j)|^2 + floor_j^2 (tdg_noise_psd.m). 'measured' = the
    % D-225 amendment method above, kept to reproduce and check the 2026-09-28 noisy set.
    cfg.noise_model = 'lowpass';                   % 'lowpass' (D-238) | 'measured' (D-225 amendment)
    % columns X1, X2, Y; fitted in temp/noise-lowfit/noise_sim_figure.m (NOISE-INJECTION.md 10.3)
    % HEURISTIC: fc and g fitted by log least squares, equal weight per decade, Y without 0.7-2 kHz,
    % g then set to the measured rms over the fitted range; floor copied (median above 5 kHz)
    cfg.noise_fc    = [988, 973, 2596];            % [Hz] low-pass cutoff
    cfg.noise_g     = [2.54e-10, 2.73e-10, 5.64e-11];   % [m/sqrt(Hz)] low-pass level
    cfg.noise_floor = [4.83e-11, 4.91e-11, 4.57e-11];   % [m/sqrt(Hz)] white floor

    % ── Output (handoff reading: two version folders) ───────────────────────
    cfg.data_root = fullfile(cfg.root, 'Thesis-writeup', 'Data');
    % D-230: the version folders of the tanh / encoder-noise data; every script reads these names
    % from here (the Karnopp / force-noise set of 2026-09-27 stays in 'Coulomb-and-MSD[-noise-free]')
    % D-238: the low-order-noise records go to a new folder; the D-225 noisy set stays in
    % 'Coulomb-tanh-and-MSD'; the noise-free set does not depend on the noise and is shared
    cfg.name_noisy = 'Coulomb-tanh-and-MSD-lowpass-noise';
    cfg.name_free  = 'Coulomb-tanh-and-MSD-noise-free';
    cfg.out_noisy = fullfile(cfg.data_root, cfg.name_noisy);
    cfg.out_free  = fullfile(cfg.data_root, cfg.name_free);
    cfg.split_dir = struct('train','Training', 'val','Validation', 'test','Test');
    cfg.disk_reserve_GB = 2.0;                     % handoff section 13: stop a stage below 2 GB free
    % run logs (one line per generated record, failures) go to the verification folder, so this
    % folder holds only the generator (user 2026-09-27)
    cfg.log_dir = fullfile(cfg.root, 'scripts', 'gantry', 'thesis-data-verification', 'logs');
end
