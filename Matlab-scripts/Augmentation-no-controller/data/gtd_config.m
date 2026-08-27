function cfg = gtd_config(TRACK, USE_MSD, MA_FRAC)
% GTD_CONFIG  Fixed constants for gantry trajectory-data generation.
%   cfg = GTD_CONFIG(TRACK, USE_MSD, MA_FRAC) returns one struct holding every
%   quantity that is identical across all records: physical parameters, the
%   hidden-MSD block, the coordinate transform P, the sample rate, record
%   timing, the enforced hardware-limit struct, and the TRACK -> band map.
%   Pure configuration: no simulation, no file writes.
%
%   TRACK    'joint' | 'augmentation'   selects the multisine band and the
%                                       output folder (the two modes are kept
%                                       in separate top-level folders).
%   USE_MSD  true  = augmented plant (baseline + hidden MSD on the payload)
%            false = baseline plant (rigid payload)
%   MA_FRAC  hidden-MSD mass fraction (only used when USE_MSD = true)
%
%   Spec: docs/trajectory-generation-spec-draft.md sections 1.1-1.4, 1.10, 2.

    if nargin < 1 || isempty(TRACK),   TRACK   = 'joint'; end
    if nargin < 2 || isempty(USE_MSD), USE_MSD = true;    end
    if nargin < 3 || isempty(MA_FRAC), MA_FRAC = 0.50;    end

    cfg.track   = TRACK;
    cfg.use_msd = USE_MSD;
    switch TRACK
        case 'joint',        cfg.track_id = 0;
        case 'augmentation', cfg.track_id = 1;
        otherwise, error('gtd_config:track', 'TRACK must be ''joint'' or ''augmentation''.');
    end

    % ── Paths (robust to the caller's cwd; repo root is 3 levels up) ─────────
    here     = fileparts(mfilename('fullpath'));        % .../Augmentation/data
    root     = fileparts(fileparts(fileparts(here)));   % repo root
    cfg.root = root;
    addpath(genpath(fullfile(root, 'kamtin-fp-model', '03 Simulink gantry')));
    % CHANGED (open-loop variant): point at THIS folder, not Matlab-scripts/Augmentation.
    % The verbatim copy inherited the parent's paths, which would silently resolve
    % gantrySystemExtended.m and the diagnostics to the frozen folder.
    olroot = fileparts(here);                            % .../Augmentation-no-controller
    addpath(olroot);
    addpath(fullfile(olroot, 'diagnostics'));

    % ── Physical parameters (identical to generate_oscillatory_multisine_data.m) ─
    cfg.mb=22.8; cfg.mh=10.1; cfg.m1=10.2; cfg.m2=10.7; cfg.Jb=1.0; cfg.Jh=0.05;
    cfg.cg1=14.5; cfg.cg2=20.3; cfg.cy=10; cfg.cb1=9; cfg.cb2=9;
    cfg.kb1=1987.5; cfg.kb2=1987.5; cfg.Lb=0.725; cfg.Lh=0.25; cfg.d=0.1;
    cfg.cc1=16.8; cfg.cc2=18.35; cfg.ccy=11.6;   % Coulomb (expected in workspace)

    % ── Hidden MSD block (payload dynamic, acts along +Y) ───────────────────
    cfg.ma_frac = MA_FRAC;
    if USE_MSD
        cfg.ma       = MA_FRAC * cfg.mh;
        cfg.mh_rigid = cfg.mh - cfg.ma;
        cfg.L0       = 0.10;                             % equilibrium offset in +Y [m]
        cfg.fa       = 150;                              % THEORY: MSD natural freq [Hz] (model param)
        cfg.ka       = cfg.ma * (2*pi*cfg.fa)^2;         % THEORY: k = m*(2*pi*f)^2
        cfg.zeta_a   = 0.05;
        cfg.ca       = 2*cfg.zeta_a*sqrt(cfg.ka*cfg.ma); % THEORY: c = 2*zeta*sqrt(k*m)
        cfg.mdl      = 'gantry_additional_state_2025a';
    else
        cfg.mdl      = 'gantry_2025a';
    end

    % ── Coordinate transform and rates ──────────────────────────────────────
    cfg.n   = 3;
    cfg.P   = [1, 1, 0; cfg.Lb/2, -cfg.Lb/2, 0; 0, 0, 1];
    cfg.fs  = 20e3;  cfg.ts = 1/cfg.fs;  cfg.fbw = 100;

    % ── Y-independent system matrices for the plant/controller build ────────
    cfg.C_damp = [cfg.cg1+cfg.cg2,             (cfg.cg1-cfg.cg2)*cfg.Lb/2,                       0;
                  (cfg.cg1-cfg.cg2)*cfg.Lb/2,  cfg.cb1+cfg.cb2+(cfg.cg1+cfg.cg2)*cfg.Lb^2/4,    0;
                  0,                            0,                                                cfg.cy];
    cfg.K = [0,0,0; 0, cfg.kb1+cfg.kb2, 0; 0,0,0];

    % ── Record timing: 0.5 s hold + 10 s active + 0.5 s hold, padded to 12 s ─
    cfg.t_hold       = 0.5;
    cfg.t_active     = 10;
    cfg.t_record     = 12;
    cfg.n_hold       = round(cfg.t_hold   / cfg.ts);
    cfg.n_hold_short = round(0.1          / cfg.ts);   % hold between p2p moves
    cfg.n_active     = round(cfg.t_active / cfg.ts);
    cfg.N_record     = round(cfg.t_record / cfg.ts);

    % ── Enforced hardware limits (TELICA spec; the lim struct is the anchor) ─
    lim.pos_X      = 0.375;
    lim.pos_Y      = 0.400;
    lim.diff       = 6e-3;                 % max |X1-X2| = 6 mm
    lim.vel        = 2.0;
    lim.acc_X      = 30.0;                 % checked on r only
    lim.acc_Y      = 50.0;                 % checked on r only
    lim.force_peak = [2000, 2000, 1420];
    lim.force_rms  = [916,  916,  656];
    cfg.lim = lim;

    % ── Multisine amplitude design (logical channels; f_anti is a torque) ───
    cfg.A_sym           = 40;     % [N]   HEURISTIC / GATE-2 default (symmetric force RMS)
    cfg.A_Y             = 30;     % [N]   HEURISTIC / GATE-2 default (Y force RMS)
    % Anti target as a TORQUE [N*m], set so the anti channel contributes the same
    % per-rail force RMS as the symmetric channel: A_anti/Lb = 0.5*A_sym.
    % This is a modest fixed level; the 2 mm yaw budget is a CEILING that can only
    % scale it down (see gtd_size_anti_amp), never a target to fill. HEURISTIC.
    cfg.A_anti          = 0.5 * cfg.A_sym * cfg.Lb;   % [N*m]
    cfg.yaw_budget      = 2e-3;   % [m]   multisine share of the 6 mm yaw ceiling (HEURISTIC, spec 1.8)
    cfg.n_ms_candidates = 30;     % crest-factor selection: keep the best of N random draws

    % ── TRACK -> multisine band map ─────────────────────────────────────────
    if USE_MSD
        switch TRACK
            case 'joint'        % broadband [1,200] Hz: theta_base + ANN everywhere
                cfg.f_low = 1;    cfg.f_high = 200;
            case 'augmentation' % narrowband [130,180] Hz: HEURISTIC, targets fa=150 +/- margin
                cfg.f_low = 130;  cfg.f_high = 180;
        end
    else
        cfg.f_low = 1;    cfg.f_high = 7;   % baseline: single ~5 Hz structural mode
    end

    % ── Open-loop generation (this folder only) ─────────────────────────────
    % No reference, no controller, no feedforward: u_total IS the applied stage
    % force. Construction follows generate_openloop_record.m (proven on this
    % plant) and scripts/ecc_2025/msd_ndof_data_generation_dynamic.py (Jan).
    cfg.openloop     = true;
    % Excitation amplitude is specified directly in STAGE coordinates here,
    % [F_X1, F_X2, F_Y] RMS, not as the logical [A_sym, A_anti, A_Y] triple the
    % closed-loop path uses: open loop there is no yaw-budget constraint to size
    % A_anti against, and stage RMS is the quantity the force limit applies to.
    cfg.ol_A_rms     = [40, 40, 30];   % [N] HEURISTIC: 2x the OL1 record, to be
                                       % settled by the probe (drift scales ~A^2)
    cfg.ol_n_periods = 2;              % periods KEPT in the saved record
    cfg.ol_n_discard = 1;              % periods simulated and then thrown away.
    % Jan's construction (msd_ndof_data_generation_dynamic.py:41-70) generates
    % periods+1 and drops the first. Measured on the OT3 probe, one period of 6 s
    % is the right amount: it holds 99.7 % of the Y rectification drift and 97.6 %
    % of the X drift, and the AC content of the kept periods matches the discarded
    % one to 0.1 %. That is the free-integrator settling time, tau_Y = mh/cy =
    % 1.01 s and tau_X = (m1+m2+mb+mh)/(cg1+cg2) = 1.55 s, so 6 s is 6 and 3.9 time
    % constants respectively. NOTE this is NOT a periodic steady state: K11 = K33 = 0
    % means position never has one. What settles is the rectified drift RATE.
    % Total simulated length is (ol_n_periods + ol_n_discard) * t_record/ol_n_periods,
    % so the SAVED record is still cfg.t_record long and comparable to the
    % closed-loop records.
    cfg.ol_ode45_tol = 0.1;            % [s] window for the RK4-vs-ode45 fidelity check.
                                       % Shorter than the 0.5 s of
                                       % generate_openloop_record.m because the
                                       % periodic construction puts 301 lines in
                                       % the band instead of 51, so each ode45
                                       % right-hand side costs 6x more.

    % ── Output directory: separate top-level folder per mode (user requirement) ─
    % Data folder is per track (joint / augmentation), under openloop/ so it can
    % never collide with the closed-loop trajectory/<track>/ records.
    if USE_MSD
        cfg.out_dir = fullfile(root, 'data', 'gantry', 'matlab', 'trajectory', 'openloop', TRACK);
    else
        cfg.out_dir = fullfile(root, 'data', 'gantry', 'matlab', 'trajectory', 'openloop', TRACK, 'baseline');
    end
    cfg.fig_dir = fullfile(cfg.out_dir, 'figures');   % per-record PNGs (separate from .mat)
end
