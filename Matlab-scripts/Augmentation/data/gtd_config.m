function cfg = gtd_config(TRACK, USE_MSD, MA_FRAC)
% GTD_CONFIG  Fixed constants for gantry trajectory-data generation.
%   cfg = GTD_CONFIG(TRACK, USE_MSD, MA_FRAC) returns one struct holding every
%   quantity that is identical across all records: physical parameters, the
%   hidden-MSD block, the coordinate transform P, the sample rate, record
%   timing, the enforced hardware-limit struct, and the TRACK -> band map.
%   Pure configuration: no simulation, no file writes.
%
%   TRACK    'joint' | 'joint_lowf' | 'augmentation'
%                                       selects the multisine band and the
%                                       output folder (the modes are kept
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
        % track_id only seeds the per-record RNG (gtd_build_records: 100*track_id + k),
        % so a distinct id is what gives joint_lowf its own multisine realisations
        % rather than reusing joint's. Added with the band case below (D-149), which
        % this validator predates and would otherwise reject before it is reached.
        case 'joint_lowf',   cfg.track_id = 2;
        otherwise, error('gtd_config:track', ...
                         'TRACK must be ''joint'', ''joint_lowf'' or ''augmentation''.');
    end

    % ── Paths (robust to the caller's cwd; repo root is 3 levels up) ─────────
    here     = fileparts(mfilename('fullpath'));        % .../Augmentation/data
    root     = fileparts(fileparts(fileparts(here)));   % repo root
    cfg.root = root;
    addpath(genpath(fullfile(root, 'kamtin-fp-model', '03 Simulink gantry')));
    addpath(fullfile(root, 'Matlab-scripts', 'Augmentation'));
    addpath(fullfile(root, 'Matlab-scripts', 'Augmentation', 'diagnostics'));

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
            case 'joint_lowf'   % as 'joint' but extended down to the record fundamental
                % WHY: the baseline model has two REAL poles (first-order corners, not
                % resonances) at cX/MX and cY/MY, i.e. the coast-down time constants
                % tau_X = (m1+m2+mb+mh)/(cg1+cg2) = 1.55 s and tau_Y = mh/cy = 1.01 s.
                % Those corners sit near 0.10 and 0.16 Hz, BELOW the 'joint' band, so a fit
                % to [1,200] Hz cannot see them: deleting them changes the FRF over that band
                % by only ~0.6 % median, under the ~1.5 % accuracy achieved. They matter for
                % long-horizon settling and for recovering cg1+cg2 and cy as physical
                % parameters in joint estimation.
                % The band start is justified from the BASELINE model's predicted corners,
                % not from the truth poles, so the experiment design is not tuned to the
                % answer. 1/t_record is the FUNDAMENTAL and therefore the lowest bin that
                % exists: gtd_make_multisine places lines on exact FFT bins, so with
                % t_record = 12 s nothing below 0.0833 Hz is reachable. Asking for a lower
                % f_low snaps to the same bin set and gains nothing; the lever for more
                % margin is t_record, not f_low.
                % Caveat carried deliberately: this gives exactly ONE line below each
                % corner, and it is the fundamental, which has a single period in the record
                % and no period-to-period averaging. Marginal but far better than none.
                cfg.f_low = 1/cfg.t_record;   cfg.f_high = 200;
            case 'augmentation' % narrowband [130,180] Hz: HEURISTIC, targets fa=150 +/- margin
                cfg.f_low = 130;  cfg.f_high = 180;
            otherwise
                % Without this, an unrecognised TRACK leaves f_low/f_high UNDEFINED and the
                % failure surfaces much later as a confusing error inside the multisine
                % synthesis rather than here.
                error('gtd_config:unknownTrack', ...
                      ['Unknown TRACK ''%s''. Valid tracks with USE_MSD = true: ' ...
                       '''joint'', ''joint_lowf'', ''augmentation''.'], TRACK);
        end
    else
        cfg.f_low = 1;    cfg.f_high = 7;   % baseline: single ~5 Hz structural mode
    end

    % ── Output directory: separate top-level folder per mode (user requirement) ─
    % Data folder is per track (joint / augmentation). Baseline (no MSD) gets a
    % subfolder so it never collides with the augmented data.
    if USE_MSD
        cfg.out_dir = fullfile(root, 'data', 'gantry', 'matlab', 'trajectory', TRACK);
    else
        cfg.out_dir = fullfile(root, 'data', 'gantry', 'matlab', 'trajectory', TRACK, 'baseline');
    end
    cfg.fig_dir = fullfile(cfg.out_dir, 'figures');   % per-record PNGs (separate from .mat)
end
