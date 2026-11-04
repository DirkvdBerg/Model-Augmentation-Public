function cfg = tdg_config()
% TDG_CONFIG  Every constant of the thesis data generator, one struct; puts the generator on the path.
%   The truth is the dual gantry with a hidden payload absorber and tanh Coulomb friction. The
%   unchanged functions of Thesis-writeup/Code/Data/vendor/ (Jasper's getss, ruleOfThumb and
%   thirdOrderSetpointETEL, and our earlier generator's gtd_enforce_limits, gtd_size_anti_amp
%   and gtd_logical_to_stage) are put on the path; kamtin-fp-model/ never is.

    here = fileparts(mfilename('fullpath'));                     % .../Clean/data_generation
    cfg.here = here;
    cfg.root = fileparts(fileparts(here));                       % repo root
    cfg.vendor_dir = fullfile(cfg.root, 'Thesis-writeup', 'Code', 'Data', 'vendor');
    addpath(here);
    addpath(cfg.vendor_dir);

    % ── Physical parameters (Garcia 2013 identified set) ────────────────────
    cfg.mb=22.8; cfg.mh=10.1; cfg.m1=10.2; cfg.m2=10.7; cfg.Jb=1.0; cfg.Jh=0.05;
    cfg.cg1=14.5; cfg.cg2=20.3; cfg.cy=10; cfg.cb1=9; cfg.cb2=9;
    cfg.kb1=1987.5; cfg.kb2=1987.5; cfg.Lb=0.725; cfg.Lh=0.25; cfg.d=0.1;
    % THEORY: garcia2013, identified Coulomb friction of X1, X2, Y (constant-velocity identification)
    cfg.cc1=16.80; cfg.cc2=18.35; cfg.ccy=11.60;
    % Friction per stage rail F_i = cc_i tanh(g v_i) (gantrySystemExtendedTanh.m); for |v| >> 1/g it
    % is Garcia's cc sign(v).
    % HEURISTIC: g = 1000 s/m (v0 = 1 mm/s), from a gain sweep: the standstill settles below the
    % measured machine level with no limit cycle, and the law stays Coulomb-like under the multisine
    % (X rails at 7 to 8 mm/s rms)
    cfg.friction_gain = 1000;                 % [s/m]

    % ── Payload absorber: a hidden mass-spring-damper on the head ───────────
    cfg.ma_frac  = 0.50;                                  % share of the head mass that is the absorber
    cfg.ma       = cfg.ma_frac * cfg.mh;
    cfg.mh_rigid = cfg.mh - cfg.ma;
    cfg.L0       = 0.10;                                  % offset of the absorber in +Y [m]
    cfg.fa       = 150;                                   % absorber natural frequency [Hz] = anti-resonance
    cfg.ka       = cfg.ma * (2*pi*cfg.fa)^2;              % THEORY: k = m (2 pi f)^2
    cfg.zeta_a   = 0.03;                                  % HEURISTIC: damping floor of jointed metal
    cfg.ca       = 2*cfg.zeta_a*sqrt(cfg.ka*cfg.ma);      % THEORY: c = 2 zeta sqrt(k m)
    cfg.f_pole   = cfg.fa * sqrt(1 + cfg.ma/cfg.mh_rigid);   % THEORY: free-free coupled pole [Hz]

    % ── Model ────────────────────────────────────────────────────────────────
    % the recorded closed loop with encoder noise, plant-force log and tanh friction; every record
    % stores the hash of this file
    cfg.mdl       = 'gantry_tdg_2025a';
    cfg.mdl_dir   = fullfile(here, 'model');
    cfg.cache_dir = fullfile(tempdir, 'tdg_slprj');       % short path: Windows 260-character build limit
    cfg.simulator = 'simulink';                           % or 'replica' (tdg_replica.m, same records bitwise)

    % ── Coordinates, rates, controller ──────────────────────────────────────
    cfg.n   = 3;
    cfg.P   = [1, 1, 0; cfg.Lb/2, -cfg.Lb/2, 0; 0, 0, 1];   % generalised = P * stage
    cfg.fs  = 20e3;  cfg.ts = 1/cfg.fs;  cfg.fbw = 100;
    cfg.C_damp = [cfg.cg1+cfg.cg2,             (cfg.cg1-cfg.cg2)*cfg.Lb/2,                       0;
                  (cfg.cg1-cfg.cg2)*cfg.Lb/2,  cfg.cb1+cfg.cb2+(cfg.cg1+cfg.cg2)*cfg.Lb^2/4,    0;
                  0,                            0,                                                cfg.cy];
    cfg.K = [0,0,0; 0, cfg.kb1+cfg.kb2, 0; 0,0,0];
    cfg.K1_design_Y = 0;          % K1 is designed once, at Y = 0, and used in every record
    cfg.K2_gain     = 1.2;        % K2-g+ = K1 gains +20 % (record E5)

    % ── Record timing ───────────────────────────────────────────────────────
    cfg.t_hold       = 0.5;
    cfg.t_active     = 10;
    cfg.t_record     = 12;
    cfg.n_hold       = round(cfg.t_hold   / cfg.ts);
    cfg.n_hold_short = round(0.1          / cfg.ts);
    cfg.n_active     = round(cfg.t_active / cfg.ts);
    cfg.N_record     = round(cfg.t_record / cfg.ts);

    % ── Limits (datasheet) ──────────────────────────────────────────────────
    lim.pos_X      = 0.375;
    lim.pos_Y      = 0.400;
    lim.diff       = 6e-3;
    lim.vel        = 2.0;
    lim.acc_X      = 30.0;
    lim.acc_Y      = 50.0;
    lim.force_peak = [2000, 2000, 1420];
    lim.force_rms  = [916,  916,  656];
    cfg.lim = lim;
    % HEURISTIC: 1 % acceleration margin for records at 1.0 x max (FIR discretisation of the setpoint)
    cfg.acc_margin = 0.01;

    % ── Multisine: band B_tr and level A_prod ───────────────────────────────
    AMP_SCALE = 6;
    cfg.A_sym  = AMP_SCALE * 40;                  % [N]   240
    cfg.A_Y    = AMP_SCALE * 30;                  % [N]   180
    cfg.A_anti = 0.5 * cfg.A_sym * cfg.Lb;        % [N m] 87
    cfg.yaw_budget      = 2e-3;
    cfg.n_ms_candidates = 30;
    cfg.f_low  = 106;  cfg.f_high = 297;          % [Hz]
    cfg.f_step = 0.5;                             % line grid [Hz] = every 6th DFT line of the 12 s record
    assert(cfg.f_low < cfg.fa && cfg.fa < cfg.f_pole && cfg.f_pole < cfg.f_high, ...
           'the band [%g %g] Hz must hold the absorber anti-resonance %.2f Hz and pole %.2f Hz', ...
           cfg.f_low, cfg.f_high, cfg.fa, cfg.f_pole);

    % ── Encoder noise: low-order model per stage axis X1, X2, Y (tdg_noise_psd.m) ──
    % HEURISTIC: fc and g fitted to the measured Telica standstill error spectrum by log least
    % squares, equal weight per decade, Y without 0.7 to 2 kHz, g then set to the measured rms over
    % the fitted range; floor = the measured median above 5 kHz
    cfg.noise_fc    = [988, 973, 2596];                 % [Hz] low-pass cutoff
    cfg.noise_g     = [2.54e-10, 2.73e-10, 5.64e-11];   % [m/sqrt(Hz)] low-pass level
    cfg.noise_floor = [4.83e-11, 4.91e-11, 4.57e-11];   % [m/sqrt(Hz)] white floor
    cfg.noise_target = fullfile(here, 'noise', 'noise_target.mat');   % the measured spectrum; its source hash is stored

    % ── Output: one folder per version ──────────────────────────────────────
    cfg.data_root = fullfile(cfg.root, 'Thesis-writeup', 'Data');
    cfg.name_noisy = 'Coulomb-tanh-and-MSD-lowpass-noise';
    cfg.name_free  = 'Coulomb-tanh-and-MSD-noise-free';
    cfg.out_noisy = fullfile(cfg.data_root, cfg.name_noisy);
    cfg.out_free  = fullfile(cfg.data_root, cfg.name_free);
    cfg.split_dir = struct('train','Training', 'val','Validation', 'test','Test');
    cfg.log_dir = fullfile(cfg.root, 'scripts', 'gantry', 'thesis-data-verification', 'logs');
end
