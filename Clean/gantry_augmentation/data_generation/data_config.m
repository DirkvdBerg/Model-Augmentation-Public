function cfg = data_config()
% DATA_CONFIG  Every constant of the data generator in one struct.
    cfg.here = fileparts(mfilename('fullpath'));

    % Plant: the dual gantry of Garcia-Herreros et al. (2013), Table I
    cfg.mb  = 22.8;     % mass of the moving cross-arm [kg]
    cfg.mh  = 10.1;     % mass of the payload on the Y axis [kg]
    cfg.m1  = 10.2;     % mass of actuator X1 [kg]
    cfg.m2  = 10.7;     % mass of actuator X2 [kg]
    cfg.Jb  = 1.0;      % rotary inertia of the cross-arm [kg m^2]
    cfg.Jh  = 0.05;     % rotary inertia of the payload [kg m^2]
    cfg.cg1 = 14.5;     % viscous friction of actuator X1 [N/(m/s)]
    cfg.cg2 = 20.3;     % viscous friction of actuator X2 [N/(m/s)]
    cfg.cy  = 10;       % viscous friction of the payload on Y [N/(m/s)]
    cfg.cb1 = 9;        % viscous friction of flexible joint 1 [N m/(rad/s)]
    cfg.cb2 = 9;        % viscous friction of flexible joint 2 [N m/(rad/s)]
    cfg.kb1 = 1987.5;   % stiffness of flexible joint 1 [N m/rad]
    cfg.kb2 = 1987.5;   % stiffness of flexible joint 2 [N m/rad]
    cfg.Lb  = 0.725;    % length of the cross-arm [m]
    cfg.d   = 0.1;      % distance between cross-arm and payload [m]
    cfg.cc1 = 16.80;    % Coulomb friction of actuator X1 [N]
    cfg.cc2 = 18.35;    % Coulomb friction of actuator X2 [N]
    cfg.ccy = 11.60;    % Coulomb friction of the payload on Y [N]
    % g = 1000 s/m makes the friction Coulomb-like above about 1 mm/s while the stage still settles at standstill
    cfg.friction_gain = 1000;       % gain g of the friction cc tanh(g v) [s/m]

    % Hidden absorber: a mass-spring-damper on the payload that the baseline model does not have
    cfg.ma_frac  = 0.50;            % share of the payload mass that is the absorber; a large share makes it clearly visible in the output
    cfg.ma       = cfg.ma_frac * cfg.mh;               % absorber mass [kg]
    cfg.mh_rigid = cfg.mh - cfg.ma;                    % rigid part of the payload [kg]; the total stays mh
    cfg.L0       = 0.10;            % chosen offset of the absorber along +Y [m]
    cfg.fa       = 150;             % chosen absorber natural frequency, the anti-resonance [Hz]
    cfg.ka       = cfg.ma * (2*pi*cfg.fa)^2;           % absorber stiffness k = m (2 pi f)^2 [N/m]
    cfg.zeta_a   = 0.03;            % chosen absorber damping ratio [-]
    cfg.ca       = 2*cfg.zeta_a*sqrt(cfg.ka*cfg.ma);   % absorber damping c = 2 zeta sqrt(k m) [N/(m/s)]
    cfg.f_pole   = cfg.fa * sqrt(1 + cfg.ma/cfg.mh_rigid);   % resonance of the payload and absorber together [Hz]

    % Coordinates, sampling and controller (Jasper's main.m)
    cfg.n   = 3;                                        % degrees of freedom X, Theta, Y of the gantry without absorber
    cfg.P   = [1, 1, 0; cfg.Lb/2, -cfg.Lb/2, 0; 0, 0, 1];   % stage positions [X1, X2, Y] = P' * [X, Theta, Y]; generalised forces = P * stage forces
    cfg.fs  = 20e3;                 % sample rate of the loop and the records [Hz]
    cfg.ts  = 1/cfg.fs;             % sample time [s]
    cfg.fbw = 100;                  % crossover of the ruleOfThumb controller [Hz]
    cfg.C_damp = [cfg.cg1+cfg.cg2,             (cfg.cg1-cfg.cg2)*cfg.Lb/2,                       0;
                  (cfg.cg1-cfg.cg2)*cfg.Lb/2,  cfg.cb1+cfg.cb2+(cfg.cg1+cfg.cg2)*cfg.Lb^2/4,    0;
                  0,                            0,                                                cfg.cy];   % viscous damping matrix of the gantry without absorber, rows [X, Theta, Y]
    cfg.K = [0,0,0; 0, cfg.kb1+cfg.kb2, 0; 0,0,0];     % stiffness matrix: only the flexible joints, on the yaw angle Theta; X and Y have no stiffness to ground
    cfg.K1_design_Y = 0;            % Y at which the controller K1 of all records is designed, the centre of the stroke [m]

    % Record timing
    cfg.t_hold   = 0.5;             % hold before and after the motion [s]
    cfg.t_active = 10;              % window that holds all motion [s]
    cfg.t_record = 12;              % record length [s]
    cfg.n_hold       = round(cfg.t_hold   / cfg.ts);
    cfg.n_hold_short = round(0.1          / cfg.ts);   % dwell between random moves, 0.1 s
    cfg.n_active     = round(cfg.t_active / cfg.ts);
    cfg.N_record     = round(cfg.t_record / cfg.ts);

    % Limits of the ASMPT machine this gantry stands for, from the ETEL Telica XYZ data sheet
    % (ASME-YGNN-08-0750-0800W3); the plant parameters stay Garcia's
    lim.pos_X      = 0.375;                 % half the X stroke of 750 mm [m]
    lim.pos_Y      = 0.400;                 % half the Y stroke of 800 mm [m]
    lim.diff       = 6e-3;                  % maximum yaw difference |X1 - X2|, given by Jasper [m]
    lim.vel        = 2.0;                   % maximum speed [m/s]
    lim.acc_X      = 30.0;                  % rated maximum acceleration X; sizes the references, the machine's own moves exceed it [m/s^2]
    lim.acc_Y      = 50.0;                  % rated maximum acceleration Y [m/s^2]
    lim.force_peak = [2000, 2000, 1420];    % peak force X1, X2, Y [N]
    lim.force_rms  = [916,  916,  656];     % continuous force X1, X2, Y, applied as the rms over the record [N]
    cfg.lim = lim;
    cfg.acc_margin = 0.01;          % a record at 100 % of the acceleration limit may exceed it by 1 %, the overshoot of the sampled S-curve

    % Multisine force on the generalised channels [X, Theta, Y]; when a record with these amplitudes would
    % exceed a machine limit, its multisine is scaled down until it fits (precheck_limits.m, generate_data.m)
    cfg.A_sym  = 240;                       % amplitude on X [N]
    cfg.A_Y    = 180;                       % amplitude on Y [N]
    cfg.A_anti = 0.5 * cfg.A_sym * cfg.Lb;  % torque on Theta; gives X1 and X2 the same force as the X channel [N m]
    cfg.n_ms_candidates = 30;               % phase draws per channel; the one with the lowest crest factor is kept
    % the band starts above the highest loop crossover, so feedback does not suppress it, and ends
    % where 90 % of the baseline's squared frequency-response error above the crossover is included
    cfg.f_low  = 106;               % lowest multisine line [Hz]
    cfg.f_high = 297;               % highest multisine line [Hz]
    cfg.f_step = 0.5;               % line spacing: every 6th DFT line of the 12 s record, so the multisine repeats every 2 s [Hz]
    assert(cfg.f_low < cfg.fa && cfg.fa < cfg.f_pole && cfg.f_pole < cfg.f_high, ...
           'the band [%g %g] Hz must hold the absorber anti-resonance %.2f Hz and resonance %.2f Hz', ...
           cfg.f_low, cfg.f_high, cfg.fa, cfg.f_pole);

    % Encoder noise per stage axis X1, X2, Y: a second-order low-pass plus a white floor, fitted to the
    % measured standstill position error of the machine
    cfg.noise_fc    = [988, 973, 2596];                 % low-pass cutoff [Hz]
    cfg.noise_g     = [2.54e-10, 2.73e-10, 5.64e-11];   % low-pass level [m/sqrt(Hz)]
    cfg.noise_floor = [4.83e-11, 4.91e-11, 4.57e-11];   % white floor [m/sqrt(Hz)]

    % Model and output
    cfg.mdl       = 'gantry_truth';                     % built by build_model.m
    cfg.cache_dir = fullfile(tempdir, 'gantry_truth_slprj');   % Simulink build files; a short path stays under the 260-character limit of Windows
    cfg.name_noisy = 'Coulomb-tanh-and-MSD-lowpass-noise';     % data set folder with encoder noise
    cfg.name_free  = 'Coulomb-tanh-and-MSD-noise-free';        % data set folder without noise
    cfg.split_dir  = struct('train', 'Training', 'val', 'Validation', 'test', 'Test');
end
