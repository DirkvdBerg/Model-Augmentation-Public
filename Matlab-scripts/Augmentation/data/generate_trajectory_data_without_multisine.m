% generate_trajectory_data_without_multisine.m
% Generates BPTT identification trajectories WITHOUT multisine injection
% for the AUGMENTED gantry system (baseline + hidden MSD on payload).
% Uses gantry_additional_state_2025a.slx as the Simulink model.
% Same motion profiles as the baseline data generation (T1-T8, V1, E1).
%
% Validation:
%   Position, velocity checked on simulated q_aug.
%   Acceleration checked on reference r (exact: piecewise polynomial).
%   Forces (peak + RMS) checked on u_total.
%
% Saved per file:
%   u            (T x 3)  plant input  [F_X1, F_X2, F_Y]              [N]
%   y            (T x 3)  plant output [X1, X2, Y]                    [m]
%   x_logical    (T x 6)  [q_logical, qdot_logical]                   [m, m/s]
%   delta_a      (T x 1)  hidden MSD relative displacement            [m]
%   r_sim        (T x 3)  reference [X1_ref, X2_ref, Y_ref]           [m]
%   Y_trajectory (T x 1)  Y(t) = y(:,3)                               [m]
%   t_sim        (T x 1)  time vector                                  [s]
%   fs           scalar   sample frequency = 20000                     [Hz]
%   dt           scalar   sample period   = 1/20000                    [s]
%   split        char     'train', 'val', or 'test'
%
% Run from repo root:
%   run('Matlab-scripts/Augmentation/data/generate_trajectory_data_without_multisine.m')

addpath(genpath(fullfile(pwd, 'kamtin-fp-model', '03 Simulink gantry')))
addpath(fullfile(pwd, 'Matlab-scripts', 'Augmentation'))

% ── Physical parameters ───────────────────────────────────────────────────
mb=22.8; mh=10.1; m1=10.2; m2=10.7; Jb=1.0; Jh=0.05;
cg1=14.5; cg2=20.3; cy=10; cb1=9; cb2=9;
kb1=1987.5; kb2=1987.5; Lb=0.725; Lh=0.25; d=0.1;
cc1=16.8; cc2=18.35; ccy=11.6;  % Coulomb -- disabled in model but expected in workspace

% ── Hidden MSD parameters (from main_augmentation.m) ─────────────────────
% mh_total = mh_rigid + ma is conserved. Baseline uses full mh as rigid mass.
% Augmented model uses mh_rigid for the Simscape body and extended ODE.
ma_frac  = 0.10;
ma       = ma_frac * mh;           % 1.01 kg  hidden MSD mass
mh_rigid = mh - ma;               % 9.09 kg  rigid part of payload
L0       = 0.10;                   % equilibrium offset of ma in +Y direction [m]
fa       = 400;                    % target MSD natural frequency [Hz]
ka       = ma * (2*pi*fa)^2;      % MSD spring stiffness [N/m]
zeta_a   = 0.05;                   % damping ratio
ca       = 2 * zeta_a * sqrt(ka * ma);  % MSD damper coefficient [Ns/m]
mh_original = mh;                      % Simulink model reads this for internal reference

C_damp = [cg1+cg2,        (cg1-cg2)*Lb/2,            0;
          (cg1-cg2)*Lb/2, cb1+cb2+(cg1+cg2)*Lb^2/4,  0;
          0,               0,                          cy];
K  = [0,0,0; 0,kb1+kb2,0; 0,0,0];
n  = 3;  P = [1,1,0; Lb/2,-Lb/2,0; 0,0,1];
fs = 20e3;  ts = 1/fs;  fbw = 100;  mdl = 'gantry_additional_state_2025a';
N_period = round(fs);       % 20000 samples = T_p = 1 s, f0 = 1 Hz
n_hold   = round(0.5/ts);  % 0.5 s settle hold at start and end of trajectory

% ── Hardware limits (TELICA spec) ─────────────────────────────────────────
lim.pos_X      = 0.375;              % [m]
lim.pos_Y      = 0.400;              % [m]
lim.diff       = sin(0.1) * Lb;     % [m] max |X1-X2| yaw limit (Garcia 2013)
lim.vel        = 2.0;               % [m/s]
lim.acc_X      = 30.0;             % [m/s^2] — checked on r only
lim.acc_Y      = 50.0;             % [m/s^2] — checked on r only
lim.force_peak = [2000, 2000, 1420]; % [N] peak [FX1,FX2,FY]
lim.force_rms  = [916,  916,  656];  % [N] RMS

% ── Trajectory definitions (T1-T8 train, V1 val, E1 test) ────────────────
% Same motion profiles as baseline data generation.
trajs(1).id='T1_Y_sweep_conservative'; trajs(1).split='train'; trajs(1).Y_initial=0.3; trajs(1).X_sym_amp=0;    trajs(1).X_anti_amp=0;     trajs(1).Y_disp=0.6; trajs(1).vmax_X=0;   trajs(1).amax_X=0;    trajs(1).vmax_Y=1.00; trajs(1).amax_Y=10.0; trajs(1).jerkTime=0.050;
trajs(2).id='T2_X_sym_Y030';          trajs(2).split='train'; trajs(2).Y_initial=0.3; trajs(2).X_sym_amp=0.15; trajs(2).X_anti_amp=0;     trajs(2).Y_disp=0;   trajs(2).vmax_X=1.5; trajs(2).amax_X=20.0; trajs(2).vmax_Y=1.00; trajs(2).amax_Y=20.0; trajs(2).jerkTime=0.030;
trajs(3).id='T3_X_sym_Y000';          trajs(3).split='train'; trajs(3).Y_initial=0.0; trajs(3).X_sym_amp=0.15; trajs(3).X_anti_amp=0;     trajs(3).Y_disp=0;   trajs(3).vmax_X=1.5; trajs(3).amax_X=20.0; trajs(3).vmax_Y=1.00; trajs(3).amax_Y=20.0; trajs(3).jerkTime=0.030;
trajs(4).id='T4_X_antisym_Y020';      trajs(4).split='train'; trajs(4).Y_initial=0.2; trajs(4).X_sym_amp=0;    trajs(4).X_anti_amp=0.030; trajs(4).Y_disp=0;   trajs(4).vmax_X=0.5; trajs(4).amax_X=8.0;  trajs(4).vmax_Y=1.00; trajs(4).amax_Y=20.0; trajs(4).jerkTime=0.040;
trajs(5).id='T5_X_sym_Y_sweep';       trajs(5).split='train'; trajs(5).Y_initial=0.2; trajs(5).X_sym_amp=0.10; trajs(5).X_anti_amp=0;     trajs(5).Y_disp=0.4; trajs(5).vmax_X=1.0; trajs(5).amax_X=15.0; trajs(5).vmax_Y=1.00; trajs(5).amax_Y=20.0; trajs(5).jerkTime=0.035;
trajs(6).id='T6_Y_sweep_aggressive';  trajs(6).split='train'; trajs(6).Y_initial=0.3; trajs(6).X_sym_amp=0;    trajs(6).X_anti_amp=0;     trajs(6).Y_disp=0.6; trajs(6).vmax_X=0;   trajs(6).amax_X=0;    trajs(6).vmax_Y=1.80; trajs(6).amax_Y=42.0; trajs(6).jerkTime=0.025;
trajs(7).id='T7_X_antisym_Y_sweep';   trajs(7).split='train'; trajs(7).Y_initial=0.3; trajs(7).X_sym_amp=0;    trajs(7).X_anti_amp=0.030; trajs(7).Y_disp=0.6; trajs(7).vmax_X=0.5; trajs(7).amax_X=8.0;  trajs(7).vmax_Y=1.50; trajs(7).amax_Y=20.0; trajs(7).jerkTime=0.040;
trajs(8).id='T8_X_sym_anti_Y_sweep';  trajs(8).split='train'; trajs(8).Y_initial=0.2; trajs(8).X_sym_amp=0.10; trajs(8).X_anti_amp=0.020; trajs(8).Y_disp=0.4; trajs(8).vmax_X=1.0; trajs(8).amax_X=8.0;  trajs(8).vmax_Y=1.20; trajs(8).amax_Y=12.0; trajs(8).jerkTime=0.035;
% V1: validation -- X symmetric + partial Y sweep. Y_initial=0.25 (interpolation holdout).
trajs(9).id='V1_X_sym_Y_mid_sweep';   trajs(9).split='val';   trajs(9).Y_initial=0.25; trajs(9).X_sym_amp=0.075; trajs(9).X_anti_amp=0;     trajs(9).Y_disp=0.30; trajs(9).vmax_X=0.8; trajs(9).amax_X=12.0; trajs(9).vmax_Y=0.90; trajs(9).amax_Y=14.0; trajs(9).jerkTime=0.040;
% E1: test -- X symmetric + X anti-symmetric + Y sweep. Y_initial=0.10 (extrapolation holdout).
trajs(10).id='E1_X_sym_anti_Y_low_offset_sweep'; trajs(10).split='test'; trajs(10).Y_initial=0.10; trajs(10).X_sym_amp=0.060; trajs(10).X_anti_amp=0.015; trajs(10).Y_disp=0.25; trajs(10).vmax_X=0.7; trajs(10).amax_X=10.0; trajs(10).vmax_Y=0.80; trajs(10).amax_Y=10.0; trajs(10).jerkTime=0.045;

% ── Output directory ──────────────────────────────────────────────────────
out_dir = fullfile(fileparts(mfilename('fullpath')),'..','Matlab-output','augmented-trajectories-no-multisine');
if ~exist(out_dir,'dir'), mkdir(out_dir); end

% ── Main loop ─────────────────────────────────────────────────────────────
for i = 1:numel(trajs)
    sp = trajs(i);
    fprintf('\n=== %d/%d  %s  [%s] ===\n', i, numel(trajs), sp.id, sp.split);

    % Controller frozen at Y_initial
    Y_op = sp.Y_initial;
    M_op = [m1+m2+mb+mh,            (m1-m2)*Lb/2-mh*Y_op,                    0;
            (m1-m2)*Lb/2-mh*Y_op,   Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y_op^2, -mh*d;
            0,                       -mh*d,                                     mh];
    sys = P.' * getss(n, M_op, C_damp, K) * P;
    Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
    for j = 1:3, Cfb(j,j) = ruleOfThumb(fbw, sys(j,j), ts); end
    G = c2d(sys, ts, 'zoh');  % discrete plant -- Simulink LTI blocks read G from workspace

    % Reference trajectory padded to integer periods
    [r_traj, t_traj] = make_ref(sp, n_hold, ts);
    [r_traj, t_traj] = pad_to_periods(r_traj, ts, N_period);
    validate_ref(r_traj, t_traj, sp.id, lim);
    N = size(r_traj, 1);

    % Swap mh to mh_rigid for augmented Simulink model.
    % The model reads mh from workspace for the rigid body mass.
    % ma, ka, ca, L0 are already in workspace from the parameter block above.
    mh = mh_rigid;

    % Simulation (no multisine injection)
    r = r_traj;  t = t_traj;  f = zeros(N, 3);  Y = Y_op;
    fprintf('  Simulating %.2f s (%d samples)...\n', t_traj(end), N);
    sim(mdl, t_traj(end));

    mh = mh_rigid + ma;  % restore full mh for next iteration

    % Reconstruct plant input: u_total = Cfb*(r - q_aug) since f=0
    [t_sim, r_sim, u_aug] = reconstruct(q_aug, r_traj, t_traj, Cfb);
    u_total = u_aug;

    % Validate -- skip trajectory if any limit violated
    if ~validate_response(q_aug, fs, lim) || ~validate_forces(u_total, lim)
        warning('%s: validation failed -- skipping.', sp.id);
        continue
    end

    % Derive nominal 6D logical state from augmented stage positions
    % q_aug is the extended ODE output in stage coordinates [X1, X2, Y] (T x 3).
    % q_logical = P^{-T} * q_aug_stage  (exact coordinate transform)
    % qdot_logical from central finite differences at 20 kHz
    q_logical = ((P') \ q_aug')';       % (T x 3)
    qdot_logical = zeros(size(q_logical));
    for j = 1:3
        qdot_logical(:,j) = gradient(q_logical(:,j), ts);
    end
    x_logical_out = [q_logical, qdot_logical];  % (T x 6)

    % Save
    u            = single(u_total);           % (T x 3) plant forces -- training
    y            = single(q_aug);             % (T x 3) stage positions -- training
    x_logical    = single(x_logical_out);     % (T x 6) 6D state projection -- encoder verify
    delta_a      = single(delta_a);           % (T x 1) hidden MSD displacement -- diagnostic
    r_sim        = single(r_sim);             % (T x 3) reference -- plotting
    Y_trajectory = single(q_aug(:,3));        % (T x 1) Y(t)
    dt           = single(ts);
    split        = sp.split;

    out_path = fullfile(out_dir, [sp.id, '.mat']);
    save(out_path, 'u','y','x_logical','delta_a','r_sim','Y_trajectory','t_sim','fs','dt','split');
    fprintf('  Saved: %s  (%d samples, %.2f s,  Y in [%.4f, %.4f] m,  delta_a max=%.4e m)\n', ...
            out_path, size(q_aug,1), t_sim(end), min(Y_trajectory), max(Y_trajectory), max(abs(double(delta_a))));
end
fprintf('\nDone.\n');

% ════════════════════════════════════════════════════════════════════════
% Local functions
% ════════════════════════════════════════════════════════════════════════

function [r, t] = make_ref(sp, n_hold, ts)
% Build [X1, X2, Y] reference (N×3) from trajectory parameters.
    r = repmat([0, 0, sp.Y_initial], n_hold, 1);
    pv_sym = []; pv_anti = []; n_sym = 0; n_anti = 0;
    if sp.X_sym_amp  > 0, pv_sym  = sp1d(sp.X_sym_amp,  sp.vmax_X, sp.amax_X, sp.jerkTime, ts); n_sym  = numel(pv_sym);  end
    if sp.X_anti_amp > 0, pv_anti = sp1d(sp.X_anti_amp, sp.vmax_X, sp.amax_X, sp.jerkTime, ts); n_anti = numel(pv_anti); end
    n_move_X = max(n_sym, n_anti);
    n_move_Y = 0; pv_Y = [];
    if sp.Y_disp > 0, pv_Y = sp1d(sp.Y_disp, sp.vmax_Y, sp.amax_Y, sp.jerkTime, ts); n_move_Y = numel(pv_Y); end
    n_main = max(n_move_X, n_move_Y);
    if n_main > 0
        X1 = zeros(n_main,1); X2 = zeros(n_main,1); Y = sp.Y_initial*ones(n_main,1);
        if n_move_X > 0
            pv_sym  = pvpad(pv_sym,  n_sym,  n_main);
            pv_anti = pvpad(pv_anti, n_anti, n_main);
            X1 = pv_sym + pv_anti;  X2 = pv_sym - pv_anti;
        end
        if n_move_Y > 0
            Y = sp.Y_initial - pvpad(pv_Y, n_move_Y, n_main);
        end
        r = [r; [X1, X2, Y]];
    end
    r = [r; repmat(r(end,:), n_hold, 1)];
    t = ts * (0:size(r,1)-1)';
end

function v = pvpad(v, n_src, n_tgt)
    if isempty(v), v = zeros(n_tgt,1); return; end
    v = [v; v(end)*ones(n_src-numel(v),1)]; v = [v; v(end)*ones(n_tgt-n_src,1)];
end

function pv = sp1d(dist, vmax, amax, jerkTime, ts)
    pv = thirdOrderSetpointETEL(dist, vmax, amax, amax/jerkTime, Inf, ts); pv = pv(:,1);
end

function [r_pad, t_pad] = pad_to_periods(r, ts, N_period)
% Pad final hold to integer number of 1 s periods — leakage-free (P&S Ch.2 §2.2.3)
    N     = size(r,1);
    N_tgt = max(2, ceil(N/N_period)) * N_period;
    r_pad = [r; repmat(r(end,:), N_tgt-N, 1)];
    t_pad = ts * (0:size(r_pad,1)-1)';
end

function validate_ref(r, t, id, lim)
% Assert reference within hardware limits. Acceleration checked here on r
% (piecewise polynomial — derivative exact via finite difference on r).
% This is the ONLY place acceleration is validated. Never check acceleration on q1.
    ts   = t(2)-t(1);
    vel  = diff(r)/ts;
    acc  = diff(vel)/ts;
    assert(max(abs(r(:,1)))          <= lim.pos_X,  '%s: X1 position limit', id);
    assert(max(abs(r(:,2)))          <= lim.pos_X,  '%s: X2 position limit', id);
    assert(max(r(:,3))               <=  lim.pos_Y, '%s: Y+ position limit', id);
    assert(min(r(:,3))               >= -lim.pos_Y, '%s: Y- position limit', id);
    assert(max(abs(r(:,1)-r(:,2)))   <= lim.diff,   '%s: yaw |X1-X2| limit', id);
    assert(max(abs(vel(:,1:2)),[],'all') <= lim.vel,   '%s: X velocity limit', id);
    assert(max(abs(vel(:,3)))        <= lim.vel,    '%s: Y velocity limit', id);
    assert(max(abs(acc(:,1:2)),[],'all') <= lim.acc_X, '%s: X acceleration limit', id);
    assert(max(abs(acc(:,3)))        <= lim.acc_Y,  '%s: Y acceleration limit', id);
    fprintf('  r OK: X1=[%+.0f %+.0f] X2=[%+.0f %+.0f] Y=[%+.0f %+.0f] mm\n', ...
            min(r(:,1))*1e3,max(r(:,1))*1e3,min(r(:,2))*1e3,max(r(:,2))*1e3, ...
            min(r(:,3))*1e3,max(r(:,3))*1e3);
end

function [t_sim, r_sim, u_aug] = reconstruct(q_aug, r, t, Cfb)
% u_aug = Cfb*(r-q_aug). Handles variable-step Simulink output via interpolation.
    Ns = size(q_aug,1);
    if Ns ~= numel(t), t_sim = linspace(0,t(end),Ns)'; r_sim = interp1(t,r,t_sim);
    else,              t_sim = t;                        r_sim = r; end
    u_aug = lsim(ss(Cfb), r_sim - q_aug, t_sim);
end

function ok = validate_response(q_aug, fs, lim)
% Position and velocity on q_aug. Acceleration NOT checked -- see header.
    vel = diff(q_aug)*fs;
    ok  =   max(abs(q_aug(:,1)))            <= lim.pos_X ...
         && max(abs(q_aug(:,2)))            <= lim.pos_X ...
         && max(abs(q_aug(:,3)))            <= lim.pos_Y ...
         && max(abs(q_aug(:,1)-q_aug(:,2))) <= lim.diff  ...
         && max(abs(vel(:,1)))              <= lim.vel   ...
         && max(abs(vel(:,2)))              <= lim.vel   ...
         && max(abs(vel(:,3)))              <= lim.vel;
    if ~ok, fprintf('  Response validation failed.\n'); end
end

function ok = validate_forces(u_total, lim)
% Peak and RMS of total actuator force u_total = u_q1 + f_sim vs TELICA limits.
    ok =   all(max(abs(u_total)) <= lim.force_peak) ...
        && all(rms(u_total)      <= lim.force_rms);
    if ~ok
        fprintf('  Force validation failed: peak=[%.0f %.0f %.0f] N  RMS=[%.0f %.0f %.0f] N\n', ...
                max(abs(u_total)), rms(u_total));
    end
end

