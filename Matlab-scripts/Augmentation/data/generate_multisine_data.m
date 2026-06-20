% generate_multisine_data.m
% Generates trajectory + multisine identification data for the gantry system.
%
% Toggle USE_MSD (below) to switch between:
%   true  — augmented system (baseline + hidden MSD on payload)
%   false — baseline system (rigid payload, no MSD)
%
% Same motion profiles as generate_trajectory_data_without_multisine.m
% (T1-T8, V1, E1) with additive multisine feedforward force injection:
%   u_total = Cfb*(r - q) + f_multisine
%
% Frequency band [1, 200] Hz covers the MSD resonance at ~150 Hz.
% See diagnostics/multisine_frequency_range.m for derivation.
%
% Per experiment:
%   1. Amplitude sweep via lsim (superposition on linear closed-loop)
%   2. Simulink simulation WITH multisine at selected amplitude
%   3. Simulink simulation WITHOUT multisine (informativeness baseline)
%      [MSD mode only]
%   4. Informativeness check: residual spectrum, delta_a comparison
%      [MSD mode only]
%
% Independent multisine realizations per split (train/val/test).
%
% Saved per file:
%   u_total      (T x 3)  actual plant input  [F_X1, F_X2, F_Y]        [N]
%   u_fb         (T x 3)  controller output   Cfb*(r - q)              [N]
%   f_sim        (T x 3)  multisine feedforward force                  [N]
%   y            (T x 3)  plant output [X1, X2, Y]                    [m]
%   x_logical    (T x 6)  [q_logical, qdot_logical]                   [m, m/s]
%   delta_a      (T x 1)  hidden MSD relative displacement [MSD only] [m]
%   r_sim        (T x 3)  reference [X1_ref, X2_ref, Y_ref]           [m]
%   Y_trajectory (T x 1)  Y(t) = y(:,3)                               [m]
%   t_sim        (T x 1)  time vector                                  [s]
%   fs           scalar   sample frequency = 20000                     [Hz]
%   dt           scalar   sample period   = 1/20000                    [s]
%   split        char     'train', 'val', or 'test'
%   amp_rms      scalar   RMS amplitude per channel                   [N]
%
% Run from repo root:
%   run('Matlab-scripts/Augmentation/data/generate_multisine_data.m')

clear; clc; close all;

%% ── MODE TOGGLE ─────────────────────────────────────────────────────────
USE_MSD = false;  % true = augmented (baseline + hidden MSD)
                   % false = baseline (rigid payload, no MSD)
MA_FRAC = 0.50;    % mass fraction of hidden MSD (only used when USE_MSD = true)
                   % 0.10 = default (saves to multisine/)
                   % 0.50 = heavy MSD  (saves to multisine/m50/)

addpath(genpath(fullfile(pwd, 'kamtin-fp-model', '03 Simulink gantry')))
addpath(fullfile(pwd, 'Matlab-scripts', 'Augmentation'))
addpath(fullfile(pwd, 'Matlab-scripts', 'Augmentation', 'diagnostics'))

%% 0. Parameters
% ── Physical parameters (identical to generate_trajectory_data_without_multisine.m)
mb=22.8; mh=10.1; m1=10.2; m2=10.7; Jb=1.0; Jh=0.05;
cg1=14.5; cg2=20.3; cy=10; cb1=9; cb2=9;
kb1=1987.5; kb2=1987.5; Lb=0.725; Lh=0.25; d=0.1;
cc1=16.8; cc2=18.35; ccy=11.6;  % Coulomb (disabled in model but expected in workspace)

% ── Hidden MSD parameters (only used when USE_MSD = true)
if USE_MSD
    ma_frac  = MA_FRAC;
    ma       = ma_frac * mh;           % ma_frac * 10.1 kg
    mh_rigid = mh - ma;               % 9.09 kg
    L0       = 0.10;                   % equilibrium offset [m]
    fa       = 150;                    % MSD natural frequency [Hz]
    ka       = ma * (2*pi*fa)^2;      % MSD spring stiffness [N/m]
    zeta_a   = 0.05;                   % damping ratio
    ca       = 2 * zeta_a * sqrt(ka * ma);  % MSD damper [Ns/m]
    mh_original = mh;
    mdl = 'gantry_additional_state_2025a';
    fprintf('MODE: augmented (baseline + MSD, fa=%d Hz, ma=%.2f kg)\n', fa, ma);
else
    mdl = 'gantry_2025a';
    fprintf('MODE: baseline (rigid payload, mh=%.1f kg)\n', mh);
end

% ── System matrices & controller
C_damp = [cg1+cg2,        (cg1-cg2)*Lb/2,            0;
          (cg1-cg2)*Lb/2, cb1+cb2+(cg1+cg2)*Lb^2/4,  0;
          0,               0,                          cy];
K  = [0,0,0; 0,kb1+kb2,0; 0,0,0];
n  = 3;  P = [1,1,0; Lb/2,-Lb/2,0; 0,0,1];
fs = 20e3;  ts = 1/fs;  fbw = 100;
N_period = round(fs);       % 20000 samples = 1 s period
n_hold   = round(0.5/ts);  % 0.5 s settle hold at start and end

% ── Hardware limits (TELICA spec)
lim.pos_X      = 0.375;              % [m]
lim.pos_Y      = 0.400;              % [m]
lim.diff       = 6e-3;              % [m] max |X1-X2| = 6 mm (from generate_data_correct_max_theta.m)
lim.vel        = 2.0;               % [m/s]
lim.acc_X      = 30.0;             % [m/s^2] (checked on r only)
lim.acc_Y      = 50.0;             % [m/s^2] (checked on r only)
lim.force_peak = [2000, 2000, 1420]; % [N] peak [FX1,FX2,FY]
lim.force_rms  = [916,  916,  656];  % [N] RMS

% ── Multisine frequency band
f_low  = 1;    % [Hz] fundamental of 1-second period
if USE_MSD
    f_high = 200;  % [Hz] covers MSD resonance at ~150 Hz + margin (from diagnostics/multisine_frequency_range_MSD.m)
else
    f_high = 7;    % [Hz] single oscillatory mode at ~5 Hz (from diagnostics/multisine_frequency_range_baseline.m)
end

% ── Amplitude sweep grid [N RMS]
amp_grid = [1, 2, 5, 10, 20, 50, 100, 200, 400];

%% 1. Cached multisines (independent realization per split)
if USE_MSD
    if ma_frac == 0.10
        out_dir = fullfile(pwd, 'data', 'gantry', 'matlab', 'multisine');
    else
        out_dir = fullfile(pwd, 'data', 'gantry', 'matlab', 'multisine', ...
                           sprintf('m%d', round(ma_frac*100)));
    end
else
    out_dir = fullfile(pwd, 'data', 'gantry', 'matlab', 'multisine', 'baseline');
end
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

n_cand = 200;
[f_unit_train, info_train] = generate_cached_multisine(fs, f_low, f_high, 1, 3, n_cand, ...
    fullfile(out_dir, 'multisine_train.mat'), 0);
[f_unit_val, info_val] = generate_cached_multisine(fs, f_low, f_high, 1, 3, n_cand, ...
    fullfile(out_dir, 'multisine_val.mat'), 1000);
[f_unit_test, info_test] = generate_cached_multisine(fs, f_low, f_high, 1, 3, n_cand, ...
    fullfile(out_dir, 'multisine_test.mat'), 2000);

fprintf('Multisine seeds: train=%d (CF=%.3f), val=%d (CF=%.3f), test=%d (CF=%.3f)\n', ...
    info_train.seed, info_train.crest_factor, ...
    info_val.seed,   info_val.crest_factor, ...
    info_test.seed,  info_test.crest_factor);

%% 2. Experiment definitions (same as generate_trajectory_data_without_multisine.m)
trajs(1).id='T1_Y_sweep_conservative'; trajs(1).split='train'; trajs(1).Y_initial=0.3; trajs(1).X_sym_amp=0;    trajs(1).X_anti_amp=0;     trajs(1).Y_disp=0.6; trajs(1).vmax_X=0;   trajs(1).amax_X=0;    trajs(1).vmax_Y=1.00; trajs(1).amax_Y=10.0; trajs(1).jerkTime=0.050;
trajs(2).id='T2_X_sym_Y030';          trajs(2).split='train'; trajs(2).Y_initial=0.3; trajs(2).X_sym_amp=0.15; trajs(2).X_anti_amp=0;     trajs(2).Y_disp=0;   trajs(2).vmax_X=1.5; trajs(2).amax_X=20.0; trajs(2).vmax_Y=1.00; trajs(2).amax_Y=20.0; trajs(2).jerkTime=0.030;
trajs(3).id='T3_X_sym_Y000';          trajs(3).split='train'; trajs(3).Y_initial=0.0; trajs(3).X_sym_amp=0.15; trajs(3).X_anti_amp=0;     trajs(3).Y_disp=0;   trajs(3).vmax_X=1.5; trajs(3).amax_X=20.0; trajs(3).vmax_Y=1.00; trajs(3).amax_Y=20.0; trajs(3).jerkTime=0.030;
trajs(4).id='T4_X_antisym_Y020';      trajs(4).split='train'; trajs(4).Y_initial=0.2; trajs(4).X_sym_amp=0;    trajs(4).X_anti_amp=0.0025; trajs(4).Y_disp=0;   trajs(4).vmax_X=0.5; trajs(4).amax_X=8.0;  trajs(4).vmax_Y=1.00; trajs(4).amax_Y=20.0; trajs(4).jerkTime=0.040;
trajs(5).id='T5_X_sym_Y_sweep';       trajs(5).split='train'; trajs(5).Y_initial=0.2; trajs(5).X_sym_amp=0.10; trajs(5).X_anti_amp=0;     trajs(5).Y_disp=0.4; trajs(5).vmax_X=1.0; trajs(5).amax_X=15.0; trajs(5).vmax_Y=1.00; trajs(5).amax_Y=20.0; trajs(5).jerkTime=0.035;
trajs(6).id='T6_Y_sweep_aggressive';  trajs(6).split='train'; trajs(6).Y_initial=0.3; trajs(6).X_sym_amp=0;    trajs(6).X_anti_amp=0;     trajs(6).Y_disp=0.6; trajs(6).vmax_X=0;   trajs(6).amax_X=0;    trajs(6).vmax_Y=1.80; trajs(6).amax_Y=42.0; trajs(6).jerkTime=0.025;
trajs(7).id='T7_X_antisym_Y_sweep';   trajs(7).split='train'; trajs(7).Y_initial=0.3; trajs(7).X_sym_amp=0;    trajs(7).X_anti_amp=0.0025; trajs(7).Y_disp=0.6; trajs(7).vmax_X=0.5; trajs(7).amax_X=8.0;  trajs(7).vmax_Y=1.50; trajs(7).amax_Y=20.0; trajs(7).jerkTime=0.040;
trajs(8).id='T8_X_sym_anti_Y_sweep';  trajs(8).split='train'; trajs(8).Y_initial=0.2; trajs(8).X_sym_amp=0.10; trajs(8).X_anti_amp=0.0018; trajs(8).Y_disp=0.4; trajs(8).vmax_X=1.0; trajs(8).amax_X=8.0;  trajs(8).vmax_Y=1.20; trajs(8).amax_Y=12.0; trajs(8).jerkTime=0.035;
trajs(9).id='V1_X_sym_Y_mid_sweep';   trajs(9).split='val';   trajs(9).Y_initial=0.25; trajs(9).X_sym_amp=0.075; trajs(9).X_anti_amp=0;     trajs(9).Y_disp=0.30; trajs(9).vmax_X=0.8; trajs(9).amax_X=12.0; trajs(9).vmax_Y=0.90; trajs(9).amax_Y=14.0; trajs(9).jerkTime=0.040;
trajs(10).id='E1_X_sym_anti_Y_low_offset_sweep'; trajs(10).split='test'; trajs(10).Y_initial=0.10; trajs(10).X_sym_amp=0.060; trajs(10).X_anti_amp=0.0013; trajs(10).Y_disp=0.25; trajs(10).vmax_X=0.7; trajs(10).amax_X=10.0; trajs(10).vmax_Y=0.80; trajs(10).amax_Y=10.0; trajs(10).jerkTime=0.045;

%% 3. Main loop
for i = 1:numel(trajs)
    sp = trajs(i);
    fprintf('\n=== %d/%d  %s  [%s] ===\n', i, numel(trajs), sp.id, sp.split);

    Y_op = sp.Y_initial;

    % ── 3a. Controller at frozen Y_initial (identical to without_multisine)
    M_op = [m1+m2+mb+mh,            (m1-m2)*Lb/2-mh*Y_op,                    0;
            (m1-m2)*Lb/2-mh*Y_op,   Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y_op^2, -mh*d;
            0,                       -mh*d,                                     mh];
    sys = P.' * getss(n, M_op, C_damp, K) * P;
    Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
    for j = 1:3, Cfb(j,j) = ruleOfThumb(fbw, sys(j,j), ts); end
    G   = c2d(sys, ts, 'zoh');

    % ── 3b. Reference trajectory
    [r_traj, t_traj] = make_ref(sp, n_hold, ts);
    [r_traj, t_traj] = pad_to_periods(r_traj, ts, N_period);
    validate_ref(r_traj, t_traj, sp.id, lim);
    N = size(r_traj, 1);

    % Select multisine realization for this split
    switch sp.split
        case 'train', f_unit = f_unit_train;
        case 'val',   f_unit = f_unit_val;
        case 'test',  f_unit = f_unit_test;
    end

    % Tile unit multisine to trajectory length
    n_tile  = ceil(N / N_period);
    f_tiled = repmat(f_unit, n_tile, 1);
    f_tiled = f_tiled(1:N, :);

    % ── 3c. Amplitude sweep via lsim (superposition on linear closed-loop)
    Cfb_ss = ss(Cfb);            % discrete controller
    sys_cl = feedback(G, Cfb_ss); % discrete closed-loop: f -> q (perturbation)

    % Trajectory-only response (compute once, reuse in superposition)
    % In perturbation coordinates: r_pert = r_traj - r_traj(1,:) for the
    % deviation from initial equilibrium. But the plant is linearized at
    % Y_op, and position states are deviations from equilibrium.
    % For the X channels, equilibrium is 0. For Y, the reference IS Y_op
    % at the start, which is the equilibrium. So r_pert = r_traj - [0,0,Y_op].
    r_eq   = [0, 0, Y_op];
    r_pert = r_traj - r_eq;
    T_cl   = feedback(G * Cfb_ss, eye(3));  % complementary sensitivity: r -> q
    q0     = lsim(T_cl, r_pert);            % trajectory-only position (perturbation)
    u0_fb  = lsim(Cfb_ss, r_pert - q0);    % trajectory-only controller force

    fprintf('  Trajectory-only force budget:\n');
    fprintf('    peak = [%.0f  %.0f  %.0f] N  (limit [%d %d %d])\n', ...
        max(abs(u0_fb)), lim.force_peak);
    fprintf('    RMS  = [%.0f  %.0f  %.0f] N  (limit [%d %d %d])\n', ...
        rms(u0_fb), lim.force_rms);

    fprintf('  Amplitude sweep (lsim, superposition):\n');
    fprintf('  %8s  %10s  %10s  %10s  %6s\n', ...
        'amp[N]', 'q_max[mm]', 'u_pk[N]', 'u_rms[N]', 'pass');
    fprintf('  %s\n', repmat('-', 1, 52));

    amp_max = 0;
    for ia = 1:length(amp_grid)
        amp = amp_grid(ia);

        % Multisine perturbation (superposition: linear system)
        q_ms  = lsim(sys_cl, amp * f_tiled);            % position from multisine
        u_ms  = lsim(Cfb_ss, -q_ms) + amp * f_tiled;   % force from multisine path

        % Total = trajectory + multisine (superposition)
        q_total = q0 + q_ms + r_eq;   % back to absolute coordinates
        u_total = u0_fb + u_ms;

        ok_resp  = validate_response(q_total, fs, lim);
        ok_force = validate_forces(u_total, lim);
        ok = ok_resp && ok_force;

        fprintf('  %8.0f  %10.2f  %10.0f  %10.0f  %6s\n', ...
            amp, max(abs(q_total), [], 'all')*1e3, ...
            max(abs(u_total), [], 'all'), max(rms(u_total)), ...
            string(ok));

        if ok
            amp_max = amp;
        else
            break
        end
    end

    if amp_max == 0
        warning('%s: no amplitude passed -- skipping.', sp.id);
        continue
    end

    % HEURISTIC: cap multisine RMS force at 40% of trajectory-only RMS force
    % to prevent multisine from dominating the actuator effort
    force_cap_frac = 0.40;
    traj_rms = rms(u0_fb);
    force_cap = force_cap_frac * min(traj_rms(traj_rms > 0));
    if force_cap > 0 && force_cap < amp_max
        fprintf('  Force cap: %.0f N RMS (%.0f%% of traj RMS %.0f N) < sweep max %.0f N\n', ...
            force_cap, force_cap_frac*100, min(traj_rms(traj_rms > 0)), amp_max);
        amp_max = force_cap;
    end

    fprintf('  Selected amplitude: %.0f N RMS\n', amp_max);

    % ── 3d. Simulink simulation WITH multisine
    f = amp_max * f_tiled;
    r = r_traj;  t = t_traj;  Y = Y_op;

    if USE_MSD
        mh = mh_rigid;
    end
    fprintf('  Simulating WITH multisine (%.2f s, %d samples)...\n', t(end), N);
    sim(mdl, t(end));
    if USE_MSD
        mh = mh_rigid + ma;
    end

    % Reconstruct force decomposition
    if USE_MSD
        [t_sim, r_sim, f_ms, q_with, da_with] = resample_sim(q_aug, delta_a, r_traj, f, t_traj);
    else
        [t_sim, r_sim, f_ms, q_with, ~] = resample_sim(q1, zeros(size(q1,1),1), r_traj, f, t_traj);
    end
    u_fb_with    = lsim(Cfb_ss, r_sim - q_with);
    u_total_with = u_fb_with + f_ms;

    if ~validate_response(q_with, fs, lim)
        warning('%s: WITH multisine response validation failed.', sp.id);
    end
    if ~validate_forces(u_total_with, lim)
        warning('%s: WITH multisine force validation failed.', sp.id);
    end

    if USE_MSD
        % ── 3e. Simulink simulation WITHOUT multisine (informativeness baseline)
        f = zeros(N, 3);

        mh = mh_rigid;
        fprintf('  Simulating WITHOUT multisine...\n');
        sim(mdl, t(end));
        mh = mh_rigid + ma;

        [~, ~, ~, q_without, da_without] = resample_sim(q_aug, delta_a, r_traj, f, t_traj);

        % ── 3f. Informativeness check
        residual = q_with - q_without;
        da_rms_with    = rms(double(da_with));
        da_rms_without = rms(double(da_without));

        [psd_res, f_psd] = pwelch(residual(:,3), hanning(N_period), N_period/2, N_period, fs);
        idx_150 = find(f_psd >= 140 & f_psd <= 160);
        psd_peak_150 = max(psd_res(idx_150));

        fprintf('  Informativeness:\n');
        fprintf('    delta_a RMS: with = %.4e m,  without = %.4e m,  ratio = %.1fx\n', ...
            da_rms_with, da_rms_without, da_rms_with / da_rms_without);
        fprintf('    residual RMS (Y)  = %.4e m\n', rms(residual(:,3)));
        fprintf('    PSD peak @150 Hz  = %.4e m^2/Hz\n', psd_peak_150);
        fprintf('  Force budget usage (with multisine):\n');
        fprintf('    peak: [%.0f  %.0f  %.0f] / [%d %d %d] N  (%.0f%% %.0f%% %.0f%%)\n', ...
            max(abs(u_total_with)), lim.force_peak, ...
            100*max(abs(u_total_with)) ./ lim.force_peak);
        fprintf('    RMS:  [%.0f  %.0f  %.0f] / [%d %d %d] N  (%.0f%% %.0f%% %.0f%%)\n', ...
            rms(u_total_with), lim.force_rms, ...
            100*rms(u_total_with) ./ lim.force_rms);
    end

    % ── 3g. Derive logical state and save
    q_logical = ((P') \ q_with')';
    qdot_logical = zeros(size(q_logical));
    for j = 1:3
        qdot_logical(:,j) = gradient(q_logical(:,j), ts);
    end

    u_total      = single(u_total_with);
    u_fb         = single(u_fb_with);
    f_sim        = single(f_ms);
    y            = single(q_with);
    x_logical    = single([q_logical, qdot_logical]);
    r_sim        = single(r_sim);
    Y_trajectory = single(q_with(:,3));
    amp_rms      = amp_max;
    dt           = single(ts);
    split        = sp.split;

    out_path = fullfile(out_dir, [sp.id, '.mat']);
    if USE_MSD
        delta_a = single(da_with);
        save(out_path, 'u_total','u_fb','f_sim','y','x_logical','delta_a', ...
             'r_sim','Y_trajectory','t_sim','fs','dt','split','amp_rms');
    else
        save(out_path, 'u_total','u_fb','f_sim','y','x_logical', ...
             'r_sim','Y_trajectory','t_sim','fs','dt','split','amp_rms');
    end
    fprintf('  Saved: %s  (%d samples, %.2f s)\n', out_path, size(q_with,1), t_sim(end));

    % ── 3h-pre. Trajectory-only data on simulation time grid
    % Use Simulink without-multisine when available (MSD mode),
    % fall back to lsim closed-loop approximation otherwise.
    if USE_MSD
        q_traj_only = q_without;
        u_traj_only = lsim(Cfb_ss, r_sim - q_without);
    elseif numel(t_sim) ~= size(q0, 1)
        q_traj_only = interp1(t_traj, q0 + r_eq, t_sim);
        u_traj_only = interp1(t_traj, u0_fb, t_sim);
    else
        q_traj_only = q0 + r_eq;
        u_traj_only = u0_fb;
    end

    % ── 3h-1. Presentation figure: motion profiles and multisine effect
    ch_labels = {'$X_1$', '$X_2$', '$Y$'};
    col_traj = [0.0 0.45 0.74];     % blue
    col_ms   = [0.85 0.33 0.10];    % orange

    figure('Name', [sp.id ' - Presentation'], ...
           'Position', [50 30 1800 900], 'Color', 'w');
    tiledlayout(3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    % Row 1: Position overlay (trajectory-only vs trajectory+multisine)
    ax_r1 = gobjects(1, 3);
    for ch = 1:3
        ax_r1(ch) = nexttile(ch);
        plot(t_sim, q_traj_only(:,ch), 'Color', col_traj, ...
             'LineWidth', 1.0, 'DisplayName', 'trajectory');
        hold on
        plot(t_sim, double(q_with(:,ch)), 'Color', col_ms, ...
             'LineWidth', 0.8, 'DisplayName', 'trajectory + multisine');
        title(ch_labels{ch}, 'Interpreter', 'latex', 'FontSize', 12);
        ylabel('Position [m]');
        grid on
        set(gca, 'XTickLabel', []);
        if ch == 1
            legend('Location', 'northeast', 'FontSize', 8);
        end
    end

    % Row 2: Position difference (multisine effect on position)
    ax_r2 = gobjects(1, 3);
    for ch = 1:3
        ax_r2(ch) = nexttile(3 + ch);
        plot(t_sim, double(q_with(:,ch)) - q_traj_only(:,ch), ...
             'Color', [0.4 0.4 0.4], 'LineWidth', 0.8);
        ylabel('\Delta position [m]');
        grid on
        set(gca, 'XTickLabel', []);
        if ch == 2
            title('Position difference (with ms - without ms)', 'FontSize', 10);
        end
    end

    % Row 3: Force overlay (trajectory-only vs trajectory+multisine)
    ax_r3 = gobjects(1, 3);
    for ch = 1:3
        ax_r3(ch) = nexttile(6 + ch);
        plot(t_sim, u_traj_only(:,ch), 'Color', col_traj, ...
             'LineWidth', 1.0, 'DisplayName', 'trajectory');
        hold on
        plot(t_sim, double(u_total_with(:,ch)), 'Color', col_ms, ...
             'LineWidth', 0.8, 'DisplayName', 'trajectory + multisine');
        ylabel('Force [N]');
        xlabel('Time [s]');
        grid on
        if ch == 1
            legend('Location', 'northeast', 'FontSize', 8);
        end
    end

    linkaxes([ax_r1, ax_r2, ax_r3], 'x');
    if USE_MSD
        sgtitle(sprintf('%s | %s | multisine %.0f N RMS | MSD f_a = %d Hz', ...
            sp.id, sp.split, amp_max, fa), 'FontSize', 12, 'FontWeight', 'bold');
    else
        sgtitle(sprintf('%s | %s | multisine %.0f N RMS | baseline', ...
            sp.id, sp.split, amp_max), 'FontSize', 12, 'FontWeight', 'bold');
    end

    % ── 3h-2. Diagnostic plots
    figure('Name', [sp.id ' - Diagnostics'], 'Position', [50 50 1400 800]);
    tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    ch_diag = {'X1', 'X2', 'Y'};

    % Stage positions (with multisine)
    nexttile; hold on
    for ch = 1:3, plot(t_sim, q_with(:,ch), 'DisplayName', ch_diag{ch}); end
    ylabel('Position [m]'); xlabel('Time [s]');
    title('Stage positions (with multisine)'); legend; grid on

    % Y-channel force decomposition
    nexttile; hold on
    plot(t_sim, double(u_total_with(:,3)), 'DisplayName', 'total');
    plot(t_sim, double(u_fb_with(:,3)),    'DisplayName', 'feedback');
    plot(t_sim, double(f_ms(:,3)),         'DisplayName', 'multisine');
    ylabel('Force [N]'); xlabel('Time [s]');
    title('Y force decomposition'); legend; grid on

    % Hidden MSD displacement (MSD only)
    nexttile; hold on
    if USE_MSD
        plot(t_sim, double(da_with),    'DisplayName', 'with multisine');
        plot(t_sim, double(da_without), 'DisplayName', 'without');
        ylabel('Displacement [m]'); xlabel('Time [s]');
        title('Hidden MSD displacement'); legend; grid on
    else
        title('No MSD'); axis off
    end

    % Force budget bar chart
    nexttile;
    pct_with = 100 * max(abs(u_total_with)) ./ lim.force_peak;
    if USE_MSD
        u_fb_without = lsim(Cfb_ss, r_sim - q_without);
        pct_without = 100 * max(abs(u_fb_without)) ./ lim.force_peak;
        bar(categorical(ch_diag), [pct_without; pct_with]');
        legend('without multisine', 'with multisine');
    else
        bar(categorical(ch_diag), pct_with');
    end
    ylabel('Peak force [% of limit]');
    title('Force budget usage'); grid on

    % PSD of Y output
    nexttile; hold on
    [psd_with, f_hz] = pwelch(q_with(:,3), hanning(N_period), N_period/2, N_period, fs);
    semilogy(f_hz, psd_with, 'DisplayName', 'with multisine');
    if USE_MSD
        psd_without = pwelch(q_without(:,3), hanning(N_period), N_period/2, N_period, fs);
        semilogy(f_hz, psd_without, 'DisplayName', 'without multisine');
        xline(150, 'r--', '150 Hz', 'LineWidth', 1);
    end
    xlim([0 500]); ylabel('PSD [m^2/Hz]'); xlabel('Frequency [Hz]');
    title('Y output PSD'); legend; grid on

    % PSD of residual (MSD only)
    nexttile;
    if USE_MSD
        semilogy(f_psd, psd_res); hold on
        xline(150, 'r--', '150 Hz', 'LineWidth', 1);
        xlim([0 500]); ylabel('PSD [m^2/Hz]'); xlabel('Frequency [Hz]');
        title('Y residual PSD (with - without)'); grid on
    else
        title('No residual'); axis off
    end

    if USE_MSD
        sgtitle(sprintf('%s | %s | %.0f N RMS | delta_a ratio = %.1fx', ...
            sp.id, sp.split, amp_max, da_rms_with / da_rms_without), ...
            'FontSize', 12, 'FontWeight', 'bold');
    else
        sgtitle(sprintf('%s | %s | %.0f N RMS | baseline', ...
            sp.id, sp.split, amp_max), 'FontSize', 12, 'FontWeight', 'bold');
    end
end

fprintf('\nDone.\n');

%% ========================================================================
% Local functions
% =========================================================================

function [t_sim, r_sim, f_sim, q_sim, da_sim] = resample_sim(q_aug, delta_a, r, f, t)
% Handle variable-step Simulink output via interpolation.
    Ns = size(q_aug, 1);
    if Ns ~= numel(t)
        t_sim = linspace(0, t(end), Ns)';
        r_sim = interp1(t, r, t_sim);
        f_sim = interp1(t, f, t_sim);
    else
        t_sim = t;
        r_sim = r;
        f_sim = f;
    end
    q_sim  = q_aug;
    da_sim = delta_a;
end

function [r, t] = make_ref(sp, n_hold, ts)
% Build [X1, X2, Y] reference (N x 3) from trajectory parameters.
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
% Pad final hold to integer number of 1 s periods.
    N     = size(r,1);
    N_tgt = max(2, ceil(N/N_period)) * N_period;
    r_pad = [r; repmat(r(end,:), N_tgt-N, 1)];
    t_pad = ts * (0:size(r_pad,1)-1)';
end

function validate_ref(r, t, id, lim)
% Assert reference within hardware limits. Acceleration checked on r only.
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

function ok = validate_response(q_aug, fs, lim)
% Position and velocity check on q_aug (stage coordinates).
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
% Peak and RMS of total actuator force vs TELICA limits.
    ok =   all(max(abs(u_total)) <= lim.force_peak) ...
        && all(rms(u_total)      <= lim.force_rms);
    if ~ok
        fprintf('  Force validation failed: peak=[%.0f %.0f %.0f] N  RMS=[%.0f %.0f %.0f] N\n', ...
                max(abs(u_total)), rms(u_total));
    end
end
