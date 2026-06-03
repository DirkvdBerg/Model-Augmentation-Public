% generate_gantry_lti_comb.m
% --------------------------
% Option-1 excitation: X symmetric motion (0.15 m) + 30 mm Y step (amax=50 m/s^2).
% Runs both baseline (gantry_2025a) and augmented (gantry_additional_state_2025a)
% then produces diagnostic plots comparing the two.
%
% Plots:
%   Figure 1 — trajectory comparison (6 rows)
%   Figure 2 — force comparison     (6 rows)
%
% Toggle:
%   save_flag — set true to write .mat files to data/gantry/matlab/
%
% Run from project root:
%   run('Matlab-scripts/Augmentation/data/generate_gantry_lti_comb.m')

clearvars
close all
clc

MDL_BASE = 'gantry_2025a';
MDL_AUG  = 'gantry_additional_state_2025a';

if bdIsLoaded(MDL_BASE), close_system(MDL_BASE, 0); end
if bdIsLoaded(MDL_AUG),  close_system(MDL_AUG,  0); end

addpath(genpath(fullfile(pwd, 'kamtin-fp-model', '03 Simulink gantry')))
addpath(fullfile(pwd, 'Matlab-scripts', 'Augmentation'))

save_flag = true;

%% Physical parameters (identical to generate scripts)
mb=22.8; mh=10.1; m1=10.2; m2=10.7; Jb=1.0; Jh=0.05;
cg1=14.5; cg2=20.3; cy=10; cb1=9; cb2=9;
kb1=1987.5; kb2=1987.5; Lb=0.725; Lh=0.25; d=0.1;
cc1=16.8; cc2=18.35; ccy=11.6;   % Coulomb — disabled in model but expected in workspace

C_damp = [cg1+cg2,        (cg1-cg2)*Lb/2,            0;
          (cg1-cg2)*Lb/2, cb1+cb2+(cg1+cg2)*Lb^2/4,  0;
          0,               0,                          cy];
K      = [0,0,0; 0,kb1+kb2,0; 0,0,0];
n = 3;  P = [1,1,0; Lb/2,-Lb/2,0; 0,0,1];
fs = 20e3;  ts = 1/fs;  fbw = 100;

%% MSD parameters (from generate_gantry_lti_augmented.m)
ma_frac  = 0.10;
ma       = ma_frac * mh;
mh_rigid = mh - ma;
L0       = 0.10;
fa       = 150;
ka       = ma * (2*pi*fa)^2;
zeta_a   = 0.05;
ca       = 2 * zeta_a * sqrt(ka * ma);

%% Controller & LTI (frozen at Y_op, full mh — controller designed for nominal plant)
Y_op = 0.3;
M_op = [m1+m2+mb+mh,          (m1-m2)*Lb/2-mh*Y_op,                    0;
        (m1-m2)*Lb/2-mh*Y_op, Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y_op^2, -mh*d;
        0,                    -mh*d,                                     mh];
sys = P.' * getss(n, M_op, C_damp, K) * P;
Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
for j = 1:3, Cfb(j,j) = ruleOfThumb(fbw, sys(j,j), ts); end
G = c2d(sys, ts, 'zoh');   % read by Simulink LTI blocks

%% Reference trajectory — rich X + scattered Y moves
n_hold = round(0.2/ts);   % 0.2 s hold at each end

% ── X: common mode (X1=X2), 4 back-and-forth trips with varying speeds ────
% Columns: [dist, vmax, amax, jerkTime, direction]
% Absolute X positions: 0 -> +0.15 -> -0.15 -> +0.12 -> -0.12 -> 0
x_profile = [
    0.15,  0.8,  15.0, 0.030, +1;   % slow   0     -> +0.15
    0.30,  1.5,  20.0, 0.030, -1;   % fast  +0.15  -> -0.15
    0.27,  1.0,  18.0, 0.035, +1;   % med   -0.15  -> +0.12
    0.24,  1.8,  22.0, 0.025, -1;   % fast  +0.12  -> -0.12
    0.12,  0.7,  12.0, 0.040, +1;   % slow  -0.12  ->  0
];

x_col = [];  x_pos = 0;  seg_len = zeros(size(x_profile,1), 1);
for k = 1:size(x_profile,1)
    pv = thirdOrderSetpointETEL(x_profile(k,1), x_profile(k,2), x_profile(k,3), ...
                                 x_profile(k,3)/x_profile(k,4), Inf, ts);
    pv    = pv(:,1);
    x_col = [x_col; x_pos + x_profile(k,5) * pv]; %#ok<AGROW>
    x_pos = x_pos + x_profile(k,5) * x_profile(k,1);
    seg_len(k) = numel(pv);
end
seg_start = [1; cumsum(seg_len(1:end-1)) + 1];  % start sample of each seg in x_col

n_move = numel(x_col);
N      = n_hold + n_move + n_hold;
t_ref  = (0:N-1)' * ts;

x_full = [zeros(n_hold,1); x_col; zeros(n_hold,1)];

% ── Y: 5 small moves around Y_op, scattered across X segments ────────────
% Absolute Y: 0.300 -> 0.310 -> 0.295 -> 0.300 -> 0.290 -> 0.300
% Columns: [dist, sign, vmax, amax, jerkTime, seg_idx, frac_within_seg]
y_profile = [
    0.010, +1, 0.30, 30, 0.010, 1, 0.50;   % +10 mm, mid  seg 1 (slow X)
    0.015, -1, 0.50, 40, 0.008, 2, 0.40;   % -15 mm, mid  seg 2 (fast X)
    0.005, +1, 0.20, 25, 0.012, 3, 0.20;   % +5  mm, early seg 3
    0.010, -1, 0.40, 35, 0.010, 4, 0.50;   % -10 mm, mid  seg 4 (fast X)
    0.010, +1, 0.40, 35, 0.010, 5, 0.60;   % +10 mm, late seg 5 (slow X)
];

y_full = Y_op * ones(N, 1);
y_pos  = Y_op;
for k = 1:size(y_profile,1)
    dist = y_profile(k,1);  sgn  = y_profile(k,2);
    vm   = y_profile(k,3);  am   = y_profile(k,4);  jt = y_profile(k,5);
    si   = y_profile(k,6);  frac = y_profile(k,7);

    pv_y    = thirdOrderSetpointETEL(dist, vm, am, am/jt, Inf, ts);
    pv_y    = pv_y(:,1);
    i_start = n_hold + seg_start(si) + round(frac * seg_len(si));
    i_end   = min(i_start + numel(pv_y) - 1, N);
    y_full(i_start:i_end) = y_pos + sgn * pv_y(1:i_end-i_start+1);
    y_pos = y_pos + sgn * dist;

    % Hold at new Y position until next move starts
    if k < size(y_profile,1)
        si_next   = y_profile(k+1,6);
        fr_next   = y_profile(k+1,7);
        i_next    = n_hold + seg_start(si_next) + round(fr_next * seg_len(si_next));
        y_full(i_end+1 : min(i_next-1, N)) = y_pos;
    else
        y_full(i_end+1:N) = y_pos;
    end
end

r = [x_full, x_full, y_full];
f = zeros(N, 3);   % no external force injection

%% Baseline simulation
fprintf('Baseline simulation...\n')
t = t_ref;  Y = Y_op;
sim(MDL_BASE, t_ref(end));

yb  = interp1((0:size(q1,1)-1)'*ts,    q1,  t_ref, 'linear', 'extrap');
u_b = lsim(ss(Cfb), r - yb, t_ref);   % reconstruct plant input

%% Augmented simulation
fprintf('Augmented simulation...\n')
mh_original = mh;   % required by MATLAB Function block in MDL_AUG
mh = mh_rigid;      % swap: Simulink reads mh for rigid body mass
t = t_ref;  Y = Y_op;
sim(MDL_AUG, t_ref(end));
mh = mh_rigid + ma;   % restore

ya   = interp1((0:size(q_aug,1)-1)'*ts,   q_aug,   t_ref, 'linear', 'extrap');
da   = interp1((0:size(delta_a,1)-1)'*ts, delta_a, t_ref, 'linear', 'extrap');
u_a  = lsim(ss(Cfb), r - ya, t_ref);   % reconstruct plant input

%% Save (toggleable)
if save_flag
    out_dir = fullfile(pwd, 'data', 'gantry', 'matlab');
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    % --- Baseline ---
    q_log_b  = ((P') \ yb')';
    dq_log_b = zeros(size(q_log_b));
    for j = 1:3, dq_log_b(:,j) = gradient(q_log_b(:,j), ts); end

    u            = single(u_b);
    y            = single(yb);
    x_logical    = single([q_log_b, dq_log_b]);
    r_sim        = single(r);
    Y_trajectory = single(yb(:,3));
    t_sim        = t_ref;
    dt           = single(ts);
    split        = 'comb';
    save(fullfile(out_dir, 'gantry_comb_baseline.mat'), ...
         'u','y','x_logical','r_sim','Y_trajectory','t_sim','fs','dt','split');
    fprintf('Saved: gantry_comb_baseline.mat\n')

    % --- Augmented ---
    q_log_a  = ((P') \ ya')';
    dq_log_a = zeros(size(q_log_a));
    for j = 1:3, dq_log_a(:,j) = gradient(q_log_a(:,j), ts); end

    u            = single(u_a);
    y            = single(ya);
    x_logical    = single([q_log_a, dq_log_a]);
    delta_a      = single(da);
    r_sim        = single(r);
    Y_trajectory = single(ya(:,3));
    t_sim        = t_ref;
    save(fullfile(out_dir, 'gantry_comb_augmented.mat'), ...
         'u','y','x_logical','delta_a','r_sim','Y_trajectory','t_sim','fs','dt','split');
    fprintf('Saved: gantry_comb_augmented.mat\n')
end

%% Summary
dy = ya - yb;
du = u_a - u_b;
fprintf('\n=== Position difference (aug - base) ===\n')
for j = 1:3
    lbl = {'X1','X2','Y '};
    fprintf('  %s:  rms=%8.2e m   max=%8.2e m\n', lbl{j}, rms(dy(:,j)), max(abs(dy(:,j))))
end
fprintf('\n=== Force difference (aug - base) ===\n')
for j = 1:3
    lbl = {'F_X1','F_X2','F_Y '};
    fprintf('  %s:  rms=%8.2e N   max=%8.2e N\n', lbl{j}, rms(du(:,j)), max(abs(du(:,j))))
end
fprintf('\n=== Hidden MSD displacement (delta_a) ===\n')
fprintf('  rms=%8.2e m   max=%8.2e m\n', rms(da), max(abs(da)))

%% Figures
plot_trajectory_comparison(t_ref, r, yb, ya, u_b, u_a, da, dy, fs, fa, Y_op)
plot_force_comparison(t_ref, u_b, u_a, du)

% =============================================================================
% Local functions
% =============================================================================

function plot_trajectory_comparison(t, r, yb, ya, ub, ua, da, dy, fs, fa, Y_op)
    ch    = {'X1','X2','Y'};
    ni    = 'none';   % shorthand: all labels use Interpreter none
    Nfft  = numel(t);
    Nhalf = floor(Nfft/2);
    f_ax  = (0:Nhalf-1) * fs / Nfft;
    f_pos = f_ax(2:end);   % drop DC bin (f=0 invalid on log axis)

    figure('Name','Trajectory comparison','Position',[50 30 1100 1100]);
    tl = tiledlayout(6, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    % ── Row 1: trajectories ───────────────────────────────────────────────
    for j = 1:3
        nexttile; hold on
        plot(t, r(:,j),  'k--', 'LineWidth', 0.8)
        plot(t, yb(:,j), 'b',   'LineWidth', 0.9)
        plot(t, ya(:,j), 'r',   'LineWidth', 0.9)
        ylabel([ch{j} ' [m]'], 'Interpreter', ni); grid on
        title([ch{j} ' trajectory'], 'Interpreter', ni)
        if j == 1
            legend('Ref','Baseline','Augmented','Location','best','FontSize',7)
        end
    end

    % ── Row 2: difference y_aug - y_base ──────────────────────────────────
    for j = 1:3
        nexttile; hold on
        plot(t, dy(:,j), 'k', 'LineWidth', 0.9)
        ylabel([ch{j} ' diff [m]'], 'Interpreter', ni); grid on
        title(sprintf('%s diff | rms=%.2e m  max=%.2e m', ...
              ch{j}, rms(dy(:,j)), max(abs(dy(:,j)))), 'Interpreter', ni)
    end

    % ── Row 3: FFT of Y-channel difference (signal spectrum) ──────────────
    Y_fft_diff = 2*abs(fft(dy(:,3)))/Nfft;
    [~, idx_fa] = min(abs(f_pos - fa));
    nexttile([1 3]); hold on
    semilogx(f_pos, 20*log10(Y_fft_diff(2:Nhalf)), 'k', 'LineWidth', 0.8)
    xline(fa, 'r--', sprintf('%g Hz', fa), 'LineWidth', 1.2, ...
          'LabelVerticalAlignment','bottom')
    text(fa, 20*log10(Y_fft_diff(idx_fa+1))+3, ...
         sprintf('%.2e m', Y_fft_diff(idx_fa+1)), ...
         'Color','r','FontSize',8,'HorizontalAlignment','center')
    xlim([f_pos(1) max(fa*4, 200)]); grid on
    xlabel('Frequency [Hz]', 'Interpreter', ni)
    ylabel('|dY| [dB re m]',  'Interpreter', ni)
    title('FFT of Y-channel difference (aug - base)', 'Interpreter', ni)

    % ── Row 4: FFT of Y-channel — baseline vs augmented ───────────────────
    Y_fft_base = 2*abs(fft(yb(:,3)))/Nfft;
    Y_fft_aug  = 2*abs(fft(ya(:,3)))/Nfft;
    nexttile([1 3]); hold on
    semilogx(f_pos, 20*log10(Y_fft_base(2:Nhalf)), 'b', 'LineWidth', 0.9)
    semilogx(f_pos, 20*log10(Y_fft_aug(2:Nhalf)),  'r', 'LineWidth', 0.9)
    xline(fa, 'k--', sprintf('%g Hz', fa), 'LineWidth', 1.2, ...
          'LabelVerticalAlignment','bottom')
    xlim([f_pos(1) max(fa*4, 200)]); grid on
    xlabel('Frequency [Hz]', 'Interpreter', ni)
    ylabel('|Y| [dB re m]',  'Interpreter', ni)
    legend('Baseline','Augmented','Location','best','FontSize',7)
    title('FFT of Y channel -- baseline vs augmented', 'Interpreter', ni)

    % ── Row 5: delta_a ────────────────────────────────────────────────────
    nexttile([1 3]); hold on
    plot(t, da, 'b', 'LineWidth', 0.9)
    ylabel('delta_a [m]', 'Interpreter', ni)
    xlabel('Time [s]',    'Interpreter', ni); grid on
    title(sprintf('delta_a  |  max=%.2e m   rms=%.2e m', max(abs(da)), rms(da)), ...
          'Interpreter', ni)

    % ── Row 6: H1 FRF — Y output / Y force input ─────────────────────────
    % H1 = FFT(y_Y) / FFT(u_Y): nonparametric FRF estimate, Y(3,3) element.
    % Reliable where u_Y has significant energy (low-frequency step content).
    H1_base = fft(yb(:,3)) ./ fft(ub(:,3));
    H1_aug  = fft(ya(:,3)) ./ fft(ua(:,3));
    nexttile([1 3]); hold on
    semilogx(f_pos, 20*log10(abs(H1_base(2:Nhalf))), 'b', 'LineWidth', 0.9)
    semilogx(f_pos, 20*log10(abs(H1_aug(2:Nhalf))),  'r', 'LineWidth', 0.9)
    xline(fa, 'k--', sprintf('%g Hz', fa), 'LineWidth', 1.2, ...
          'LabelVerticalAlignment','bottom')
    xlim([1, 4*fa]); grid on
    xlabel('Frequency [Hz]',      'Interpreter', ni)
    ylabel('|Y/F_Y| [dB re m/N]', 'Interpreter', ni)
    legend('Baseline','Augmented','Location','best','FontSize',7)
    title('H1 FRF -- Y/F_Y  (interpret above ~20 Hz with caution: low input energy)', ...
          'Interpreter', ni)

    title(tl, sprintf('Rich trajectory  |  Y_op=%.2f m  |  fa=%g Hz', Y_op, fa), ...
          'Interpreter', ni)
end

function plot_force_comparison(t, ub, ua, du)
    ch  = {'F_X1','F_X2','F_Y'};
    ni  = 'none';

    figure('Name','Force comparison','Position',[100 30 900 950]);
    tl = tiledlayout(6, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    for j = 1:3
        % Baseline vs augmented overlaid
        nexttile; hold on
        plot(t, ub(:,j), 'b', 'LineWidth', 0.9)
        plot(t, ua(:,j), 'r', 'LineWidth', 0.9)
        ylabel([ch{j} ' [N]'], 'Interpreter', ni); grid on
        title(sprintf('%s -- baseline vs augmented', ch{j}), 'Interpreter', ni)
        if j == 1
            legend('Baseline','Augmented','Location','best','FontSize',7)
        end

        % Force difference
        nexttile; hold on
        plot(t, du(:,j), 'k', 'LineWidth', 0.9)
        ylabel(['d' ch{j} ' [N]'], 'Interpreter', ni); grid on
        title(sprintf('%s diff | rms=%.2e N  max=%.2e N', ...
              ch{j}, rms(du(:,j)), max(abs(du(:,j)))), 'Interpreter', ni)
        if j == 3, xlabel('Time [s]', 'Interpreter', ni); end
    end

    title(tl, 'Forces -- baseline vs augmented', 'Interpreter', ni)
end
