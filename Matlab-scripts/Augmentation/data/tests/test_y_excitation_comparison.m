% test_y_excitation_comparison.m
% Compares two Y step excitation strategies for exciting the hidden MSD.
% Each option is run on both baseline (no MSD) and augmented (with MSD) models.
%
% Option 1: Y step 30 mm, amax = 50 m/s^2
% Option 2: Y step 80 mm, amax = 50 m/s^2
%
% Decision metric: FFT of y_aug - y_base at Y channel.
% A visible peak near fa confirms the MSD is excited and observable.
%
% Run from project root:
%   run('Matlab-scripts/Augmentation/data/tests/test_y_excitation_comparison.m')

clearvars
close all
clc

MDL_BASE = 'gantry_2025a';
MDL_AUG  = 'gantry_additional_state_2025a';

% Close Simulink models if open
if bdIsLoaded(MDL_BASE), close_system(MDL_BASE, 0); end
if bdIsLoaded(MDL_AUG),  close_system(MDL_AUG,  0); end

addpath(genpath(fullfile(pwd, 'kamtin-fp-model', '03 Simulink gantry')))
addpath(fullfile(pwd, 'Matlab-scripts', 'Augmentation'))

%% Physical parameters (identical to generate scripts)
mb=22.8; mh=10.1; m1=10.2; m2=10.7; Jb=1.0; Jh=0.05;
cg1=14.5; cg2=20.3; cy=10; cb1=9; cb2=9;
kb1=1987.5; kb2=1987.5; Lb=0.725; Lh=0.25; d=0.1;
cc1=16.8; cc2=18.35; ccy=11.6;   % Coulomb — disabled in model but expected in workspace
C_damp = [cg1+cg2,        (cg1-cg2)*Lb/2,            0;
          (cg1-cg2)*Lb/2, cb1+cb2+(cg1+cg2)*Lb^2/4,  0;
          0,               0,                          cy];
K = [0,0,0; 0,kb1+kb2,0; 0,0,0];
n=3; P=[1,1,0; Lb/2,-Lb/2,0; 0,0,1];
fs=20e3; ts=1/fs; fbw=100; Y_op=0.3;

% MSD parameters (same as generate_gantry_lti_augmented.m)
ma_frac=0.10; ma=ma_frac*mh; mh_rigid=mh-ma; L0=0.10;
fa=150; ka=ma*(2*pi*fa)^2; zeta_a=0.05; ca=2*zeta_a*sqrt(ka*ma);

% Controller and plant (full mh, frozen at Y_op)
M_op = [m1+m2+mb+mh,          (m1-m2)*Lb/2-mh*Y_op,                    0;
        (m1-m2)*Lb/2-mh*Y_op, Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y_op^2, -mh*d;
        0,                    -mh*d,                                     mh];
sys = P.' * getss(n, M_op, C_damp, K) * P;
Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
for j = 1:3, Cfb(j,j) = ruleOfThumb(fbw, sys(j,j), ts); end
G = c2d(sys, ts, 'zoh');   % Simulink LTI block reads G from workspace

%% Reference trajectories
% Shared X: symmetric motion identical to training trajectories
n_hold = round(0.5/ts);
pv_x = thirdOrderSetpointETEL(0.15, 1.5, 20.0, 20.0/0.030, Inf, ts);
pv_x = pv_x(:,1);
N    = n_hold + numel(pv_x) + n_hold;
t_ref = (0:N-1)' * ts;
x_col = [zeros(n_hold,1); pv_x; zeros(n_hold,1)];
f = zeros(N, 3);   % no external force injection

% Option 1: Y step move — 30 mm, amax = 50 m/s^2
pv_y1 = thirdOrderSetpointETEL(0.03, 0.5, 50.0, 50.0/0.005, Inf, ts);
pv_y1 = pv_y1(:,1);
r1 = [x_col, x_col, Y_op*ones(N,1)];
i0 = n_hold+1;  i1 = min(i0+numel(pv_y1)-1, N);
r1(i0:i1, 3) = Y_op + pv_y1(1:i1-i0+1);

% Option 2: Y step move — 80 mm, amax = 50 m/s^2
pv_y2 = thirdOrderSetpointETEL(0.08, 1.0, 50.0, 50.0/0.005, Inf, ts);
pv_y2 = pv_y2(:,1);
r2 = [x_col, x_col, Y_op*ones(N,1)];
i0 = n_hold+1;  i1 = min(i0+numel(pv_y2)-1, N);
r2(i0:i1, 3) = Y_op + pv_y2(1:i1-i0+1);

% Option 3: broadband multisine force injection on all 3 channels (1-100 Hz)
% Reference frozen at Y_op, X=0.  Force signal injected via the 'f' port.
% Frequency grid: integer harmonics 1..100 Hz, random phases, amplitude 50 N per channel.
rng(42);   % reproducible
f_lo = 1; f_hi = 200; F_amp = 50;   % [Hz, Hz, N]
freqs = (f_lo:f_hi)';               % 100 harmonics
phi   = 2*pi*rand(numel(freqs), 3); % (100 x 3) random phases per channel
t_ms  = (0:N-1)' * ts;
f3    = zeros(N, 3);
for ch3 = 1:3
    % (N x 100): each column is one harmonic
    f3(:,ch3) = F_amp * sum(sin(2*pi*t_ms*freqs' + phi(:,ch3)'), 2) / numel(freqs);
end
% f3 is (N x 3): broadband force, RMS per channel = F_amp/sqrt(2)
r3    = [x_col, x_col, Y_op*ones(N,1)];  % constant Y reference

%% Run six simulations
% sim() must be at script level so Simulink can read/write the base workspace.
% mh swapped to mh_rigid before augmented sim, restored with mh_rigid+ma after.

fprintf('Option 1 - Baseline...\n')
r=r1; t=t_ref; Y=Y_op;
sim(MDL_BASE, t_ref(end));
Q1b=q1; T1b=(0:size(Q1b,1)-1)'*ts;

fprintf('Option 1 - Augmented...\n')
r=r1; t=t_ref; Y=Y_op; mh_original=mh; mh=mh_rigid;
sim(MDL_AUG, t_ref(end));
mh=mh_rigid+ma;
Q1a=q_aug; DA1=delta_a; T1a=(0:size(Q1a,1)-1)'*ts;

fprintf('Option 2 - Baseline...\n')
r=r2; t=t_ref; Y=Y_op;
sim(MDL_BASE, t_ref(end));
Q2b=q1; T2b=(0:size(Q2b,1)-1)'*ts;

fprintf('Option 2 - Augmented...\n')
r=r2; t=t_ref; Y=Y_op; mh_original=mh; mh=mh_rigid;
sim(MDL_AUG, t_ref(end));
mh=mh_rigid+ma;
Q2a=q_aug; DA2=delta_a; T2a=(0:size(Q2a,1)-1)'*ts;

fprintf('Option 3 - Baseline (multisine)...\n')
r=r3; t=t_ref; f=f3; Y=Y_op;
sim(MDL_BASE, t_ref(end));
Q3b=q1; T3b=(0:size(Q3b,1)-1)'*ts;

fprintf('Option 3 - Augmented (multisine)...\n')
r=r3; t=t_ref; f=f3; Y=Y_op; mh_original=mh; mh=mh_rigid;
sim(MDL_AUG, t_ref(end));
mh=mh_rigid+ma;
f=zeros(N,3);   % restore f for safety
Q3a=q_aug; DA3=delta_a; T3a=(0:size(Q3a,1)-1)'*ts;

%% Align all results to common time grid
yb1 = interp1(T1b, Q1b, t_ref, 'linear', 'extrap');
ya1 = interp1(T1a, Q1a, t_ref, 'linear', 'extrap');
da1 = interp1(T1a, DA1, t_ref, 'linear', 'extrap');

yb2 = interp1(T2b, Q2b, t_ref, 'linear', 'extrap');
ya2 = interp1(T2a, Q2a, t_ref, 'linear', 'extrap');
da2 = interp1(T2a, DA2, t_ref, 'linear', 'extrap');

yb3 = interp1(T3b, Q3b, t_ref, 'linear', 'extrap');
ya3 = interp1(T3a, Q3a, t_ref, 'linear', 'extrap');
da3 = interp1(T3a, DA3, t_ref, 'linear', 'extrap');

%% Summary
ch = {'X1','X2','Y'};
for iOpt = 1:3
    if     iOpt==1; yb=yb1; ya=ya1; da=da1; lbl='Option 1 (30mm step)';
    elseif iOpt==2; yb=yb2; ya=ya2; da=da2; lbl='Option 2 (80mm step)';
    else;           yb=yb3; ya=ya3; da=da3; lbl='Option 3 (multisine F, 1-100 Hz)'; end
    dy = ya - yb;
    Nfft=numel(t_ref); Nhalf=floor(Nfft/2);
    f_ax=(0:Nhalf-1)*fs/Nfft;
    Y_fft=2*abs(fft(dy(:,3)))/Nfft;
    [~,idx]=min(abs(f_ax-fa));
    fprintf('\n=== %s ===\n', lbl)
    fprintf('  fa = %g Hz\n', fa)
    fprintf('  max|delta_a| = %.4e m\n',  max(abs(da)))
    fprintf('  rms(delta_a) = %.4e m\n',  rms(da))
    for j=1:3
        fprintf('  rms(d%s)     = %.4e m\n', ch{j}, rms(dy(:,j)))
        fprintf('  max|d%s|     = %.4e m\n', ch{j}, max(abs(dy(:,j))))
    end
    fprintf('  FFT(dY) at %g Hz = %.4e m\n', fa, Y_fft(idx))
    fprintf('  FFT(dY) DC       = %.4e m\n', Y_fft(1))
end

%% Plots
plot_comparison(t_ref, r1, yb1, ya1, da1, fs, fa, ...
    'Option 1: Y step 30 mm (amax=50 m/s^2)')
plot_comparison(t_ref, r2, yb2, ya2, da2, fs, fa, ...
    'Option 2: Y step 80 mm (amax=50 m/s^2)')
plot_comparison(t_ref, r3, yb3, ya3, da3, fs, fa, ...
    'Option 3: multisine force all channels (1-100 Hz, 50 N)')

%% FRF comparison: baseline vs augmented at Y = Y_op
fprintf('\nFRF comparison: baseline vs augmented at Y=%.2f m...\n', Y_op)

frf_f_lines  = 1:500;                   % 1-500 Hz at 1 Hz resolution
frf_line_idx = frf_f_lines + 1;         % FFT bin indices (bin 1 = DC)
frf_N_period = fs;                      % 1 s period = 20000 samples
frf_N_settle = 1;
frf_N_drop   = 2;
frf_N_clean  = 5;
frf_N_record = frf_N_drop + frf_N_clean;
frf_N_total  = (frf_N_settle + frf_N_record) * frf_N_period;
t_frf        = (0:frf_N_total-1)' * ts;

frf_modes(1).name='common'; frf_modes(1).f_vec=[1, 1, 0]; frf_modes(1).rms=25;
frf_modes(2).name='diff';   frf_modes(2).f_vec=[1,-1, 0]; frf_modes(2).rms=25;
frf_modes(3).name='y';      frf_modes(3).f_vec=[0, 0, 1]; frf_modes(3).rms=15;

rng(99);
t_per = (0:frf_N_period-1)'/fs;
for im = 1:3
    phases = 2*pi*rand(1, numel(frf_f_lines));
    sig = sum(cos(2*pi*frf_f_lines(:)' .* t_per + phases), 2);
    frf_modes(im).one_period = frf_modes(im).rms * sig / rms(sig);
end

nFreqFRF = numel(frf_f_lines);
Yb_runs  = complex(nan(nFreqFRF, 3, 3));
Ub_runs  = complex(nan(nFreqFRF, 3, 3));
Ya_runs  = complex(nan(nFreqFRF, 3, 3));
Ua_runs  = complex(nan(nFreqFRF, 3, 3));

r_frf     = repmat([0, 0, Y_op], frf_N_total, 1);
rec_start = frf_N_settle*frf_N_period + 1;

for im = 1:3
    mode  = frf_modes(im);
    f_frf = zeros(frf_N_total, 3);
    f_frf(rec_start:end,:) = repmat(mode.one_period, frf_N_record, 1) * mode.f_vec;

    fprintf('  FRF baseline  mode %-7s\n', mode.name)
    r=r_frf; t=t_frf; f=f_frf; Y=Y_op;
    sim(MDL_BASE, t_frf(end));
    q1u = interp1((0:size(q1,1)-1)'*ts, q1, t_frf, 'linear', 'extrap');
    ub  = lsim(ss(Cfb), r_frf-q1u, t_frf) + f_frf;
    [Yb_runs(:,:,im), Ub_runs(:,:,im)] = frf_avg(q1u, ub, frf_N_period, frf_N_settle, frf_N_drop, frf_N_clean, frf_line_idx);

    fprintf('  FRF augmented mode %-7s\n', mode.name)
    r=r_frf; t=t_frf; f=f_frf; Y=Y_op; mh_original=mh; mh=mh_rigid;
    sim(MDL_AUG, t_frf(end));
    mh=mh_rigid+ma;
    qau = interp1((0:size(q_aug,1)-1)'*ts, q_aug, t_frf, 'linear', 'extrap');
    ua  = lsim(ss(Cfb), r_frf-qau, t_frf) + f_frf;
    [Ya_runs(:,:,im), Ua_runs(:,:,im)] = frf_avg(qau, ua, frf_N_period, frf_N_settle, frf_N_drop, frf_N_clean, frf_line_idx);
end
f = zeros(N, 3);   % restore f

G_base = frf_compute(Yb_runs, Ub_runs);
G_aug  = frf_compute(Ya_runs, Ua_runs);

plot_frf_comparison(frf_f_lines, G_base, G_aug, fa, Y_op)

%% Linearize-based FRF comparison (MATLAB linearize workflow)
% Follows: linearize(mdl, io) with linio I/O specification.
% open-loop output breaks the feedback path so we get the plant FRF.
fprintf('\nLinearizing baseline and augmented models at Y=%.2f m...\n', Y_op)

io_b(1) = linio([MDL_BASE '/Gain3'],         1, 'input');
io_b(2) = linio([MDL_BASE '/To Workspace1'], 1, 'openoutput');
sys_base_lin = linearize(MDL_BASE, io_b);

io_a(1) = linio([MDL_AUG '/Gain3'],         1, 'input');
io_a(2) = linio([MDL_AUG '/To Workspace1'], 1, 'openoutput');
mh = mh_rigid;
sys_aug_lin = linearize(MDL_AUG, io_a);
mh = mh_rigid + ma;

plot_linearized_frf(sys_base_lin, sys_aug_lin, fa, Y_op)

%% =========================================================================
function plot_comparison(t, r, y_base, y_aug, da, fs, fa, ttl)
    dy = y_aug - y_base;
    ch = {'X1','X2','Y'};
    figure('Name', ttl, 'Position', [100 50 1100 820]);

    % Row 1: overlaid trajectories (reference, baseline, augmented)
    for j = 1:3
        subplot(5,3,j); hold on
        plot(t, r(:,j),      'k--', 'LineWidth', 0.8)
        plot(t, y_base(:,j), 'b',   'LineWidth', 0.9)
        plot(t, y_aug(:,j),  'r',   'LineWidth', 0.9)
        ylabel([ch{j} ' [m]']); grid on
        title([ch{j} ' trajectory'])
        if j==1, legend('Ref','Baseline','Augmented','Location','best'); end
    end

    % Row 2: difference y_aug - y_base per channel
    for j = 1:3
        subplot(5,3,3+j)
        plot(t, dy(:,j), 'k', 'LineWidth', 0.9)
        ylabel([ch{j} ' diff [m]']); grid on
        title(sprintf('%s diff | rms=%.2e m', ch{j}, rms(dy(:,j))))
    end

    Nfft  = numel(t);
    Nhalf = floor(Nfft/2);
    f_ax  = (0:Nhalf-1) * fs / Nfft;
    Y_fft_diff = 2*abs(fft(dy(:,3)))/Nfft;
    Y_fft_base = 2*abs(fft(y_base(:,3)))/Nfft;
    Y_fft_aug  = 2*abs(fft(y_aug(:,3)))/Nfft;
    [~,idx_fa] = min(abs(f_ax - fa));

    % Row 3: FFT of Y-channel difference
    subplot(5,3,7:9)
    semilogy(f_ax, Y_fft_diff(1:Nhalf), 'k', 'LineWidth', 0.8); hold on
    xline(fa, 'r--', sprintf('%g Hz', fa), 'LineWidth', 1.2, 'LabelVerticalAlignment', 'bottom')
    text(fa, Y_fft_diff(idx_fa)*2, sprintf('%.2e m', Y_fft_diff(idx_fa)), ...
         'Color','r', 'FontSize', 8, 'HorizontalAlignment','center')
    xlim([0 max(fa*4, 200)]); grid on
    xlabel('Frequency [Hz]'); ylabel('|\DeltaY| [m]')
    title('FFT of Y-channel difference (aug - base)')

    % Row 4: FFT of Y channel — baseline vs augmented overlaid
    subplot(5,3,10:12)
    semilogy(f_ax, Y_fft_base(1:Nhalf), 'b', 'LineWidth', 0.9); hold on
    semilogy(f_ax, Y_fft_aug(1:Nhalf),  'r', 'LineWidth', 0.9)
    xline(fa, 'k--', sprintf('%g Hz', fa), 'LineWidth', 1.2, 'LabelVerticalAlignment', 'bottom')
    xlim([0 max(fa*4, 200)]); grid on
    xlabel('Frequency [Hz]'); ylabel('|Y| [m]')
    legend('Baseline','Augmented','Location','best')
    title('FFT of Y channel — baseline vs augmented')

    % Row 5: delta_a — ground truth MSD displacement
    subplot(5,3,13:15)
    plot(t, da, 'b', 'LineWidth', 0.9)
    ylabel('\delta_a [m]'); xlabel('Time [s]'); grid on
    title(sprintf('\\delta_a  |  max = %.2e m   rms = %.2e m', ...
                  max(abs(da)), rms(da)))

    sgtitle(ttl, 'Interpreter', 'none')
end

function [Y_avg, U_avg] = frf_avg(y, u, N_period, n_settle, n_drop, n_clean, line_idx)
    i0 = (n_settle + n_drop)*N_period + 1;
    y3 = permute(reshape(y(i0:i0+n_clean*N_period-1,:), N_period, n_clean, 3), [1 3 2]);
    u3 = permute(reshape(u(i0:i0+n_clean*N_period-1,:), N_period, n_clean, 3), [1 3 2]);
    Y_avg = mean(fft(y3,[],1), 3);
    U_avg = mean(fft(u3,[],1), 3);
    Y_avg = Y_avg(line_idx,:);
    U_avg = U_avg(line_idx,:);
end

function G = frf_compute(Y_runs, U_runs)
    % Y_runs, U_runs: (nFreq x 3 x 3) -- third dim indexes the 3 modal inputs
    Y3 = permute(Y_runs, [2 3 1]);   % (3 x 3 x nFreq)
    U3 = permute(U_runs, [2 3 1]);
    G  = permute(pagemrdivide(Y3, U3), [3 1 2]);  % (nFreq x 3 x 3)
end

function plot_linearized_frf(sys_base, sys_aug, fa, Y_op)
    out_names = {'X_1','X_2','Y'};
    in_names  = {'F_1','F_2','F_Y'};
    freq_hz   = logspace(0, log10(4*fa), 3000);   % 1 Hz to 4*fa
    [mag_b, ~, w] = bode(sys_base, freq_hz*2*pi);
    [mag_a, ~]    = bode(sys_aug,  freq_hz*2*pi);
    f_hz = w / (2*pi);
    figure('Name', sprintf('Linearized FRF Y=%.2fm', Y_op), ...
           'Position', [100 50 1100 850]);
    tiledlayout(3, 3, 'TileSpacing','compact', 'Padding','compact');
    for iy = 1:3
        for iu = 1:3
            nexttile; hold on
            semilogx(f_hz, 20*log10(squeeze(mag_b(iy,iu,:))), 'b', 'LineWidth', 1.0)
            semilogx(f_hz, 20*log10(squeeze(mag_a(iy,iu,:))), 'r', 'LineWidth', 1.0)
            xline(fa, 'k--', sprintf('%g Hz', fa), 'LineWidth', 1.0, ...
                  'LabelVerticalAlignment','bottom')
            grid on; xlim([f_hz(1) f_hz(end)])
            title(sprintf('%s / %s', out_names{iy}, in_names{iu}))
            if iu==1; ylabel('Mag [dB re m/N]'); end
            if iy==3; xlabel('Frequency [Hz]'); end
            if iy==1 && iu==1
                legend('Baseline','Augmented','Location','best','FontSize',7)
            end
        end
    end
    sgtitle(sprintf('Linearized FRF: Baseline vs Augmented  |  Y = %.2f m  |  fa = %g Hz', Y_op, fa), ...
            'Interpreter','none')
end

function plot_frf_comparison(f, G_base, G_aug, fa, Y_op)
    out_names = {'X_1','X_2','Y'};
    in_names  = {'F_1','F_2','F_Y'};
    figure('Name', sprintf('FRF baseline vs augmented Y=%.2fm', Y_op), ...
           'Position', [100 50 1100 850]);
    tiledlayout(3, 3, 'TileSpacing','compact', 'Padding','compact');
    for iy = 1:3
        for iu = 1:3
            nexttile; hold on
            mag_b = 20*log10(abs(squeeze(G_base(:,iy,iu))));
            mag_a = 20*log10(abs(squeeze(G_aug(:,iy,iu))));
            semilogx(f, mag_b, 'b', 'LineWidth', 1.0)
            semilogx(f, mag_a, 'r', 'LineWidth', 1.0)
            xline(fa, 'k--', sprintf('%g Hz', fa), 'LineWidth', 1.0, ...
                  'LabelVerticalAlignment','bottom')
            grid on; xlim([f(1) f(end)])
            title(sprintf('%s / %s', out_names{iy}, in_names{iu}))
            if iu==1; ylabel('Mag [dB re m/N]'); end
            if iy==3; xlabel('Frequency [Hz]'); end
            if iy==1 && iu==1
                legend('Baseline','Augmented','Location','best','FontSize',7)
            end
        end
    end
    sgtitle(sprintf('FRF: Baseline vs Augmented  |  Y = %.2f m  |  fa = %g Hz', Y_op, fa), ...
            'Interpreter','none')
end
