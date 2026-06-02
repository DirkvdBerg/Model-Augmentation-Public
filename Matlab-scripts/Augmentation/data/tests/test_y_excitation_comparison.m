% test_y_excitation_comparison.m
% Compares two Y excitation strategies for exciting the hidden 400 Hz MSD.
% Each option is run on both baseline (no MSD) and augmented (with MSD) models.
%
% Option 1: small Y step moves via thirdOrderSetpointETEL (high acceleration pulse)
% Option 2: slow Y sinusoid (5 mm at 1 Hz)
%
% Decision metric: FFT of y_aug - y_base at Y channel.
% A visible peak near 400 Hz confirms the MSD is excited and observable.
%
% Run from project root:
%   run('Matlab-scripts/Augmentation/data/tests/test_y_excitation_comparison.m')

addpath(genpath(fullfile(pwd, 'kamtin-fp-model', '03 Simulink gantry')))
addpath(fullfile(pwd, 'Matlab-scripts', 'Augmentation'))

MDL_BASE = 'gantry_2025a';
MDL_AUG  = 'gantry_additional_state_2025a';

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
fa=50; ka=ma*(2*pi*fa)^2; zeta_a=0.05; ca=2*zeta_a*sqrt(ka*ma);

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

% Option 1: Y step move using same generator as X (10 mm, amax = 50 m/s^2)
pv_y = thirdOrderSetpointETEL(0.01, 0.2, 50.0, 50.0/0.005, Inf, ts);
pv_y = pv_y(:,1);
r1 = [x_col, x_col, Y_op*ones(N,1)];
i0 = n_hold+1;  i1 = min(i0+numel(pv_y)-1, N);
r1(i0:i1, 3) = Y_op + pv_y(1:i1-i0+1);

% Option 2: slow Y sinusoid (5 mm, 1 Hz)
r2 = [x_col, x_col, Y_op + 0.005*sin(2*pi*1.0*t_ref)];

%% Run four simulations
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

%% Align all results to common time grid
yb1 = interp1(T1b, Q1b, t_ref, 'linear', 'extrap');
ya1 = interp1(T1a, Q1a, t_ref, 'linear', 'extrap');
da1 = interp1(T1a, DA1, t_ref, 'linear', 'extrap');

yb2 = interp1(T2b, Q2b, t_ref, 'linear', 'extrap');
ya2 = interp1(T2a, Q2a, t_ref, 'linear', 'extrap');
da2 = interp1(T2a, DA2, t_ref, 'linear', 'extrap');

%% Summary
fprintf('\n=== Summary ===\n')
fprintf('Option 1  max|delta_a|=%.3f um  rms(dY)=%.4f um\n', ...
        max(abs(da1))*1e6, rms(ya1(:,3)-yb1(:,3))*1e6)
fprintf('Option 2  max|delta_a|=%.3f um  rms(dY)=%.4f um\n', ...
        max(abs(da2))*1e6, rms(ya2(:,3)-yb2(:,3))*1e6)

%% Plots
plot_comparison(t_ref, r1, yb1, ya1, da1, fs, ...
    'Option 1: Y step moves (10 mm, amax=50 m/s^2)')
plot_comparison(t_ref, r2, yb2, ya2, da2, fs, ...
    'Option 2: Y sinusoid (5 mm, 1 Hz)')

%% =========================================================================
function plot_comparison(t, r, y_base, y_aug, da, fs, ttl)
    dy = y_aug - y_base;
    ch = {'X1','X2','Y'};
    figure('Name', ttl, 'Position', [100 50 1100 820]);

    % Row 1: overlaid trajectories (reference, baseline, augmented)
    for j = 1:3
        subplot(4,3,j); hold on
        plot(t, r(:,j)*1e3,      'k--', 'LineWidth', 0.8)
        plot(t, y_base(:,j)*1e3, 'b',   'LineWidth', 0.9)
        plot(t, y_aug(:,j)*1e3,  'r',   'LineWidth', 0.9)
        ylabel([ch{j} ' [mm]']); grid on
        title([ch{j} ' trajectory'])
        if j==1, legend('Ref','Baseline','Augmented','Location','best'); end
    end

    % Row 2: difference y_aug - y_base per channel
    for j = 1:3
        subplot(4,3,3+j)
        plot(t, dy(:,j)*1e6, 'k', 'LineWidth', 0.9)
        ylabel([ch{j} ' diff [\mum]']); grid on
        title(sprintf('%s diff | rms=%.4f \\mum', ch{j}, rms(dy(:,j))*1e6))
    end

    % Row 3: FFT of Y-channel difference — key metric for 400 Hz visibility
    subplot(4,3,7:9)
    Nfft  = numel(t);
    Nhalf = floor(Nfft/2);
    f_ax  = (0:Nhalf-1) * fs / Nfft;
    Y_fft = 2*abs(fft(dy(:,3)))/Nfft * 1e6;   % micrometers, two-sided corrected
    semilogy(f_ax, Y_fft(1:Nhalf), 'k', 'LineWidth', 0.8); hold on
    xline(400, 'r--', '400 Hz', 'LineWidth', 1.2, 'LabelVerticalAlignment', 'bottom')
    xlim([0 800]); grid on
    xlabel('Frequency [Hz]'); ylabel('|\DeltaY| [\mum]')
    title('FFT of Y-channel difference (aug - base)')

    % Row 4: delta_a — ground truth MSD displacement
    subplot(4,3,10:12)
    plot(t, da*1e6, 'b', 'LineWidth', 0.9)
    ylabel('\delta_a [\mum]'); xlabel('Time [s]'); grid on
    title(sprintf('\\delta_a  |  max = %.3f \\mum   rms = %.4f \\mum', ...
                  max(abs(da))*1e6, rms(da)*1e6))

    sgtitle(ttl, 'Interpreter', 'none')
end
