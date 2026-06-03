% frf_open_closed_loop.m
% Linearize baseline and augmented gantry models at Y_op.
% Produces both open-loop (plant G) and closed-loop (disturbance-to-output)
% FRFs using the MATLAB linearize/linio workflow.
%
% Open-loop:   Cfb zeroed before linearize() -> plant FRF G (force to position).
% Closed-loop: Cfb active during linearize() -> S*G from disturbance force to output.
%
% Run from project root:
%   run('Matlab-scripts/Augmentation/frf_open_closed_loop.m')

clearvars
close all
clc

MDL_BASE = 'gantry_2025a';
MDL_AUG  = 'gantry_additional_state_2025a';

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

% MSD parameters
ma_frac=0.10; ma=ma_frac*mh; mh_rigid=mh-ma; L0=0.10;
fa=150; ka=ma*(2*pi*fa)^2; zeta_a=0.05; ca=2*zeta_a*sqrt(ka*ma);

% Controller and plant (frozen at Y_op, full mh)
M_op = [m1+m2+mb+mh,          (m1-m2)*Lb/2-mh*Y_op,                    0;
        (m1-m2)*Lb/2-mh*Y_op, Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y_op^2, -mh*d;
        0,                    -mh*d,                                     mh];
sys = P.' * getss(n, M_op, C_damp, K) * P;
Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
for j = 1:3, Cfb(j,j) = ruleOfThumb(fbw, sys(j,j), ts); end
G = c2d(sys, ts, 'zoh');

%% Shared linio specs (same for both cases)
io_b(1) = linio([MDL_BASE '/Gain3'], 1, 'input');
io_b(2) = linio([MDL_BASE '/Gain4'], 1, 'output');
io_a(1) = linio([MDL_AUG  '/Gain3'], 1, 'input');
io_a(2) = linio([MDL_AUG  '/Gain4'], 1, 'output');

Y = Y_op;                              % required by Prismatic Joint block
t = [0; 1];                            % column vector — model blocks use [t, r(:,col)]
r = repmat([0, 0, Y_op], 2, 1);        % constant reference at operating point
f = zeros(2, 3);                       % no force injection

%% Open-loop linearization — controller zeroed
fprintf('\nLinearizing: open-loop (Cfb = 0)...\n')
Cfb_orig = Cfb;
Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));  % turn off controller

sys_base_ol = linearize(MDL_BASE, io_b);

mh_original = mh; mh = mh_rigid;
sys_aug_ol  = linearize(MDL_AUG,  io_a);
mh = mh_rigid + ma;

Cfb = Cfb_orig;   % restore controller

%% Closed-loop linearization — controller active
fprintf('Linearizing: closed-loop (Cfb active)...\n')

sys_base_cl = linearize(MDL_BASE, io_b);

mh_original = mh; mh = mh_rigid;
sys_aug_cl  = linearize(MDL_AUG,  io_a);
mh = mh_rigid + ma;

%% Report state counts
fprintf('\nState counts:\n')
fprintf('  Baseline  OL: %d states\n', order(sys_base_ol))
fprintf('  Augmented OL: %d states\n', order(sys_aug_ol))
fprintf('  Baseline  CL: %d states\n', order(sys_base_cl))
fprintf('  Augmented CL: %d states\n', order(sys_aug_cl))

%% Plot — 4 separate figures
freq_hz = logspace(0, log10(4*fa), 3000);
plot_frf_pair(sys_base_ol, sys_aug_ol, freq_hz, fa, Y_op, 'Open-loop FRF')
plot_frf_pair(sys_base_cl, sys_aug_cl, freq_hz, fa, Y_op, 'Closed-loop FRF')
plot_frf_diff_single(sys_base_ol, sys_aug_ol, freq_hz, fa, Y_op, 'Open-loop FRF difference (aug - base)')
plot_frf_diff_single(sys_base_cl, sys_aug_cl, freq_hz, fa, Y_op, 'Closed-loop FRF difference (aug - base)')

%% =========================================================================
function plot_frf_pair(G_base, G_aug, freq_hz, fa, Y_op, ttl)
    out_names = {'X_1','X_2','Y'};
    in_names  = {'F_1','F_2','F_Y'};
    [mag_b, ~, w] = bode(G_base, freq_hz*2*pi);
    [mag_a]       = bode(G_aug,  freq_hz*2*pi);
    f_hz = w / (2*pi);

    figure('Name', sprintf('%s  Y=%.2fm', ttl, Y_op), 'Position', [50 50 1200 900]);
    tiledlayout(3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    for iy = 1:3
        for iu = 1:3
            nexttile; hold on
            plot(f_hz, 20*log10(squeeze(mag_b(iy,iu,:))), 'b', 'LineWidth', 1.0)
            plot(f_hz, 20*log10(squeeze(mag_a(iy,iu,:))), 'r', 'LineWidth', 1.0)
            xline(fa, 'k:', sprintf('%g Hz', fa), 'LineWidth', 1.0, ...
                  'LabelVerticalAlignment', 'bottom')
            set(gca, 'XScale', 'log')
            grid on; xlim([f_hz(1) f_hz(end)])
            title(sprintf('%s / %s', out_names{iy}, in_names{iu}), 'Interpreter', 'none')
            if iu == 1; ylabel('Mag [dB re m/N]'); end
            if iy == 3; xlabel('Frequency [Hz]'); end
            if iy == 1 && iu == 1
                legend('Baseline', 'Augmented', 'Location', 'best', 'FontSize', 7)
            end
        end
    end
    sgtitle(sprintf('%s  |  Y = %.2f m  |  f_a = %g Hz', ttl, Y_op, fa), ...
            'Interpreter', 'none')
end

function plot_frf_diff_single(G_base, G_aug, freq_hz, fa, Y_op, ttl)
    out_names = {'X_1','X_2','Y'};
    in_names  = {'F_1','F_2','F_Y'};
    [mag_b, ~, w] = bode(G_base, freq_hz*2*pi);
    [mag_a]       = bode(G_aug,  freq_hz*2*pi);
    f_hz  = w / (2*pi);
    d_mag = 20*log10(mag_a) - 20*log10(mag_b);   % aug - base in dB

    figure('Name', sprintf('%s  Y=%.2fm', ttl, Y_op), 'Position', [100 50 1200 900]);
    tiledlayout(3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    for iy = 1:3
        for iu = 1:3
            nexttile; hold on
            plot(f_hz, squeeze(d_mag(iy,iu,:)), 'k', 'LineWidth', 1.0)
            yline(0, 'k:', 'LineWidth', 0.8)
            xline(fa, 'r:', sprintf('%g Hz', fa), 'LineWidth', 1.0, ...
                  'LabelVerticalAlignment', 'bottom')
            set(gca, 'XScale', 'log')
            grid on; xlim([f_hz(1) f_hz(end)])
            title(sprintf('%s / %s', out_names{iy}, in_names{iu}), 'Interpreter', 'none')
            if iu == 1; ylabel('\Delta Mag [dB]'); end
            if iy == 3; xlabel('Frequency [Hz]'); end
        end
    end
    sgtitle(sprintf('%s  |  Y = %.2f m  |  f_a = %g Hz', ttl, Y_op, fa), ...
            'Interpreter', 'none')
end
