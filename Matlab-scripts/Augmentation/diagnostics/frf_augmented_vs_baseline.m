%% frf_augmented_vs_baseline.m
% Compare augmented (with MSD) vs baseline (rigid) plant FRFs analytically.
%
% Shows WHERE the MSD adds information: resonance + anti-resonance pair
% near fa, visible as a local bump/dip in the transfer function.
%
% Also computes the FRF ratio |H_aug/H_base| to quantify the MSD effect
% as a function of frequency — independent of trajectory or excitation.
%
% Run from repo root:
%   run('Matlab-scripts/Augmentation/diagnostics/frf_augmented_vs_baseline.m')

clear; clc; close all;

%% ── Parameters (same as generate_multisine_data.m) ─────────────────────
mb=22.8; m1=10.2; m2=10.7; Jb=1.0; Jh=0.05;
cg1=14.5; cg2=20.3; cy=10; cb1=9; cb2=9;
kb1=1987.5; kb2=1987.5; Lb=0.725; d=0.1;

mh_total = 10.1;
fa       = 150;                          % MSD natural frequency [Hz]
zeta_a   = 0.05;                         % MSD damping ratio
L0       = 0.10;                         % equilibrium offset [m]

% Coordinate transform: logical <-> stage
P = [1, 1, 0; Lb/2, -Lb/2, 0; 0, 0, 1];

Y_op = 0.3;       % frozen operating point
delta_a0 = 0;

in_names  = {'F_{X1}', 'F_{X2}', 'F_Y'};
out_names = {'X1', 'X2', 'Y'};

%% ── Mass fractions to compare ──────────────────────────────────────────
ma_fracs = [0.10, 0.50];
colors   = {'b', 'r'};
freq_hz  = logspace(-1, log10(1000), 5000);
w        = 2*pi * freq_hz;

%% ── Build baseline (no MSD: ma=0, rigid payload mh=mh_total) ──────────
[A_base, B_base, C_base] = build_ss_baseline(Y_op, ...
    m1, m2, mb, mh_total, Lb, Jb, Jh, d, ...
    cg1, cg2, cb1, cb2, cy, kb1, kb2);
B_base_stg = B_base * P;
C_base_stg = P' * C_base;
sys_base = ss(A_base, B_base_stg, C_base_stg, zeros(3));
H_base = freqresp(sys_base, w);

%% ── Figure 1: FRF overlay (Y/F_Y channel) ─────────────────────────────
figure('Position', [50 50 900 600]);
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% Magnitude
ax_m = nexttile; hold on;
plot(freq_hz, 20*log10(squeeze(abs(H_base(3,3,:)))), 'k', ...
    'LineWidth', 1.5, 'DisplayName', 'baseline (rigid)');

for mi = 1:length(ma_fracs)
    ma_frac = ma_fracs(mi);
    ma = ma_frac * mh_total;
    mh = mh_total - ma;
    ka = ma * (2*pi*fa)^2;                         % THEORY: ka = ma*omega_a^2
    ca = 2 * zeta_a * sqrt(ka * ma);               % THEORY: ca = 2*zeta*sqrt(k*m)

    [A_aug, B_aug, C_aug] = build_ss_augmented(Y_op, delta_a0, ...
        m1, m2, mb, mh, Lb, Jb, Jh, d, ...
        cg1, cg2, cb1, cb2, cy, kb1, kb2, ma, ka, ca, L0);
    B_stg = B_aug * P;
    C_stg = P' * C_aug;
    sys_aug = ss(A_aug, B_stg, C_stg, zeros(3));
    H_aug = freqresp(sys_aug, w);

    pct = round(ma_frac * 100);
    plot(freq_hz, 20*log10(squeeze(abs(H_aug(3,3,:)))), colors{mi}, ...
        'LineWidth', 1.2, 'DisplayName', sprintf('m_a = %d%% (%.2f kg)', pct, ma));

    % Store for ratio plot
    H_augs{mi} = H_aug; %#ok<SAGROW>
end

xline(fa, 'r--', sprintf('f_a = %d Hz', fa), 'LineWidth', 1, ...
    'LabelVerticalAlignment', 'bottom');
set(gca, 'XScale', 'log'); grid on;
xlim([1 500]); ylabel('Magnitude [dB]');
legend('Location', 'southwest');
title('Y / F_Y  transfer function');

% Phase
ax_p = nexttile; hold on;
ph_base = rad2deg(unwrap(angle(squeeze(H_base(3,3,:)))));
plot(freq_hz, ph_base, 'k', 'LineWidth', 1.5);

for mi = 1:length(ma_fracs)
    ph_aug = rad2deg(unwrap(angle(squeeze(H_augs{mi}(3,3,:)))));
    plot(freq_hz, ph_aug, colors{mi}, 'LineWidth', 1.2);
end

xline(fa, 'r--', 'LineWidth', 1);
set(gca, 'XScale', 'log'); grid on;
xlim([1 500]); ylabel('Phase [deg]'); xlabel('Frequency [Hz]');

linkaxes([ax_m, ax_p], 'x');
sgtitle(sprintf('Augmented vs baseline plant FRF  (Y_{op} = %.1f m, f_a = %d Hz)', ...
    Y_op, fa));

%% ── Figure 2: FRF ratio |H_aug / H_base| for all 9 I/O pairs ─────────
figure('Position', [100 100 1200 800]);
tiledlayout(3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

for iy = 1:3
    for iu = 1:3
        nexttile; hold on;
        h_base_iu = squeeze(abs(H_base(iy, iu, :)));

        for mi = 1:length(ma_fracs)
            h_aug_iu = squeeze(abs(H_augs{mi}(iy, iu, :)));
            ratio_db = 20*log10(h_aug_iu ./ h_base_iu);
            pct = round(ma_fracs(mi) * 100);
            plot(freq_hz, ratio_db, colors{mi}, 'LineWidth', 1.2, ...
                'DisplayName', sprintf('%d%%', pct));
        end

        yline(0, 'k:', 'LineWidth', 0.5);
        xline(fa, 'r--', 'LineWidth', 0.8);
        set(gca, 'XScale', 'log'); grid on;
        xlim([1 500]);
        ylabel('dB');
        title(sprintf('%s / %s', out_names{iy}, in_names{iu}));
        if iy == 1 && iu == 3, legend('Location', 'best'); end
    end
end
sgtitle(sprintf('FRF ratio: |H_{aug}| / |H_{base}|  [dB]   (0 dB = no change)'));

%% ── Figure 3: Zoom on Y/F_Y around MSD frequency ──────────────────────
figure('Position', [150 150 800 400]);
hold on;

for mi = 1:length(ma_fracs)
    h_base_YFY = squeeze(abs(H_base(3,3,:)));
    h_aug_YFY  = squeeze(abs(H_augs{mi}(3,3,:)));
    ratio_db = 20*log10(h_aug_YFY ./ h_base_YFY);
    pct = round(ma_fracs(mi) * 100);
    plot(freq_hz, ratio_db, colors{mi}, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('m_a = %d%%', pct));
end

yline(0, 'k:', 'LineWidth', 0.5);
xline(fa, 'r--', sprintf('f_a = %d Hz', fa), 'LineWidth', 1);
set(gca, 'XScale', 'log'); grid on;
xlim([50 400]); ylabel('|H_{aug}/H_{base}| [dB]');
xlabel('Frequency [Hz]');
legend('Location', 'best');
title('Y/F_Y ratio — MSD fingerprint');

%% ── Print summary ─────────────────────────────────────────────────────
fprintf('\n=== MSD effect on Y/F_Y transfer function ===\n');
fprintf('%-8s  %12s  %12s  %12s  %12s  %12s\n', ...
    'ma [%]', 'res [dB]', 'f_res [Hz]', 'anti-res[dB]', 'f_ar [Hz]', 'broadband[dB]');
fprintf('%s\n', repmat('-', 1, 75));

for mi = 1:length(ma_fracs)
    h_base_YFY = squeeze(abs(H_base(3,3,:)));
    h_aug_YFY  = squeeze(abs(H_augs{mi}(3,3,:)));
    ratio_db = 20*log10(h_aug_YFY ./ h_base_YFY);

    % Resonance and anti-resonance near fa
    idx_near = (freq_hz >= (fa-30) & freq_hz <= (fa+30))';
    ratio_near = ratio_db(idx_near);
    f_near = freq_hz(idx_near);

    [res_db, ix_res]   = max(ratio_near);     % resonance peak (positive bump)
    [ar_db,  ix_ar]    = min(ratio_near);      % anti-resonance dip (negative notch)
    f_res = f_near(ix_res);
    f_ar  = f_near(ix_ar);

    % Broadband deviation (away from fa)
    idx_far = (freq_hz >= 1 & freq_hz <= 500)' & ~idx_near;
    mean_dev = mean(abs(ratio_db(idx_far)));

    pct = round(ma_fracs(mi) * 100);
    fprintf('  %4d    %+10.1f    %10.1f    %+10.1f    %10.1f    %10.1f\n', ...
        pct, res_db, f_res, ar_db, f_ar, mean_dev);
end

fprintf('\n  Interpretation:\n');
fprintf('    res [dB]       = MSD resonance amplification (positive = louder)\n');
fprintf('    anti-res [dB]  = MSD anti-resonance absorption (negative = quieter)\n');
fprintf('    broadband [dB] = mean |ratio| away from f_a (mass redistribution effect)\n');
fprintf('    If |res| >> broadband, the MSD resonance is a distinguishable feature.\n');
fprintf('    If |res| ~ broadband, the MSD effect is dominated by mass change.\n');

%% =========================================================================
function [A, B_log, C_log] = build_ss_augmented(Y, da, ...
    m1, m2, mb, mh, Lb, Jb, Jh, d, ...
    cg1, cg2, cb1, cb2, cy, kb1, kb2, ma, ka, ca, L0)
% 8-state augmented gantry (with MSD). Same as multisine_frequency_range.m.

    M = [m1+m2+mb+mh+ma, (m1-m2)*Lb/2-(mh+ma)*Y-ma*L0-ma*da,            0,     0;
         (m1-m2)*Lb/2-(mh+ma)*Y-ma*L0-ma*da, ...
             Jb+Jh+(m1+m2)*Lb^2/4+(mh+ma)*d^2+mh*Y^2+ma*(Y+L0+da)^2, ...
             -(mh+ma)*d, -ma*d;
         0,  -(mh+ma)*d,  mh+ma,  ma;
         0,  -ma*d,       ma,     ma];

    C4 = [cg1+cg2,         (cg1-cg2)*Lb/2,               0,  0;
          (cg1-cg2)*Lb/2,  cb1+cb2+(cg1+cg2)*Lb^2/4,     0,  0;
          0,               0,                             cy,  0;
          0,               0,                              0, ca];

    K4 = [0, 0,       0,  0;
          0, kb1+kb2, 0,  0;
          0, 0,       0,  0;
          0, 0,       0, ka];

    A     = [zeros(4), eye(4); -M\K4, -M\C4];
    B_log = [zeros(4,3); M \ [eye(3); zeros(1,3)]];
    C_log = [eye(3), zeros(3,1), zeros(3,4)];
end

function [A, B_log, C_log] = build_ss_baseline(Y, ...
    m1, m2, mb, mh, Lb, Jb, Jh, d, ...
    cg1, cg2, cb1, cb2, cy, kb1, kb2)
% 6-state baseline gantry (rigid payload, no MSD).

    M = [m1+m2+mb+mh, (m1-m2)*Lb/2 - mh*Y,       0;
         (m1-m2)*Lb/2 - mh*Y, ...
             Jb+Jh+(m1+m2)*Lb^2/4 + mh*d^2 + mh*Y^2, ...
             -mh*d;
         0,  -mh*d,  mh];

    C3 = [cg1+cg2,         (cg1-cg2)*Lb/2,               0;
          (cg1-cg2)*Lb/2,  cb1+cb2+(cg1+cg2)*Lb^2/4,     0;
          0,               0,                             cy];

    K3 = [0, 0,       0;
          0, kb1+kb2, 0;
          0, 0,       0];

    A     = [zeros(3), eye(3); -M\K3, -M\C3];
    B_log = [zeros(3); M \ eye(3)];
    C_log = [eye(3), zeros(3)];
end
