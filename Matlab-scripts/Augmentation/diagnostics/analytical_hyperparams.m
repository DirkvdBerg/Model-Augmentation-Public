%% analytical_hyperparams.m
% Derive training hyperparameters (fs, nf, na_nb) analytically from the
% known 8-state augmented gantry model. Replaces empirical training sweeps.
%
% Sections:
%   1. Eigenvalue analysis   -> fs, nf, na_nb recommendations
%   2. FRF comparison        -> MSD visibility per I/O channel
%   3. Observability         -> PBH test, Gramian, per-mode observability
%   4. Summary table         -> derived vs current hyperparameters
%
% Run from repo root:
%   run('Matlab-scripts/Augmentation/diagnostics/analytical_hyperparams.m')

clear; clc; close all;

%% ========================================================================
%  PARAMETERS (from generate_multisine_data.m, NOT main_augmentation.m)
%  ========================================================================
mb = 22.8;   m1 = 10.2;   m2 = 10.7;
Jb = 1.0;    Jh = 0.05;
Lb = 0.725;  d  = 0.1;
cg1 = 14.5;  cg2 = 20.3;  cy = 10;
cb1 = 9;     cb2 = 9;
kb1 = 1987.5; kb2 = 1987.5;

mh_total = 10.1;
ma_frac  = 0.10;
ma       = ma_frac * mh_total;          % 1.01 kg
mh       = mh_total - ma;               % 9.09 kg (rigid part)
fa       = 150;                          % MSD natural frequency [Hz]
ka       = ma * (2*pi*fa)^2;            % MSD spring stiffness [N/m]
zeta_a   = 0.05;                         % MSD damping ratio
ca       = 2 * zeta_a * sqrt(ka * ma);  % MSD damper [Ns/m]
L0       = 0.10;                         % equilibrium offset [m]

P = [1, 1, 0; Lb/2, -Lb/2, 0; 0, 0, 1];  % logical <-> stage

Y_op     = 0.3;       % frozen operating point [m]
delta_a0 = 0;          % MSD at equilibrium

% Current training hyperparameters (from gantry_interconnect_dynamic.py)
fs_current   = 4000;    % [Hz]
nf_current   = 1200;    % [samples] = 0.300 s
nanb_current = 400;     % [samples] = 0.100 s
up_sample    = 10;      % RK4 sub-steps per Ts in training pipeline

%% ========================================================================
%  BUILD STATE-SPACE MODELS
%  ========================================================================
[A_aug, B_aug, C_aug] = build_ss_augmented(Y_op, delta_a0, ...
    m1, m2, mb, mh, Lb, Jb, Jh, d, ...
    cg1, cg2, cb1, cb2, cy, kb1, kb2, ma, ka, ca, L0);

[A_base, B_base, C_base] = build_ss_baseline(Y_op, ...
    m1, m2, mb, mh_total, Lb, Jb, Jh, d, ...
    cg1, cg2, cb1, cb2, cy, kb1, kb2);

% Stage-coordinate systems
B_aug_stg  = B_aug  * P;   C_aug_stg  = P' * C_aug;
B_base_stg = B_base * P;   C_base_stg = P' * C_base;

sys_aug  = ss(A_aug,  B_aug_stg,  C_aug_stg,  zeros(3));
sys_base = ss(A_base, B_base_stg, C_base_stg, zeros(3));

in_names  = {'F_{X1}', 'F_{X2}', 'F_Y'};
out_names = {'X_1', 'X_2', 'Y'};

%% ========================================================================
%  SECTION 1: EIGENVALUE ANALYSIS
%  ========================================================================
fprintf('\n');
fprintf('================================================================\n');
fprintf('  SECTION 1: EIGENVALUE ANALYSIS\n');
fprintf('================================================================\n\n');

eigs_aug  = eig(A_aug);
eigs_base = eig(A_base);

% Select positive-imaginary eigenvalues (one per conjugate pair)
idx_pos_aug  = imag(eigs_aug) > 0;
eigs_pairs   = eigs_aug(idx_pos_aug);
idx_pos_base = imag(eigs_base) > 0;
eigs_base_pairs = eigs_base(idx_pos_base);

% Include real eigenvalues (if any) as standalone modes
idx_real_aug = imag(eigs_aug) == 0;
eigs_real    = eigs_aug(idx_real_aug);

n_osc_modes  = length(eigs_pairs);
n_real_modes = length(eigs_real);

% Compute properties for oscillatory modes
freq_hz  = abs(imag(eigs_pairs)) / (2*pi);
omega_n  = abs(eigs_pairs);
damping  = -real(eigs_pairs) ./ omega_n;
tau      = -1 ./ real(eigs_pairs);
settling = 5 * tau;  % 5*tau ~ 1% settling

% Sort by frequency
[freq_hz, isort] = sort(freq_hz);
eigs_pairs = eigs_pairs(isort);
omega_n    = omega_n(isort);
damping    = damping(isort);
tau        = tau(isort);
settling   = settling(isort);

% Label gantry vs MSD modes by matching against baseline eigenvalues
base_freqs = sort(abs(imag(eigs_base_pairs)) / (2*pi));
mode_labels = cell(n_osc_modes, 1);
for k = 1:n_osc_modes
    [min_dist, ~] = min(abs(freq_hz(k) - base_freqs));
    if min_dist < 1  % within 1 Hz: gantry mode
        mode_labels{k} = 'gantry';
    else
        mode_labels{k} = 'MSD';
    end
end

% Print oscillatory modes
fprintf('  Oscillatory modes (conjugate pairs):\n');
fprintf('  %-6s  %10s  %10s  %10s  %10s  %8s\n', ...
    'Mode', 'f_n [Hz]', 'zeta', 'tau [ms]', '5*tau [ms]', 'Type');
fprintf('  %s\n', repmat('-', 1, 65));
for k = 1:n_osc_modes
    fprintf('  %-6d  %10.2f  %10.4f  %10.2f  %10.1f  %8s\n', ...
        k, freq_hz(k), damping(k), tau(k)*1e3, settling(k)*1e3, mode_labels{k});
end

% Print real eigenvalues (overdamped modes)
if n_real_modes > 0
    fprintf('\n  Real eigenvalues (overdamped/integrator modes):\n');
    fprintf('  %-6s  %12s  %10s\n', 'Index', 'lambda', 'tau [ms]');
    fprintf('  %s\n', repmat('-', 1, 35));
    for k = 1:n_real_modes
        lam = eigs_real(k);
        if lam == 0
            fprintf('  %-6d  %12.4f  %10s\n', k, lam, 'inf');
        else
            fprintf('  %-6d  %12.4f  %10.2f\n', k, lam, -1/lam*1e3);
        end
    end
end

% Derived hyperparameters
f_max = max(freq_hz);
fs_nyquist = 2 * f_max;
fs_rk4     = 20 * f_max / up_sample;  % effective dt = Ts/up_sample

% nf: capture settling of slowest augmented mode
idx_msd = strcmp(mode_labels, 'MSD');
if any(idx_msd)
    tau_msd      = max(tau(idx_msd));
    settling_msd = 5 * tau_msd;
    T_msd        = 1 / min(freq_hz(idx_msd));  % period of MSD mode
else
    warning('No MSD mode identified. Using slowest mode overall.');
    tau_msd      = max(tau);
    settling_msd = 5 * tau_msd;
    T_msd        = 1 / min(freq_hz);
end

fs_recommended = max(fs_nyquist, fs_rk4);
nf_derived     = ceil(settling_msd * fs_current);
nanb_derived   = ceil(2 * T_msd * fs_current);

fprintf('\n  Derived hyperparameter bounds:\n');
fprintf('    f_max (MSD)      = %.1f Hz\n', f_max);
fprintf('    fs_nyquist       = %.0f Hz (2 * f_max)\n', fs_nyquist);
fprintf('    fs_rk4 (20x/up)  = %.0f Hz (20*f_max / %d)\n', fs_rk4, up_sample);
fprintf('    fs_recommended   = %.0f Hz\n', fs_recommended);
fprintf('    tau_msd          = %.2f ms\n', tau_msd*1e3);
fprintf('    settling_msd     = %.1f ms (5*tau)\n', settling_msd*1e3);
fprintf('    T_msd            = %.2f ms (1/f_msd)\n', T_msd*1e3);
fprintf('    nf_derived       = %d samples (%.1f ms at %d Hz)\n', ...
    nf_derived, nf_derived/fs_current*1e3, fs_current);
fprintf('    na_nb_derived    = %d samples (%.1f ms at %d Hz)\n', ...
    nanb_derived, nanb_derived/fs_current*1e3, fs_current);

%% ========================================================================
%  SECTION 1 PLOT: Pole map
%  ========================================================================
figure('Position', [50 50 700 500]);
hold on;

% Baseline poles
plot(real(eigs_base), imag(eigs_base), 'ko', 'MarkerSize', 8, ...
    'LineWidth', 1.5, 'DisplayName', 'Baseline (6-state)');

% Augmented poles
plot(real(eigs_aug), imag(eigs_aug), 'bx', 'MarkerSize', 10, ...
    'LineWidth', 2, 'DisplayName', 'Augmented (8-state)');

grid on; axis equal;
xlabel('Re(\lambda) [rad/s]');
ylabel('Im(\lambda) [rad/s]');
title(sprintf('Continuous-time poles  (Y_{op} = %.1f m, f_a = %d Hz)', Y_op, fa));
legend('Location', 'best');

% Add frequency circles for reference
theta_circ = linspace(0, 2*pi, 200);
for f_ref = [10, 50, 100, 150]
    w_ref = 2*pi*f_ref;
    plot(w_ref*cos(theta_circ), w_ref*sin(theta_circ), ':', ...
        'Color', [0.7 0.7 0.7], 'HandleVisibility', 'off');
    text(w_ref*0.72, w_ref*0.72, sprintf('%d Hz', f_ref), ...
        'Color', [0.5 0.5 0.5], 'FontSize', 8);
end

saveas(gcf, 'poles_augmented.png');

%% ========================================================================
%  SECTION 2: FRF COMPARISON
%  ========================================================================
fprintf('\n');
fprintf('================================================================\n');
fprintf('  SECTION 2: FRF COMPARISON\n');
fprintf('================================================================\n\n');

freq_hz_frf = logspace(-1, log10(1000), 5000);
w           = 2*pi * freq_hz_frf;

H_aug  = freqresp(sys_aug,  w);   % (3, 3, N)
H_base = freqresp(sys_base, w);   % (3, 3, N)

% MSD power contribution per I/O channel: integral |H_aug - H_base|^2 df
power_delta = zeros(3, 3);
for iy = 1:3
    for iu = 1:3
        dH = squeeze(abs(H_aug(iy, iu, :)) - abs(H_base(iy, iu, :)));
        power_delta(iy, iu) = trapz(freq_hz_frf, dH.^2);
    end
end

% Normalize: fraction of total output power per channel
power_base = zeros(3, 3);
for iy = 1:3
    for iu = 1:3
        power_base(iy, iu) = trapz(freq_hz_frf, squeeze(abs(H_base(iy, iu, :))).^2);
    end
end
power_frac = power_delta ./ power_base;

fprintf('  MSD power contribution (integral |H_aug - H_base|^2 / |H_base|^2):\n');
fprintf('  %12s', '');
for iu = 1:3, fprintf('  %12s', in_names{iu}); end
fprintf('\n');
for iy = 1:3
    fprintf('  %12s', out_names{iy});
    for iu = 1:3
        fprintf('  %12.2e', power_frac(iy, iu));
    end
    fprintf('\n');
end

[max_frac, max_idx] = max(power_frac(:));
[max_iy, max_iu] = ind2sub([3,3], max_idx);
fprintf('\n  Most sensitive channel: %s / %s (%.2e)\n', ...
    out_names{max_iy}, in_names{max_iu}, max_frac);

%% Section 2 Plot: 3x3 FRF overlay
figure('Position', [100 100 1200 800]);
tiledlayout(3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

for iy = 1:3
    for iu = 1:3
        nexttile; hold on;
        plot(freq_hz_frf, 20*log10(squeeze(abs(H_base(iy,iu,:)))), 'k', ...
            'LineWidth', 1.2, 'DisplayName', 'baseline');
        plot(freq_hz_frf, 20*log10(squeeze(abs(H_aug(iy,iu,:)))), 'b', ...
            'LineWidth', 1.0, 'DisplayName', 'augmented');
        xline(fa, 'r--', 'LineWidth', 0.8, 'HandleVisibility', 'off');
        set(gca, 'XScale', 'log'); grid on;
        xlim([1 500]);
        ylabel('dB');
        title(sprintf('%s / %s', out_names{iy}, in_names{iu}));
        if iy == 1 && iu == 1, legend('Location', 'southwest'); end
    end
end
sgtitle(sprintf('FRF: baseline vs augmented  (f_a = %d Hz, m_a = %d%%)', ...
    fa, round(ma_frac*100)));
saveas(gcf, 'frf_3x3_overlay.png');

%% Section 2 Plot: 3x3 FRF ratio
figure('Position', [150 100 1200 800]);
tiledlayout(3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

for iy = 1:3
    for iu = 1:3
        nexttile; hold on;
        h_b = squeeze(abs(H_base(iy,iu,:)));
        h_a = squeeze(abs(H_aug(iy,iu,:)));
        ratio_db = 20*log10(h_a ./ h_b);
        plot(freq_hz_frf, ratio_db, 'b', 'LineWidth', 1.2);
        yline(0, 'k:', 'LineWidth', 0.5);
        xline(fa, 'r--', 'LineWidth', 0.8);
        set(gca, 'XScale', 'log'); grid on;
        xlim([1 500]); ylabel('dB');
        title(sprintf('%s / %s', out_names{iy}, in_names{iu}));
    end
end
sgtitle('|H_{aug}| / |H_{base}|  [dB]   (0 dB = no change)');
saveas(gcf, 'frf_ratio_3x3.png');

%% Section 2 Plot: Y/F_Y zoom
figure('Position', [200 150 800 400]);
hold on;
h_b_yfy = squeeze(abs(H_base(3,3,:)));
h_a_yfy = squeeze(abs(H_aug(3,3,:)));
plot(freq_hz_frf, 20*log10(h_b_yfy), 'k', 'LineWidth', 1.5, 'DisplayName', 'baseline');
plot(freq_hz_frf, 20*log10(h_a_yfy), 'b', 'LineWidth', 1.5, 'DisplayName', 'augmented');
xline(fa, 'r--', sprintf('f_a = %d Hz', fa), 'LineWidth', 1, ...
    'LabelVerticalAlignment', 'bottom');
set(gca, 'XScale', 'log'); grid on;
xlim([10 500]); ylabel('Magnitude [dB]'); xlabel('Frequency [Hz]');
legend('Location', 'southwest');
title(sprintf('Y / F_Y  (m_a = %d%% = %.2f kg)', round(ma_frac*100), ma));
saveas(gcf, 'frf_yfy_zoom.png');

%% ========================================================================
%  SECTION 3: OBSERVABILITY ANALYSIS
%  ========================================================================
fprintf('\n');
fprintf('================================================================\n');
fprintf('  SECTION 3: OBSERVABILITY ANALYSIS\n');
fprintf('================================================================\n\n');

% Discretize at training sample rate
Ts = 1 / fs_current;
sys_aug_log = ss(A_aug, B_aug, C_aug, zeros(3,3));
sys_d = c2d(sys_aug_log, Ts, 'zoh');
A_d = sys_d.A;  C_d = sys_d.C;

nx = 8;

% 3a. Observability matrix rank
O = obsv(A_d, C_d);
obs_rank = rank(O);
fprintf('  Observability matrix rank: %d / %d', obs_rank, nx);
if obs_rank == nx
    fprintf('  -> FULLY OBSERVABLE\n');
else
    fprintf('  -> RANK DEFICIENT (unobservable subspace dim = %d)\n', nx - obs_rank);
end

% 3b. PBH test per eigenvalue
eigs_d = eig(A_d);
pbh_sigma = zeros(nx, 1);
for k = 1:nx
    PBH_mat = [A_d - eigs_d(k)*eye(nx); C_d];
    s_vals = svd(PBH_mat);
    pbh_sigma(k) = s_vals(end);  % minimum singular value
end

% Map discrete eigenvalues to continuous frequencies for labeling
eigs_ct = log(eigs_d) / Ts;  % continuous-time equivalent
freq_ct = abs(imag(eigs_ct)) / (2*pi);

fprintf('\n  PBH test (per eigenvalue of A_d):\n');
fprintf('  %-6s  %12s  %12s  %12s  %10s\n', ...
    'Mode', 'Re(z)', 'Im(z)', 'f_ct [Hz]', 'sigma_min');
fprintf('  %s\n', repmat('-', 1, 60));
for k = 1:nx
    fprintf('  %-6d  %12.6f  %12.6f  %12.2f  %10.4e\n', ...
        k, real(eigs_d(k)), imag(eigs_d(k)), freq_ct(k), pbh_sigma(k));
end

pbh_threshold = nx * eps(norm(A_d, 1));
if min(pbh_sigma) > pbh_threshold
    fprintf('\n  PBH PASS: min(sigma_min) = %.4e > threshold %.4e\n', ...
        min(pbh_sigma), pbh_threshold);
else
    fprintf('\n  PBH WARN: min(sigma_min) = %.4e <= threshold %.4e\n', ...
        min(pbh_sigma), pbh_threshold);
end

% 3c. Per-mode observability: |C*v_i| for each eigenvector
[V, D_eig] = eig(A_d);
mode_obs = zeros(nx, 1);
for k = 1:nx
    mode_obs(k) = norm(C_d * V(:, k));
end

fprintf('\n  Per-mode observability |C*v_i|:\n');
fprintf('  %-6s  %12s  %12s\n', 'Mode', 'f_ct [Hz]', '|C*v|');
fprintf('  %s\n', repmat('-', 1, 35));
for k = 1:nx
    fprintf('  %-6d  %12.2f  %12.4e\n', k, freq_ct(k), mode_obs(k));
end

[min_obs, min_idx] = min(mode_obs);
fprintf('\n  Least observable mode: %d (f = %.1f Hz, |C*v| = %.4e)\n', ...
    min_idx, freq_ct(min_idx), min_obs);

% 3d. Observability Gramian
Wo = dlyap(A_d', C_d' * C_d);
wo_eigs = sort(real(eig(Wo)));  % should be real and positive
obs_cond = wo_eigs(end) / wo_eigs(1);

fprintf('\n  Observability Gramian eigenvalues:\n');
for k = 1:nx
    fprintf('    lambda_%d = %.4e\n', k, wo_eigs(k));
end
fprintf('  Condition number (max/min): %.2e\n', obs_cond);
if wo_eigs(1) < 0
    fprintf('  WARNING: negative Gramian eigenvalue, system may be marginally observable\n');
end

%% Section 3 Plot: PBH sigma per mode
figure('Position', [250 150 700 400]);
bar(pbh_sigma);
hold on;
yline(pbh_threshold, 'r--', 'threshold', 'LineWidth', 1);
xlabel('Eigenvalue index');
ylabel('\sigma_{min}');
title('PBH observability test: min singular value per eigenvalue');
% Add frequency labels
xtick_labels = cell(nx, 1);
for k = 1:nx
    xtick_labels{k} = sprintf('%.0f Hz', freq_ct(k));
end
set(gca, 'XTickLabel', xtick_labels);
xtickangle(45);
grid on;
saveas(gcf, 'observability_pbh.png');

%% Section 3 Plot: Gramian eigenvalues
figure('Position', [300 150 600 400]);
bar(wo_eigs);
xlabel('Direction index');
ylabel('Gramian eigenvalue');
title('Observability Gramian eigenvalues (larger = more observable)');
set(gca, 'YScale', 'log');
grid on;
saveas(gcf, 'observability_gramian.png');

%% ========================================================================
%  SECTION 4: SUMMARY TABLE
%  ========================================================================
fprintf('\n');
fprintf('================================================================\n');
fprintf('  SECTION 4: SUMMARY TABLE\n');
fprintf('================================================================\n\n');

% Status logic
if fs_current >= fs_recommended
    fs_status = 'OK';
else
    fs_status = 'INCREASE';
end
if nf_current >= nf_derived
    nf_status = 'OK';
else
    nf_status = 'INCREASE';
end
if nanb_current >= nanb_derived
    nanb_status = 'OK';
else
    nanb_status = 'INCREASE';
end

fprintf('  %-10s  %-40s  %15s  %15s  %8s\n', ...
    'Param', 'Physics argument', 'Derived', 'Current', 'Status');
fprintf('  %s\n', repmat('-', 1, 95));
fprintf('  %-10s  %-40s  %12.0f Hz  %12.0f Hz  %8s\n', ...
    'fs', sprintf('f_max=%.0f Hz, Nyquist+RK4(%dx sub)', f_max, up_sample), ...
    fs_recommended, fs_current, fs_status);
fprintf('  %-10s  %-40s  %8d samp  %8d samp  %8s\n', ...
    'nf', sprintf('tau_msd=%.1f ms, 5*tau=%.0f ms', tau_msd*1e3, settling_msd*1e3), ...
    nf_derived, nf_current, nf_status);
fprintf('  %-10s  %-40s  %8d samp  %8d samp  %8s\n', ...
    'na_nb', sprintf('T_msd=%.1f ms, 2 periods', T_msd*1e3), ...
    nanb_derived, nanb_current, nanb_status);

fprintf('\n  Key findings:\n');
fprintf('    MSD mode at %.1f Hz, damping ratio %.4f, settling %.1f ms\n', ...
    freq_hz(idx_msd), damping(idx_msd), settling(idx_msd)*1e3);
if obs_rank == nx
    fprintf('    System is fully observable from [X, Theta, Y] measurements\n');
else
    fprintf('    WARNING: system is NOT fully observable\n');
end
fprintf('    Least observable mode: %.0f Hz (sigma_min = %.4e)\n', ...
    freq_ct(min_idx), min_obs);
fprintf('    Most MSD-sensitive I/O channel: %s / %s\n', ...
    out_names{max_iy}, in_names{max_iu});

fprintf('\n  Note: RK4 convergence test deferred to Python script\n');
fprintf('        (scripts/gantry/verification/verify_rk4_convergence.py)\n');

%% ========================================================================
%  LOCAL FUNCTIONS (ported from frf_augmented_vs_baseline.m lines 193-242)
%  ========================================================================
function [A, B_log, C_log] = build_ss_augmented(Y, da, ...
    m1, m2, mb, mh, Lb, Jb, Jh, d, ...
    cg1, cg2, cb1, cb2, cy, kb1, kb2, ma, ka, ca, L0)
% 8-state augmented gantry with hidden MSD.
% States: [X, Theta, Y, delta_a, dX, dTheta, dY, vdelta_a]

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
