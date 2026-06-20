%% multisine_frequency_range_baseline.m
% Determine multisine frequency range from the baseline gantry plant model.
%
% Builds the 6-state baseline state-space at a frozen operating point, then:
%   1. Eigendecomposition: modal frequencies and damping ratios
%   2. PBH observability test per eigenvalue
%   3. Modal participation factors: which I/O channels couple to which modes
%   4. Frequency range recommendation (f_low, f_high)
%   5. Bode plot verification
%   6. Sensitivity of modal frequencies to Y operating point
%
% State:  x = [X, Theta, Y, dX, dTheta, dY]  (6)
% Input:  u = [F_X1, F_X2, F_Y]  (stage forces, 3)
% Output: y = [X1, X2, Y]        (stage positions, 3)
%
% Run from repo root:
%   run('Matlab-scripts/Augmentation/diagnostics/multisine_frequency_range_baseline.m')

clear; clc; close all;

%% 1. Parameters (from main.m / gantry_ss.py)
mb=22.8; mh=10.1; m1=10.2; m2=10.7; Jb=1.0; Jh=0.05;
cg1=14.5; cg2=20.3; cy=10; cb1=9; cb2=9;
kb1=1987.5; kb2=1987.5; Lb=0.725; d=0.1;

% Coordinate transform: logical <-> stage
P = [1, 1, 0; Lb/2, -Lb/2, 0; 0, 0, 1];

Y_op = 0.3;      % frozen Y position [m]

%% 2. Build baseline state-space at operating point
[A, B_log, C_log] = build_ss_baseline(Y_op, ...
    m1, m2, mb, mh, Lb, Jb, Jh, d, ...
    cg1, cg2, cb1, cb2, cy, kb1, kb2);

B_stg = B_log * P;       % (6x3) stage force inputs:    u_logical = P * u_stage
C_stg = P' * C_log;      % (3x6) stage position outputs: y_stage = P' * y_logical
n = size(A, 1);           % 6

%% 3. Eigenvalue decomposition
[V, D_eig] = eig(A);
lambdas = diag(D_eig);

fprintf('\n=== Modal frequencies and damping (Y = %.2f m) ===\n', Y_op);
fprintf('%-5s  %10s  %8s  %10s  %s\n', ...
    'Mode', 'f_n [Hz]', 'zeta', 'BW_hp [Hz]', 'Type');
fprintf('%s\n', repmat('-', 1, 60));

modes = struct('idx',{}, 'fn',{}, 'zeta',{}, 'bw',{}, 'osc',{});
seen = false(n, 1);
mi = 0;

for k = 1:n
    if seen(k), continue; end
    lam = lambdas(k);
    cj = find(abs(lambdas - conj(lam)) < 1e-8 & (1:n)' ~= k, 1);

    mi = mi + 1;
    if ~isempty(cj) && abs(imag(lam)) > 1e-6
        % Oscillatory mode (complex conjugate pair)
        seen([k, cj]) = true;
        fn = abs(imag(lam)) / (2*pi);
        z  = -real(lam) / abs(lam);
        bw = 2 * z * fn;
        modes(mi) = struct('idx',k, 'fn',fn, 'zeta',z, 'bw',bw, 'osc',true);
        fprintf('  %2d   %10.2f  %8.4f  %10.2f  oscillatory\n', mi, fn, z, bw);
    else
        % Overdamped mode (real pole)
        seen(k) = true;
        tau = -1 / real(lam);
        modes(mi) = struct('idx',k, 'fn',abs(real(lam))/(2*pi), ...
            'zeta',Inf, 'bw',Inf, 'osc',false);
        fprintf('  %2d   %10.2f  %8s  %10s  overdamped (tau=%.4f s)\n', ...
            mi, modes(mi).fn, 'Inf', '-', tau);
    end
end

%% 4. PBH observability test
fprintf('\n=== PBH observability test ===\n');
tol = n * eps(norm(A, 1));
all_pass = true;

for k = 1:n
    s_min = min(svd([A - lambdas(k)*eye(n); C_stg]));
    if s_min <= tol
        fprintf('  FAIL at lambda(%d) = %.4f%+.4fj  sigma_min=%.2e\n', ...
            k, real(lambdas(k)), imag(lambdas(k)), s_min);
        all_pass = false;
    end
end
if all_pass
    fprintf('  PASS: all %d eigenvalues observable from [X1, X2, Y].\n', n);
end

%% 5. Modal participation factors
modal_B = V \ B_stg;    % (6x3) controllability: rows=modes, cols=[F_X1,F_X2,F_Y]
modal_C = C_stg * V;    % (3x6) observability:   rows=[X1,X2,Y], cols=modes

in_names  = {'F_X1', 'F_X2', 'F_Y'};
out_names = {'X1', 'X2', 'Y'};

fprintf('\n=== Controllability: |V\\B| (which input excites which mode) ===\n');
fprintf('  Normalized per mode (1.0 = strongest channel for that mode)\n');
fprintf('%-5s %8s   %8s %8s %8s\n', 'Mode', 'f [Hz]', in_names{:});
fprintf('%s\n', repmat('-', 1, 45));
for m = 1:length(modes)
    row = abs(modal_B(modes(m).idx, :));
    rn  = row / max(row);
    fprintf('  %2d  %7.1f   %8.4f %8.4f %8.4f\n', m, modes(m).fn, rn);
end

fprintf('\n=== Observability: |C*V| (which output sees which mode) ===\n');
fprintf('  Normalized per mode (1.0 = strongest channel for that mode)\n');
fprintf('%-5s %8s   %8s %8s %8s\n', 'Mode', 'f [Hz]', out_names{:});
fprintf('%s\n', repmat('-', 1, 45));
for m = 1:length(modes)
    col = abs(modal_C(:, modes(m).idx));
    cn  = col / max(col);
    fprintf('  %2d  %7.1f   %8.4f %8.4f %8.4f\n', m, modes(m).fn, cn);
end

%% 6. Frequency range recommendation
osc_idx = find([modes.osc]);
osc_fn  = [modes(osc_idx).fn];
osc_bw  = [modes(osc_idx).bw];
osc_z   = [modes(osc_idx).zeta];

fn_lo = min(osc_fn);   bw_lo = max(osc_bw(osc_fn == fn_lo));
fn_hi = max(osc_fn);   bw_hi = max(osc_bw(osc_fn == fn_hi));
z_hi  = max(osc_z(osc_fn == fn_hi));

% f_low: fundamental frequency of 1-second period multisine
f_low  = 1;

% f_high: 2.5 * half-power BW above highest mode = fn*(1 + 5*zeta).
% At this point the 2nd-order resonance contribution is negligible.
f_high = ceil(fn_hi + 2.5 * bw_hi);

% Anti-resonance from transmission zeros of Y/F_Y
sys_YFY = ss(A, B_stg(:,3), C_stg(3,:), 0);
z_YFY = tzero(sys_YFY);
z_osc = z_YFY(abs(imag(z_YFY)) > 1);
f_antires = sort(abs(imag(z_osc)) / (2*pi));
f_antires = unique(round(f_antires, 2));

fprintf('\n=== Frequency range recommendation ===\n');
fprintf('  Lowest oscillatory mode:   %.2f Hz (BW = %.2f Hz)\n', fn_lo, bw_lo);
fprintf('  Highest oscillatory mode:  %.2f Hz (BW = %.2f Hz, zeta = %.4f)\n', fn_hi, bw_hi, z_hi);
if ~isempty(f_antires)
    fprintf('  Anti-resonance (Y/F_Y):    ');
    fprintf('%.2f Hz  ', f_antires);
    fprintf('\n');
end
fprintf('  f_high margin: fn + 2.5*BW = fn*(1+5*zeta) = %.1f Hz\n', fn_hi + 2.5*bw_hi);
fprintf('  Recommended:  f_low = %d Hz,  f_high = %d Hz\n', f_low, f_high);

%% 7. FRF verification (magnitude + phase)
sys = ss(A, B_stg, C_stg, zeros(3));
sys.InputName  = in_names;
sys.OutputName = out_names;

freq_hz = logspace(-1, log10(50), 2000);
H = freqresp(sys, freq_hz * 2*pi);

figure('Position', [50 50 1200 950]);
tiledlayout(6, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

mag_axes = gobjects(3, 3);
ph_axes  = gobjects(3, 3);

for iy = 1:3
    for iu = 1:3
        % Magnitude (rows 1, 3, 5)
        ax_m = nexttile(2*(iy-1)*3 + iu); hold on
        plot(freq_hz, 20*log10(squeeze(abs(H(iy,iu,:)))), 'b', 'LineWidth', 1.2);
        for m = 1:length(modes)
            if modes(m).osc
                xline(modes(m).fn, 'r--', sprintf('%.0f', modes(m).fn), ...
                    'FontSize', 7, 'LineWidth', 0.8, ...
                    'LabelVerticalAlignment', 'bottom');
            end
        end
        for j = 1:length(f_antires)
            xline(f_antires(j), 'k:', sprintf('AR %.0f', f_antires(j)), ...
                'FontSize', 6, 'LineWidth', 0.8, ...
                'LabelVerticalAlignment', 'top');
        end
        set(gca, 'XScale', 'log'); grid on; xlim([0.1 50]);
        yl = ylim;
        patch([f_low f_high f_high f_low], [yl(1) yl(1) yl(2) yl(2)], ...
            [0.8 1 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        ylim(yl);
        title(sprintf('%s / %s', out_names{iy}, in_names{iu}));
        if iu == 1, ylabel('Mag [dB]'); end
        mag_axes(iy, iu) = ax_m;

        % Phase (rows 2, 4, 6)
        ax_p = nexttile((2*iy-1)*3 + iu); hold on
        ph = rad2deg(unwrap(angle(squeeze(H(iy,iu,:)))));
        plot(freq_hz, ph, 'b', 'LineWidth', 1.2);
        for m = 1:length(modes)
            if modes(m).osc
                xline(modes(m).fn, 'r--', 'LineWidth', 0.8);
            end
        end
        for j = 1:length(f_antires)
            xline(f_antires(j), 'k:', 'LineWidth', 0.8);
        end
        set(gca, 'XScale', 'log'); grid on; xlim([0.1 50]);
        yl = ylim;
        patch([f_low f_high f_high f_low], [yl(1) yl(1) yl(2) yl(2)], ...
            [0.8 1 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        ylim(yl);
        if iu == 1, ylabel('Phase [deg]'); end
        if iy == 3, xlabel('Frequency [Hz]'); end
        ph_axes(iy, iu) = ax_p;
    end
end

linkaxes([mag_axes(:); ph_axes(:)], 'x');
sgtitle(sprintf(['Baseline plant FRF  |  Y = %.2f m  |  ' ...
    'f_{low} = %d Hz   f_{high} = %d Hz  (shaded)'], Y_op, f_low, f_high));

%% 8. Sensitivity to Y operating point
fprintf('\n=== Modal frequencies vs Y operating point ===\n');
Y_sweep = [0.0, 0.1, 0.2, 0.3];

% Header: only oscillatory mode columns
osc_labels = arrayfun(@(m) sprintf('f_%d [Hz]', m), osc_idx, 'UniformOutput', false);
fprintf('%-8s', 'Y [m]');
fprintf('%12s', osc_labels{:});
fprintf('\n%s\n', repmat('-', 1, 8 + 12*length(osc_idx)));

for Y_s = Y_sweep
    A_s = build_ss_baseline(Y_s, m1, m2, mb, mh, Lb, Jb, Jh, d, ...
        cg1, cg2, cb1, cb2, cy, kb1, kb2);
    lam_s = eig(A_s);
    % Keep only oscillatory, deduplicate conjugate pairs
    osc_f = sort(unique(round( ...
        abs(imag(lam_s(abs(imag(lam_s)) > 1))) / (2*pi), 2)));
    fprintf('  %-6.1f', Y_s);
    for j = 1:length(osc_f)
        fprintf('%12.2f', osc_f(j));
    end
    fprintf('\n');
end

%% =========================================================================
function [A, B_log, C_log] = build_ss_baseline(Y, ...
    m1, m2, mb, mh, Lb, Jb, Jh, d, ...
    cg1, cg2, cb1, cb2, cy, kb1, kb2)
% Build 6-state baseline gantry state-space at frozen Y.
%
% Returns logical-coordinate I/O:
%   B_log: (6x3) maps [F_X, F_Theta, F_Y] to xdot
%   C_log: (3x6) maps x to [X, Theta, Y]

    M = [m1+m2+mb+mh,              (m1-m2)*Lb/2 - mh*Y,         0;
         (m1-m2)*Lb/2 - mh*Y,  Jb+Jh+(m1+m2)*Lb^2/4 + mh*d^2 + mh*Y^2,  -mh*d;
         0,                                                    -mh*d,       mh];

    C4 = [cg1+cg2,            (cg1-cg2)*Lb/2,                   0;
          (cg1-cg2)*Lb/2,  cb1+cb2+(cg1+cg2)*Lb^2/4,            0;
          0,                0,                                   cy];

    K4 = [0,  0,        0;
          0,  kb1+kb2,  0;
          0,  0,        0];

    A     = [zeros(3), eye(3); -M\K4, -M\C4];
    B_log = [zeros(3); M \ eye(3)];
    C_log = [eye(3), zeros(3)];
end
