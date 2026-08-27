%% poles_lpv_msd_vs_baseline.m
% Frozen-Y pole loci of the gantry, baseline (6 states) vs hidden-MSD (8 states).
%
% WHY FROZEN POLES
%   The gantry is quasi-LPV: the scheduling variable Y enters only the mass
%   matrix, M(Y) = M0 + M1*Y + M2*Y^2, while C and K are constant. There is no
%   single pole set. The analysis is to freeze Y, form the LTI A(Y), take its
%   eigenvalues, and sweep Y over the stroke, giving eigenvalue loci.
%
%   Two properties make this exact rather than heuristic for this model:
%     1. The LFR realisation of M(Y)^-1 = N(Y)/d(Y) is a STATIC Delta feedback,
%        so it adds no states. Freezing Y and closing that loop reproduces
%        A(Y) = [0, I; -M(Y)\K, -M(Y)\C] exactly.
%     2. Y is a state (q3), so the Jacobian normally picks up a term
%        d(M^-1)/dY * (K*q + C*qdot - f). At an equilibrium that bracket is
%        zero, and the equilibrium manifold here is exactly {f_X = f_Y = 0,
%        Theta = f_Theta/(kb1+kb2), Y free}. So the frozen family IS the
%        linearisation along the equilibrium family. Same at delta_a = 0.
%
%   What does NOT follow: all-frozen-Hurwitz does not imply LPV stability under
%   fast Ydot. That needs a parameter-dependent Lyapunov function and a rate
%   bound. This script is descriptive, not a stability certificate.
%
% MODELS
%   baseline : 6 states, x = [X, Theta, Y, dX, dTheta, dY], payload rigid (mh)
%   MSD      : 8 states, x = [X, Theta, Y, delta_a, dX, dTheta, dY, vdelta_a],
%              payload split into mh_rigid + ma on (ka, ca) at offset L0.
%              M, C4, K4 replicate gantrySystemExtended.m, frozen at delta_a = 0.
%
% PARAMETERS come from gtd_config so there is one source of truth. MA_FRAC is
% 0.10 to match the augmentation-track training data, NOT the gtd_config default.
%
% RUN (from anywhere):
%   run('Matlab-scripts/Augmentation/diagnostics/poles_lpv_msd_vs_baseline.m')
%
% OUTPUT (suffix _ol open loop, _cl closed loop, so the two never overwrite)
%   Matlab-scripts/Augmentation/Matlab-output/poles_lpv_splane_ol.png
%   Matlab-scripts/Augmentation/Matlab-output/poles_lpv_vs_Y_ol.png

clear; clc; close all;

%% ------------------------------------------------------------------------
%  0. Options
% -------------------------------------------------------------------------
MA_FRAC = 0.10;     % augmentation-track training data (memory: ma_frac = 0.10)
USE_CL  = false;    % false = open-loop plant poles, true = frozen-Cfb closed loop
NY      = 41;       % number of frozen operating points
Y_MAX   = 0.400;    % stroke limit [m], cfg.lim.pos_Y

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'data'));
cfg = gtd_config('augmentation', true, MA_FRAC);

out_dir = fullfile(cfg.root, 'Matlab-scripts', 'Augmentation', 'Matlab-output');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% Separate file names per loop configuration, so the two never overwrite.
tag       = ternary(USE_CL, '_cl', '_ol');
png_splane = fullfile(out_dir, ['poles_lpv_splane' tag '.png']);
png_vs_Y   = fullfile(out_dir, ['poles_lpv_vs_Y'   tag '.png']);

Yv = linspace(-Y_MAX, Y_MAX, NY);
k_th = cfg.kb1 + cfg.kb2;

%% ------------------------------------------------------------------------
%  1. Frozen-Y sweep
% -------------------------------------------------------------------------
pol_b = cell(1, NY);  pol_a = cell(1, NY);
det_b = zeros(1, NY); det_a = zeros(1, NY);
w_closed_form = zeros(1, NY);   % analytic structural mode, baseline

if USE_CL
    % Controller designed ONCE at the nominal operating point and frozen for
    % the whole sweep (D-039), matching how the training data was generated.
    Cfb = gtd_build_plant(0, cfg).Cfb;
end

for i = 1:NY
    Y = Yv(i);

    [Ab, Bb, Cb, Mb] = build_baseline(Y, cfg);
    [Aa, Ba, Ca, Ma] = build_msd(Y, 0, cfg);

    det_b(i) = det(Mb);
    det_a(i) = det(Ma);

    % Analytic structural mode of the baseline:
    %   det(K - w^2*M) = w^4 * [ -w^2*det(M) + k*(M11*M33 - M13*M31) ] = 0
    % The cofactor is Y-independent (M11 = alpha, M33 = mh, M13 = 0), so all of
    % the Y dependence of this mode sits in det M(Y).
    w_closed_form(i) = sqrt(k_th * (Mb(1,1)*Mb(3,3) - Mb(1,3)*Mb(3,1)) / det(Mb));

    if USE_CL
        pol_b{i} = cl_poles(Ab, Bb, Cb, Cfb, cfg);
        pol_a{i} = cl_poles(Aa, Ba, Ca, Cfb, cfg);
    else
        pol_b{i} = eig(Ab);
        pol_a{i} = eig(Aa);
    end
end

%% ------------------------------------------------------------------------
%  2. Verification checks
% -------------------------------------------------------------------------
fprintf('=== Verification ===\n');

% (a) M(Y) must equal the M0 + M1*Y + M2*Y^2 decomposition used by the LFR in
%     model_augmentation/systems/gantry_ss.py. If this fails, the MATLAB pole
%     plot and the Python baseline are not the same system.
[M0, M1, M2] = lfr_mass_decomposition(cfg);
res_dec = 0;
for i = 1:NY
    [~, ~, ~, Mb] = build_baseline(Yv(i), cfg);
    res_dec = max(res_dec, max(abs(Mb - (M0 + M1*Yv(i) + M2*Yv(i)^2)), [], 'all'));
end
fprintf('  M(Y) vs LFR M0+M1*Y+M2*Y^2 : max abs residual = %.2e\n', res_dec);

% (b) ma -> 0 must recover the baseline 3x3 blocks exactly.
% Setting ma = 0 makes the 4x4 mass matrix singular by construction (the
% absorber state disappears), so only the 3x3 block is meaningful here.
cfg0 = cfg; cfg0.ma = 0; cfg0.mh_rigid = cfg.mh; cfg0.ka = 0; cfg0.ca = 0;
ws = warning('off', 'MATLAB:singularMatrix');
[~, ~, ~, Ma0] = build_msd(0.3, 0, cfg0);
warning(ws);
[~, ~, ~, Mb3] = build_baseline(0.3, cfg);
fprintf('  ma -> 0 recovery of M(1:3,1:3)  : max abs residual = %.2e\n', ...
    max(abs(Ma0(1:3,1:3) - Mb3), [], 'all'));

% (c) Well-posedness: the rational LFR form needs det M(Y) ~= 0 over the sweep.
fprintf('  min det M(Y) baseline / MSD     : %.4g / %.4g  (must stay > 0)\n', ...
    min(det_b), min(det_a));

% (d) Closed form vs numerical eigenvalue, baseline structural mode.
if ~USE_CL
    res_cf = 0;
    for i = 1:NY
        p  = pol_b{i};
        po = p(imag(p) > 1e-9);
        res_cf = max(res_cf, abs(abs(po(1)) - w_closed_form(i)));
    end
    fprintf('  closed-form w_n vs eig(A(Y))    : max abs error = %.2e rad/s\n', res_cf);
end

%% ------------------------------------------------------------------------
%  3. Mode table at the sweep extremes
% -------------------------------------------------------------------------
fprintf('\n=== Oscillatory modes (%s) ===\n', ternary(USE_CL, 'closed loop', 'open loop'));
for Yq = [-Y_MAX, 0, Y_MAX]
    [~, i] = min(abs(Yv - Yq));
    print_modes(sprintf('Y = %+.3f  baseline', Yv(i)), pol_b{i});
    print_modes(sprintf('Y = %+.3f  MSD     ', Yv(i)), pol_a{i});
end

% Relative shift of the lowest oscillatory mode across the stroke.
f_b = arrayfun(@(i) osc_freq(pol_b{i}, 1), 1:NY);
f_a = arrayfun(@(i) osc_freq(pol_a{i}, 1), 1:NY);
fprintf('\nMode 1 across the stroke: baseline %.3f -> %.3f Hz (%.1f%% spread)\n', ...
    min(f_b), max(f_b), 100*(max(f_b)-min(f_b))/max(f_b));
if any(cellfun(@(p) sum(imag(p) > 1e-9), pol_a) >= 2)
    f_a2 = arrayfun(@(i) osc_freq(pol_a{i}, 2), 1:NY);
    fprintf('Mode 2 across the stroke: MSD      %.3f -> %.3f Hz (%.1f%% spread)\n', ...
        min(f_a2), max(f_a2), 100*(max(f_a2)-min(f_a2))/max(f_a2));
end

%% ------------------------------------------------------------------------
%  4. Figure 1: s-plane loci
% -------------------------------------------------------------------------
col_b = [0.00, 0.45, 0.70];
col_a = [0.84, 0.19, 0.15];

fig1 = figure('Name', 'Frozen-Y pole loci', 'Position', [80, 80, 1100, 470]);

subplot(1, 2, 1); hold on; grid on;
plot_loci(pol_b, col_b, 'o');
plot_loci(pol_a, col_a, 'x');
xlabel('Re(s)  [1/s]'); ylabel('Im(s)  [rad/s]');
title('All poles');
legend({'baseline (6 states)', 'MSD (8 states)'}, 'Location', 'east');

subplot(1, 2, 2); hold on; grid on;
plot_loci(pol_b, col_b, 'o');
plot_loci(pol_a, col_a, 'x');
xlabel('Re(s)  [1/s]'); ylabel('Im(s)  [rad/s]');
title(sprintf('Zoom: structural mode, Y = %+.2f .. %+.2f m', -Y_MAX, Y_MAX));
ylim([0, 1.25*2*pi*max(f_b)]);
xlim([-1.6*max(abs(real(cell2mat(pol_b(:)'))), [], 'all'), 1]);

sgtitle(sprintf(['Frozen-Y pole loci, gantry baseline vs hidden MSD ' ...
    '(ma_{frac} = %.2f, %s)'], MA_FRAC, ternary(USE_CL, 'closed loop', 'open loop')));
exportgraphics(fig1, png_splane, 'Resolution', 150);

%% ------------------------------------------------------------------------
%  5. Figure 2: frequency, damping and det M against Y
%     This is the panel that actually shows the LPV behaviour: the s-plane
%     view is dominated by two fixed poles at the origin.
% -------------------------------------------------------------------------
[F_b, Z_b] = mode_tracks(pol_b);
[F_a, Z_a] = mode_tracks(pol_a);

fig2 = figure('Name', 'Modes vs Y', 'Position', [80, 80, 1300, 400]);

% Modes sit at 5 Hz and 158 Hz, so absolute frequencies on one axis show
% nothing. Plot each mode relative to its own value at Y = 0; that is the
% quantity the LPV baseline has to reproduce.
i0  = (NY+1)/2;
subplot(1, 3, 1); hold on; grid on;
lg  = {};
plot(Yv, 100*(F_b(:,1)/F_b(i0,1) - 1), '-', 'Color', col_b, 'LineWidth', 1.6);
lg{end+1} = sprintf('baseline, %.2f Hz at Y=0', F_b(i0,1));
nm_a = size(F_a, 2);
for j = 1:nm_a
    sh = 0.6 * (j-1) / max(1, nm_a-1);      % shade later modes towards white
    plot(Yv, 100*(F_a(:,j)/F_a(i0,j) - 1), '--', ...
        'Color', col_a*(1-sh) + sh*[1 1 1], 'LineWidth', 1.6);
    lg{end+1} = sprintf('MSD mode %d, %.2f Hz at Y=0', j, F_a(i0,j));
end
xlabel('Y  [m]'); ylabel('\Deltaf_n relative to Y = 0  [%]');
title('Modal frequency shift');
legend(lg, 'Location', 'south', 'FontSize', 8);

subplot(1, 3, 2); hold on; grid on;
plot(Yv, Z_b, '-',  'Color', col_b, 'LineWidth', 1.6);
plot(Yv, Z_a, '--', 'Color', col_a, 'LineWidth', 1.6);
xlabel('Y  [m]'); ylabel('\zeta  [-]');
title('Modal damping');

subplot(1, 3, 3); hold on; grid on;
plot(Yv, det_b/det_b((NY+1)/2), '-',  'Color', col_b, 'LineWidth', 1.6);
plot(Yv, det_a/det_a((NY+1)/2), '--', 'Color', col_a, 'LineWidth', 1.6);
xlabel('Y  [m]'); ylabel('det M(Y) / det M(0)  [-]');
title('Well-posedness of M(Y)');

sgtitle('Frozen-Y modal properties: all Y dependence enters through M(Y)');
exportgraphics(fig2, png_vs_Y, 'Resolution', 150);

fprintf('\nSaved:\n  %s\n  %s\n', png_splane, png_vs_Y);

%% ========================================================================
%  Local functions
% =========================================================================
function [A, B, C, M] = build_baseline(Y, cfg)
% 6-state baseline at frozen Y. Rigid payload, full mh.
    M = [cfg.m1+cfg.m2+cfg.mb+cfg.mh,      (cfg.m1-cfg.m2)*cfg.Lb/2 - cfg.mh*Y,   0;
         (cfg.m1-cfg.m2)*cfg.Lb/2 - cfg.mh*Y, ...
             cfg.Jb+cfg.Jh + (cfg.m1+cfg.m2)*cfg.Lb^2/4 + cfg.mh*cfg.d^2 + cfg.mh*Y^2, ...
             -cfg.mh*cfg.d;
         0,  -cfg.mh*cfg.d,  cfg.mh];

    A = [zeros(3), eye(3); -M\cfg.K, -M\cfg.C_damp];
    B = [zeros(3); M\eye(3)] * cfg.P;          % stage forces -> xdot
    C = [cfg.P.', zeros(3)];                   % state -> stage positions
end

function [A, B, C, M] = build_msd(Y, da, cfg)
% 8-state hidden-MSD model at frozen (Y, delta_a).
% M, C4, K4 replicate gantrySystemExtended.m; cfg.mh_rigid is the rigid payload.
    mh = cfg.mh_rigid; ma = cfg.ma; Lb = cfg.Lb; d = cfg.d; L0 = cfg.L0;
    s  = Y + L0 + da;

    M = [cfg.m1+cfg.m2+cfg.mb+mh+ma,  (cfg.m1-cfg.m2)*Lb/2 - mh*Y - ma*s,  0,  0;
         (cfg.m1-cfg.m2)*Lb/2 - mh*Y - ma*s, ...
             cfg.Jb+cfg.Jh + (cfg.m1+cfg.m2)*Lb^2/4 + (mh+ma)*d^2 + mh*Y^2 + ma*s^2, ...
             -(mh+ma)*d,  -ma*d;
         0,  -(mh+ma)*d,  mh+ma,  ma;
         0,  -ma*d,       ma,     ma];

    C4 = blkdiag(cfg.C_damp, cfg.ca);
    K4 = blkdiag(cfg.K,      cfg.ka);

    A = [zeros(4), eye(4); -M\K4, -M\C4];
    B = [zeros(4,3); M \ [eye(3); zeros(1,3)]] * cfg.P;
    C = [cfg.P.', zeros(3,1), zeros(3,4)];
end

function [M0, M1, M2] = lfr_mass_decomposition(cfg)
% M(Y) = M0 + M1*Y + M2*Y^2, as used by the LFR in gantry_ss.py.
    M0 = zeros(3);
    M0(1,1) = cfg.m1+cfg.m2+cfg.mb+cfg.mh;
    M0(1,2) = (cfg.m1-cfg.m2)*cfg.Lb/2;  M0(2,1) = M0(1,2);
    M0(2,2) = cfg.Jb+cfg.Jh + (cfg.m1+cfg.m2)*cfg.Lb^2/4 + cfg.mh*cfg.d^2;
    M0(2,3) = -cfg.mh*cfg.d;             M0(3,2) = M0(2,3);
    M0(3,3) = cfg.mh;

    M1 = zeros(3); M1(1,2) = -cfg.mh; M1(2,1) = -cfg.mh;
    M2 = zeros(3); M2(2,2) =  cfg.mh;
end

function p = cl_poles(A, B, C, Cfb, cfg)
% Frozen-Y closed-loop poles, mapped back to continuous time.
% The controller is discrete, so the loop is closed in discrete time and the
% poles are mapped by s = log(z)/ts. Poles at z = 0 (deadbeat modes introduced
% by the discretisation) have no continuous-time image and are dropped.
    G  = c2d(ss(A, B, C, zeros(3)), cfg.ts, 'zoh');
    z  = eig(feedback(G, Cfb));
    z  = z(abs(z) > 1e-12);
    p  = log(z) / cfg.ts;
end

function f = osc_freq(p, k)
% Natural frequency [Hz] of the k-th oscillatory mode, ordered by frequency.
    po = sort(p(imag(p) > 1e-9), 'ComparisonMethod', 'abs');
    if numel(po) < k, f = NaN; else, f = abs(po(k)) / (2*pi); end
end

function [F, Z] = mode_tracks(pol)
% Frequency [Hz] and damping of every oscillatory mode, per operating point.
% Modes are ordered by frequency at each Y, which is unambiguous here because
% the modes are well separated over the whole stroke.
    NY = numel(pol);
    nm = max(cellfun(@(p) sum(imag(p) > 1e-9), pol));
    F  = nan(NY, nm); Z = nan(NY, nm);
    for i = 1:NY
        po = sort(pol{i}(imag(pol{i}) > 1e-9), 'ComparisonMethod', 'abs');
        F(i, 1:numel(po)) = abs(po) / (2*pi);
        Z(i, 1:numel(po)) = -real(po) ./ abs(po);
    end
end

function plot_loci(pol, col, mk)
% Scatter every frozen-Y pole, shaded light (Y = -Ymax) to dark (Y = +Ymax).
    NY = numel(pol);
    for i = 1:NY
        w = 0.25 + 0.75*(i-1)/(NY-1);
        plot(real(pol{i}), imag(pol{i}), mk, ...
            'Color', col*w + (1-w)*[1 1 1], 'MarkerSize', 5, 'LineWidth', 1.1, ...
            'HandleVisibility', ternary(i == 1, 'on', 'off'));
    end
end

function print_modes(label, p)
    po = sort(p(imag(p) > 1e-9), 'ComparisonMethod', 'abs');
    fprintf('  %s : ', label);
    for j = 1:numel(po)
        fprintf('%7.2f Hz (zeta = %.4f)   ', abs(po(j))/(2*pi), -real(po(j))/abs(po(j)));
    end
    fprintf('\n');
end

function out = ternary(c, a, b)
    if c, out = a; else, out = b; end
end
