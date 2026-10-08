% NOISE_SIM_FIGURE  Encoder-noise model after Jasper's feedback (2026-10-06), test only. Simulation based.
%   Same structure on X1, X2 and Y (order 2):
%       v = g * filter(b, a, w1) + floor * w2,   [b, a] = butter(2, fc / (fs/2), 'low')
%   - floor:  copied from the measured spectrum (median above 5 kHz)
%   - fc, g:  fitted by simulation: v through the closed loop at standstill (e = -S v, lsim), the error
%             analysed with the same estimator as the measurement (pwelch, Hann 2048, 50 % overlap,
%             20 kHz), log spectrum fitted to the log measured spectrum (lsqnonlin), all bins from
%             9.77 Hz, equal weight per decade; for Y, 0.7 to 2 kHz excluded. Fixed noise draws during
%             the fit, so the cost is deterministic.
%   - g:      then set so the simulated error rms over the fitted range equals the measured rms there
%   Closed loop: S = (I + G K1)^-1, truth linearised at rest at Y = 0 (tanh slope), K1.
%   Run from this folder: matlab -batch "noise_sim_figure"

repo = fullfile(fileparts(mfilename('fullpath')), '..', '..');
addpath(fullfile(repo, 'Thesis-writeup', 'Code', 'Data'));
cfg = tdg_config();
fs = cfg.fs;  ts = cfg.ts;  nw = 2048;

T  = load(cfg.noise_target);
fw = T.f(:);  df = fw(2) - fw(1);  f = fw(2:end);
Pm = zeros(numel(f), 3);
for j = 1:3, Pm(:, j) = real(T.P(2:end, j, j)); end

Gd = c2d(tdg_truth_lin(0, cfg, cfg.friction_gain), ts, 'zoh');
plant0 = gtd_build_plant(0, cfg, 1);
S  = feedback(eye(3), Gd * plant0.Cfb);

Tsim = 20;  N = round(Tsim * fs);  keep = round(2 * fs) + 1:N;   % 20 s, first 2 s dropped
t = (0:N - 1)' * ts;
names = {'X1', 'X2', 'Y'};
excl  = {[], [], [700 2000]};                             % [Hz] left out of fit and rms
opts  = optimoptions('lsqnonlin', 'Display', 'off');
lb = log([50, 1e-13]);  ub = log([5000, 1e-7]);           % [fc g]
rng(20261006);  W = randn(N, 6) * sqrt(fs / 2);           % fixed draws, unit one-sided PSD
mask = true(numel(f), 3);  par = zeros(3, 3);  v = zeros(N, 3);
for j = 1:3
    if ~isempty(excl{j}), mask(:, j) = f < excl{j}(1) | f > excl{j}(2); end
    m = mask(:, j);
    Pfloor = median(Pm(f > 5000, j));
    Sj = S(j, j);  w1 = W(:, 2 * j - 1);  w2 = W(:, 2 * j);
    simP = @(p) welch_e(Sj, noise(exp(p(1)), exp(p(2)), Pfloor, w1, w2, fs), t, keep, nw, fs);
    wt = sqrt(1 ./ f(m));                                 % equal weight per decade (bins are linear in f)
    res = @(p) wt .* (log(sel(simP(p), m)) - log(Pm(m, j)));
    tic;  p = lsqnonlin(res, log([1000, 2e-10]), lb, ub, opts);
    fc = exp(p(1));
    % level: simulated error rms over the fitted range equals the measured rms there
    gap = @(lg) sqrt(sum(sel(simP([log(fc), lg]), m)) * df) - sqrt(sum(Pm(m, j)) * df);
    lg = fzero(gap, p(2));
    par(j, :) = [fc, exp(lg), sqrt(Pfloor)];
    v(:, j) = noise(fc, exp(lg), Pfloor, w1, w2, fs);
    fprintf('%-2s fc %5.0f Hz | g fit %.2e -> rms-matched %.2e | floor %.2e m/sqrt(Hz) | %.0f s\n', ...
            names{j}, fc, exp(p(2)), exp(lg), sqrt(Pfloor), toc);
end

% final run: all three axes together through the full 3 x 3 loop
e = -lsim(S, v, t);
[Pe, fe] = pwelch(e(keep, :), hann(nw), nw / 2, nw, fs);  Pe = Pe(2:end, :);
rmsM = sqrt(sum(Pm .* mask) * df);  rmsS = sqrt(sum(Pe .* mask) * df);
for j = 1:3
    fprintf('%-2s rms fitted range: measured %.2e, simulated %.2e | full: measured %.2e, simulated %.2e\n', ...
            names{j}, rmsM(j), rmsS(j), sqrt(sum(Pm(:, j)) * df), sqrt(sum(Pe(:, j)) * df));
end

fig = figure('Visible', 'off', 'Position', [100 100 820 920]);
tl = tiledlayout(3, 1, 'TileSpacing', 'compact');
for j = 1:3
    nexttile;
    if ~isempty(excl{j})
        patch([excl{j}(1) excl{j}(2) excl{j}(2) excl{j}(1)], [1e-12 1e-12 1e-8 1e-8], ...
              [0.92 0.92 0.92], 'EdgeColor', 'none', 'HandleVisibility', 'off'); hold on;
    end
    h = loglog(f, sqrt(Pm(:, j)), 'k', 'LineWidth', 1.3); hold on;
    h(2) = loglog(f, sqrt(Pe(:, j)), 'Color', [0.16 0.47 0.84], 'LineWidth', 1.4);
    set(gca, 'XScale', 'log', 'YScale', 'log');
    lab = {sprintf('measured, rms %.2e m', rmsM(j)), ...
           sprintf('simulated: low-pass %.0f Hz + floor, rms %.2e m', par(j, 1), rmsS(j))};
    if ~isempty(excl{j})
        lab{1} = [lab{1} ' (without 0.7-2 kHz)'];  lab{2} = [lab{2} ' (without 0.7-2 kHz)'];
    end
    xlim([9 1e4]);  ylim([1e-11 5e-10]);  grid on;
    ylabel(sprintf('%s  [m/\\surdHz]', names{j}));
    legend(h, lab, 'Location', 'southwest', 'FontSize', 8);
end
xlabel(tl, 'frequency [Hz]');
title(tl, {'Servo error at standstill: Telica measurement and simulation', ...
           'Simulation at Y = 0: encoder noise = low-pass (order 2) + white floor, closed loop with K_1', ...
           'Fit from 10 Hz (equal weight per decade), rms matched, same spectrum estimator; Y: grey band left out'}, ...
      'FontSize', 9);
exportgraphics(fig, fullfile(fileparts(mfilename('fullpath')), 'noise_sim_jasper.png'), 'Resolution', 150);
fprintf('figure written\n');

function v = noise(fc, g, Pfloor, w1, w2, fs)
% encoder noise: low-pass (order 2) on white noise plus a white floor
    [b, a] = butter(2, fc / (fs / 2), 'low');
    v = g * filter(b, a, w1) + sqrt(Pfloor) * w2;
end

function P = welch_e(Sj, v, t, keep, nw, fs)
% standstill error of one axis (e = -S_jj v) and its Welch spectrum, as the measurement
    e = lsim(Sj, v, t);
    P = pwelch(e(keep), hann(nw), nw / 2, nw, fs);
    P = P(2:end);
end

function y = sel(x, m), y = x(m); end
