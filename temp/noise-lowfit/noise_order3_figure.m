% NOISE_ORDER3_FIGURE  Encoder-noise model after Jasper's feedback (2026-10-06), test only.
%   Same structure on X1, X2 and Y (order 3 in total):
%       v = gL * filter(bL, aL, w1) + g * filter(b, a, w2) + floor * w3
%       [bL, aL] = butter(1, fL / (fs/2), 'low')   low-frequency slope term
%       [b, a]   = butter(2, fc / (fs/2), 'low')   level and roll-off
%   - floor:        copied from the measured spectrum (median above 5 kHz)
%   - fL, gL, fc, g: least-squares fit (lsqnonlin) of log(|S_jj|^2 phi_v) to the log of the measured
%                   standstill error spectrum, all bins from 9.77 Hz, weighted 1/f (equal weight per
%                   decade); for Y, 0.7 to 2 kHz excluded
%   - gL, g:        then scaled by one common factor so the error rms over the fitted range equals
%                   the measured rms there
%   S = (I + G K1)^-1 from the state space (truth at rest, Y = 0, tanh slope) and K1.
%   The figure shows calculated spectra (fine frequency grid); a time simulation (lsim) checks the rms.
%   Run from this folder: matlab -batch "noise_order3_figure"

repo = fullfile(fileparts(mfilename('fullpath')), '..', '..');
addpath(fullfile(repo, 'Thesis-writeup', 'Code', 'Data'));
cfg = tdg_config();
fs = cfg.fs;  ts = cfg.ts;

T  = load(cfg.noise_target);
fw = T.f(:);  df = fw(2) - fw(1);  f = fw(2:end);
Pm = zeros(numel(f), 3);
for j = 1:3, Pm(:, j) = real(T.P(2:end, j, j)); end

Gd = c2d(tdg_truth_lin(0, cfg, cfg.friction_gain), ts, 'zoh');
plant0 = gtd_build_plant(0, cfg, 1);
S  = feedback(eye(3), Gd * plant0.Cfb);
Sf = freqresp(S, 2 * pi * f);
ff = logspace(log10(f(1)), log10(f(end)), 2000)';         % fine grid for the plotted model
Sff = freqresp(S, 2 * pi * ff);

names = {'X1', 'X2', 'Y'};
excl  = {[], [], [700 2000]};                             % [Hz] left out of fit and rms
opts  = optimoptions('lsqnonlin', 'Display', 'off');
lb = log([9.77, 1e-13, 50, 1e-13]);  ub = log([500, 1e-6, 5000, 1e-7]);   % [fL gL fc g]; fL >= lowest measured bin (no data below)
N = round(12 * fs);  rng(20261006);
v = zeros(N, 3);  eC = zeros(numel(f), 3);  eF = zeros(numel(ff), 3);
mask = true(numel(f), 3);  par = zeros(3, 5);
for j = 1:3
    S2 = abs(squeeze(Sf(j, j, :))).^2;  S2f = abs(squeeze(Sff(j, j, :))).^2;
    if ~isempty(excl{j}), mask(:, j) = f < excl{j}(1) | f > excl{j}(2); end
    m = mask(:, j);
    Pfloor = median(Pm(f > 5000, j));
    wt = sqrt(1 ./ f(m));                                 % equal weight per decade (bins are linear in f)
    phi = @(p, fq) exp(2 * p(2)) * lp(1, exp(p(1)), fq, fs) + exp(2 * p(4)) * lp(2, exp(p(3)), fq, fs) + Pfloor;
    res = @(p) wt .* (log(S2(m) .* phi(p, f(m))) - log(Pm(m, j)));
    best = inf;                                           % a few starts for the slope corner
    for fL0 = [10 20 40 80]
        [p, rn] = lsqnonlin(res, log([fL0, 2e-10, 1000, 2e-10]), lb, ub, opts);
        if rn < best, best = rn;  pb = p; end
    end
    p = pb;
    % one common factor k on gL and g: rms over the fitted range equals the measured rms there
    shape = exp(2 * p(2)) * lp(1, exp(p(1)), f, fs) + exp(2 * p(4)) * lp(2, exp(p(3)), f, fs);
    k = sqrt((sum(Pm(m, j)) - sum(S2(m) * Pfloor)) / sum(S2(m) .* shape(m)));
    p([2 4]) = p([2 4]) + log(k);
    eC(:, j) = S2 .* phi(p, f);  eF(:, j) = S2f .* phi(p, ff);
    fL = exp(p(1));  gL = exp(p(2));  fc = exp(p(3));  g = exp(p(4));
    par(j, :) = [fL, gL, fc, g, sqrt(Pfloor)];
    fprintf('%-2s fL %6.1f Hz gL %.2e | fc %5.0f Hz g %.2e | floor %.2e m/sqrt(Hz) | rms scale k %.2f\n', ...
            names{j}, fL, gL, fc, g, sqrt(Pfloor), k);
    [bL, aL] = butter(1, fL / (fs / 2), 'low');  [b, a] = butter(2, fc / (fs / 2), 'low');
    w = randn(N, 3) * sqrt(fs / 2);                       % unit one-sided PSD
    v(:, j) = gL * filter(bL, aL, w(:, 1)) + g * filter(b, a, w(:, 2)) + sqrt(Pfloor) * w(:, 3);
end
rmsM = sqrt(sum(Pm .* mask) * df);  rmsC = sqrt(sum(eC .* mask) * df);

% time-simulation check of the calculated spectra
e = -lsim(S, v, (0:N - 1)' * ts);
keep = round(2 * fs) + 1:N;
[Pe, fe] = pwelch(e(keep, :), hann(2048), 1024, 2048, fs);
for j = 1:3
    ms = fe > 0;
    if ~isempty(excl{j}), ms = ms & (fe < excl{j}(1) | fe > excl{j}(2)); end
    fprintf('%-2s rms fitted range: measured %.2e, calculated %.2e, time sim %.2e | full measured %.2e\n', ...
            names{j}, rmsM(j), rmsC(j), sqrt(sum(Pe(ms, j)) * (fe(2) - fe(1))), sqrt(sum(Pm(:, j)) * df));
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
    h(2) = loglog(ff, sqrt(eF(:, j)), 'Color', [0.16 0.47 0.84], 'LineWidth', 2);
    set(gca, 'XScale', 'log', 'YScale', 'log');
    lab = {sprintf('measured, rms %.2e m', rmsM(j)), sprintf('model, rms %.2e m', rmsC(j))};
    if ~isempty(excl{j})
        lab{1} = [lab{1} ' (without 0.7-2 kHz)'];  lab{2} = [lab{2} ' (without 0.7-2 kHz)'];
    end
    xlim([9 1e4]);  ylim([1e-11 5e-10]);  grid on;
    ylabel(sprintf('%s  [m/\\surdHz]', names{j}));
    legend(h, lab, 'Location', 'southwest', 'FontSize', 8);
end
xlabel(tl, 'frequency [Hz]');
title(tl, {'Servo error at standstill: Telica measurement and model (calculated)', ...
           'Model at Y = 0: encoder noise = low-pass (order 1) + low-pass (order 2) + white floor, error = S \times noise, S = (I + G K_1)^{-1}', ...
           'Fitted from 10 Hz with equal weight per decade, rms matched; Y: 0.7-2 kHz (grey) left out of the fit and the rms'}, ...
      'FontSize', 9);
exportgraphics(fig, fullfile(fileparts(mfilename('fullpath')), 'noise_order3_jasper.png'), 'Resolution', 150);
fprintf('figure written\n');

function L2 = lp(n, fc, f, fs)
% |H|^2 of butter(n, fc/(fs/2), 'low') at frequencies f
    [b, a] = butter(n, fc / (fs / 2), 'low');
    L2 = abs(freqz(b, a, f, fs)).^2;
end
