% NOISE_LOWPASS_FIGURE  Encoder-noise model after Jasper's feedback (2026-10-06), test only.
%   Same structure on X1, X2 and Y:
%       v = g * filter(b, a, w1) + floor * w2,   [b, a] = butter(2, fc / (fs/2), 'low')
%   - floor:  copied from the measured spectrum (median above 5 kHz)
%   - fc, g:  least-squares fit (lsqnonlin) of log(|S_jj|^2 phi_v) to the log of the measured
%             standstill error spectrum, all bins from 9.77 Hz, weighted 1/f (equal weight per
%             decade, so 10-100 Hz counts as much as 1-10 kHz); for Y, 0.7 to 2 kHz excluded
%   - g:      then scaled so the error rms over the fitted range equals the measured rms there
%   S = (I + G K1)^-1 from the state space (truth at rest, Y = 0, tanh slope) and K1.
%   The figure shows calculated spectra; a time simulation (lsim) checks the rms.
%   Run from this folder: matlab -batch "noise_lowpass_figure"

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

names = {'X1', 'X2', 'Y'};
excl  = {[], [], [700 2000]};                             % [Hz] left out of fit and rms
opts  = optimoptions('lsqnonlin', 'Display', 'off');
N = round(12 * fs);  rng(20261006);
v = zeros(N, 3);  eC = zeros(numel(f), 3);  mask = true(numel(f), 3);  par = zeros(3, 3);
for j = 1:3
    S2 = abs(squeeze(Sf(j, j, :))).^2;
    if ~isempty(excl{j}), mask(:, j) = f < excl{j}(1) | f > excl{j}(2); end
    m = mask(:, j);
    Pfloor = median(Pm(f > 5000, j));
    wt = sqrt(1 ./ f(m));                                 % equal weight per decade (bins are linear in f)
    res = @(p) wt .* (log(S2(m) .* (exp(2 * p(2)) * lp2(exp(p(1)), f(m), fs) + Pfloor)) - log(Pm(m, j)));
    p = lsqnonlin(res, log([500, 2e-10]), log([10, 1e-13]), log([5000, 1e-7]), opts);
    fc = exp(p(1));  L2 = lp2(fc, f, fs);
    % scale g: rms over the fitted range equals the measured rms there
    g = sqrt((sum(Pm(m, j)) - sum(S2(m) * Pfloor)) / sum(S2(m) .* L2(m)));
    fprintf('%s fit g %.2e -> rms-scaled g %.2e\n', names{j}, exp(p(2)), g);
    eC(:, j) = S2 .* (g^2 * L2 + Pfloor);
    par(j, :) = [fc, g, sqrt(Pfloor)];
    [b, a] = butter(2, fc / (fs / 2), 'low');
    w = randn(N, 2) * sqrt(fs / 2);                       % unit one-sided PSD
    v(:, j) = g * filter(b, a, w(:, 1)) + sqrt(Pfloor) * w(:, 2);
end
rmsM = sqrt(sum(Pm .* mask) * df);                        % measured, fitted range
rmsC = sqrt(sum(eC .* mask) * df);                        % calculated, fitted range
rmsMfull = sqrt(sum(Pm) * df);  rmsCfull = sqrt(sum(eC) * df);

% time-simulation check of the calculated spectra
e = -lsim(S, v, (0:N - 1)' * ts);
keep = round(2 * fs) + 1:N;
[Pe, fe] = pwelch(e(keep, :), hann(2048), 1024, 2048, fs);
for j = 1:3
    ms = true(size(fe));
    if ~isempty(excl{j}), ms = fe < excl{j}(1) | fe > excl{j}(2); end
    ms = ms & fe > 0;
    fprintf(['%-2s fc %4.0f Hz | g %.2e | floor %.2e m/sqrt(Hz) | rms fitted range: measured %.2e, ' ...
             'calculated %.2e, time sim %.2e | full: measured %.2e, calculated %.2e\n'], names{j}, ...
            par(j, 1), par(j, 2), par(j, 3), rmsM(j), rmsC(j), sqrt(sum(Pe(ms, j)) * (fe(2) - fe(1))), ...
            rmsMfull(j), rmsCfull(j));
end

fig = figure('Visible', 'off', 'Position', [100 100 800 900]);
tl = tiledlayout(3, 1, 'TileSpacing', 'compact');
for j = 1:3
    nexttile;
    if ~isempty(excl{j})
        patch([excl{j}(1) excl{j}(2) excl{j}(2) excl{j}(1)], [1e-12 1e-12 1e-8 1e-8], ...
              [0.92 0.92 0.92], 'EdgeColor', 'none', 'HandleVisibility', 'off'); hold on;
    end
    h = loglog(f, sqrt(Pm(:, j)), 'k', 'LineWidth', 1.3); hold on;
    h(2) = loglog(f, sqrt(eC(:, j)), 'Color', [0.16 0.47 0.84], 'LineWidth', 2);
    set(gca, 'XScale', 'log', 'YScale', 'log');
    lab = {sprintf('measured, rms %.2e m', rmsM(j)), ...
           sprintf('model: low-pass (fc %.0f Hz) + floor, rms %.2e m', par(j, 1), rmsC(j))};
    if ~isempty(excl{j})
        lab{1} = [lab{1} ' (without 0.7-2 kHz)'];  lab{2} = [lab{2} ' (without 0.7-2 kHz)'];
    end
    xlim([9 1e4]);  ylim([1e-11 5e-10]);  grid on;
    ylabel(sprintf('%s  [m/\\surdHz]', names{j}));
    legend(h, lab, 'Location', 'southwest', 'FontSize', 8);
end
xlabel(tl, 'frequency [Hz]');
title(tl, {'Servo error at standstill: Telica measurement and model (calculated)', ...
           'Model at Y = 0: encoder noise = low-pass filter (order 2) + white floor, error = S \times noise, S = (I + G K_1)^{-1}', ...
           'Fitted from 10 Hz with equal weight per decade, rms matched; Y: 0.7-2 kHz (grey) left out of the fit and the rms'}, ...
      'FontSize', 10);
exportgraphics(fig, fullfile(fileparts(mfilename('fullpath')), 'noise_lowpass_jasper.png'), 'Resolution', 150);
fprintf('figure written\n');

function L2 = lp2(fc, f, fs)
% |H|^2 of butter(2, fc/(fs/2), 'low') at frequencies f
    [b, a] = butter(2, fc / (fs / 2), 'low');
    L2 = abs(freqz(b, a, f, fs)).^2;
end

