% NOISE_FINAL_FIGURE  Encoder-noise model after Jasper's feedback (2026-10-06), test only.
%   Same structure on X1, X2 and Y (order 2):
%       v = g * filter(b, a, w1) + floor * w2,   [b, a] = butter(2, fc / (fs/2), 'low')
%   - floor:  copied from the measured spectrum (median above 5 kHz)
%   - fc, g:  least-squares fit (lsqnonlin) of the log model spectrum to the log measured standstill
%             error spectrum, all bins from 9.77 Hz, weighted 1/f (equal weight per decade);
%             for Y, 0.7 to 2 kHz excluded
%   - g:      then scaled so the error rms over the fitted range equals the measured rms there
%   Like with like: the measured spectrum is a Welch estimate (Hann 2048, 20 kHz, 9.77 Hz bins), which
%   averages each bin over about +-20 Hz. The model spectrum |S_jj|^2 phi_v is passed through the same
%   window (convolution with the Hann window's spectral kernel) before it is compared; this matters
%   only near 10 Hz, where S changes quickly. A time simulation (lsim + pwelch) checks it.
%   S = (I + G K1)^-1 from the state space (truth at rest, Y = 0, tanh slope) and K1.
%   Run from this folder: matlab -batch "noise_final_figure"

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

% fine grid and the Hann window's spectral kernel (the estimator of the measurement)
dg = 0.25;  fg = (0:dg:fs / 2)';
Sg = freqresp(S, 2 * pi * fg);
kf = (-200:dg:200)';
W2 = abs(freqz(hann(nw), 1, kf, fs)).^2;  W2 = W2 / (sum(W2) * dg);
ib = round(f / dg) + 1;                                   % bin positions on the fine grid
welchify = @(Pg) smear(Pg, W2, kf, dg, ib);

names = {'X1', 'X2', 'Y'};
excl  = {[], [], [700 2000]};                             % [Hz] left out of fit and rms
opts  = optimoptions('lsqnonlin', 'Display', 'off');
lb = log([50, 1e-13]);  ub = log([5000, 1e-7]);           % [fc g]
N = round(12 * fs);  rng(20261006);
v = zeros(N, 3);  eW = zeros(numel(f), 3);  mask = true(numel(f), 3);  par = zeros(3, 3);
for j = 1:3
    S2g = abs(squeeze(Sg(j, j, :))).^2;
    if ~isempty(excl{j}), mask(:, j) = f < excl{j}(1) | f > excl{j}(2); end
    m = mask(:, j);
    Pfloor = median(Pm(f > 5000, j));
    wt = sqrt(1 ./ f(m));                                 % equal weight per decade (bins are linear in f)
    model = @(p) welchify(S2g .* (exp(2 * p(2)) * lp2(exp(p(1)), fg, fs) + Pfloor));
    res = @(p) wt .* (log(sel(model(p), m)) - log(Pm(m, j)));
    p = lsqnonlin(res, log([1000, 2e-10]), lb, ub, opts);
    % scale g: rms over the fitted range equals the measured rms there
    L = welchify(S2g .* lp2(exp(p(1)), fg, fs));  F = welchify(S2g * Pfloor);
    g = sqrt((sum(Pm(m, j)) - sum(F(m))) / sum(L(m)));
    fprintf('%-2s fc %5.0f Hz | g fit %.2e -> rms-scaled %.2e | floor %.2e m/sqrt(Hz)\n', ...
            names{j}, exp(p(1)), exp(p(2)), g, sqrt(Pfloor));
    par(j, :) = [exp(p(1)), g, sqrt(Pfloor)];
    eW(:, j) = g^2 * L + F;
    [b, a] = butter(2, par(j, 1) / (fs / 2), 'low');
    w = randn(N, 2) * sqrt(fs / 2);                       % unit one-sided PSD
    v(:, j) = g * filter(b, a, w(:, 1)) + sqrt(Pfloor) * w(:, 2);
end
rmsM = sqrt(sum(Pm .* mask) * df);  rmsC = sqrt(sum(eW .* mask) * df);

% time-simulation check: lsim + the same Welch estimator
e = -lsim(S, v, (0:N - 1)' * ts);
keep = round(2 * fs) + 1:N;
[Pe, fe] = pwelch(e(keep, :), hann(nw), nw / 2, nw, fs);
for j = 1:3
    ms = mask(:, j);
    fprintf(['%-2s rms fitted range: measured %.2e, model %.2e, time sim %.2e | ' ...
             '9.77 Hz bin: measured %.1e, model %.1e, time sim %.1e m/sqrt(Hz)\n'], names{j}, rmsM(j), ...
            rmsC(j), sqrt(sum(Pe([false; ms], j)) * df), sqrt(Pm(1, j)), sqrt(eW(1, j)), sqrt(Pe(2, j)));
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
    h(2) = loglog(f, sqrt(eW(:, j)), 'Color', [0.16 0.47 0.84], 'LineWidth', 2);
    set(gca, 'XScale', 'log', 'YScale', 'log');
    lab = {sprintf('measured, rms %.2e m', rmsM(j)), ...
           sprintf('model: low-pass %.0f Hz + floor, rms %.2e m', par(j, 1), rmsC(j))};
    if ~isempty(excl{j})
        lab{1} = [lab{1} ' (without 0.7-2 kHz)'];  lab{2} = [lab{2} ' (without 0.7-2 kHz)'];
    end
    xlim([9 1e4]);  ylim([1e-11 5e-10]);  grid on;
    ylabel(sprintf('%s  [m/\\surdHz]', names{j}));
    legend(h, lab, 'Location', 'southwest', 'FontSize', 8);
end
xlabel(tl, 'frequency [Hz]');
title(tl, {'Servo error at standstill: Telica measurement and model (calculated)', ...
           'Model at Y = 0: encoder noise = low-pass (order 2) + white floor, error = S \times noise, S = (I + G K_1)^{-1}', ...
           'Fit from 10 Hz (equal weight per decade), rms matched, same Welch estimator as the measurement; Y: grey band left out'}, ...
      'FontSize', 9);
exportgraphics(fig, fullfile(fileparts(mfilename('fullpath')), 'noise_final_jasper.png'), 'Resolution', 150);
fprintf('figure written\n');

function L2 = lp2(fc, f, fs)
% |H|^2 of butter(2, fc/(fs/2), 'low') at frequencies f
    [b, a] = butter(2, fc / (fs / 2), 'low');
    L2 = abs(freqz(b, a, f, fs)).^2;
end

function Pb = smear(Pg, W2, kf, dg, ib)
% expected Welch estimate at the bins: the one-sided model spectrum, mirrored to negative frequency,
% convolved with the window's normalised spectral kernel (two-sided density = one-sided / 2 for f > 0)
    nk = (numel(kf) - 1) / 2;
    P2 = [flipud(Pg(2:nk + 1)); Pg];                      % symmetric extension below 0 Hz
    C  = conv(P2, W2 * dg, 'same');
    C  = C(nk + 1:end);
    Pb = C(ib);
end

function y = sel(x, m), y = x(m); end
