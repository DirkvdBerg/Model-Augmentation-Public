% NOISE_OPTIONS_FIGURE  Clean figure for Jasper: measured standstill error vs the expected
%   simulated standstill error for two encoder-noise options (same model as noise_filter_test.m).
%   Option A: band-pass (butter, order 2) + white floor on X1, X2 and Y, gain set by the measured rms
%   Option B: as A on X1 and X2; Y floor only (white at the copied floor level)
%   Simulated curves are expected spectra |S_jj|^2 phi_v, S = (I + G K1)^-1 at Y = 0 from the
%   state space and K1 (no realisation, no FRF).
%   Run from this folder: matlab -batch "noise_options_figure"

repo = fullfile(fileparts(mfilename('fullpath')), '..', '..');
addpath(fullfile(repo, 'Thesis-writeup', 'Code', 'Data'));
cfg = tdg_config();
fs = cfg.fs;

T  = load(cfg.noise_target);
fw = T.f(:);  df = fw(2) - fw(1);  f = fw(2:end);
Pm = zeros(numel(f), 3);
for j = 1:3, Pm(:, j) = real(T.P(2:end, j, j)); end
rms_meas = sqrt(sum(Pm) * df);

Gd = c2d(tdg_truth_lin(0, cfg, cfg.friction_gain), cfg.ts, 'zoh');
plant0 = gtd_build_plant(0, cfg, 1);
S  = feedback(eye(3), Gd * plant0.Cfb);
Sf = freqresp(S, 2 * pi * f);

names = {'X1', 'X2', 'Y'};
band  = [55 1500; 40 1100; 950 1450];                     % [Hz] band-pass cutoffs
eA = zeros(numel(f), 3);
for j = 1:3
    S2 = abs(squeeze(Sf(j, j, :))).^2;
    [b, a] = butter(1, band(j, :) / (fs / 2), 'bandpass');
    Hbp2 = abs(freqz(b, a, f, fs)).^2;
    Pfloor = median(Pm(f > 5000, j));
    g2 = (rms_meas(j)^2 - sum(S2 * Pfloor) * df) / (sum(S2 .* Hbp2) * df);
    eA(:, j) = S2 .* (g2 * Hbp2 + Pfloor);
    if j == 3, eB = S2 * Pfloor; end
end
rA = sqrt(sum(eA) * df);  rB = sqrt(sum(eB) * df);

fig = figure('Visible', 'off', 'Position', [100 100 800 900]);
tl = tiledlayout(3, 1, 'TileSpacing', 'compact');
for j = 1:3
    nexttile;
    h = loglog(f, sqrt(Pm(:, j)), 'k', 'LineWidth', 1.3); hold on;
    h(2) = loglog(f, sqrt(eA(:, j)), 'Color', [0.16 0.47 0.84], 'LineWidth', 2);
    lab = {sprintf('measured, rms %.2e m', rms_meas(j)), ...
           sprintf('%s, rms %.2e m', ternary(j < 3, 'option A = B: band-pass + floor', ...
                   'option A: band-pass + floor'), rA(j))};
    if j == 3
        h(3) = loglog(f, sqrt(eB), 'Color', [0.10 0.55 0.25], 'LineWidth', 2);
        lab{3} = sprintf('option B: floor only, rms %.2e m', rB);
    end
    xlim([10 1e4]); grid on;
    ylabel(sprintf('%s  [m/\\surdHz]', names{j}));
    legend(h, lab, 'Location', 'southwest', 'FontSize', 8);
end
xlabel(tl, 'frequency [Hz]');
title(tl, {'Servo error at standstill: Telica measurement and model (calculated)', ...
           'Model at Y = 0: encoder noise = band-pass filter (order 2) + white floor, error = S \times noise, S = (I + G K_1)^{-1}'}, ...
      'FontSize', 10);
exportgraphics(fig, fullfile(fileparts(mfilename('fullpath')), 'noise_options_jasper_v3.png'), 'Resolution', 150);
fprintf('A rms %s | B (Y) %.2e | measured %s\n', mat2str(rA, 3), rB, mat2str(rms_meas, 3));

function out = ternary(c, a, b)
    if c, out = a; else, out = b; end
end
