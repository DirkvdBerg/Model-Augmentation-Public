% NOISE_MODEL_FIGURE  The injected encoder-noise model itself (D-238), per stage axis, for Jasper.
%   Amplitude spectral density of v = g * butter(2, fc) on white noise + floor * white noise, with the
%   values the generator uses (tdg_config: noise_fc, noise_g, noise_floor). Low-pass part, floor, total,
%   and fc marked. Plus one generator draw (tdg_noise_psd + tdg_noise_draw) as a check.
%   Run from this folder: matlab -batch "noise_model_figure"

repo = fullfile(fileparts(mfilename('fullpath')), '..', '..');
addpath(fullfile(repo, 'Thesis-writeup', 'Code', 'Data'));
cfg = tdg_config();  fs = cfg.fs;
f = logspace(log10(5), log10(fs / 2 - 1), 3000)';
v = tdg_noise_draw(tdg_noise_psd(0, cfg, []), 20261007);  % one draw from the generator itself
[Pv, fv] = pwelch(v, hann(2048), 1024, 2048, fs);

names = {'X1', 'X2', 'Y'};
fig = figure('Visible', 'off', 'Position', [100 100 820 920]);
tl = tiledlayout(3, 1, 'TileSpacing', 'compact');
for j = 1:3
    [b, a] = butter(2, cfg.noise_fc(j) / (fs / 2), 'low');
    lp = cfg.noise_g(j) * abs(freqz(b, a, f, fs));
    fl = cfg.noise_floor(j) * ones(size(f));
    tot = sqrt(lp.^2 + fl.^2);
    nexttile;
    h = loglog(fv(2:end), sqrt(Pv(2:end, j)), 'Color', [0.80 0.80 0.80], 'LineWidth', 0.8); hold on;
    h(2) = loglog(f, lp, '--', 'Color', [0.16 0.47 0.84], 'LineWidth', 1.5);
    h(3) = loglog(f, fl, '--', 'Color', [0.88 0.54 0.12], 'LineWidth', 1.5);
    h(4) = loglog(f, tot, 'k', 'LineWidth', 2);
    xline(cfg.noise_fc(j), ':', sprintf('f_c = %.0f Hz', cfg.noise_fc(j)), 'LabelVerticalAlignment', 'bottom', ...
          'HandleVisibility', 'off');
    xlim([10 1e4]);  ylim([1e-11 5e-10]);  grid on;
    ylabel(sprintf('%s  [m/\\surdHz]', names{j}));
    legend(h, {'one generated noise signal (spectrum)', ...
               sprintf('low-pass part: g = %.2e m/\\surdHz, order 2', cfg.noise_g(j)), ...
               sprintf('white floor: %.2e m/\\surdHz', cfg.noise_floor(j)), 'total noise model'}, ...
           'Location', 'southwest', 'FontSize', 8);
end
xlabel(tl, 'frequency [Hz]');
title(tl, {'Encoder noise model per axis (injected noise, before the control loop)', ...
           'noise = g \times low-pass(f_c, order 2) on white noise + white floor'}, 'FontSize', 10);
exportgraphics(fig, fullfile(fileparts(mfilename('fullpath')), 'noise_model_jasper.png'), 'Resolution', 150);
fprintf('figure written; std of the draw %s m\n', mat2str(std(v), 3));
