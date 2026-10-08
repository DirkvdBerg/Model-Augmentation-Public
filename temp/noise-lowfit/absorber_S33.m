% ABSORBER_S33  Where the Y bump and dip at 130-190 Hz come from: |S33| of the standstill loop (K1, Y = 0)
%   with the hidden MSD absorber, and with the absorber rigidly attached (ka x 1e6: same mass, no resonance).
repo = fullfile(pwd, '..', '..');  addpath(fullfile(repo, 'Thesis-writeup', 'Code', 'Data'));
cfg = tdg_config();  plant0 = gtd_build_plant(0, cfg, 1);
f = logspace(log10(20), log10(2000), 1500)';
cr = cfg;  cr.ka = cfg.ka * 1e6;
lab = {sprintf('with MSD absorber (%.0f Hz)', sqrt(cfg.ka / cfg.ma) / 2 / pi), 'absorber rigidly attached (same mass)'};
col = {[0.16 0.47 0.84], [0.45 0.45 0.45]};
fig = figure('Visible', 'off', 'Position', [100 100 760 420]);
for c = 1:2
    cc = cfg;  if c == 2, cc = cr; end
    S = feedback(eye(3), c2d(tdg_truth_lin(0, cc, cc.friction_gain), cfg.ts, 'zoh') * plant0.Cfb);
    H = squeeze(freqresp(S(3, 3), 2 * pi * f));
    semilogx(f, abs(H), 'Color', col{c}, 'LineWidth', 2); hold on;
    [~, i130] = min(abs(f - 130));  [~, i185] = min(abs(f - 185));
    fprintf('%-40s |S33| 130 Hz %.2f, 185 Hz %.2f\n', lab{c}, abs(H(i130)), abs(H(i185)));
end
xline(sqrt(cfg.ka / cfg.ma) / 2 / pi, ':', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
grid on;  xlim([20 2000]);  xlabel('frequency [Hz]');  ylabel('|S_{33}|  [-]');
legend(lab, 'Location', 'southeast');
title({'Y loop sensitivity at standstill (K_1, Y = 0)', 'the 130-190 Hz bump and dip come from the MSD absorber'});
exportgraphics(fig, 'absorber_S33_fig.png', 'Resolution', 150);
fprintf('figure written\n');
