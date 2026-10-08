% NOISE_FILTER_TEST  Low-order encoder noise as Jasper suggested (2026-10-05), test only.
%   Per stage axis (X1, X2, Y): white noise through a 2nd-order Butterworth band-pass
%   (butter, order 1 per edge) plus a white floor:
%       v = g * filter(b, a, w1) + floor * w2
%   - cutoffs: chosen to follow the envelope of the measured standstill spectrum
%     (rounded from the -3 dB edges of the earlier Python fit, fit_plot.py)
%   - floor:   the flat end of the measured spectrum (median above 5 kHz), copied
%   - g:       set so the simulated standstill error e = -S v has the measured rms
%   S = (I + G K1)^-1 from the controller and the state space (truth at rest at Y = 0,
%   tanh friction slope), no FRF. Standstill, linear loop, one realisation.
%   Run from this folder: matlab -batch "noise_filter_test"

repo = fullfile(fileparts(mfilename('fullpath')), '..', '..');
addpath(fullfile(repo, 'Thesis-writeup', 'Code', 'Data'));
cfg = tdg_config();
fs = cfg.fs;  ts = cfg.ts;

% measured standstill error spectrum (one-sided PSD, m^2/Hz, Welch bins 9.77 Hz)
T  = load(cfg.noise_target);
fw = T.f(:);  df = fw(2) - fw(1);
Pm = zeros(numel(fw), 3);
for j = 1:3, Pm(:, j) = real(T.P(:, j, j)); end
rms_meas = sqrt(sum(Pm(2:end, :)) * df);

% sensitivity S = (I + G K1)^-1, standard functions only
Gd = c2d(tdg_truth_lin(0, cfg, cfg.friction_gain), ts, 'zoh');
plant0 = gtd_build_plant(0, cfg, 1);  K1 = plant0.Cfb;
S  = feedback(eye(3), Gd * K1);
Sf = freqresp(S, 2 * pi * fw(2:end));
S2 = zeros(numel(fw) - 1, 3);
for j = 1:3, S2(:, j) = abs(squeeze(Sf(j, j, :))).^2; end

% noise model per axis
names = {'X1', 'X2', 'Y'};
band  = [55 1500; 40 1100; 950 1450];                    % [Hz] band-pass cutoffs
N = round(12 * fs);  rng(20261005);
v = zeros(N, 3);  phi_e = zeros(numel(fw) - 1, 3);  par = zeros(3, 2);
for j = 1:3
    [b, a] = butter(1, band(j, :) / (fs / 2), 'bandpass');
    Hbp2   = abs(freqz(b, a, fw(2:end), fs)).^2;
    Pfloor = median(Pm(fw > 5000, j));                    % copied floor level, m^2/Hz
    % rms of e: sum |S|^2 (g^2 |Hbp|^2 + Pfloor) df = rms_meas^2  ->  g
    g = sqrt((rms_meas(j)^2 - sum(S2(:, j) * Pfloor) * df) / (sum(S2(:, j) .* Hbp2) * df));
    par(j, :) = [g, sqrt(Pfloor)];
    phi_v = g^2 * Hbp2 + Pfloor;                          % injected v, one-sided PSD
    phi_e(:, j) = S2(:, j) .* phi_v;                      % expected error (per axis)
    w = randn(N, 2) * sqrt(fs / 2);                       % unit one-sided PSD
    v(:, j) = g * filter(b, a, w(:, 1)) + sqrt(Pfloor) * w(:, 2);
    fprintf('%-2s band %4d-%4d Hz | floor %.2e m/sqrt(Hz) | band-pass gain %.2e m/sqrt(Hz)\n', ...
            names{j}, band(j, 1), band(j, 2), sqrt(Pfloor), g);
end

% Y variant: floor only (white at the copied floor level, no band-pass, no rms matching)
vYf = v;  vYf(:, 3) = par(3, 2) * randn(N, 1) * sqrt(fs / 2);
phi_eYf = S2(:, 3) * par(3, 2)^2;

% one realisation through the loop, e = -S v
e = -lsim(S, v, (0:N - 1)' * ts);
eYf = -lsim(S, vYf, (0:N - 1)' * ts);  eYf = eYf(:, 3);
keep = round(2 * fs) + 1:N;                               % drop the 2 s start transient
[Pe, fe] = pwelch(e(keep, :), hann(2048), 1024, 2048, fs);
[Pv, ~]  = pwelch(v(keep, :), hann(2048), 1024, 2048, fs);
[PeYf, ~] = pwelch(eYf(keep), hann(2048), 1024, 2048, fs);

fprintf('\nrms of the standstill error [m]: measured | simulated (total, and < 1.6 kHz)\n');
for j = 1:3
    ib = fw > 0 & fw < 1600;  is = fe > 0 & fe < 1600;
    fprintf('%-2s total %.2e | %.2e    in band %.2e | %.2e\n', names{j}, rms_meas(j), ...
            std(e(keep, j)), sqrt(sum(Pm(ib, j)) * df), sqrt(sum(Pe(is, j)) * (fe(2) - fe(1))));
end
fprintf('Y floor only total %.2e | %.2e    in band %.2e | %.2e\n', rms_meas(3), std(eYf(keep)), ...
        sqrt(sum(Pm(ib, 3)) * df), sqrt(sum(PeYf(is)) * (fe(2) - fe(1))));

% figure: measured, expected (clean), one realisation, injected noise
fig = figure('Visible', 'off', 'Position', [100 100 820 980]);
for j = 1:3
    subplot(3, 1, j);
    loglog(fw(2:end), sqrt(Pm(2:end, j)), 'k', 'LineWidth', 1.4); hold on;
    loglog(fe(2:end), sqrt(Pe(2:end, j)), 'Color', [0.65 0.78 0.95], 'LineWidth', 0.8);
    loglog(fw(2:end), sqrt(phi_e(:, j)), 'Color', [0.16 0.47 0.84], 'LineWidth', 1.8);
    loglog(fe(2:end), sqrt(Pv(2:end, j)), '--', 'Color', [0.88 0.54 0.12], 'LineWidth', 0.9);
    xline(band(j, :), ':', 'Color', [0.5 0.5 0.5]);
    if j == 3
        loglog(fe(2:end), sqrt(PeYf(2:end)), 'Color', [0.70 0.88 0.70], 'LineWidth', 0.8);
        loglog(fw(2:end), sqrt(phi_eYf), 'Color', [0.10 0.55 0.25], 'LineWidth', 1.8);
        legend({'measured', 'band-pass + floor, one realisation', 'band-pass + floor, expected', ...
                'injected v (band-pass + floor)', 'band-pass cutoffs', '', ...
                'floor only, one realisation', 'floor only, expected'}, 'Location', 'southwest', 'FontSize', 7);
    end
    xlim([9 1e4]); grid on;
    ylabel(sprintf('%s  [m/\\surdHz]', names{j}));
    if j == 1
        legend({sprintf('measured error (Telica standstill), rms %.2e m', rms_meas(1)), ...
                'simulated error, one realisation', 'simulated error, expected', ...
                'injected encoder noise v', 'band-pass cutoffs'}, 'Location', 'southwest', 'FontSize', 7);
        title('Standstill servo error at Y = 0: band-pass + floor encoder noise (butter, order 2)');
    end
end
xlabel('frequency [Hz]');
exportgraphics(fig, fullfile(fileparts(mfilename('fullpath')), 'noise_filter_test_Yfloor.png'), 'Resolution', 150);
fprintf('figure written\n');
