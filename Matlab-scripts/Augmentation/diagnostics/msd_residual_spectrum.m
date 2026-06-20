%% msd_residual_spectrum.m
% Check whether the hidden MSD is measurable in the output data.
%
% Compares paired datasets for each mass fraction:
%   augmented: data with hidden MSD (ma_frac of payload mass)
%   baseline:  data without MSD (rigid payload)
%
% Both were generated with the same controller, same trajectories, same
% multisine realization. The only difference is the MSD.
%
% Diagnostic:
%   1. Time-domain residual: e(t) = y_augmented - y_baseline
%   2. Power spectral density of e(t) via pwelch
%   3. SNR at the MSD frequency: PSD peak at fa vs broadband noise floor
%
% The MSD is measurable if the residual PSD shows a clear peak at fa
% that stands above the broadband floor.
%
% To generate 50% data (if missing):
%   In generate_multisine_data.m, set USE_MSD = true, MA_FRAC = 0.50, run.
%   Data saves to data/gantry/matlab/multisine/m50/
%
% Usage (from repo root):
%   run('Matlab-scripts/Augmentation/diagnostics/msd_residual_spectrum.m')

clear; clc; close all;

%% ── Configuration ───────────────────────────────────────────────────────
fa = 150;   % MSD natural frequency [Hz] (must match generate_multisine_data.m)

% Mass fractions to analyse and their data directories
configs = struct( ...
    'ma_frac', {0.10, 0.50}, ...
    'dir_aug', { ...
        fullfile(pwd, 'data', 'gantry', 'matlab', 'multisine'), ...
        fullfile(pwd, 'data', 'gantry', 'matlab', 'multisine', 'm50') ...
    } ...
);
dir_baseline = fullfile(pwd, 'data', 'gantry', 'matlab', 'multisine', 'baseline');

% Trajectories to analyse (must exist in both directories)
files = {
    'T1_Y_sweep_conservative.mat'
    'T2_X_sym_Y030.mat'
    'T3_X_sym_Y000.mat'
    'T4_X_antisym_Y020.mat'
    'T5_X_sym_Y_sweep.mat'
    'T6_Y_sweep_aggressive.mat'
    'T7_X_antisym_Y_sweep.mat'
    'T8_X_sym_anti_Y_sweep.mat'
    'V1_X_sym_Y_mid_sweep.mat'
    'E1_X_sym_anti_Y_low_offset_sweep.mat'
};

ch_names = {'X1', 'X2', 'Y'};

%% ── Analyse each mass fraction ─────────────────────────────────────────
all_configs_summary = struct();

for ci = 1:numel(configs)
    cfg = configs(ci);
    pct = round(cfg.ma_frac * 100);

    fprintf('\n%s\n', repmat('=', 1, 80));
    fprintf('  Mass fraction: %d%%  (ma = %.2f kg)\n', pct, cfg.ma_frac * 10.1);
    fprintf('  Augmented dir: %s\n', cfg.dir_aug);
    fprintf('%s\n', repmat('=', 1, 80));

    % Check if data exists
    first_file = fullfile(cfg.dir_aug, files{1});
    if ~isfile(first_file)
        fprintf('  DATA NOT FOUND. Generate it first:\n');
        fprintf('    In generate_multisine_data.m, set:\n');
        fprintf('      USE_MSD = true;\n');
        fprintf('      MA_FRAC = %.2f;\n', cfg.ma_frac);
        fprintf('    Then run the script. Data saves to: %s\n', cfg.dir_aug);
        continue
    end

    % Summary table header
    fprintf('\n%-40s  %10s  %10s  %10s  %10s  %10s\n', ...
        'Trajectory', 'e_rms_Y[m]', 'y_rms_Y[m]', 'ratio[%]', ...
        'SNR_Y[dB]', 'peak_f[Hz]');
    fprintf('%s\n', repmat('-', 1, 95));

    summary = struct('file', {}, 'e_rms', {}, 'snr_db', {}, 'peak_f', {});

    for i = 1:numel(files)
        fname = files{i};
        path_aug  = fullfile(cfg.dir_aug, fname);
        path_base = fullfile(dir_baseline, fname);

        if ~isfile(path_aug) || ~isfile(path_base)
            fprintf('%-40s  SKIPPED (file missing)\n', fname);
            continue
        end

        d_aug  = load(path_aug);
        d_base = load(path_base);

        y_aug  = double(d_aug.y);     % (T x 3) augmented system output [m]
        y_base = double(d_base.y);    % (T x 3) baseline system output [m]
        fs     = double(d_aug.fs);    % 20000 Hz

        % Trim to same length (should be identical, but guard)
        T = min(size(y_aug,1), size(y_base,1));
        y_aug  = y_aug(1:T, :);
        y_base = y_base(1:T, :);

        % ── Residual ─────────────────────────────────────────────────────
        e = y_aug - y_base;   % (T x 3) MSD-induced difference

        % ── PSD via pwelch ───────────────────────────────────────────────
        % pwelch: splits signal into overlapping windows, computes FFT of
        % each, averages the squared magnitudes. Gives a smooth estimate of
        % power spectral density (power per Hz).
        % 1-second windows (= fs samples) give 1 Hz frequency resolution.
        N_win = min(round(fs), T);
        n_overlap = round(N_win / 2);
        [psd_e, f_hz] = pwelch(e(:,3), hanning(N_win), n_overlap, N_win, fs);

        % ── SNR at fa ────────────────────────────────────────────────────
        % Peak: max PSD in [fa-10, fa+10] Hz band
        idx_peak = f_hz >= (fa - 10) & f_hz <= (fa + 10);
        [psd_peak, pk_ix] = max(psd_e(idx_peak));
        f_peak_band = f_hz(idx_peak);
        f_peak = f_peak_band(pk_ix);

        % Noise floor: median PSD outside the peak band [fa-30, fa+30]
        idx_noise = f_hz >= 10 & f_hz <= 500 & ~(f_hz >= (fa-30) & f_hz <= (fa+30));
        psd_floor = median(psd_e(idx_noise));

        % THEORY: SNR = signal power / noise power [dB]
        snr_db = 10 * log10(psd_peak / psd_floor);

        % ── RMS metrics ──────────────────────────────────────────────────
        e_rms = rms(e);        % (1 x 3)
        y_rms = rms(y_aug);   % (1 x 3)
        ratio_pct = 100 * e_rms ./ y_rms;

        % ── Print summary row ────────────────────────────────────────────
        [~, name_only, ~] = fileparts(fname);
        fprintf('%-40s  %10.2e  %10.2e  %10.2f  %10.1f  %10.1f\n', ...
            name_only, e_rms(3), y_rms(3), ratio_pct(3), snr_db, f_peak);

        summary(end+1).file  = name_only;  %#ok<SAGROW>
        summary(end).e_rms   = e_rms;
        summary(end).snr_db  = snr_db;
        summary(end).peak_f  = f_peak;

        % ── Per-trajectory plot ──────────────────────────────────────────
        figure('Name', sprintf('m%d_%s', pct, name_only), ...
               'Position', [50 50 1200 700]);
        tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

        t = (0:T-1)' / fs;

        % Row 1: time-domain residual per channel
        for ch = 1:3
            nexttile(ch)
            plot(t, e(:, ch) * 1e6);
            ylabel(sprintf('%s residual [um]', ch_names{ch}));
            xlabel('Time [s]');
            grid on
        end

        % Row 2: PSD of residual per channel
        for ch = 1:3
            nexttile(3 + ch)
            [psd_ch, f_ch] = pwelch(e(:, ch), hanning(N_win), n_overlap, N_win, fs);
            semilogy(f_ch, psd_ch, 'b', 'LineWidth', 0.8); hold on

            % Mark MSD frequency
            xline(fa, 'r--', sprintf('%d Hz', fa), 'LineWidth', 1.2, ...
                'LabelVerticalAlignment', 'bottom');

            % Noise floor
            idx_n = f_ch >= 10 & f_ch <= 500 & ~(f_ch >= (fa-30) & f_ch <= (fa+30));
            floor_ch = median(psd_ch(idx_n));
            yline(floor_ch, 'k:', 'noise floor', 'LineWidth', 0.8);

            xlim([0 500]);
            ylabel(sprintf('%s PSD [m^2/Hz]', ch_names{ch}));
            xlabel('Frequency [Hz]');
            grid on

            if ch == 3
                idx_p = f_ch >= (fa-10) & f_ch <= (fa+10);
                pk_val = max(psd_ch(idx_p));
                snr_ch = 10*log10(pk_val / floor_ch);
                title(sprintf('SNR = %.1f dB', snr_ch));
            end
        end

        sgtitle(sprintf('m%d%%  %s:  e_{rms,Y} = %.2e m,  SNR_Y = %.1f dB @ %.0f Hz', ...
            pct, name_only, e_rms(3), snr_db, f_peak), 'FontSize', 11);
    end

    % ── Per-config summary ───────────────────────────────────────────────
    if numel(summary) > 0
        all_snr = [summary.snr_db];
        all_erms_Y = arrayfun(@(s) s.e_rms(3), summary);

        all_configs_summary(ci).pct       = pct;
        all_configs_summary(ci).mean_snr  = mean(all_snr);
        all_configs_summary(ci).min_snr   = min(all_snr);
        all_configs_summary(ci).max_snr   = max(all_snr);
        all_configs_summary(ci).mean_erms = mean(all_erms_Y);

        fprintf('\n  Summary (Y channel, m%d%%):\n', pct);
        fprintf('    Mean residual RMS:  %.2e m\n', mean(all_erms_Y));
        fprintf('    SNR at %d Hz:  min=%.1f  mean=%.1f  max=%.1f dB\n', ...
            fa, min(all_snr), mean(all_snr), max(all_snr));

        if min(all_snr) > 10
            fprintf('    VERDICT: MSD is clearly measurable (SNR > 10 dB)\n');
        elseif min(all_snr) > 3
            fprintf('    VERDICT: MSD is marginally measurable (3 < SNR < 10 dB)\n');
        else
            fprintf('    VERDICT: MSD is NOT measurable (SNR < 3 dB)\n');
        end
    end
end

%% ── Cross-comparison ────────────────────────────────────────────────────
valid = arrayfun(@(s) isfield(s, 'pct') && ~isempty(s.pct), all_configs_summary);
if sum(valid) >= 2
    fprintf('\n%s\n', repmat('=', 1, 80));
    fprintf('  COMPARISON ACROSS MASS FRACTIONS\n');
    fprintf('%s\n', repmat('=', 1, 80));
    fprintf('  %6s  %12s  %12s  %12s\n', 'ma[%]', 'mean_SNR', 'min_SNR', 'mean_e_rms');
    for ci = find(valid)
        s = all_configs_summary(ci);
        fprintf('  %6d  %12.1f  %12.1f  %12.2e\n', ...
            s.pct, s.mean_snr, s.min_snr, s.mean_erms);
    end
    fprintf('\n');

    % Scaling check: if MSD is linear, residual should scale with ma_frac
    if all_configs_summary(1).mean_erms > 0
        ratio = all_configs_summary(2).mean_erms / all_configs_summary(1).mean_erms;
        frac_ratio = configs(2).ma_frac / configs(1).ma_frac;
        fprintf('  Residual RMS ratio: %.1fx  (mass ratio: %.1fx)\n', ratio, frac_ratio);
    end
end
fprintf('%s\n', repmat('=', 1, 80));
