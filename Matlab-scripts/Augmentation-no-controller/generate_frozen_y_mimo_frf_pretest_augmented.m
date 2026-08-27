% generate_frozen_y_mimo_frf_pretest_augmented.m
% Frozen-Y MIMO FRF pretest for the augmented gantry model (with hidden MSD).
%
% Same structure as generate_frozen_y_mimo_frf_pretest.m but uses:
%   - gantry_additional_state_2025a.slx (Simscape + extended ODE)
%   - Simscape output q as position (physical truth including MSD dynamics)
%   - MSD parameters: fa = 400 Hz, zeta = 0.05
%   - Frequency range extended to 500 Hz to capture the 400 Hz MSD mode
%
% Run from repo root:
%   run('Matlab-scripts/generate_frozen_y_mimo_frf_pretest_augmented.m')

clearvars
close all
clc

addpath(genpath(fullfile(pwd, 'kamtin-fp-model', '03 Simulink gantry')))
addpath(genpath(fileparts(mfilename('fullpath'))))

%% Configuration

mdl = 'gantry_additional_state_2025a';
fs = 20e3;
ts = 1/fs;
fbw = 100;

df = 1;
T_period = 1/df;
N_period = round(fs/df);
assert(abs(N_period - fs/df) < 10*eps(fs/df), 'fs/df must be integer.');

f_low_pre = 1;
f_high_pre = 800;
f_lines = f_low_pre:df:f_high_pre;
line_idx = round(f_lines/df) + 1;

Y_grid = [-0.30, -0.25, -0.15, 0.00, 0.15, 0.25, 0.30];
Y_ctrl = 0.25;

N_settle_periods = 1;   % no multisine, lets the frozen operating point settle
N_drop_periods = 2;     % multisine periods discarded before FRF averaging
N_clean_periods = 10;   % multisine periods used for the FRF
N_record_periods = N_drop_periods + N_clean_periods;
N_total = (N_settle_periods + N_record_periods) * N_period;
t = (0:N_total-1)' * ts;

N_candidates = 50;
rng_base = 20260518;

% Absolute low-force starting amplitudes [N RMS] per modal input. These are
% intentionally not percentages of hardware limits; limits are checked below.
mode_rms.common = 25;
mode_rms.diff = 25;
mode_rms.y = 15;

% Set environment variable SAVE_FRF_TIME_SERIES=1 to save full 20 kHz run
% histories. The default keeps output compact and saves spectra/metadata only.
save_time_series = strcmpi(getenv('SAVE_FRF_TIME_SERIES'), '1');

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', ...
                   'Matlab-output', 'frf-identification');
plot_dir = fullfile(out_dir, 'plots');
run_dir = fullfile(out_dir, 'runs');
ensure_dir(out_dir);
ensure_dir(plot_dir);
ensure_dir(run_dir);

%% Physical parameters and limits

mb=22.8; mh=10.1; m1=10.2; m2=10.7; Jb=1.0; Jh=0.05;
cg1=14.5; cg2=20.3; cy=10; cb1=9; cb2=9;
kb1=1987.5; kb2=1987.5; Lb=0.725; Lh=0.25; d=0.1;

% Hidden MSD parameters (Option A: mh_total = mh_rigid + ma, conserved)
ma_frac  = 0.10;
ma       = ma_frac * mh;             % 1.01 kg  hidden MSD mass
mh_rigid = mh - ma;                  % 9.09 kg  rigid part of payload
L0       = 0.10;                     % equilibrium offset of ma in +Y (m)
fa       = 400;                      % target MSD natural frequency (Hz)
ka       = ma * (2*pi*fa)^2;         % MSD spring stiffness (N/m)
zeta_a   = 0.05;                     % damping ratio (metal structures with joints)
ca       = 2 * zeta_a * sqrt(ka*ma); % MSD damper coefficient (Ns/m)
mh_original = mh;                    % keep original mh for baseline block (q1)
mh          = mh_rigid;              % extended ODE block uses mh_rigid

C_damp = [cg1+cg2,        (cg1-cg2)*Lb/2,           0;
          (cg1-cg2)*Lb/2, cb1+cb2+(cg1+cg2)*Lb^2/4, 0;
          0,              0,                         cy];
K = [0,0,0; 0,kb1+kb2,0; 0,0,0];
n = 3;
P = [1,1,0; Lb/2,-Lb/2,0; 0,0,1];

lim.pos_X = 0.375;
lim.pos_Y = 0.400;
lim.diff = sin(0.1) * Lb;
lim.vel = 2.0;
lim.acc_X = 30.0;
lim.acc_Y = 50.0;
lim.force_peak = [2000, 2000, 1420];
lim.force_rms = [916, 916, 656];

%% Modal definitions and fixed controller

modes(1).name = 'common'; modes(1).f_vec = [1,  1, 0]; modes(1).rms = mode_rms.common;
modes(2).name = 'diff';   modes(2).f_vec = [1, -1, 0]; modes(2).rms = mode_rms.diff;
modes(3).name = 'y';      modes(3).f_vec = [0,  0, 1]; modes(3).rms = mode_rms.y;

M_ctrl = mass_matrix(Y_ctrl, m1, m2, mb, mh, Jb, Jh, Lb, d);
sys_ctrl = P.' * getss(n, M_ctrl, C_damp, K) * P;
Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)), ts);
for j = 1:3
    Cfb(j,j) = ruleOfThumb(fbw, sys_ctrl(j,j), ts);
end

%% Low-crest-factor periodic multisines

fprintf('Multisine design: f = %g\x2013%g Hz, df = %g Hz, %d lines, %d candidates, seed %d\n', ...
        f_low_pre, f_high_pre, df, numel(f_lines), N_candidates, rng_base);
for im = 1:numel(modes)
    seed = rng_base + 100*im;
    [one_period, cf_best] = best_random_multisine_period( ...
        N_period, fs, f_lines, N_candidates, modes(im).f_vec, seed);
    one_period = modes(im).rms * one_period / rms(one_period);

    modes(im).one_period = one_period;
    modes(im).crest_factor = cf_best;
    modes(im).force_peak = max(abs(one_period * modes(im).f_vec), [], 1);
    modes(im).force_rms = rms(one_period * modes(im).f_vec, 1);

    fprintf('mode %-7s: RMS %5.1f N | CF %.3f | peak [%5.1f %5.1f %5.1f] N | rms [%5.1f %5.1f %5.1f] N\n', ...
            modes(im).name, modes(im).rms, modes(im).crest_factor, ...
            modes(im).force_peak, modes(im).force_rms);
end

%% Simulation and FRF estimation

nY = numel(Y_grid);
nModes = numel(modes);
nFreq = numel(f_lines);

G = complex(nan(nFreq, 3, 3, nY));
U_avg = complex(nan(nFreq, 3, 3, nY));
Y_avg = complex(nan(nFreq, 3, 3, nY));
rank_U = nan(nFreq, nY);
cond_U = nan(nFreq, nY);
fprintf('\nFrozen-Y MIMO FRF pretest: %d Y points x %d modes\n', nY, nModes);

for iY = 1:nY
    Y0 = Y_grid(iY);
    fprintf('\nY = %+0.2f m\n', Y0);

    Y_runs = complex(nan(nFreq, 3, nModes));
    U_runs = complex(nan(nFreq, 3, nModes));

    for im = 1:nModes
        mode = modes(im);
        fprintf('  running %-7s input...\n', mode.name);

        r = repmat([0, 0, Y0], N_total, 1);
        f = zeros(N_total, 3);
        sig_record = repmat(mode.one_period, N_record_periods, 1);
        record_start = N_settle_periods*N_period + 1;
        f(record_start:end,:) = sig_record * mode.f_vec;
        Y = Y0; % Used by the Simulink model.

        sim(mdl, t(end));

        [t_sim, ~, u_controller] = reconstruct_controller(q, r, t, Cfb);
        q_uniform = interp1(t_sim, q, t, 'linear', 'extrap');
        u_ctrl_uniform = interp1(t_sim, u_controller, t, 'linear', 'extrap');
        u_total = u_ctrl_uniform + f;

        validity(iY, im) = validate_run(q_uniform, u_total, lim);
        run_summary(iY, im) = summarize_run(Y0, mode.name, q_uniform, u_total, ...
                                            validity(iY, im));

        [Y_runs(:,:,im), U_runs(:,:,im)] = period_average_spectra( ...
            q_uniform, u_total, N_period, N_settle_periods, ...
            N_drop_periods, N_clean_periods, line_idx);

        if save_time_series
            save_run_time_series(run_dir, Y0, mode.name, fs, t, q_uniform, ...
                                 u_total, f, validity(iY, im));
        end
    end

    Y_avg(:,:,:,iY) = Y_runs;
    U_avg(:,:,:,iY) = U_runs;

    % Vectorise FRF inversion over all frequency bins (requires R2022a+).
    Y3 = permute(Y_runs, [2 3 1]);            % (3, 3, nFreq)
    U3 = permute(U_runs, [2 3 1]);            % (3, 3, nFreq)
    G(:,:,:,iY) = permute(pagemrdivide(Y3, U3), [3 1 2]);
    for k = 1:nFreq
        Uk = U3(:,:,k);
        rank_U(k,iY) = rank(Uk);
        cond_U(k,iY) = cond(Uk);
    end
    bad = rank_U(:,iY) < 3 | ~isfinite(cond_U(:,iY));
    G(bad,:,:,iY) = NaN;

    fprintf('  U_all condition number: median %.2e, max %.2e\n', ...
            median(cond_U(:,iY), 'omitnan'), max(cond_U(:,iY), [], 'omitnan'));
end

config = struct();
config.fs = fs;
config.df = df;
config.T_period = T_period;
config.N_period = N_period;
config.f_lines = f_lines;
config.Y_grid = Y_grid;
config.Y_ctrl = Y_ctrl;
config.N_settle_periods = N_settle_periods;
config.N_drop_periods = N_drop_periods;
config.N_clean_periods = N_clean_periods;
config.N_candidates = N_candidates;
config.mode_rms = mode_rms;
config.save_time_series = save_time_series;
config.output_dir = out_dir;

save(fullfile(out_dir, 'frozen_y_mimo_frf_pretest.mat'), ...
     'G', 'U_avg', 'Y_avg', 'f_lines', 'Y_grid', 'modes', 'config', ...
     'lim', 'rank_U', 'cond_U', 'validity', 'run_summary', '-v7.3');

make_frf_plots(G, f_lines, Y_grid, plot_dir);
make_condition_plot(cond_U, f_lines, Y_grid, plot_dir);

fprintf('\nSaved: %s\n', fullfile(out_dir, 'frozen_y_mimo_frf_pretest.mat'));
fprintf('Saved plots in: %s\n', plot_dir);

%% Local functions

function M = mass_matrix(Y, m1, m2, mb, mh, Jb, Jh, Lb, d)
    M = [m1+m2+mb+mh,            (m1-m2)*Lb/2-mh*Y,                   0;
         (m1-m2)*Lb/2-mh*Y,      Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y^2, -mh*d;
         0,                      -mh*d,                               mh];
end

function [period, cf_best] = best_random_multisine_period(N_period, fs, f_lines, n_candidates, f_vec, seed)
    rng(seed);
    t_period = (0:N_period-1)' / fs;
    best = [];
    cf_best = inf;

    for ic = 1:n_candidates
        phases = 2*pi*rand(1, numel(f_lines));
        candidate = sum(cos(2*pi * f_lines(:)' .* t_period + phases), 2);
        candidate = candidate / rms(candidate);

        f_phys = candidate * f_vec;
        cf = max(max(abs(f_phys), [], 1) ./ max(rms(f_phys, 1), eps));
        if cf < cf_best
            best = candidate;
            cf_best = cf;
        end
    end

    period = best;
end

function [t_sim, r_sim, u_controller] = reconstruct_controller(q1, r, t, Cfb)
    assert(size(q1,1) == numel(t), ...
        'q1 has %d rows but t has %d elements. Check Simulink fixed-step settings.', ...
        size(q1,1), numel(t));
    t_sim = t;
    r_sim = r;
    u_controller = lsim(ss(Cfb), r - q1, t);
end

function validity = validate_run(q, u_total, lim)
    validity.pos_X_ok = max(abs(q(:,1))) <= lim.pos_X && max(abs(q(:,2))) <= lim.pos_X;
    validity.pos_Y_ok = max(abs(q(:,3))) <= lim.pos_Y;
    validity.diff_ok = max(abs(q(:,1)-q(:,2))) <= lim.diff;
    validity.force_peak_ok = all(max(abs(u_total), [], 1) <= lim.force_peak);
    validity.force_rms_ok = all(rms(u_total, 1) <= lim.force_rms);
    validity.ok = validity.pos_X_ok && validity.pos_Y_ok && validity.diff_ok && ...
                  validity.force_peak_ok && validity.force_rms_ok;
end

function summary = summarize_run(Y0, mode_name, q, u_total, validity)
    summary.Y_fixed = Y0;
    summary.mode = mode_name;
    summary.valid = validity.ok;
    summary.q_min = min(q, [], 1);
    summary.q_max = max(q, [], 1);
    summary.u_peak = max(abs(u_total), [], 1);
    summary.u_rms = rms(u_total, 1);
end

function [Y_avg, U_avg] = period_average_spectra(y_modal, u_modal, N_period, n_settle, n_drop, n_clean, line_idx)
    i0 = (n_settle + n_drop) * N_period + 1;
    % reshape(., N_period, n_clean, 3) maps each channel to a page (column-major).
    % permute([1 3 2]) gives (N_period, 3, n_clean): y3(:,c,p) = channel c, period p.
    y3 = permute(reshape(y_modal(i0 : i0 + n_clean*N_period - 1, :), N_period, n_clean, 3), [1 3 2]);
    u3 = permute(reshape(u_modal(i0 : i0 + n_clean*N_period - 1, :), N_period, n_clean, 3), [1 3 2]);
    Y_avg = mean(fft(y3, [], 1), 3);
    U_avg = mean(fft(u3, [], 1), 3);
    Y_avg = Y_avg(line_idx, :);
    U_avg = U_avg(line_idx, :);
end

function save_run_time_series(run_dir, Y0, mode_name, fs, t, q, u_total, f_multisine, validity)
    safe_Y = sprintf('%+0.2f', Y0);
    safe_Y = strrep(strrep(safe_Y, '+', 'p'), '-', 'm');
    file_name = sprintf('run_Y%s_%s.mat', safe_Y, mode_name);

    t = single(t);
    q = single(q);
    u_total = single(u_total);
    f_multisine = single(f_multisine);

    save(fullfile(run_dir, file_name), ...
         'Y0', 'mode_name', 'fs', 't', 'q', 'u_total', 'f_multisine', ...
         'validity', '-v7.3');
end

function make_frf_plots(G, f, Y_grid, plot_dir)
    % Input/output labels: forces in, positions out.
    out_names = {'$X_1$', '$X_2$', '$Y$'};
    in_names  = {'$F_1$', '$F_2$', '$F_Y$'};

    plot_per_Y_matrix(G, f, Y_grid, out_names, in_names, plot_dir, ...
        @(x) 20*log10(abs(x)), 'Magnitude (dB re m/N)', 'frf_magnitude_matrix');
    plot_per_Y_matrix(G, f, Y_grid, out_names, in_names, plot_dir, ...
        @(x) unwrap(angle(x))*180/pi, 'Phase (deg)', 'frf_phase_matrix');

    [f_low, f_high, f_res, f_antires] = detect_frequency_range(G, f);
    fprintf('  Detected frequency range: f_low = %.1f Hz, f_high = %.1f Hz\n', f_low, f_high);

    make_diagonal_overlay(G, f, Y_grid, f_low, f_high, f_res, f_antires, plot_dir);
    make_frf_overlay_matrix(G, f, Y_grid, out_names, in_names, f_low, f_high, f_res, f_antires, plot_dir);
end

function plot_per_Y_matrix(G, f, Y_grid, out_names, in_names, plot_dir, tfm, ylabel_str, file_prefix)
    for iY = 1:numel(Y_grid)
        Y_mm = round(Y_grid(iY) * 1000);
        fig = figure('Visible','off', ...
                     'Name', sprintf('%s Y=%+dmm', file_prefix, Y_mm), ...
                     'Position', [0 0 1100 850]);
        tiledlayout(3, 3, 'TileSpacing','compact', 'Padding','compact');
        for iy = 1:3
            for iu = 1:3
                nexttile
                semilogx(f, tfm(squeeze(G(:,iy,iu,iY))), 'LineWidth', 1.2);
                grid on; xlim([f(1) f(end)])
                title(sprintf('%s / %s', out_names{iy}, in_names{iu}), ...
                      'Interpreter', 'latex')
                if iy == 3
                    xlabel('Frequency (Hz)')
                end
                if iu == 1
                    ylabel(ylabel_str)
                end
            end
        end
        if Y_mm >= 0
            sgtitle(sprintf('FRF matrix | Y = +%d mm', Y_mm))
        else
            sgtitle(sprintf('FRF matrix | Y = %d mm', Y_mm))
        end
        save_plot(fig, plot_dir, sprintf('%s_Y%+dmm', file_prefix, Y_mm));
    end
end

function [f_low, f_high, f_res, f_antires] = detect_frequency_range(G, f)
    % Detect identification frequency bounds from G11 and G22 diagonal FRFs.
    % G33 (Y/FY) is excluded: K[2,2] = 0 so Y/FY has no resonance or anti-resonance.
    %
    % f_low     -- min across (channels, Y) of left  half-prominence onset  of first feature
    % f_high    -- max across (channels, Y) of right half-prominence descent of last  feature
    % f_res     -- (3 x nY) cell; resonance     (peak)   frequencies, row 3 always empty
    % f_antires -- (3 x nY) cell; anti-resonance (trough) frequencies, row 3 always empty
    %
    % Features = resonances (magnitude peaks) and anti-resonances (magnitude troughs).
    % THEORY: half-prominence criterion -- Pintelon & Schoukens (2001), consistent with
    %         MATLAB findpeaks 'halfprom' width reference.
    nY = size(G, 4);
    f  = f(:);
    f_low_cands  = nan(4, nY);   % rows 1-2: peaks j=1,2; rows 3-4: troughs j=1,2
    f_high_cands = nan(4, nY);
    f_res     = cell(3, nY);
    f_antires = cell(3, nY);

    for iY = 1:nY
        for j = 1:2
            mag_dB = 20*log10(abs(squeeze(G(:,j,j,iY))));

            % Resonances (magnitude peaks)
            [pks, locs, ~, proms] = findpeaks(mag_dB, f);
            keep = proms >= 1;
            pks = pks(keep); locs = locs(keep); proms = proms(keep);
            if ~isempty(pks)
                f_res{j,iY} = locs;
                [f_low_cands(j,iY), f_high_cands(j,iY)] = ...
                    feature_bounds(mag_dB, f, pks, locs, proms);
            end

            % Anti-resonances (magnitude troughs via inverted signal)
            [pks_t, locs_t, ~, proms_t] = findpeaks(-mag_dB, f);
            keep_t = proms_t >= 1;
            pks_t = pks_t(keep_t); locs_t = locs_t(keep_t); proms_t = proms_t(keep_t);
            if ~isempty(pks_t)
                f_antires{j,iY} = locs_t;
                [f_low_cands(j+2,iY), f_high_cands(j+2,iY)] = ...
                    feature_bounds(-mag_dB, f, pks_t, locs_t, proms_t);
            end
        end
    end

    f_low  = min(f_low_cands(:),  [], 'omitnan');
    f_high = max(f_high_cands(:), [], 'omitnan');
end

function [f_lo, f_hi] = feature_bounds(mag_dB, f, pks, locs, proms)
    % Left half-prominence onset of first feature and right descent of last.
    % Pass mag_dB directly for peaks, pass -mag_dB for troughs.
    half = pks(1) - proms(1)/2;
    lf   = f(f <= locs(1));
    lm   = mag_dB(f <= locs(1));
    idx  = find(lm < half, 1, 'last');
    if ~isempty(idx) && idx < numel(lf)
        f_lo = interp1(lm(idx:idx+1), lf(idx:idx+1), half);
    else
        f_lo = f(1);
    end
    half = pks(end) - proms(end)/2;
    rf   = f(f >= locs(end));
    rm   = mag_dB(f >= locs(end));
    idx  = find(rm < half, 1, 'first');
    if ~isempty(idx) && idx > 1
        f_hi = interp1(rm(idx-1:idx), rf(idx-1:idx), half);
    else
        f_hi = f(end);
    end
end

function make_diagonal_overlay(G, f, Y_grid, f_low, f_high, f_res, f_antires, plot_dir)
    colors     = cool(numel(Y_grid));
    line_color = [0.15 0.15 0.15];
    diag_labels = {'$G_{11}$: $X_1 / F_1$', '$G_{22}$: $X_2 / F_2$', '$G_{33}$: $Y / F_Y$'};

    fig = figure('Visible','off', 'Name', 'Diagonal FRF overlay', ...
                 'Position', [0 0 700 750]);
    tiledlayout(3, 1, 'TileSpacing','compact', 'Padding','compact');

    for j = 1:3
        nexttile; hold on; set(gca, 'XScale', 'log')
        for iY = 1:numel(Y_grid)
            mag_dB = 20*log10(abs(squeeze(G(:,j,j,iY))));
            semilogx(f, mag_dB, 'LineWidth', 1.2, 'Color', colors(iY,:), ...
                     'DisplayName', sprintf('Y = %+d mm', round(Y_grid(iY)*1000)));
            if j <= 2
                add_feature_markers(f, mag_dB, f_res{j,iY},     '^', colors(iY,:));
                add_feature_markers(f, mag_dB, f_antires{j,iY}, 'v', colors(iY,:));
            end
        end
        grid on; xlim([f(1) f(end)])
        if j == 1
            xline(f_low,  '--', '$f_{\rm low}$',  'Color', line_color, 'LineWidth', 1.5, ...
                  'HandleVisibility','off', 'Interpreter','latex', ...
                  'LabelVerticalAlignment','top', 'LabelHorizontalAlignment','right');
            xline(f_high, '--', '$f_{\rm high}$', 'Color', line_color, 'LineWidth', 1.5, ...
                  'HandleVisibility','off', 'Interpreter','latex', ...
                  'LabelVerticalAlignment','top', 'LabelHorizontalAlignment','left');
        else
            xline(f_low,  '--', 'Color', line_color, 'LineWidth', 1.5, 'HandleVisibility','off');
            xline(f_high, '--', 'Color', line_color, 'LineWidth', 1.5, 'HandleVisibility','off');
        end
        ylabel('Magnitude (dB re m/N)')
        title(diag_labels{j}, 'Interpreter','latex')
        if j == 1; legend('Location','best', 'FontSize', 8); end
        if j == 3; xlabel('Frequency (Hz)'); end
    end
    sgtitle('Diagonal FRF elements - all Y positions')
    save_plot(fig, plot_dir, 'frf_diagonal_overlay');
end

function make_frf_overlay_matrix(G, f, Y_grid, out_names, in_names, f_low, f_high, f_res, f_antires, plot_dir)
    nY         = numel(Y_grid);
    colors     = cool(nY);
    line_color = [0.15 0.15 0.15];

    fig = figure('Visible','off', 'Name', 'FRF matrix overlay', ...
                 'Position', [0 0 1100 850]);
    tiledlayout(3, 3, 'TileSpacing','compact', 'Padding','compact');

    for iy = 1:3
        for iu = 1:3
            nexttile; hold on; set(gca, 'XScale', 'log')
            for iY = 1:nY
                mag_dB = 20*log10(abs(squeeze(G(:,iy,iu,iY))));
                semilogx(f, mag_dB, 'LineWidth', 1.2, 'Color', colors(iY,:), ...
                         'DisplayName', sprintf('Y = %+d mm', round(Y_grid(iY)*1000)));
                % Resonance and anti-resonance markers on diagonal panels G11 and G22
                if iy == iu && iy <= 2
                    add_feature_markers(f, mag_dB, f_res{iy,iY},     '^', colors(iY,:));
                    add_feature_markers(f, mag_dB, f_antires{iy,iY}, 'v', colors(iY,:));
                end
            end
            grid on; xlim([f(1) f(end)])
            if iy == 1 && iu == 1
                xline(f_low,  '--', '$f_{\rm low}$',  'Color', line_color, 'LineWidth', 1.5, ...
                      'HandleVisibility','off', 'Interpreter','latex', ...
                      'LabelVerticalAlignment','top', 'LabelHorizontalAlignment','right');
                xline(f_high, '--', '$f_{\rm high}$', 'Color', line_color, 'LineWidth', 1.5, ...
                      'HandleVisibility','off', 'Interpreter','latex', ...
                      'LabelVerticalAlignment','top', 'LabelHorizontalAlignment','left');
            else
                xline(f_low,  '--', 'Color', line_color, 'LineWidth', 1.5, 'HandleVisibility','off');
                xline(f_high, '--', 'Color', line_color, 'LineWidth', 1.5, 'HandleVisibility','off');
            end
            title(sprintf('%s / %s', out_names{iy}, in_names{iu}), 'Interpreter','latex')
            if iu == 1; ylabel('Magnitude (dB re m/N)'); end
            if iy == 3; xlabel('Frequency (Hz)'); end
            if iy == 1 && iu == 3; legend('Location','best', 'FontSize', 7); end
        end
    end
    sgtitle('FRF matrix - all Y positions')
    save_plot(fig, plot_dir, 'frf_matrix_overlay');
end

function add_feature_markers(f, mag_dB, f_feat, marker, color)
    % Plot up/down triangle markers at feature frequencies (resonances or anti-resonances).
    % marker = '^' for resonances, 'v' for anti-resonances.
    if isempty(f_feat); return; end
    for ip = 1:numel(f_feat)
        [~, ki] = min(abs(f - f_feat(ip)));
        plot(f(ki), mag_dB(ki), marker, 'MarkerSize', 6, ...
             'MarkerFaceColor', color, 'MarkerEdgeColor', 'none', ...
             'HandleVisibility', 'off');
    end
end

function make_condition_plot(cond_U, f, Y_grid, plot_dir)
    colors = cool(numel(Y_grid));
    fig = figure('Visible','off', 'Name', 'Input matrix condition number', ...
                 'Position', [0 0 700 400]);
    hold on
    for iY = 1:numel(Y_grid)
        plot(f, cond_U(:,iY), 'LineWidth', 1.2, 'Color', colors(iY,:), ...
             'DisplayName', sprintf('Y = %+d mm', round(Y_grid(iY)*1000)));
    end
    set(gca, 'XScale', 'log', 'YScale', 'log')
    grid on; xlim([f(1) f(end)])
    xlabel('Frequency (Hz)')
    ylabel('Condition number')
    title('Excitation input matrix condition number')
    legend('Location','best')
    save_plot(fig, plot_dir, 'input_condition_number');
end

function save_plot(fig, plot_dir, base_name)
    safe = strrep(strrep(base_name, '+', 'p'), '-', 'm');
    saveas(fig, fullfile(plot_dir, [safe, '.png']));
    close(fig)
end

function ensure_dir(path_name)
    if ~exist(path_name, 'dir')
        mkdir(path_name);
    end
end
