% diagnostics_frf_probe.m
% Closed-loop FRF probe for post-controller force-multisine design.
%
% Purpose
% -------
% This is a preparatory experiment, not the final identification experiment.
% It measures the closed-loop disturbance path that the final force multisine
% will use:
%
%     d_probe -> q
%
% The probe estimates MIMO FRM columns from one input direction at a time:
% common X, differential X, and Y. The outputs are always inspected in all
% modal coordinates:
%
%     q_common = (X1 + X2)/2,  q_diff = X1 - X2,  q_Y = Y - Y0.
%
% Lecture-backed structure
% ------------------------
% System Identification lecture support:
%   - Lecture 9 slide 6: preparatory experiments obtain dynamics information.
%   - Lecture 9 slide 8: frequency/impulse-response analysis gives frequency
%     range of interest and length of experiment.
%   - Lecture 3 slide 30: multisine input is used for FRF estimation.
%   - Lecture 3 slides 39 and 59: periodic transient-free data avoids
%     leakage/transient contribution R(n).
%   - Lecture 9 slide 47: MIMO frequency response matrix Y(f) = G(f)U(f).
%   - Lecture 9 slides 49 and 51: zippered/orthogonal multisines are MIMO
%     extensions; this first probe uses simpler one-input-at-a-time columns.
%
% Engineering choices
% -------------------
% The exact probe band, amplitude, convergence metric, relevance thresholds,
% and impulse-response tail cutoffs are not fixed by the lectures. They are
% reported explicitly below and should be refined from the measured results.
%
% Run from repo root:
%   run('Matlab-scripts/diagnostics_frf_probe.m')

clearvars
close all
clc

addpath(genpath(fullfile(pwd, 'kamtin-fp-model', '03 Simulink gantry')))

% Physical parameters. Keep these synchronized with the experiment generator.
mb=22.8; mh=10.1; m1=10.2; m2=10.7; Jb=1.0; Jh=0.05;
cg1=14.5; cg2=20.3; cy=10; cb1=9; cb2=9;
kb1=1987.5; kb2=1987.5; Lb=0.725; Lh=0.25; d=0.1; % Lh is used by the Simulink model.
cc1=16.8; cc2=18.35; ccy=11.6; % Simulink workspace parameters

C_damp = [cg1+cg2,         (cg1-cg2)*Lb/2,             0;
          (cg1-cg2)*Lb/2,  cb1+cb2+(cg1+cg2)*Lb^2/4,   0;
          0,                0,                           cy];
K_stiff = [0,0,0; 0,kb1+kb2,0; 0,0,0];
n   = 3;
P   = [1,1,0; Lb/2,-Lb/2,0; 0,0,1];
fs  = 20e3;
ts  = 1/fs;
fbw_hz = 100;       % Controller design prior, not a hard FRF band edge.
mdl = 'gantry_2025a';

% Hardware limits (TELICA spec). Used only for validity checks.
lim.pos_X      = 0.375;               % [m]
lim.pos_Y      = 0.400;               % [m]
lim.diff       = sin(0.1) * Lb;       % [m]
lim.force_peak = [2000, 2000, 1420];  % [N] peak [FX1, FX2, FY]
lim.force_rms  = [916,  916,  656];   % [N] RMS

% Probe settings. These are engineering initial values, initialized from
% controller/model prior knowledge and meant to be refined from the measured FRF.
Y_vals          = [-0.35, 0.0, 0.35]; % SYSTEM-SPECIFIC: interior frozen LPV positions
F_PROBE_LOW_HZ  = 5;                 % ENGINEERING: broad low edge for first scan
F_PROBE_HIGH_HZ = 200;               % ENGINEERING: margin above expected CL modes
DF_PROBE_HZ     = 5;                 % ENGINEERING: coarse scan, T_probe = 0.2 s
N_PERIODS       = 10;                % ENGINEERING: period budget, not fixed discard
A_FRAC          = 0.05;              % ENGINEERING: small-signal force RMS start
MIN_KEEP        = 3;                 % ENGINEERING: minimum periods for averaging
LOCAL_Y_MAX_M   = 0.02;              % ENGINEERING: frozen-Y locality check

% Data-analysis engineering settings. Report sensitivity rather than treating
% one value as theory.
TAIL_LEVELS = [0.10, 0.05, 0.01];
FRF_RELATIVE_FLOOR = 0.01;   % ENGINEERING: candidate visible-band marker only

N_period = round(fs / DF_PROBE_HZ);
T_probe  = N_period / fs;
f0       = fs / N_period;
f_lines  = F_PROBE_LOW_HZ:f0:F_PROBE_HIGH_HZ;

if abs(f0 - DF_PROBE_HZ) > 10*eps
    error('DF_PROBE_HZ must divide fs cleanly for this minimal script.');
end

modes(1).name = 'common'; modes(1).fv = [1, 1, 0]; modes(1).F_lim = min(lim.force_peak(1:2));
modes(2).name = 'diff';   modes(2).fv = [1,-1, 0]; modes(2).F_lim = min(lim.force_peak(1:2));
modes(3).name = 'Y';      modes(3).fv = [0, 0, 1]; modes(3).F_lim = lim.force_peak(3);

out_names = {'common', 'diff', 'Y'};
results = struct('Y0',{},'input',{},'f',{},'G_dq',{},'G_uq_diag',{}, ...
                 'period_error',{},'keep_periods',{},'tail_times',{}, ...
                 'visible_band',{},'validity',{},'amp_rms',{},'amp_peak',{});

fprintf('\n%s\nClosed-loop FRF probe\n%s\n', repmat('=',1,72), repmat('=',1,72));
fprintf('Probe band %.1f-%.1f Hz, Df=%.1f Hz, T_probe=%.3f s, periods=%d\n', ...
        F_PROBE_LOW_HZ, F_PROBE_HIGH_HZ, f0, T_probe, N_PERIODS);
fprintf('Controller bandwidth prior: %.1f Hz (not used as a hard cutoff)\n', fbw_hz);

res_idx = 0;
for iY = 1:numel(Y_vals)
    Y0 = Y_vals(iY);

    M_op = [m1+m2+mb+mh,           (m1-m2)*Lb/2-mh*Y0,                   0;
            (m1-m2)*Lb/2-mh*Y0,    Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y0^2, -mh*d;
            0,                      -mh*d,                                  mh];
    sys_ct = P.' * getss(n, M_op, C_damp, K_stiff) * P;

    Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)), ts);
    for j = 1:3
        Cfb(j,j) = ruleOfThumb(fbw_hz, sys_ct(j,j), ts);
    end

    fprintf('\nY0 = %+.1f m\n', Y0);
    for im = 1:numel(modes)
        md = modes(im);
        A_rms = A_FRAC * md.F_lim;
        N_total = N_PERIODS * N_period;
        t_vec = (0:N_total-1)' * ts;
        r = repmat([0, 0, Y0], N_total, 1);
        d_scalar = random_phase_multisine(N_period, N_PERIODS, fs, f_lines, 100*iY + im);
        d_scalar = A_rms * d_scalar / rms(d_scalar);
        f = d_scalar * md.fv;
        t = t_vec;
        Y = Y0; % Used by the Simulink model.

        sim(mdl, t_vec(end));

        [t_sim, r_sim, u_ctrl] = reconstruct(q1, r, t_vec, Cfb);
        q_uniform = interp1(t_sim, q1, t_vec, 'linear', 'extrap');
        u_ctrl_uniform = interp1(t_sim, u_ctrl, t_vec, 'linear', 'extrap');
        f_uniform = f;
        u_total = u_ctrl_uniform + f_uniform;
        q_modal = [(q_uniform(:,1)+q_uniform(:,2))/2, ...
                    q_uniform(:,1)-q_uniform(:,2), ...
                    q_uniform(:,3)-Y0];

        validity = validate_probe(q_uniform, u_total, Y0, lim, LOCAL_Y_MAX_M);
        [keep_periods, period_error] = choose_periods(q_modal, N_period, MIN_KEEP);
        [G_dq, G_uq_diag] = estimate_probe_frf(q_modal, f_uniform*md.fv', ...
                                               u_total*md.fv', N_period, ...
                                               keep_periods, fs, f_lines);
        [tail_times, tail_curves] = frf_memory_tail(G_dq, TAIL_LEVELS, fs, f_lines, N_period);
        visible_band = suggest_visible_band(G_dq, f_lines, FRF_RELATIVE_FLOOR);

        res_idx = res_idx + 1;
        results(res_idx).Y0 = Y0;
        results(res_idx).input = md.name;
        results(res_idx).f = f_lines;
        results(res_idx).G_dq = G_dq;
        results(res_idx).G_uq_diag = G_uq_diag;
        results(res_idx).period_error = period_error;
        results(res_idx).keep_periods = keep_periods;
        results(res_idx).tail_times = tail_times;
        results(res_idx).tail_curves = tail_curves;
        results(res_idx).visible_band = visible_band;
        results(res_idx).validity = validity;
        results(res_idx).amp_rms = rms(f_uniform);
        results(res_idx).amp_peak = max(abs(f_uniform), [], 1);

        fprintf('  %-8s keep periods %s | valid=%d | visible %.1f-%.1f Hz | tail 5%% max %.4f s\n', ...
                md.name, mat2str(keep_periods), validity.ok, ...
                visible_band.f_low, visible_band.f_high, max(tail_times(:,2)));
    end
end

summary = summarize_results(results, TAIL_LEVELS);

fprintf('\n%s\nFRF-probe summary\n%s\n', repmat('=',1,72), repmat('-',1,72));
fprintf('  Candidate visible band: %.1f-%.1f Hz\n', summary.f_low, summary.f_high);
fprintf('  Candidate Fs_new lower bound: %.1f Hz (10*f_high, lecture-style sampling margin)\n', ...
        summary.Fs_new_candidate);
for k = 1:numel(TAIL_LEVELS)
    fprintf('  Worst impulse-tail memory at %.0f%%: %.4f s\n', ...
            100*TAIL_LEVELS(k), summary.T_memory(k));
end
fprintf('  NOTE: visible-band floor and tail cutoffs are engineering diagnostics, not lecture constants.\n');
fprintf('%s\n', repmat('=',1,72));

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'Matlab-output');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
plot_dir = fullfile(out_dir, 'frf-probe-plots');
if ~exist(plot_dir, 'dir'), mkdir(plot_dir); end
make_probe_plots(results, TAIL_LEVELS, plot_dir, out_names, fs);
save(fullfile(out_dir, 'frf_probe_outputs.mat'), ...
     'results', 'summary', 'Y_vals', 'F_PROBE_LOW_HZ', 'F_PROBE_HIGH_HZ', ...
     'DF_PROBE_HZ', 'T_probe', 'N_PERIODS', 'A_FRAC', 'TAIL_LEVELS', ...
     'FRF_RELATIVE_FLOOR', 'LOCAL_Y_MAX_M', 'fbw_hz', 'plot_dir');
fprintf('Saved: Matlab-output/frf_probe_outputs.mat\n');
fprintf('Saved plots in: %s\n', plot_dir);

% Local functions ----------------------------------------------------------

function sig = random_phase_multisine(N_period, N_periods, fs, f_lines, seed)
% ENGINEERING: random phases are adequate for the diagnostic probe. The final
% identification multisine may later use Schroeder, zippered, or orthogonal
% structure. This function returns one unit-RMS period tiled N_periods times.
    rng(seed);
    t = (0:N_period-1)' / fs;
    phase = 2*pi*rand(1, numel(f_lines));
    one_period = zeros(N_period, 1);
    for k = 1:numel(f_lines)
        one_period = one_period + cos(2*pi*f_lines(k)*t + phase(k));
    end
    one_period = one_period / rms(one_period);
    sig = repmat(one_period, N_periods, 1);
end

function [t_sim, r_sim, u_q1] = reconstruct(q1, r, t, Cfb)
% Reconstruct feedback-controller output u_q1 = Cfb*(r-q1). FRF analysis
% requires the data to be placed back on the command grid after this step.
    Ns = size(q1, 1);
    if Ns == numel(t)
        t_sim = t;
        r_sim = r;
    elseif evalin('caller', 'exist(''tout'', ''var'')')
        t_sim = evalin('caller', 'tout');
        t_sim = t_sim(:);
        if numel(t_sim) ~= Ns
            error('Logged tout length (%d) does not match q1 length (%d).', numel(t_sim), Ns);
        end
        r_sim = interp1(t, r, t_sim, 'linear', 'extrap');
    else
        error('q1 length differs from command grid, but no logged tout was found.');
    end
    u_q1 = lsim(ss(Cfb), r_sim - q1, t_sim);
end

function validity = validate_probe(q, u_total, Y0, lim, local_y_max)
% Time-domain validity gate. FRFs from nonphysical runs should not be used for
% final multisine design.
    validity.pos_ok = max(abs(q(:,1))) <= lim.pos_X && ...
                      max(abs(q(:,2))) <= lim.pos_X && ...
                      max(abs(q(:,3))) <= lim.pos_Y && ...
                      max(abs(q(:,1)-q(:,2))) <= lim.diff;
    validity.local_y_ok = max(abs(q(:,3)-Y0)) <= local_y_max;
    validity.force_peak_ok = all(max(abs(u_total), [], 1) <= lim.force_peak);
    validity.force_rms_ok = all(rms(u_total, 1) <= lim.force_rms);
    validity.ok = validity.pos_ok && validity.local_y_ok && ...
                  validity.force_peak_ok && validity.force_rms_ok;
end

function [keep_periods, period_error] = choose_periods(q_modal, N_period, min_keep)
% Lecture principle: use transient-free periods. Engineering implementation:
% report period-to-period convergence and keep the last min_keep periods. This
% avoids pretending a hard convergence tolerance is lecture-derived.
    N_periods = floor(size(q_modal,1) / N_period);
    period_error = nan(N_periods, 1);
    for p = 2:N_periods
        a = q_modal((p-1)*N_period + (1:N_period), :);
        b = q_modal((p-2)*N_period + (1:N_period), :);
        period_error(p) = norm(a(:)-b(:)) / max(norm(a(:)), eps);
    end
    keep_periods = max(1, N_periods-min_keep+1):N_periods;
end

function [G_dq, G_uq_diag] = estimate_probe_frf(q_modal, d_scalar, u_scalar, ...
                                                N_period, keep_periods, fs, f_lines)
% Lecture 3 periodic-data estimate: Ghat = sum_p Y_p / sum_p U_p on excited
% DFT lines. Here d_scalar is the injected force along the active input
% direction, so the result is one MIMO FRM column.
    N_freq = numel(f_lines);
    G_dq = nan(3, N_freq);
    G_uq_diag = nan(3, N_freq);
    line_idx = round(f_lines / (fs/N_period)) + 1;

    sum_D = zeros(1, N_freq);
    sum_U = zeros(1, N_freq);
    sum_Q = zeros(3, N_freq);
    for p = keep_periods
        idx = (p-1)*N_period + (1:N_period);
        D = fft(d_scalar(idx));
        U = fft(u_scalar(idx));
        for iy = 1:3
            Q = fft(q_modal(idx,iy));
            sum_Q(iy,:) = sum_Q(iy,:) + Q(line_idx).';
        end
        sum_D = sum_D + D(line_idx).';
        sum_U = sum_U + U(line_idx).';
    end

    for iy = 1:3
        G_dq(iy,:) = sum_Q(iy,:) ./ sum_D;
        G_uq_diag(iy,:) = sum_Q(iy,:) ./ sum_U;
    end
end

function [tail_times, tail_curves] = frf_memory_tail(G_dq, tail_levels, fs, f_lines, N_period)
% Engineering extension to Lecture 9 slide 8 impulse-response analysis:
% build a Hermitian spectrum from measured positive-frequency FRF lines and
% inspect normalized impulse-response energy tail. This is a diagnostic memory
% estimate, not a proof of BPTT segment optimality.
    tail_times = nan(3, numel(tail_levels));
    tail_curves = cell(3,1);
    line_idx = round(f_lines / (fs/N_period)) + 1;
    for iy = 1:3
        H = zeros(N_period, 1);
        H(line_idx) = G_dq(iy,:).';
        neg_idx = N_period - line_idx + 2;
        valid = neg_idx >= 1 & neg_idx <= N_period;
        H(neg_idx(valid)) = conj(G_dq(iy,valid).');
        g = real(ifft(H));
        e = abs(g).^2;
        if sum(e) <= eps
            tail_curves{iy} = nan(N_period,1);
            continue
        end
        tail = flipud(cumsum(flipud(e))) / sum(e);
        tail_curves{iy} = tail;
        for k = 1:numel(tail_levels)
            idx = find(tail <= tail_levels(k), 1, 'first');
            if ~isempty(idx)
                tail_times(iy,k) = (idx-1) / fs;
            else
                tail_times(iy,k) = N_period / fs;
            end
        end
    end
end

function band = suggest_visible_band(G_dq, f_lines, rel_floor)
% Engineering marker for a candidate visible band. The threshold is only for
% automation; plots/FRF repeatability should decide the final band.
    mag = vecnorm(abs(G_dq), 2, 1);
    if max(mag) <= eps
        band.f_low = NaN;
        band.f_high = NaN;
        return
    end
    keep = mag >= rel_floor * max(mag);
    band.f_low = min(f_lines(keep));
    band.f_high = max(f_lines(keep));
end

function summary = summarize_results(results, tail_levels)
    f_lows = arrayfun(@(r) r.visible_band.f_low, results);
    f_highs = arrayfun(@(r) r.visible_band.f_high, results);
    valid = ~isnan(f_lows) & ~isnan(f_highs) & arrayfun(@(r) r.validity.ok, results);
    if any(valid)
        summary.f_low = min(f_lows(valid));
        summary.f_high = max(f_highs(valid));
    else
        warning('No fully valid FRF probe run found; summarizing all runs for inspection.');
        summary.f_low = min(f_lows, [], 'omitnan');
        summary.f_high = max(f_highs, [], 'omitnan');
    end
    summary.Fs_new_candidate = 10 * summary.f_high;
    summary.T_memory = nan(1, numel(tail_levels));
    for k = 1:numel(tail_levels)
        vals = [];
        for i = 1:numel(results)
            vals = [vals; results(i).tail_times(:,k)]; %#ok<AGROW>
        end
        summary.T_memory(k) = max(vals, [], 'omitnan');
    end
end

function make_probe_plots(results, tail_levels, plot_dir, out_names, fs)
% Diagnostic plots are part of the method, not decoration. The automatic
% visible-band threshold is only a marker; these plots are what should be
% inspected before choosing the final multisine band.
    y_vals = unique([results.Y0]);
    input_names = unique({results.input}, 'stable');

    for iy = 1:numel(y_vals)
        Y0 = y_vals(iy);
        rows = find([results.Y0] == Y0);

        fig = figure('Visible','off', 'Name', sprintf('FRF Y0=%+.2f', Y0));
        tiledlayout(numel(rows), 1, 'TileSpacing','compact');
        for ir = 1:numel(rows)
            r = results(rows(ir));
            nexttile
            semilogy(r.f, abs(r.G_dq(1,:)), '-o', ...
                     r.f, abs(r.G_dq(2,:)), '-s', ...
                     r.f, abs(r.G_dq(3,:)), '-^', 'LineWidth', 1.0);
            grid on
            xlabel('Frequency [Hz]')
            ylabel('|Q/D| [m/N]')
            title(sprintf('Y0=%+.2f, input=%s, valid=%d', Y0, r.input, r.validity.ok))
            legend(out_names, 'Location','best')
        end
        save_plot(fig, plot_dir, sprintf('frf_magnitude_Y%+.2f', Y0));
    end

    for ii = 1:numel(input_names)
        in_name = input_names{ii};
        rows = find(strcmp({results.input}, in_name));

        fig = figure('Visible','off', 'Name', sprintf('Period convergence %s', in_name));
        hold on
        for ir = 1:numel(rows)
            r = results(rows(ir));
            plot(1:numel(r.period_error), r.period_error, '-o', ...
                 'DisplayName', sprintf('Y0=%+.2f', r.Y0));
        end
        grid on
        xlabel('Period index')
        ylabel('||q_p - q_{p-1}|| / ||q_p||')
        title(sprintf('Period-to-period convergence, input=%s', in_name))
        legend('Location','best')
        save_plot(fig, plot_dir, sprintf('period_convergence_%s', in_name));
    end

    for iy = 1:numel(y_vals)
        Y0 = y_vals(iy);
        rows = find([results.Y0] == Y0);

        fig = figure('Visible','off', 'Name', sprintf('Impulse tails Y0=%+.2f', Y0));
        tiledlayout(numel(rows), 1, 'TileSpacing','compact');
        for ir = 1:numel(rows)
            r = results(rows(ir));
            nexttile
            hold on
            t_tail = (0:numel(r.tail_curves{1})-1) / fs;
            for io = 1:3
                semilogy(t_tail, r.tail_curves{io}, 'LineWidth', 1.0, ...
                         'DisplayName', out_names{io});
            end
            for k = 1:numel(tail_levels)
                yline(tail_levels(k), ':', sprintf('%.0f%%', 100*tail_levels(k)));
            end
            grid on
            xlabel('Time [s]')
            ylabel('Tail energy [-]')
            title(sprintf('Y0=%+.2f, input=%s', Y0, r.input))
            legend('Location','best')
        end
        save_plot(fig, plot_dir, sprintf('impulse_tail_Y%+.2f', Y0));
    end
end

function save_plot(fig, plot_dir, stem)
    safe_stem = strrep(stem, '+', 'p');
    safe_stem = strrep(safe_stem, '-', 'm');
    safe_stem = strrep(safe_stem, '.', 'p');
    png_path = fullfile(plot_dir, [safe_stem, '.png']);
    fig_path = fullfile(plot_dir, [safe_stem, '.fig']);
    exportgraphics(fig, png_path, 'Resolution', 150);
    savefig(fig, fig_path);
    close(fig);
end
