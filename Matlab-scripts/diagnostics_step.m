% diagnostics_step.m
% Step response preparatory experiment for closed-loop BPTT data.
%
% This is not an identification experiment. It estimates the slowest visible
% transient in the closed-loop data path used for training:
%
%     u_total -> q
%
% The force step is injected after the controller, at the same location as the
% force multisine. Therefore the measured response is the closed-loop
% disturbance-to-output path, not the naked open-loop plant.
%
% Primary outputs:
%   T_ms   = max(Tsettle_worst, 10*tau_slowest)  multisine-period lower bound [s]
%   Df     = 1/T_ms                              frequency resolution [Hz]
%   Fs_new = 10*fmax                             resampling-rate lower bound [Hz]
%   f_low  = fbw_hz                              initial lower band edge [Hz]
%   f_high = fres_max, or f_tau if no oscillation is visible [Hz]
%
% N_seg is not an output. Determine it later by segment-length sweep.
%
% Run from repo root:
%   run('Matlab-scripts/diagnostics_step.m')

clearvars
addpath(genpath(fullfile(pwd, 'kamtin-fp-model', '03 Simulink gantry')))

% Physical parameters. Keep these synchronized with the experiment generator.
mb=22.8; mh=10.1; m1=10.2; m2=10.7; Jb=1.0; Jh=0.05;
cg1=14.5; cg2=20.3; cy=10; cb1=9; cb2=9;
kb1=1987.5; kb2=1987.5; Lb=0.725; Lh=0.25; d=0.1;
cc1=16.8; cc2=18.35; ccy=11.6; % Simulink workspace parameters
C_damp = [cg1+cg2,         (cg1-cg2)*Lb/2,             0;
          (cg1-cg2)*Lb/2,  cb1+cb2+(cg1+cg2)*Lb^2/4,   0;
          0,                0,                           cy];
K_stiff = [0,0,0; 0,kb1+kb2,0; 0,0,0];
n   = 3;
P   = [1,1,0; Lb/2,-Lb/2,0; 0,0,1];
fs  = 20e3;
ts  = 1/fs;
fbw_hz = 100;                 % ruleOfThumb expects Hz and internally uses 2*pi*fbw
wbw    = 2*pi*fbw_hz;         % angular bandwidth [rad/s], only for angular formulas
mdl = 'gantry_2025a';

% Hardware limits (TELICA spec).
F_peak = [2000, 2000, 1420];  % [N] peak per actuator [FX1, FX2, FY]
F_rms  = [916,  916,  656];   % [N] RMS per actuator [FX1, FX2, FY]

% Experiment settings.
A_FRAC       = 0.10;    % HEURISTIC: 10% peak force step
T_HOLD       = 0.10;    % [s] pre-step hold
T_TOTAL_INIT = 0.40;    % [s] first try
T_TOTAL_MAX  = 3.20;    % [s] adaptive upper limit
THR_PCT      = 0.05;    % Lecture 9 slide 9: tau_set,95
COUPLING_THR = 0.05;    % HEURISTIC: reliability filter for weak coupled q channels
USE_COUPLING_FILTER = true;
ABS_MIN_Q_M  = 1e-5;    % [m] q channels below this are treated as not reliably visible
ABS_MIN_U_N  = 1.0;     % [N] u_total projection below this is treated as not visible

% Force directions in physical actuator coordinates [FX1, FX2, FY].
modes(1).name = 'common';  modes(1).fv = [1, 1, 0];  modes(1).F_lim = min(F_peak(1:2));
modes(2).name = 'diff';    modes(2).fv = [1,-1, 0];  modes(2).F_lim = min(F_peak(1:2));
modes(3).name = 'Y';       modes(3).fv = [0, 0, 1];  modes(3).F_lim = F_peak(3);

Y_vals = [-0.4, 0.0, 0.4];
ch_names = {'common', 'diff', 'Y', 'u_total'};

results = struct('peak',{},'Tsettle',{},'tau',{},'fres',{},'visible',{}, ...
                 'settled_by_end',{},'Y0',{},'input',{},'output',{}, ...
                 'kind',{},'coupling_ratio',{});
force_reports = struct('Y0',{},'input',{},'peak',{},'rms',{},'ok_peak',{},'ok_rms',{});
res_idx = 0;
force_idx = 0;

warning('off', 'signal:findpeaks:largeMinPeakHeight');

fprintf('\n%s\nStep response diagnostic (%d Y-points x %d modes)\n%s\n', ...
        repmat('=',1,72), numel(Y_vals), numel(modes), repmat('=',1,72));
fprintf('Controller bandwidth: fbw_hz = %.1f Hz (wbw = %.1f rad/s)\n', fbw_hz, wbw);

for i = 1:numel(Y_vals)
    Y0 = Y_vals(i);

    M_op = [m1+m2+mb+mh,           (m1-m2)*Lb/2-mh*Y0,                   0;
            (m1-m2)*Lb/2-mh*Y0,    Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y0^2, -mh*d;
            0,                      -mh*d,                                  mh];
    sys_ct = P.' * getss(n, M_op, C_damp, K_stiff) * P;

    % ruleOfThumb returns a discrete-time controller with sample time ts.
    Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)), ts);
    for j = 1:3
        Cfb(j,j) = ruleOfThumb(fbw_hz, sys_ct(j,j), ts);
    end

    fprintf('\nY0 = %+.1f m\n', Y0);

    for m = 1:numel(modes)
        md = modes(m);
        A = A_FRAC * md.F_lim;

        run_ok = false;
        T_TOTAL = T_TOTAL_INIT;
        last_run = struct();

        while ~run_ok
            [t_vec, f_step, r] = build_step_run(T_TOTAL, T_HOLD, fs, Y0, md.fv, A);
            t = t_vec;
            f = f_step;
            Y = Y0;

            sim(mdl, t_vec(end));

            [t_sim, ~, u_ctrl] = reconstruct(q1, r, t_vec, Cfb);
            f_out = resample_to(f_step, t_vec, t_sim);
            u_tot = u_ctrl + f_out;

            q_modal = [(q1(:,1)+q1(:,2))/2,  q1(:,1)-q1(:,2),  q1(:,3)-Y0];
            ch_sigs = {q_modal(:,1), q_modal(:,2), q_modal(:,3), u_tot * md.fv'};
            ch_kind = {'q','q','q','u'};
            ch_abs_min = [ABS_MIN_Q_M, ABS_MIN_Q_M, ABS_MIN_Q_M, ABS_MIN_U_N];

            tmp = repmat(empty_result(), 1, 4);
            for c = 1:4
                tmp(c) = extract_channel(ch_sigs{c}, t_sim, T_HOLD, THR_PCT, ch_abs_min(c));
                tmp(c).Y0 = Y0;
                tmp(c).input = md.name;
                tmp(c).output = ch_names{c};
                tmp(c).kind = ch_kind{c};
            end

            visible_settled = [tmp.visible] & [tmp.settled_by_end];
            visible_unsettled = [tmp.visible] & ~[tmp.settled_by_end];
            run_ok = ~any(visible_unsettled);

            last_run.tmp = tmp;
            last_run.u_tot = u_tot;

            if ~run_ok
                if T_TOTAL >= T_TOTAL_MAX
                    warning('%s at Y0=%+.1f did not settle by %.2f s; keeping flagged results.', ...
                            md.name, Y0, T_TOTAL);
                    run_ok = true;
                else
                    T_TOTAL = min(2*T_TOTAL, T_TOTAL_MAX);
                end
            end

            % If nothing is visible, do not keep doubling forever.
            if ~any([tmp.visible]) && T_TOTAL >= T_TOTAL_INIT
                run_ok = true;
            end

            if ~any(visible_settled) && any([tmp.visible]) && T_TOTAL < T_TOTAL_MAX && ~run_ok
                % Continue adaptive loop.
            end
        end

        for c = 1:4
            res_idx = res_idx + 1;
            results(res_idx) = last_run.tmp(c);
        end

        force_idx = force_idx + 1;
        force_reports(force_idx) = summarize_forces(Y0, md.name, last_run.u_tot, F_peak, F_rms);
        if ~force_reports(force_idx).ok_peak || ~force_reports(force_idx).ok_rms
            warning('%s at Y0=%+.1f exceeds force limits: peak=[%.0f %.0f %.0f], rms=[%.0f %.0f %.0f]', ...
                    md.name, Y0, force_reports(force_idx).peak, force_reports(force_idx).rms);
        end

        fprintf('  %-8s  A = %5.0f N  T_total = %.2f s\n', md.name, A, T_TOTAL);
    end
end

% Coupling ratios for q outputs. u_total is always retained as a separate input
% transient diagnostic.
all_Y0  = [results.Y0];
all_in  = {results.input};
all_out = {results.output};
all_pk  = [results.peak];

for k = 1:numel(results)
    if strcmp(results(k).output, 'u_total')
        results(k).coupling_ratio = 1.0;
        continue
    end

    is_diag = all_Y0 == results(k).Y0 & ...
              strcmp(all_in,  results(k).input) & ...
              strcmp(all_out, results(k).input);
    if any(is_diag) && all_pk(find(is_diag,1)) > 0
        results(k).coupling_ratio = results(k).peak / all_pk(find(is_diag,1));
    else
        results(k).coupling_ratio = 0;
    end
end

visible = results([results.visible]);
settled_visible = [results.visible] & [results.settled_by_end];
coupling_mask = ([results.coupling_ratio] >= COUPLING_THR) | ~USE_COUPLING_FILTER;
relevant = results(settled_visible & coupling_mask);

if isempty(relevant)
    error('No relevant settled channels found. Increase T_TOTAL_MAX/A_FRAC or lower COUPLING_THR.');
end

q_relevant = relevant(strcmp({relevant.kind}, 'q'));
u_relevant = relevant(strcmp({relevant.kind}, 'u'));

Tsettle_q_all = [q_relevant.Tsettle]; Tsettle_q_all = Tsettle_q_all(~isnan(Tsettle_q_all));
Tsettle_u_all = [u_relevant.Tsettle]; Tsettle_u_all = Tsettle_u_all(~isnan(Tsettle_u_all));
tau_all       = [relevant.tau];       tau_all       = tau_all(~isnan(tau_all));
tau_q_all     = [q_relevant.tau];     tau_q_all     = tau_q_all(~isnan(tau_q_all));
fres_all      = [q_relevant.fres];    fres_all      = fres_all(~isnan(fres_all));

if isempty(tau_all) || isempty(tau_q_all) || isempty([Tsettle_q_all, Tsettle_u_all])
    error('Relevant channels exist, but tau or Tsettle extraction failed. Inspect results.');
end

Tsettle_q_worst = max_or_nan(Tsettle_q_all);
Tsettle_u_worst = max_or_nan(Tsettle_u_all);
Tsettle_worst   = max([Tsettle_q_all, Tsettle_u_all]);
tau_slowest     = max(tau_all);
tau_fastest     = min(tau_all);
tau_q_fastest   = min(tau_q_all);
fres_max        = max_or_nan(fres_all);

% HEURISTIC: Use Lecture 9's experiment-length rule as a multisine-period lower
% bound for the first design. This is not the BPTT segment length.
T_ms = max(Tsettle_worst, 10*tau_slowest);
Df   = 1/T_ms;

% fmax is based on output dynamics. u_total contributes to settling/memory
% checks, but fast controller force transients should not by themselves set the
% output resampling bandwidth for plant identification.
f_tau = 1/(2*pi*tau_q_fastest);
if isnan(fres_max)
    fmax = f_tau;
    f_high_raw = f_tau;
    f_high_source = 'f_tau_no_visible_oscillation';
else
    fmax = max(f_tau, fres_max);
    f_high_raw = fres_max;
    f_high_source = 'fres_max';
end
Fs_new = 10*fmax;

% Initial lower band edge from controller bandwidth. This should later be
% checked against measured/estimated sensitivity survival.
f_low = fbw_hz;
band_valid_from_step = f_high_raw > f_low;
if band_valid_from_step
    f_high = f_high_raw;
else
    f_high = f_low;
    warning(['Step response did not produce a usable upper band edge: ', ...
             'raw f_high=%.2f Hz <= f_low=%.2f Hz. Use analytical eigenvalues ', ...
             'or multisine/FRF diagnostics to set f_high.'], f_high_raw, f_low);
end

fprintf('\n%s\nDesign outputs\n%s\n', repmat('=',1,72), repmat('-',1,72));
fprintf('  Tsettle_q_worst = %.4f s\n', Tsettle_q_worst);
fprintf('  Tsettle_u_worst = %.4f s\n', Tsettle_u_worst);
fprintf('  Tsettle_worst   = %.4f s\n', Tsettle_worst);
fprintf('  tau_slowest     = %.4f s\n', tau_slowest);
fprintf('  tau_fastest     = %.4f s\n', tau_fastest);
fprintf('  tau_q_fastest   = %.4f s\n', tau_q_fastest);
if isnan(fres_max)
    fprintf('  fres_max        = NaN (no oscillation visible)\n');
else
    fprintf('  fres_max        = %.2f Hz\n', fres_max);
end
fprintf('%s\n', repmat('-',1,72));
fprintf('  T_ms   = %.4f s   (multisine-period lower bound)\n', T_ms);
fprintf('  Df     = %.4f Hz\n', Df);
fprintf('  Fs_new = %.1f Hz  (lower bound; round up to standard rate)\n', Fs_new);
fprintf('  f_low  = %.1f Hz  (controller bandwidth by design)\n', f_low);
if band_valid_from_step
    fprintf('  f_high = %.1f Hz  (%s)\n', f_high, f_high_source);
else
    fprintf('  f_high = %.1f Hz  (INVALID STEP BAND: raw %.1f Hz <= f_low; use eigenvalue/FRF check)\n', ...
            f_high, f_high_raw);
end
fprintf('%s\n', repmat('=',1,72));

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'Matlab-output');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save(fullfile(out_dir, 'step1_outputs.mat'), ...
     'T_ms', 'Df', 'Fs_new', 'f_low', 'f_high', 'f_high_source', 'fres_max', ...
     'f_high_raw', 'band_valid_from_step', ...
     'Tsettle_worst', 'Tsettle_q_worst', 'Tsettle_u_worst', ...
     'tau_slowest', 'tau_fastest', 'tau_q_fastest', 'results', 'relevant', 'visible', ...
     'force_reports', 'Y_vals', 'A_FRAC', 'THR_PCT', ...
     'COUPLING_THR', 'USE_COUPLING_FILTER', 'fbw_hz', 'wbw');
fprintf('Saved: Matlab-output/step1_outputs.mat\n');

% Local functions ----------------------------------------------------------

function r = empty_result()
    r = struct('peak',NaN,'Tsettle',NaN,'tau',NaN,'fres',NaN, ...
               'visible',false,'settled_by_end',false, ...
               'Y0',NaN,'input','','output','','kind','','coupling_ratio',NaN);
end

function [t_vec, f_step, r] = build_step_run(T_TOTAL, T_HOLD, fs, Y0, fv, A)
    ts = 1/fs;
    N_sim = round(T_TOTAL*fs);
    N_hold = round(T_HOLD*fs);
    t_vec = (0:N_sim-1)'*ts;
    f_step = zeros(N_sim, 3);
    f_step(N_hold+1:end,:) = repmat(fv*A, N_sim-N_hold, 1);
    r = repmat([0, 0, Y0], N_sim, 1);
end

function res = extract_channel(sig, t, t_step_s, thr_pct, abs_min)
    res = empty_result();
    idx0 = find(t >= t_step_s, 1);
    if isempty(idx0)
        return
    end

    n_tail = max(1, round(0.2*numel(sig)));
    q_inf = mean(sig(end-n_tail+1:end));
    sig_rel = sig - q_inf;
    sig_post = sig_rel(idx0:end);
    t_post = t(idx0:end);

    peak = max(abs(sig_post));
    res.peak = peak;

    if peak < abs_min
        return
    end
    res.visible = true;

    band = thr_pct*peak;
    last_above = find(abs(sig_post) > band, 1, 'last');
    if isempty(last_above)
        res.Tsettle = 0;
        res.settled_by_end = true;
    else
        res.Tsettle = t_post(last_above) - t_step_s;
        tail_start = max(1, numel(sig_post) - n_tail + 1);
        res.settled_by_end = last_above < tail_start;
    end

    [res.tau, res.fres] = extract_tau_fres(sig_post, t_post, peak);
end

function [tau, fres] = extract_tau_fres(sig, t, peak)
    fres = NaN;
    tau = NaN;
    if peak <= 0
        return
    end

    min_h = 0.03*peak;
    [pks_p, locs_p] = findpeaks( sig, 'MinPeakHeight', min_h);
    [pks_n, locs_n] = findpeaks(-sig, 'MinPeakHeight', min_h);

    if numel(locs_p) >= 2 || numel(locs_n) >= 2
        if numel(locs_p) >= 2
            fres = 1/(t(locs_p(2)) - t(locs_p(1)));
            t_env = t(locs_p);
            v_env = log(pks_p);
        else
            fres = 1/(t(locs_n(2)) - t(locs_n(1)));
            t_env = t(locs_n);
            v_env = log(pks_n);
        end
        p = polyfit(t_env - t_env(1), v_env, 1);
        if p(1) < 0
            tau = -1/p(1);
        end
    else
        [~, idx_peak] = max(abs(sig));
        idx_63 = find(abs(sig(idx_peak:end)) <= exp(-1)*peak, 1, 'first');
        if ~isempty(idx_63)
            tau = t(idx_peak + idx_63 - 1) - t(idx_peak);
        end
    end
end

function [t_sim, r_sim, u_q1] = reconstruct(q1, r, t, Cfb)
    Ns = size(q1, 1);
    if evalin('base', 'exist(''tout'', ''var'')') || evalin('caller', 'exist(''tout'', ''var'')')
        try
            t_logged = evalin('caller', 'tout');
        catch
            t_logged = evalin('base', 'tout');
        end
        t_sim = t_logged(:);
        if numel(t_sim) ~= Ns
            error('Logged tout length (%d) does not match q1 length (%d).', numel(t_sim), Ns);
        end
        r_sim = interp1(t, r, t_sim, 'linear', 'extrap');
    elseif Ns == numel(t)
        t_sim = t;
        r_sim = r;
    else
        error(['q1 length differs from command grid, but no logged tout was found. ', ...
               'Log simulation time or force fixed-step output before reconstructing u_total.']);
    end
    u_q1 = lsim(ss(Cfb), r_sim - q1, t_sim);
end

function y_out = resample_to(y, t_src, t_tgt)
    if size(y,1) == numel(t_tgt)
        y_out = y;
    else
        y_out = interp1(t_src, y, t_tgt, 'linear', 'extrap');
    end
end

function rep = summarize_forces(Y0, input_name, u_total, F_peak, F_rms)
    rep.Y0 = Y0;
    rep.input = input_name;
    rep.peak = max(abs(u_total), [], 1);
    rep.rms = rms(u_total, 1);
    rep.ok_peak = all(rep.peak <= F_peak);
    rep.ok_rms = all(rep.rms <= F_rms);
end

function x = max_or_nan(v)
    if isempty(v)
        x = NaN;
    else
        x = max(v);
    end
end
