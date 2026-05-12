% diagnostics_step.m
% Step response preparatory experiment.
%
% Injects a pure step force in each modal direction at three frozen Y
% operating points. Extracts settled time (Tsettle), time constant (tau),
% and resonance frequency (fres) for every (Y0, input_mode, output_channel)
% pair in closed loop. Outputs initialise the multisine design.
%
% Primary outputs:
%   T_ms   = max(Tsettle_worst, 10·tau_slowest)   multisine period [s]
%   Df     = 1 / T_ms                             frequency resolution [Hz]
%   Fs_new = 10 · fmax                            resampling rate lower bound [Hz]
%   f_low  = fbw / (2·pi)                         multisine band lower bound [Hz]
%   f_high = fres_max                             highest oscillatory pole [Hz]
%
% N_seg is NOT an output — determine empirically via segment-length sweep.
%
% Analytical verification: run analytical_cl_eigenvalues.m and compare.
% Agreement within ~30% confirms the step response is trustworthy.
%
% Run from repo root:
%   run('Matlab-scripts/diagnostics_step.m')

addpath(genpath(fullfile(pwd, 'kamtin-fp-model', '03 Simulink gantry')))

% ── Physical parameters (identical to generate_identification_experiment.m) ──
mb=22.8; mh=10.1; m1=10.2; m2=10.7; Jb=1.0; Jh=0.05;
cg1=14.5; cg2=20.3; cy=10; cb1=9; cb2=9;
kb1=1987.5; kb2=1987.5; Lb=0.725; Lh=0.25; d=0.1;
C_damp = [cg1+cg2,         (cg1-cg2)*Lb/2,             0;
          (cg1-cg2)*Lb/2,  cb1+cb2+(cg1+cg2)*Lb^2/4,   0;
          0,                0,                           cy];
K   = [0,0,0; 0,kb1+kb2,0; 0,0,0];
n   = 3;
P   = [1,1,0; Lb/2,-Lb/2,0; 0,0,1];
fs  = 20e3;  ts = 1/fs;  fbw = 100;  mdl = 'gantry_2025a';

% ── Hardware limits (TELICA spec) ─────────────────────────────────────────────
F_peak = [2000, 2000, 1420];   % [N] peak per actuator [FX1, FX2, FY]

% ── Experiment settings ───────────────────────────────────────────────────────
A_FRAC       = 0.10;   % HEURISTIC: 10% of peak force — keeps frozen-LTI valid locally
T_HOLD       = 0.10;   % [s] hold before step for controller to settle
T_TOTAL      = 0.40;   % [s] total simulation — well past expected Tsettle ~50 ms
THR_PCT      = 0.05;   % settling threshold 5%  (Lecture 9 slide 9: tau_set,95)
COUPLING_THR = 0.05;   % HEURISTIC: exclude q channels below 5% of main response
ABS_MIN_M    = 1e-5;   % [m] minimum visible peak — below this treat channel as dead

% ── Mode definitions ─────────────────────────────────────────────────────────
modes(1).name = 'common';  modes(1).fv = [1, 1, 0];  modes(1).F_lim = min(F_peak(1:2));
modes(2).name = 'diff';    modes(2).fv = [1,-1, 0];  modes(2).F_lim = min(F_peak(1:2));
modes(3).name = 'Y';       modes(3).fv = [0, 0, 1];  modes(3).F_lim = F_peak(3);

Y_vals   = [-0.4, 0.0, 0.4];
N_sim    = round(T_TOTAL * fs);
N_hold   = round(T_HOLD  * fs);
t_vec    = (0:N_sim-1)' * ts;
t_step_s = T_HOLD;

ch_names = {'common', 'diff', 'Y', 'u_total'};

% ── Main loop ─────────────────────────────────────────────────────────────────
results = struct('peak',{},'Tsettle',{},'tau',{},'fres',{},'visible',{}, ...
                 'Y0',{},'input',{},'output',{});
res_idx = 0;

warning('off', 'signal:findpeaks:largeMinPeakHeight');

fprintf('\n%s\nStep response diagnostic  (%d Y-points x %d modes)\n%s\n', ...
        repmat('=',1,64), numel(Y_vals), numel(modes), repmat('=',1,64));

for i = 1:numel(Y_vals)
    Y0 = Y_vals(i);

    % Plant + controller — identical construction to generate_identification_experiment.m
    M_op = [m1+m2+mb+mh,           (m1-m2)*Lb/2-mh*Y0,                   0;
            (m1-m2)*Lb/2-mh*Y0,    Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y0^2, -mh*d;
            0,                      -mh*d,                                  mh];
    sys_ct = P.' * getss(n, M_op, C_damp, K) * P;
    Cfb    = tf(num2cell(zeros(3)), num2cell(ones(3)));
    for j = 1:3, Cfb(j,j) = ruleOfThumb(fbw, sys_ct(j,j), ts); end

    fprintf('\nY0 = %+.1f m\n', Y0);

    for m = 1:numel(modes)
        md = modes(m);
        A  = A_FRAC * md.F_lim;

        % Pure step: zeros until t_step_s, then A * mode_direction held to T_TOTAL
        f_step                  = zeros(N_sim, 3);
        f_step(N_hold+1:end, :) = repmat(md.fv * A, N_sim - N_hold, 1);

        % Simulate
        r = repmat([0, 0, Y0], N_sim, 1);
        f = f_step;
        t = t_vec;
        Y = Y0;
        sim(mdl, t_vec(end));

        % Reconstruct u_total (handles variable-step Simulink output)
        [t_sim, ~, u_ctrl] = reconstruct(q1, r, t_vec, Cfb);
        f_out = resample_to(f_step, t_vec, t_sim);
        u_tot = u_ctrl + f_out;

        % Four channels: three modal q outputs + u_total projected onto input direction
        q_modal   = [(q1(:,1)+q1(:,2))/2,  q1(:,1)-q1(:,2),  q1(:,3)-Y0];
        ch_sigs   = {q_modal(:,1), q_modal(:,2), q_modal(:,3), u_tot * md.fv'};

        for c = 1:4
            res        = extract_channel(ch_sigs{c}, t_sim, t_step_s, THR_PCT, ABS_MIN_M);
            res.Y0     = Y0;
            res.input  = md.name;
            res.output = ch_names{c};
            res_idx    = res_idx + 1;
            results(res_idx) = res;
        end

        fprintf('  %-8s  A = %5.0f N\n', md.name, A);
    end
end

% ── Coupling filter ───────────────────────────────────────────────────────────
% Diagonal peak: (Y0, input_mode) where output channel name == input mode name.
% u_total is always included (coupling_ratio = 1).
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
              strcmp(all_out, results(k).input);   % output name == input mode name
    if any(is_diag)
        results(k).coupling_ratio = results(k).peak / all_pk(find(is_diag,1));
    else
        results(k).coupling_ratio = 0;
    end
end

relevant = results([results.visible] & [results.coupling_ratio] >= COUPLING_THR);

if isempty(relevant)
    error('No relevant channels found. Increase A_FRAC or lower COUPLING_THR.');
end

% ── Worst-case extraction ─────────────────────────────────────────────────────
tau_all     = [relevant.tau];      tau_all     = tau_all(    ~isnan(tau_all));
Tsettle_all = [relevant.Tsettle];  Tsettle_all = Tsettle_all(~isnan(Tsettle_all));
fres_all    = [relevant.fres];     fres_all    = fres_all(   ~isnan(fres_all));

Tsettle_worst = max(Tsettle_all);
tau_slowest   = max(tau_all);
tau_fastest   = min(tau_all);
fres_max      = NaN;
if ~isempty(fres_all), fres_max = max(fres_all); end

% ── Design outputs ────────────────────────────────────────────────────────────
% THEORY Lecture 9 slide 8:  T_ms >= 10 * tau_slowest
T_ms = max(Tsettle_worst, 10 * tau_slowest);
Df   = 1 / T_ms;

% THEORY Lecture 9 slide 12: Fs_new >= 10 * fmax  (10*omega_b <= omega_s)
f_tau = 1 / (2*pi * tau_fastest);
fmax  = f_tau;
if ~isnan(fres_max), fmax = max(fmax, fres_max); end
Fs_new = 10 * fmax;

% THEORY: S(jw) ≈ 0 for w < fbw — post-controller injection suppressed below fbw
f_low  = fbw / (2*pi);   % [Hz]
f_high = fres_max;        % [Hz] — NaN if no oscillation visible

% ── Print ─────────────────────────────────────────────────────────────────────
fprintf('\n%s\nDesign outputs\n%s\n', repmat('=',1,64), repmat('-',1,64));
fprintf('  Tsettle_worst = %.4f s\n', Tsettle_worst);
fprintf('  tau_slowest   = %.4f s\n', tau_slowest);
fprintf('  tau_fastest   = %.4f s\n', tau_fastest);
if ~isnan(fres_max)
    fprintf('  fres_max      = %.2f Hz\n', fres_max);
else
    fprintf('  fres_max      = NaN  (no oscillation visible — expected if controller damps resonances)\n');
end
fprintf('%s\n', repmat('-',1,64));
fprintf('  T_ms   = %.4f s     (multisine period)\n',            T_ms);
fprintf('  Df     = %.4f Hz\n',                                  Df);
fprintf('  Fs_new = %.1f Hz    (round up to standard rate)\n',   Fs_new);
fprintf('  f_low  = %.1f Hz    (fbw = %.0f rad/s by design)\n',  f_low, fbw);
if ~isnan(f_high)
    fprintf('  f_high = %.1f Hz    (fres_max)\n',                f_high);
else
    fprintf('  f_high = %.1f Hz    (1/(2pi*tau_fastest), no oscillation)\n', f_tau);
end
fprintf('%s\n', repmat('=',1,64));

% ── Save ──────────────────────────────────────────────────────────────────────
out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'Matlab-output');
save(fullfile(out_dir, 'step1_outputs.mat'), ...
     'T_ms', 'Df', 'Fs_new', 'f_low', 'f_high', 'fres_max', ...
     'Tsettle_worst', 'tau_slowest', 'tau_fastest', ...
     'results', 'Y_vals', 'A_FRAC', 'THR_PCT', 'fbw');
fprintf('Saved: Matlab-output/step1_outputs.mat\n');

% ════════════════════════════════════════════════════════════════════════
% Local functions
% ════════════════════════════════════════════════════════════════════════

function res = extract_channel(sig, t, t_step_s, thr_pct, abs_min)
% Extract Tsettle, tau, fres from one scalar response channel.
    idx0     = find(t >= t_step_s, 1);
    q_inf    = mean(sig(end - round(0.2*numel(sig)) + 1 : end));
    sig_rel  = sig - q_inf;
    sig_post = sig_rel(idx0:end);
    t_post   = t(idx0:end);

    peak = max(abs(sig_post));

    res.peak    = peak;
    res.Tsettle = NaN;
    res.tau     = NaN;
    res.fres    = NaN;

    if peak < abs_min
        res.visible = false;
        return
    end
    res.visible = true;

    % Tsettle: last sample outside thr_pct band around q_inf
    last_above = find(abs(sig_post) > thr_pct * peak, 1, 'last');
    if ~isempty(last_above)
        res.Tsettle = t_post(last_above) - t_step_s;
    end

    [res.tau, res.fres] = extract_tau_fres(sig_post, t_post, peak);
end

function [tau, fres] = extract_tau_fres(sig, t, peak)
% Estimate tau and fres from a post-step zero-centred signal.
% Oscillatory:  fres from consecutive same-sign peak spacing;
%               tau  from log-linear fit of peak envelope.
% Overdamped:   fres = NaN;
%               tau  from first crossing of 0.368*peak (63.2% settled).
    fres = NaN;
    tau  = NaN;
    MIN_H = 0.03 * peak;   % HEURISTIC: ignore peaks below 3% of max — captures lightly-damped resonances that the controller nearly suppresses

    [pks_p, locs_p] = findpeaks( sig, 'MinPeakHeight', MIN_H);
    [pks_n, locs_n] = findpeaks(-sig, 'MinPeakHeight', MIN_H);

    if numel(locs_p) >= 2 || numel(locs_n) >= 2
        % ── Oscillatory ──────────────────────────────────────────────────
        % THEORY: damped natural frequency fd = 1/T_d, T_d = t_peak2 - t_peak1
        % THEORY: envelope ~ C*exp(-t/tau) => log-linear slope = -1/tau
        if numel(locs_p) >= 2
            fres  = 1 / (t(locs_p(2)) - t(locs_p(1)));
            t_env = t(locs_p);
            v_env = log(pks_p);
        else
            fres  = 1 / (t(locs_n(2)) - t(locs_n(1)));
            t_env = t(locs_n);
            v_env = log(pks_n);
        end
        p = polyfit(t_env - t_env(1), v_env, 1);
        if p(1) < 0
            tau = -1 / p(1);
        end
    else
        % ── Overdamped ───────────────────────────────────────────────────
        % THEORY: first-order system: |x(tau)| = exp(-1) * peak = 0.368 * peak
        % Search only after the peak — crossing on the rising edge gives tau=0.
        [~, idx_peak] = max(abs(sig));
        idx_63 = find(abs(sig(idx_peak:end)) <= 0.368 * peak, 1, 'first');
        if ~isempty(idx_63)
            tau = t(idx_peak + idx_63 - 1) - t(idx_peak);
        end
    end
end

function [t_sim, r_sim, u_q1] = reconstruct(q1, r, t, Cfb)
% Reconstruct u_q1 = Cfb*(r - q1). Handles variable-step Simulink output.
    Ns = size(q1, 1);
    if Ns ~= numel(t)
        t_sim = linspace(0, t(end), Ns)';
        r_sim = interp1(t, r, t_sim);
    else
        t_sim = t;
        r_sim = r;
    end
    u_q1 = lsim(ss(Cfb), r_sim - q1, t_sim);
end

function y_out = resample_to(y, t_src, t_tgt)
% Resample signal y from t_src grid onto t_tgt grid.
    if size(y, 1) == numel(t_tgt), y_out = y; return; end
    y_out = interp1(t_src, y, t_tgt, 'linear', 'extrap');
end
