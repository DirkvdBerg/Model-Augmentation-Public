% diagnostics_system.m
% Pre-analysis: nonparametric estimation of directional sensitivity Ŝ(jω)
% and plant FRF Ĝ(jω) from closed-loop probe simulations.
%
% Theory:
%   Ŝ_c(jω) = FFT(u_modal) / FFT(f_modal)  [feedback algebra: U_total = S × F_sim,
%              valid at excited frequencies where r has no content]
%   Ĝ_c(jω) = FFT(q_modal) / FFT(u_modal)  [plant equation: Q = G × U_total,
%              directional FRF — not necessarily a modal FRF]
%   Projection: x_modal = x * f_vec'        [projects signals onto excitation direction]
%
% Outputs (worst-case across 3 modes and 5 Y operating points):
%   f_low  — lowest freq where |Ŝ|² > 0.1   [HEURISTIC: -10 dB survival threshold]
%   f_high — last resonance peak in |Ĝ|      [HEURISTIC: above this G is mass-dominated]
%   A_max  — 0.4 × hardware RMS limit        [HEURISTIC: 40% actuator capacity]
%
% Run from repo root:
%   run('Matlab-scripts/diagnostics_system.m')

addpath(genpath(fullfile(pwd, 'kamtin-fp-model', '03 Simulink gantry')))

% ── Physical parameters ───────────────────────────────────────────────────
mb=22.8; mh=10.1; m1=10.2; m2=10.7; Jb=1.0; Jh=0.05;
cg1=14.5; cg2=20.3; cy=10; cb1=9; cb2=9;
kb1=1987.5; kb2=1987.5; Lb=0.725; Lh=0.25; d=0.1;
cc1=16.8; cc2=18.35; ccy=11.6;

C_damp = [cg1+cg2,         (cg1-cg2)*Lb/2,             0;
          (cg1-cg2)*Lb/2,  cb1+cb2+(cg1+cg2)*Lb^2/4,   0;
          0,                0,                           cy];
K   = [0,0,0; 0,kb1+kb2,0; 0,0,0];
n   = 3;
P   = [1,1,0; Lb/2,-Lb/2,0; 0,0,1];
fs  = 20e3; ts = 1/fs; fbw = 100; mdl = 'gantry_2025a';

% ── Probe signal ──────────────────────────────────────────────────────────
% T_p = 1 s → f0 = 1 Hz: leakage-free condition requires excited frequencies
%   to be integer multiples of 1/T_p (P&S Ch.2 §2.2.3)
% Odd harmonics 1 Hz to Nyquist: broadband — needed to observe G rolloff and
%   locate f_high. Diagnostic probe only; identification multisine will be narrower.
% 2 periods: discard first (transient), use second (steady state)
% Schroeder phases: minimize crest factor (Schroeder 1970) to avoid actuator
%   saturation from coherent peak of zero-phase cosine sum
% 50 N RMS: HEURISTIC — sufficient probe level, well within all actuator limits
N_period  = round(fs);           % 20000 samples, T_p = 1 s
N_periods = 2;                   % HEURISTIC: 1 transient + 1 steady-state period
N         = N_periods * N_period;
t         = (0:N-1)' * ts;
f0        = fs / N_period;       % = 1 Hz
k         = (1:2:N_period/2);   % odd harmonic numbers: 1, 3, 5, ..., 9999
F_bins    = numel(k);
freqs     = k * f0;              % excited frequencies [Hz]
amp_rms   = 50;                  % [N RMS] HEURISTIC: probe amplitude

% Schroeder 1970 eq.3: phi_n = -pi * n*(n-1) / F
phi_sch  = -pi * (1:F_bins) .* (0:F_bins-1) / F_bins;
t_period = (0:N_period-1)' * ts;
sig_p    = sum(cos(2*pi*t_period*freqs + phi_sch), 2);
sig_p    = sig_p / rms(sig_p) * amp_rms;
sig      = repmat(sig_p, N_periods, 1);

% ── Mode definitions ─────────────────────────────────────────────────────
% Force direction in motor coordinates [FX1, FX2, FY].
% Projection x_modal = x * f_vec' gives a scalar directional signal per mode.
% Not necessarily a decoupled modal channel — called directional FRF.
ch(1).name = 'common';  ch(1).f_vec = [1, 1, 0];   % X symmetric (rigid body)
ch(2).name = 'diff';    ch(2).f_vec = [1,-1, 0];   % theta tilt — kb resonance here
ch(3).name = 'y';       ch(3).f_vec = [0, 0, 1];   % Y translation (free mass)
nCh = 3;

% ── Hardware limits (TELICA spec) ─────────────────────────────────────────
F_limit_rms = [916, 916, 656];   % [N RMS] per actuator [FX1, FX2, FY]

% ── Y operating points ───────────────────────────────────────────────────
% HEURISTIC: 5 uniformly spaced points across hardware limits ±0.4 m
Y_vals = [-0.4, -0.2, 0.0, 0.2, 0.4];
nY     = numel(Y_vals);
bins   = k + 1;   % 1-indexed DFT bins: bin m → frequency (m-1)*fs/N_period = k*f0

S_hat = zeros(F_bins, nY, nCh);
G_hat = zeros(F_bins, nY, nCh);

% ── Probe loop: 3 modes × 5 Y points = 15 simulations ───────────────────
for i = 1:nY
    Y_op = Y_vals(i);
    fprintf('Y = %+.1f m (%d/%d)\n', Y_op, i, nY);

    % Controller at this operating point (frozen at Y_op for static probe)
    M_op = [m1+m2+mb+mh,            (m1-m2)*Lb/2-mh*Y_op,                   0;
            (m1-m2)*Lb/2-mh*Y_op,   Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y_op^2, -mh*d;
            0,                       -mh*d,                                    mh];
    sys = P.' * getss(n, M_op, C_damp, K) * P;
    Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
    for j = 1:3, Cfb(j,j) = ruleOfThumb(fbw, sys(j,j), ts); end

    for c = 1:nCh
        fv = ch(c).f_vec;
        fprintf('  mode: %s\n', ch(c).name);

        r = repmat([0, 0, Y_op], N, 1);   % static reference at Y_op
        f = sig * fv;                      % N×3: inject into this mode only
        Y = Y_op;                          % Simulink workspace: prismatic joint IC
        sim(mdl, t(end));

        % Reconstruct u_total (handles variable-step Simulink output)
        Ns   = size(q1, 1);
        t_s  = linspace(0, t(end), Ns)';
        r_s  = interp1(t, r, t_s);
        f_s  = interp1(t, f, t_s);
        u_q1 = lsim(ss(Cfb), r_s - q1, t_s);
        u_tot = u_q1 + f_s;

        % Project onto mode direction → scalar SISO signals
        f_modal = f_s   * fv';
        u_modal = u_tot * fv';
        q_modal = q1    * fv';

        % FRF estimation from last period (steady state only)
        idx_ss = Ns - N_period + 1 : Ns;
        F_f = fft(f_modal(idx_ss));
        U_f = fft(u_modal(idx_ss));
        Q_f = fft(q_modal(idx_ss));

        % THEORY: Ŝ = U_total/F_sim — feedback algebra, exact at excited freqs
        % THEORY: Ĝ = Q/U_total — directional plant FRF, exact in noiseless simulation
        S_hat(:,i,c) = abs(U_f(bins) ./ F_f(bins));
        G_hat(:,i,c) = abs(Q_f(bins) ./ U_f(bins));
    end
end

% ── Extract outputs ───────────────────────────────────────────────────────

% f_low: lowest freq where |Ŝ|² > threshold, worst-case across modes and Y
% HEURISTIC: threshold 0.1 (-10 dB) — below this >90% of force is suppressed
S_THRESHOLD = 0.1;
f_low_all = zeros(nCh, nY);
for c = 1:nCh
    for i = 1:nY
        hit = find(S_hat(:,i,c).^2 > S_THRESHOLD, 1, 'first');
        if ~isempty(hit), f_low_all(c,i) = freqs(hit); end
    end
end
f_low = max(f_low_all(:));

% f_high: cumulative energy criterion on |Ĝ × Ŝ|²
% Q/F_sim = G × S (feedback algebra) is the closed-loop force-to-position FRF.
% f_high is the highest frequency where the tail energy above that frequency
% still exceeds ε of the total energy in the usable band (freqs >= f_low).
% Below f_low: S ≈ 0, so G_hat = Q/U_total is noise-dominated — excluded.
% Per (mode, Y): normalise independently so no mode dominates by energy scale.
% Worst-case across modes and Y: conservative, covers all modes.
% HEURISTIC: ε = 0.05 — accept 5% energy loss above f_high (declared tolerance)
ENERGY_TAIL_THRESH = 0.05;
valid       = freqs >= f_low;
freqs_valid = freqs(valid);
f_high_all  = zeros(nCh, nY);
for c = 1:nCh
    for i = 1:nY
        gs_sq = (G_hat(valid,i,c) .* S_hat(valid,i,c)).^2;  % |G×S|² per freq
        cumE  = cumsum(gs_sq(end:-1:1)); cumE = cumE(end:-1:1);  % tail energy
        idx   = find(cumE / cumE(1) >= ENERGY_TAIL_THRESH, 1, 'last');
        if ~isempty(idx), f_high_all(c,i) = freqs_valid(idx); end
    end
end
f_high = max(f_high_all(:));

% A_max per mode: HEURISTIC — 40% of hardware RMS limit per mode
% Declared engineering choice: leaves 60% actuator capacity for tracking and safety
ALPHA = 0.4;
mode_F_limit = [min(F_limit_rms(1:2)), min(F_limit_rms(1:2)), F_limit_rms(3)];
A_max = ALPHA * mode_F_limit;   % [N RMS]: [common, diff, y]

% ── Print ─────────────────────────────────────────────────────────────────
fprintf('\n=== Pre-analysis outputs ===\n');
fprintf('f_low  = %6.1f Hz   [HEURISTIC: |S|^2 > %.1f, worst-case across modes and Y]\n', f_low, S_THRESHOLD);
fprintf('f_high = %6.1f Hz   [HEURISTIC: tail |G×S|² < %.0f%% of total, worst-case across modes and Y]\n', f_high, ENERGY_TAIL_THRESH*100);
fprintf('A_max  = common=%.0f N  diff=%.0f N  y=%.0f N   [HEURISTIC: %.0f%% of hardware limit]\n', ...
        A_max(1), A_max(2), A_max(3), ALPHA*100);

% ── Plots ─────────────────────────────────────────────────────────────────
mode_names = {ch.name};
Y_lgd = arrayfun(@(y) sprintf('Y=%+.1f m', y), Y_vals, 'UniformOutput', false);

for c = 1:nCh
    figure(c); clf;
    subplot(2,1,1);
    semilogx(freqs, 20*log10(S_hat(:,:,c)));
    hold on
    yline(10*log10(S_THRESHOLD), 'k--', sprintf('|S|^2=%.0f dB', 10*log10(S_THRESHOLD)));
    xline(f_low, 'r--', sprintf('f_{low}=%.0f Hz', f_low));
    xlabel('Frequency [Hz]'); ylabel('|S| [dB]');
    title(sprintf('Sensitivity Ŝ — %s mode', mode_names{c}));
    legend(Y_lgd); grid on;

    subplot(2,1,2);
    semilogx(freqs, 20*log10(G_hat(:,:,c)));
    hold on
    xline(f_high, 'r--', sprintf('f_{high}=%.0f Hz', f_high));
    xlabel('Frequency [Hz]'); ylabel('|G| [dB]');
    title(sprintf('Directional plant FRF Ĝ — %s mode', mode_names{c}));
    legend(Y_lgd); grid on;
end

% ── Save ──────────────────────────────────────────────────────────────────
out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'Matlab-output');
save(fullfile(out_dir, 'step0_outputs.mat'), ...
     'f_low', 'f_high', 'A_max', 'freqs', 'S_hat', 'G_hat', 'Y_vals', ...
     'S_THRESHOLD', 'ENERGY_TAIL_THRESH', 'ALPHA', 'F_limit_rms');
fprintf('Saved: Matlab-output/step0_outputs.mat\n');
