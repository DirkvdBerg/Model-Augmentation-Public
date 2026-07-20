%% v1f: open-loop DC + AC comparison, plant with vs without the hidden MSD
%
% Spec + rationale: HANDOFF-2026-07-17.md, ZERO-MEAN-TESTING-METHODOLOGY.md
% (identifiability section 5, Volterra rectification section 6), README.md.
% Redesigned 2026-07-17 after a theory-alignment check: the first draft drove a
% pure sustained force and was blind to the second-order rectification DC (see
% below). This version excites BOTH DC mechanisms in one run.
%
% The question (Jan's Theme A / gate G-A): does the with-MSD gantry carry a
% non-zero-mean (DC) component that the baseline lacks, and does it justify the
% DC the ANN learns? Two DISTINCT DC mechanisms answer that, and a single
% waveform must excite both or the verdict is only half the story:
%   A  STATIC-GAIN DC  -- a difference in the 0 Hz equilibrium. Needs power at
%      DC (a sustained offset). Physics predicts it is ZERO: at steady state
%      qddot = 0, so the mass matrix M (the ONLY object differing between the two
%      plants; both share C, K, total mass) drops out of C*qdot+K*q=F, and the
%      MSD spring is UNPRELOADED (ka only on the delta_a diagonal; delta_a relaxes
%      to 0 with no constant force on the main axes). Predetermined ~0.
%   B  RECTIFICATION DC (methodology section 6) -- the truth's quadratic inertia
%      term ma*(Y+L0+delta_a)^2 rectifies an OSCILLATING input into a nonzero-mean
%      OUTPUT. This is the truth-only DC that SURVIVES truth-minus-baseline (the
%      common M(Y) part cancels). It needs an AC component at the MSD resonance
%      (fa = 150 Hz) to appear, and it is the mechanism that bears on the ANN,
%      which trains on the 130-180 Hz multisine. A pure-DC input is blind to it.
%
% Instrument (open-loop, one identical input to both plants; the closed loop
% rejects DC and gives each plant a different input, so it cannot do this):
%   drive per axis = sustained OFFSET (covers A + identifiability at 0 Hz)
%                  + a sustained single TONE at fa = 150 Hz (excites delta_a, so B
%                    appears as a surviving mean in the truth-minus-baseline output)
%   both with a smooth raised-cosine onset to avoid a jerk transient.
%   truth    = gantrySystemExtended (8-state, mh_rigid + hidden MSD ma;
%              Matlab-scripts/Augmentation/, behind gantry_additional_state_2025a)
%   baseline = gantrySystem (6-state, rigid mh_total; kamtin-fp-model, READ ONLY,
%              behind gantry_2025a)
%   readout  = on X / Y (K=0 free integrators) the velocity; on Theta the angle.
%              The steady DC is extracted by HARMONIC LEAST SQUARES (constant +
%              first two harmonics of fa) over the record tail, NOT a raw mean:
%              150 Hz is not commensurate with fs = 20 kHz, so a plain tail-mean
%              leaves a tone residual comparable to the rectification DC.
%              The DC is read on ALL THREE logical channels (dX, Theta, dY) for
%              EVERY drive, not only the driven one, because B lands in Theta (the
%              rectifying quadratic sits in the Theta inertia M(2,2)); delta_a tail
%              RMS is also reported, to prove the tone actually excited the MSD.
%   solver   = fixed-step RK4 at the native rate (same as v1d).
%
% Read-out interpretation per axis:
%   baseline DC  vs the analytic static value F/c or torque/(kb1+kb2)  -> mechanism A
%                (the static-gain path; must match, confirming A carries no offset)
%   truth DC minus baseline DC                                          -> mechanism B
%                (the surviving rectification DC; a floor value = no truth-only DC;
%                 a resolved value that GROWS with the tone level is a real MSD DC).
% Both outcomes are defensible. Note section 6 predicts B is real but tiny
% (~1e-4 / 1e-5), i.e. far below the ANN's learned DC, which would point at the
% estimator / training (gates G-B / G-C) rather than the physics.
%
% Extension: repeat from several initial Y so any operating-point / LPV
% dependence of B is visible.
%
% Location per folder convention: scripts/gantry/gantry-zero-mean/; run in MATLAB
% from this folder. Outputs: figures -> ./figures/v1f_<axis>_*.png,
% results -> ./data/v1f_results.mat, console -> ./data/v1f_console.txt (diary).
%
% STOP for user review after this run; do not chain into training or Python.

clear; clc;

%% Configuration
MA_FRAC   = 0.10;    % augmentation data generated with 0.10 (user-confirmed 2026-07-07;
                     % gtd_config default is 0.50, must pass this)
Y0_LIST   = [-0.30, 0.00, +0.10, +0.30];   % initial Y operating points (extension)
T_RAMP    = 0.5;     % HEURISTIC: raised-cosine onset [s], smooth (jerk-limited) rise of BOTH offset and tone
T_SIM     = 5.0;     % HEURISTIC: total sim time [s] (~3 tau_X; delta_a settles in ~0.1 s)
TAIL_FRAC = 0.40;    % HEURISTIC: final fraction of the record used for the DC read-out
DEC       = 20;      % HEURISTIC: plot decimation only (analysis uses full data)

here = fileparts(mfilename('fullpath'));            % .../scripts/gantry/gantry-zero-mean
root = fileparts(fileparts(fileparts(here)));       % repo root
addpath(fullfile(root, 'Matlab-scripts', 'Augmentation', 'data'));   % gtd_config
cfg  = gtd_config('augmentation', true, MA_FRAC);   % params + addpath (EOM functions)
figDir = fullfile(here, 'figures');
datDir = fullfile(here, 'data');
if ~exist(figDir, 'dir'), mkdir(figDir); end
if ~exist(datDir, 'dir'), mkdir(datDir); end

% capture everything printed below to a text log next to the results
logFile = fullfile(datDir, 'v1f_console.txt');
if exist(logFile, 'file'), delete(logFile); end
diary(logFile);
diaryGuard = onCleanup(@() diary('off'));           % diary closes even if the run errors
fprintf('v1f_dc_excitation_openloop | run started %s | MA_FRAC = %.2f\n', ...
    datestr(now, 'yyyy-mm-dd HH:MM:SS'), MA_FRAC);
fprintf('IDENTIFIABILITY: offset gives power at 0 Hz (mechanism A, static gain);\n');
fprintf('  the %g Hz tone excites delta_a so the truth-only rectification DC (mechanism B)\n', cfg.fa);
fprintf('  appears in the output difference. Each mechanism needs its own excitation.\n');

ts = cfg.ts;  fs = cfg.fs;  fa = cfg.fa;            % fa = 150 Hz MSD natural freq (THEORY: cfg.fa)

% plant EOMs, parameters frozen from cfg (single source, same as the generator / v1d)
f_truth = @(x, u) gantrySystemExtended(u, x, cfg.m1, cfg.m2, cfg.mb, cfg.mh_rigid, ...
    cfg.Lb, cfg.Jb, cfg.Jh, cfg.d, cfg.cg1, cfg.cg2, cfg.cb1, cfg.cb2, cfg.cy, ...
    cfg.kb1, cfg.kb2, cfg.ma, cfg.ka, cfg.ca, cfg.L0);
f_base  = @(x, u) gantrySystem(u, x, cfg.m1, cfg.m2, cfg.mb, cfg.mh, ...
    cfg.Lb, cfg.Jb, cfg.Jh, cfg.d, cfg.cg1, cfg.cg2, cfg.cb1, cfg.cb2, cfg.cy, ...
    cfg.kb1, cfg.kb2);

% ── DC offset levels (HEURISTIC sizing, handoff spec) ───────────────────────
% Sized to a small target so the K=0 position excursion stays roughly in range:
F_X     = 0.03 * (cfg.cg1 + cfg.cg2);   % HEURISTIC: target static dX ~ 0.03 m/s
F_Y     = 0.03 * cfg.cy;                % HEURISTIC: target static dY ~ 0.03 m/s
T_theta = 1e-3 * (cfg.kb1 + cfg.kb2);   % HEURISTIC: target static Theta ~ 1e-3 rad

% ── Per-axis spec: DC offset, AC tone amplitude/channel, read-out, analytic ─
% AC amplitude = the training multisine channel level (cfg.A_sym / A_anti / A_Y),
%   so the rectification is representative of what the ANN actually sees.
%   HEURISTIC: single-tone amplitude at the training channel level.
% Read-out index: truth 8-state x=[X Th Y da dX dTh dY vda]; baseline 6-state
%   x=[X Th Y dX dTh dY]. X/Y are K=0 -> read velocity; Theta has stiffness -> read angle.
% Analytic value = steady solution of C*qdot+K*q=F at qddot=0 (mechanism-A path).
AX(1) = struct('name','X',     'dc',[F_X;0;0],     'ac_amp',cfg.A_sym,  'ac_ch',1, ...
    'it',5, 'ib',4, 'unit','m/s', 'readname','dX velocity', ...
    'analytic', F_X/(cfg.cg1+cfg.cg2));                              % THEORY: F/c on K=0 axis
AX(2) = struct('name','Theta', 'dc',[0;T_theta;0], 'ac_amp',cfg.A_anti,'ac_ch',2, ...
    'it',2, 'ib',2, 'unit','rad', 'readname','Theta angle', ...
    'analytic', T_theta/(cfg.kb1+cfg.kb2));                         % THEORY: torque/(kb1+kb2)
AX(3) = struct('name','Y',     'dc',[0;0;F_Y],     'ac_amp',cfg.A_Y,   'ac_ch',3, ...
    'it',7, 'ib',6, 'unit','m/s', 'readname','dY velocity', ...
    'analytic', F_Y/cfg.cy);                                       % THEORY: F/c on K=0 axis

% ── Time base, raised-cosine onset, tone, tail window, DC-extraction basis ──
N   = round(T_SIM/ts) + 1;
t   = (0:N-1)' * ts;
env = ones(N, 1);                                    % onset envelope, 0 -> 1
ir  = t <= T_RAMP;
env(ir) = 0.5 * (1 - cos(pi * t(ir) / T_RAMP));      % HEURISTIC: raised-cosine (Hann) C1-smooth onset
w    = 2*pi*fa;
tone = sin(w * t);                                   % the 150 Hz tone (shared across axes)
i_tail = find(t >= (1 - TAIL_FRAC) * T_SIM);         % steady-read window
tt   = t(i_tail);
% harmonic-regression basis: constant + first two harmonics of fa (isolates DC from the tone)
Phi  = [ones(numel(tt),1), sin(w*tt), cos(w*tt), sin(2*w*tt), cos(2*w*tt)];   % THEORY: local DFT / harmonic LS
mw   = round(fs / fa);                               % HEURISTIC: moving-average window = one tone period (plot only)

% ── Fixed logical read-out set (stationary quantity per channel) ────────────
% Mechanism B (rectification DC) lands in Theta: the quadratic ma*(Y+L0+delta_a)^2
% sits in the Theta inertia M(2,2); the linear -ma*delta_a in M(1,2) averages to
% zero. X/Y are K=0 so their stationary quantity is velocity. Reading ALL three
% channels for EVERY drive captures the "drive Y (max delta_a), read Theta (where
% B lands)" combination the driven-axis-only read-out was blind to.
RD(1) = struct('name','dX',    'it',5, 'ib',4, 'unit','m/s');
RD(2) = struct('name','Theta', 'it',2, 'ib',2, 'unit','rad');
RD(3) = struct('name','dY',    'it',7, 'ib',6, 'unit','m/s');
IDA   = 4;    % truth-state index of delta_a (excitation proof)

results = struct();

for ai = 1:numel(AX)
    a    = AX(ai);
    drv  = a.ac_ch;                                  % driven channel index = 1/2/3 (X/Theta/Y ordering)
    e_ch = zeros(3,1);  e_ch(a.ac_ch) = 1;           % AC channel selector
    fprintf('\n=== axis %s | DC offset [%.4g %.4g %.4g] + %g Hz tone amp %.4g on ch %d ===\n', ...
        a.name, a.dc(1), a.dc(2), a.dc(3), fa, a.ac_amp, a.ac_ch);
    fprintf('  driven read-out: %-6s | analytic static (mechanism A) = %+.6e %s\n', ...
        RD(drv).name, a.analytic, RD(drv).unit);
    fprintf('  %-7s %14s %14s | %13s %13s %13s | %12s\n', 'Y0 [m]', ...
        'baseDC_drv', 'truthDC_drv', 'B[dX] m/s', 'B[Theta] rad', 'B[dY] m/s', 'delta_a RMS');

    nY  = numel(Y0_LIST);
    DCT = zeros(3,nY);  DCB = zeros(3,nY);  DCD = zeros(3,nY);  DAR = zeros(1,nY);
    Ddrv = zeros(N,nY);  Dth = zeros(N,nY);          % diff traces for figures (driven + Theta)

    for yi = 1:nY
        Y0 = Y0_LIST(yi);
        x_t = [0; 0; Y0; 0;  0; 0; 0; 0];            % [X Th Y da dX dTh dY vda], at rest
        x_b = [0; 0; Y0;     0; 0; 0];               % [X Th Y dX dTh dY], at rest

        Xt = zeros(N,8);  Xb = zeros(N,6);
        Xt(1,:) = x_t';   Xb(1,:) = x_b';
        for k = 1:N-1
            uk = env(k) * (a.dc + e_ch * a.ac_amp * tone(k));   % same input to both plants
            k1  = f_truth(x_t, uk);              k1b = f_base(x_b, uk);
            k2  = f_truth(x_t + ts/2*k1, uk);    k2b = f_base(x_b + ts/2*k1b, uk);
            k3  = f_truth(x_t + ts/2*k2, uk);    k3b = f_base(x_b + ts/2*k2b, uk);
            k4  = f_truth(x_t + ts*k3, uk);      k4b = f_base(x_b + ts*k3b, uk);
            x_t = x_t + ts/6 * (k1 + 2*k2 + 2*k3 + k4);   % THEORY: classical 4th-order Runge-Kutta
            x_b = x_b + ts/6 * (k1b + 2*k2b + 2*k3b + k4b);
            Xt(k+1,:) = x_t';  Xb(k+1,:) = x_b';
        end

        % harmonic-LS DC on ALL three logical channels, truth and baseline
        for c = 1:3
            bt = Phi \ Xt(i_tail, RD(c).it);  DCT(c,yi) = bt(1);   % THEORY: DC = intercept of harmonic LS fit
            bb = Phi \ Xb(i_tail, RD(c).ib);  DCB(c,yi) = bb(1);
        end
        DCD(:,yi) = DCT(:,yi) - DCB(:,yi);                        % mechanism B per channel
        DAR(yi)   = sqrt(mean(Xt(i_tail, IDA).^2));               % delta_a tail RMS (excitation proof)
        Ddrv(:,yi) = Xt(:,RD(drv).it) - Xb(:,RD(drv).ib);
        Dth(:,yi)  = Xt(:,RD(2).it)   - Xb(:,RD(2).ib);

        fprintf('  %+6.2f %14.6e %14.6e | %13.3e %13.3e %13.3e | %12.3e\n', ...
            Y0, DCB(drv,yi), DCT(drv,yi), DCD(1,yi), DCD(2,yi), DCD(3,yi), DAR(yi));
    end

    results.(a.name) = struct('Y0', Y0_LIST, 'dc_truth', DCT, 'dc_base', DCB, ...
        'dc_diff_mechB', DCD, 'delta_a_rms', DAR, 'chan_order', {{RD.name}}, ...
        'driven_chan', drv, 'analytic_mechA', a.analytic, 'unit', RD(drv).unit, ...
        'dc_offset', a.dc', 'ac_amp', a.ac_amp, 'ac_ch', a.ac_ch, 'fa', fa);

    % ── figure: tone-removed truth-minus-baseline diff, driven channel (top) and
    %    Theta home channel (bottom), per Y0 ─────────────────────────────────
    ip = 1:DEC:N;  tp = t(ip);
    fh = figure('Visible', 'off', 'Position', [40 40 380*nY 620]);
    for yi = 1:nY
        Ld = movmean(Ddrv(:,yi), mw);   Lt = movmean(Dth(:,yi), mw);   % HEURISTIC: one-period MA removes the tone (plot)
        subplot(2, nY, yi); hold on; grid on;
        plot(tp, Ld(ip), 'Color', [0.2 0.5 0.2], 'LineWidth', 0.8);
        yline(0, 'k-');  yline(DCD(drv,yi), 'r--', 'LineWidth', 1.0);
        title(sprintf('Y0 %+.2f: driven %s B %+.2e', Y0_LIST(yi), RD(drv).name, DCD(drv,yi)));
        if yi == 1, ylabel(sprintf('driven %s diff [%s]', RD(drv).name, RD(drv).unit)); end
        ax = gca; ax.YAxis.Exponent = 0;

        subplot(2, nY, nY + yi); hold on; grid on;
        plot(tp, Lt(ip), 'Color', [0.3 0.3 0.7], 'LineWidth', 0.8);
        yline(0, 'k-');  yline(DCD(2,yi), 'r--', 'LineWidth', 1.0);
        title(sprintf('Theta B %+.2e | \\delta_a RMS %.2e', DCD(2,yi), DAR(yi)));
        if yi == 1, ylabel('Theta diff [rad]'); end
        xlabel('time [s]');
        ax = gca; ax.YAxis.Exponent = 0;
    end
    sgtitle({sprintf('v1f axis %s: offset + %g Hz tone, open loop, same input; tone-removed truth-minus-baseline', a.name, fa), ...
        'top = driven channel | bottom = Theta (rectification home channel); red dashed = harmonic-LS DC'});
    exportgraphics(fh, fullfile(figDir, sprintf('v1f_%s_dcac_response.png', a.name)), 'Resolution', 150);
    close(fh);
    fprintf('  figure -> %s\n', fullfile(figDir, sprintf('v1f_%s_dcac_response.png', a.name)));
end

save(fullfile(datDir, 'v1f_results.mat'), '-struct', 'results');
fprintf('\nresults saved to %s\n', fullfile(datDir, 'v1f_results.mat'));
fprintf('console log saved to %s\n', logFile);
diary off;
