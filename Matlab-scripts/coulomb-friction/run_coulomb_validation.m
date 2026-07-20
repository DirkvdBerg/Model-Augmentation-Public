% run_coulomb_validation.m
% -------------------------
% Phase 1 of the Coulomb-friction plan (Matlab-scripts/coulomb-friction/PLAN.md):
% verify the analytic Coulomb EOM against the Simscape multibody model.
%
% Method (closed loop on both sides, difference design):
%   The Simscape model gantry_2025a.slx runs closed loop (reference + feedback).
%   We run it twice:
%     q_sim_OFF = sim('gantry_2025a')          % original model, Coulomb blocks disabled
%     q_sim_ON  = sim('gantry_2025a_coulomb')  % copy with the 6 Coulomb blocks enabled
%   We run the analytic EOM (gantrySystemCoriolisCentripetal + gantrySystemCoulomb)
%   in a matched closed loop with the SAME controller and reference:
%     q_eom_OFF = closed_loop_analytic(..., cc = 0)
%     q_eom_ON  = closed_loop_analytic(..., cc = [cc1;cc2;ccy])
%
%   Difference design isolates the Coulomb term and cancels the (small,
%   documented) Simscape-vs-EOM geometry residual:
%     dSim = q_sim_ON - q_sim_OFF
%     dEOM = q_eom_ON - q_eom_OFF
%   GOOD WHEN: dEOM matches dSim to a small tolerance across the trajectory.
%
% Prerequisite (manual, one time): copy gantry_2025a.slx into this folder as
% gantry_2025a_coulomb.slx and un-comment the 6 Coulomb blocks (3 Signum +
% cc1/cc2/ccy gains) in the Simscape subsystem. The original gantry_2025a
% (Coulomb disabled) is used as-is from kamtin-fp-model for the OFF run.
%
% Run from repo root (from MATLAB):
%   cd('<path-to>/Baseline-LPV-Augmentation')
%   run('Matlab-scripts/coulomb-friction/run_coulomb_validation.m')

here = fileparts(mfilename('fullpath'));   % Matlab-scripts/coulomb-friction
repo = fileparts(fileparts(here));         % repo root (up two)
addpath(genpath(fullfile(repo, 'kamtin-fp-model', '03 Simulink gantry')))
addpath(here)

% ------------------------------------------------------------------
% 1. Physical parameters (identical to export_lpv_sim.m / main.m)
% ------------------------------------------------------------------
mb  = 22.8;  mh  = 10.1;  m1  = 10.2;  m2  = 10.7;
Jb  = 1.0;   Jh  = 0.05;
cg1 = 14.5;  cg2 = 20.3;  cy  = 10;
cb1 = 9;     cb2 = 9;
kb1 = 1987.5; kb2 = 1987.5;
Lb  = 0.725; Lh  = 0.25;  d = 0.1;

% THEORY: garcia2013 identified Coulomb forces [N]
cc1 = 16.8;  cc2 = 18.35; ccy = 11.6;
cc_on  = [cc1; cc2; ccy];
cc_off = [0; 0; 0];

Y    = 0.3;   % design operating point (main.m line 49); Simscape reads it
Y_op = Y;

% Matrices at the operating point (constant C, K)
M_op   = [m1+m2+mb+mh,           (m1-m2)*Lb/2 - mh*Y_op,               0;
          (m1-m2)*Lb/2 - mh*Y_op, Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y_op^2, -mh*d;
          0,                                                    -mh*d,  mh];
C_damp = [           cg1+cg2,               (cg1-cg2)*Lb/2,  0;
          (cg1-cg2)*Lb/2, cb1+cb2+(cg1+cg2)*Lb^2/4,          0;
                       0,                            0,       cy];
K      = [0,        0, 0;
          0, kb1+kb2, 0;
          0,        0, 0];

% ------------------------------------------------------------------
% 2. State-space, coordinate transform, controller (main.m lines 88-207)
% ------------------------------------------------------------------
n = 3;
sys_logical = getss(n, M_op, C_damp, K);

P = [1,    1,    0;
     Lb/2, -Lb/2, 0;
     0,    0,    1];                      % logical -> stage via P'
StageCoordinatesSystem = P.' * sys_logical * P;

fs = 20e3;
ts = 1 / fs;

fbw = 100;
Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
for j = 1:3
    Cfb(j,j) = ruleOfThumb(fbw, StageCoordinatesSystem(j,j), ts);
end
Cfb_ss = ss(Cfb);                          % discrete state space for manual stepping

% ------------------------------------------------------------------
% 3. Test trajectory: Y step 0.3 -> -0.3 m with holds, X at rest
%    (holds are where Coulomb bites: velocity ~ 0)
% ------------------------------------------------------------------
pmax_Y   = 0.6;  vmax_Y = 1.0;  amax_Y = 10.0;  jerkTime = 0.05;
jmax_Y   = amax_Y / jerkTime;  smax = Inf;
pvajs_Y  = thirdOrderSetpointETEL(pmax_Y, vmax_Y, amax_Y, jmax_Y, smax, ts);
n_move   = size(pvajs_Y, 1);
n_hold   = round(0.5 / ts);
nt       = n_hold + n_move + n_hold;
t        = ts * (0:nt-1)';

r = zeros(nt, 3);
r(:, 3) = 0.3;
r(n_hold + (1:n_move), 3) = 0.3 - pvajs_Y(:, 1);
r(n_hold + n_move + 1 : end, 3) = -0.3;

f = zeros(nt, 3);                          % no feedforward
G = c2d(StageCoordinatesSystem, ts, 'zoh');% frozen LTI, read by the Simulink model

% ------------------------------------------------------------------
% 4. Analytic closed-loop EOM (Coriolis base + Coulomb), OFF and ON
% ------------------------------------------------------------------
baseFn = @(uu, xx) gantrySystemCoriolisCentripetal(uu, xx, ...
             m1, m2, mb, mh, Jb, Jh, d, Lb, kb1, kb2, cg1, cg2, cb1, cb2, cy);

x0 = [0; 0; Y; 0; 0; 0];                   % rest at the initial reference (logical)

fprintf('Analytic closed-loop EOM: Coulomb OFF ...\n');
q_eom_OFF = closed_loop_analytic(baseFn, x0, r, Cfb_ss, P, Lb, ts, cc_off);
fprintf('Analytic closed-loop EOM: Coulomb ON  ...\n');
q_eom_ON  = closed_loop_analytic(baseFn, x0, r, Cfb_ss, P, Lb, ts, cc_on);

% ------------------------------------------------------------------
% 5. Simscape runs (OFF = original model, ON = enabled copy)
% ------------------------------------------------------------------
have_simscape = false;
q_sim_OFF = []; q_sim_ON = [];
if exist('gantry_2025a_coulomb', 'file') == 4
    try
        fprintf('Simscape: gantry_2025a (Coulomb disabled) ...\n');
        sim('gantry_2025a', t(end));
        q_sim_OFF = align_to_grid(q, tout, t);

        fprintf('Simscape: gantry_2025a_coulomb (Coulomb enabled) ...\n');
        sim('gantry_2025a_coulomb', t(end));
        q_sim_ON = align_to_grid(q, tout, t);
        have_simscape = true;
    catch ME
        warning('Simscape run failed (%s). Continuing with analytic-only.', ME.message);
    end
else
    warning(['gantry_2025a_coulomb.slx not found on path. ' ...
             'Copy gantry_2025a.slx into this folder, enable the 6 Coulomb ' ...
             'blocks, then re-run. Continuing with analytic-only.']);
end

% ------------------------------------------------------------------
% 6. Compare and report
% ------------------------------------------------------------------
axes_lbl = {'X1', 'X2', 'Y'};
dEOM = q_eom_ON - q_eom_OFF;               % Coulomb effect, analytic
fprintf('\nAnalytic Coulomb effect (ON - OFF), RMS per channel [m]:\n');
for i = 1:3
    fprintf('  %-3s  %.4e\n', axes_lbl{i}, rmsval(dEOM(:, i)));
end

if have_simscape
    dSim = q_sim_ON - q_sim_OFF;           % Coulomb effect, Simscape
    resid = dEOM - dSim;                   % should be ~0 if the term matches
    base_resid = q_eom_OFF - q_sim_OFF;    % geometry residual (Coriolis vs multibody)
    fprintf('\nVerification (difference design):\n');
    fprintf('  %-3s  %-14s  %-14s  %-14s\n', 'Ch', 'RMS dEOM', 'RMS dSim', 'RMS(dEOM-dSim)');
    for i = 1:3
        fprintf('  %-3s  %-14.4e  %-14.4e  %-14.4e\n', ...
                axes_lbl{i}, rmsval(dEOM(:, i)), rmsval(dSim(:, i)), rmsval(resid(:, i)));
    end
    fprintf('\n  Baseline geometry residual (q_eom_OFF - q_sim_OFF), RMS per channel [m]:\n');
    for i = 1:3
        fprintf('    %-3s  %.4e\n', axes_lbl{i}, rmsval(base_resid(:, i)));
    end
    fprintf(['\n  GOOD WHEN: RMS(dEOM - dSim) is small relative to RMS(dSim) on\n' ...
             '  every channel, i.e. the analytic Coulomb term reproduces the\n' ...
             '  Simscape Coulomb effect.\n']);

    figure('Name', 'Coulomb effect: analytic vs Simscape');
    for i = 1:3
        subplot(3, 1, i);
        plot(t, dSim(:, i), 'b', 'LineWidth', 1.0); hold on
        plot(t, dEOM(:, i), 'r--', 'LineWidth', 1.0);
        ylabel([axes_lbl{i} ' [m]']); grid on
        if i == 1
            legend('Simscape ON-OFF', 'Analytic ON-OFF', 'Location', 'best');
            title('Coulomb effect (ON - OFF): Simscape vs analytic EOM');
        end
    end
    xlabel('Time [s]');
else
    figure('Name', 'Coulomb effect: analytic only');
    for i = 1:3
        subplot(3, 1, i);
        plot(t, q_eom_OFF(:, i), 'k', 'LineWidth', 0.8); hold on
        plot(t, q_eom_ON(:, i), 'r', 'LineWidth', 0.8);
        ylabel([axes_lbl{i} ' [m]']); grid on
        if i == 1
            legend('Coulomb OFF', 'Coulomb ON', 'Location', 'best');
            title('Analytic EOM: Coulomb OFF vs ON (Simscape not available)');
        end
    end
    xlabel('Time [s]');
end

% ------------------------------------------------------------------
% 7. Export reference for the Python Phase-2 cross-check
% ------------------------------------------------------------------
out_path = fullfile(here, 'coulomb_validation_ref.mat');
save(out_path, 't', 'r', 'x0', 'P', 'Lb', 'cc_on', 'cc_off', ...
     'q_eom_OFF', 'q_eom_ON', 'q_sim_OFF', 'q_sim_ON', 'have_simscape', ...
     'fs', 'ts');
fprintf('\nReference saved: %s\n', out_path);


% ======================================================================
% Local functions
% ======================================================================
function q_out = closed_loop_analytic(baseFn, x0, r_stage, Cfb_ss, P, Lb, ts, cc)
% Closed-loop emulation: discrete controller (stepped at ts) + continuous
% plant (RK4 at ts, ZOH on the force). Mirrors the Telica _run_closed_loop
% pattern. Controller acts on the same-sample stage error (no extra delay).
    [Ac, Bc, Cc, Dc] = ssdata(Cfb_ss);
    xc = zeros(size(Ac, 1), 1);
    x  = x0;
    nt = size(r_stage, 1);
    q_out = zeros(nt, 3);
    for k = 1:nt
        q_log   = x(1:3);
        q_stage = P.' * q_log;             % logical -> stage positions
        q_out(k, :) = q_stage.';
        e  = r_stage(k, :).' - q_stage;    % stage tracking error
        yc = Cc * xc + Dc * e;             % controller output (stage force)
        xc = Ac * xc + Bc * e;             % controller state update
        u_log = P * yc;                    % stage force -> logical
        f1 = gantrySystemCoulomb(baseFn, u_log, x,             Lb, cc);
        f2 = gantrySystemCoulomb(baseFn, u_log, x + ts/2 * f1, Lb, cc);
        f3 = gantrySystemCoulomb(baseFn, u_log, x + ts/2 * f2, Lb, cc);
        f4 = gantrySystemCoulomb(baseFn, u_log, x + ts   * f3, Lb, cc);
        x  = x + ts/6 * (f1 + 2*f2 + 2*f3 + f4);
    end
end

function r = rmsval(v)
% Root-mean-square without the Signal Processing Toolbox dependency.
    r = sqrt(mean(v(:).^2));
end

function q_grid = align_to_grid(q_raw, t_raw, t_grid)
% Resample a Simulink output onto the fixed analysis grid if the solver used
% variable steps (main.m note: sim() can return interpolated outputs).
    if numel(t_raw) == numel(t_grid) && max(abs(t_raw(:) - t_grid(:))) < 1e-12
        q_grid = q_raw;
    else
        q_grid = interp1(t_raw, q_raw, t_grid, 'linear', 'extrap');
    end
end
