% export_lpv_sim.m
% ----------------
% Export varying-Y Simulink simulation data for LPV ZOH discretization
% validation (Task 2.2 Export 2).
%
% ---------------------------------------------------------------------------
% Simulink output variables -- what each q is and how it was verified
% ---------------------------------------------------------------------------
% The gantry_2025a Simulink model contains four ToWorkspace blocks. Their
% contents were verified by inspecting the .slx XML directly (gantry_2025a.slx
% is a ZIP; simulink/systems/system_root.xml lists ToWorkspace VariableNames,
% and simulink/systems/system_47.xml -- the Simscape subsystem -- lists the
% Coulomb gain blocks with <P Name="Commented">on</P>, confirming they are
% disabled).
%
%   q          Simscape full nonlinear model.
%              Coulomb friction gains (cc1, cc2, ccy) are present in the
%              subsystem XML but carry <P Name="Commented">on</P> -- they are
%              DISABLED. Coriolis-centripetal forces are included (Simscape
%              integrates the full multibody equations). No Coulomb.
%
%   q1         CT quasi-LPV, gantrySystem.m integrated by Simulink.
%              Self-schedules Y = x(3) at every ODE step -- M(Y) varies
%              continuously as Y moves 0.3 -> 0.1 m.
%              No Coulomb, no Coriolis-centripetal.
%              PRIMARY comparison target for Python LPV-LFR baseline.
%
%   q2         CT quasi-LPV with Coriolis-centripetal,
%              gantrySystemCoriolisCentripetal.m integrated by Simulink.
%              Self-schedules Y = x(3). No Coulomb.
%              NOT exported to .mat (not needed for current validation).
%
%   q3         Frozen LTI lsim at Y = 0.3 m (operating point), computed in
%              main.m AFTER sim() via lsim(T, ...). ZOH discrete-time.
%              NOT a Simulink ToWorkspace output -- generated post-hoc.
%              NOT exported here.
%
% Residual interpretation (from main.m figure titles):
%   q - q1  = Coriolis-centripetal effect (both have no Coulomb)
%   q - q2  = residual nonlinear geometry beyond Coriolis (small at 16 kHz)
%   q - q3  = Coriolis + LPV scheduling benefit (frozen M(Y=0.3) vs varying)
%
% ---------------------------------------------------------------------------
% What this proves:
%   Python RK4 vs q1 -- integration method mismatch only (RK4 fixed-step vs
%   ode45 adaptive). Both integrate the same CT quasi-LPV ODE. Expected error
%   ~1e-14 m. Validates that Python correctly implements gantrySystem.m physics.
%
%   Frozen LTI vs q1 -- ZOH error + frozen M(Y=0.3) scheduling error combined.
%   Gap between the two = LPV benefit from tracking M(Y).
%
% Test trajectory (Option B -- Y step, X at rest):
%   Y: smooth 3rd-order step from 0.3 m to 0.1 m (200 mm in negative direction).
%   X1, X2: hold at 0 m throughout.
%   Direction: negative, consistent with main.m (r(:,3) = -pvajs + 0.3 goes 0.3->-0.1).
%   Moving positive (toward 0.5 m) risks reaching the physical beam end.
%   Why: isolates Y dynamics, decouples X-Y coupling, Y stays within safe range.
%
% Variables exported (Matlab-output/lpv_sim_varying_y.mat):
%   t_sim          (N x 1)  time vector [s]
%   fs             (1 x 1)  sample frequency = 16000 Hz
%   r_sim          (N x 3)  reference [X1_ref, X2_ref, Y_ref] stage coords [m]
%   u_q1           (N x 3)  force applied to q1 path [F_X1, F_X2, F_Y] [N]
%   q1             (N x 3)  CT quasi-LPV output [X1, X2, Y] [m]  -- PRIMARY target
%   q_simscape     (N x 3)  Simscape output [X1, X2, Y] [m]  (Coulomb disabled)
%   Y_trajectory   (N x 1)  absolute Y(t) = q1(:,3) [m]  -- scheduling variable
%
% Does NOT modify any file in kamtin-fp-model/.
%
% Run from repo root (from MATLAB):
%   cd('<path-to>/Baseline-LPV-Augmentation')
%   run('Matlab-scripts/export_lpv_sim.m')

addpath(genpath(fullfile(pwd, 'kamtin-fp-model', '03 Simulink gantry')))

% ------------------------------------------------------------------
% 1. Physical parameters (identical to main.m lines 12-49)
% ------------------------------------------------------------------
mb  = 22.8;    % Mass of moving cross-arm         [kg]
mh  = 10.1;    % Mass of payload (Y-axis)         [kg]
m1  = 10.2;    % Mass of actuator X1              [kg]
m2  = 10.7;    % Mass of actuator X2              [kg]

Jb  = 1.0;     % Rotary inertia of cross-arm      [kg.m^2]
Jh  = 0.05;    % Rotary inertia of payload        [kg.m^2]

cg1 = 14.5;    % Viscous friction X1              [N/(m/s)]
cg2 = 20.3;    % Viscous friction X2              [N/(m/s)]
cy  = 10;      % Viscous friction Y               [N/(m/s)]

cb1 = 9;       % Viscous friction joint 1         [Nm/(rad/s)]
cb2 = 9;       % Viscous friction joint 2         [Nm/(rad/s)]

kb1 = 1987.5;  % Stiffness joint 1                [N.m/rad]
kb2 = 1987.5;  % Stiffness joint 2                [N.m/rad]

Lb  = 0.725;   % Length of moving cross-arm       [m]
Lh  = 0.25;    % Length of payload                [m]
d   = 0.1;     % Distance cross-arm to payload    [m]

% Coulomb friction (main.m lines 27-29) -- NOT in SS model, but used by Simscape.
cc1 = 16.8;    % Coulomb friction actuator X1     [N]
cc2 = 18.35;   % Coulomb friction actuator X2     [N]
ccy = 11.6;    % Coulomb friction payload Y       [N]

% Design operating point (main.m line 49): Y = 0.3 m.
% Set as plain 'Y' so the Simulink model can read it from the workspace.
Y    = 0.3;
Y_op = Y;

% Mass matrix at operating point (main.m lines 52-54)
M_op = [m1+m2+mb+mh,                          (m1-m2)*Lb/2 - mh*Y_op,       0;
        (m1-m2)*Lb/2 - mh*Y_op, Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y_op^2, -mh*d;
        0,                                                       -mh*d,    mh];

% Viscous damping matrix (constant, main.m lines 57-59)
C_damp = [           cg1+cg2,               (cg1-cg2)*Lb/2,  0;
          (cg1-cg2)*Lb/2, cb1+cb2+(cg1+cg2)*Lb^2/4,         0;
                       0,                            0,      cy];

% Stiffness matrix (constant, main.m lines 62-64)
K = [0,         0, 0;
     0, kb1+kb2,  0;
     0,         0, 0];

% ------------------------------------------------------------------
% 2. State-space and controller (identical to main.m lines 88-207)
% ------------------------------------------------------------------
n = 3;
sys_logical = getss(n, M_op, C_damp, K);

% Stage coordinate transform (main.m lines 98-100)
P = [1,    1,    0;
     Lb/2, -Lb/2, 0;
     0,    0,    1];
StageCoordinatesSystem = P.' * sys_logical * P;

fs = 16e3;
ts = 1 / fs;

% Discrete feedback controller (main.m lines 199-202, fbw=100 Hz)
fbw = 100;
Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
for j = 1:3
    Cfb(j,j) = ruleOfThumb(fbw, StageCoordinatesSystem(j,j), ts);
end

% ------------------------------------------------------------------
% 3. Test trajectory: Y step 0.3 -> 0.1 m, X at rest
% ------------------------------------------------------------------
% Y displacement: 0.2 m in the NEGATIVE direction (same as main.m convention).
% main.m: r(:,3) = -pvajs + 0.3 moves Y from 0.3 down to -0.1 m.
% Moving positive (toward 0.5 m) risks reaching the physical beam end-stop.
% X1=X2 reference stays at zero throughout.
%
% Parameters chosen for the Y axis (slower than X):
%   vmax  = 0.3 m/s  -- conservative for Y axis
%   amax  = 3   m/s^2
%   jmax  = amax / jerkTime = 3 / 0.05 = 60 m/s^3
% Expected motion duration: ~1.1 s total.
pmax_Y    = 0.2;     % [m]    Y displacement magnitude (0.3 -> 0.1 m)
vmax_Y    = 0.3;     % [m/s]
amax_Y    = 3.0;     % [m/s^2]
jerkTime  = 0.05;    % [s]
jmax_Y    = amax_Y / jerkTime;   % [m/s^3]
smax      = Inf;

[pvajs_Y] = thirdOrderSetpointETEL(pmax_Y, vmax_Y, amax_Y, jmax_Y, smax, ts);
n_move = size(pvajs_Y, 1);  % number of samples for the Y motion

% Hold periods: 0.5 s before (settle at Y=0.3) and 0.5 s after (steady state).
n_hold = round(0.5 / ts);
nt = n_hold + n_move + n_hold;
t  = ts * (0:nt-1)';

% Reference matrix [X1_ref, X2_ref, Y_ref] in stage coordinates [m].
% X1, X2: hold at 0.  Y: hold at 0.3, then ramp DOWN to 0.1, then hold at 0.1.
r = zeros(nt, 3);
r(:, 3) = 0.3;                                         % Y starts at 0.3 m
r(n_hold + (1:n_move), 3) = 0.3 - pvajs_Y(:, 1);      % Y moves 0.3 -> 0.1 m
r(n_hold + n_move + 1 : end, 3) = 0.1;                 % Y holds at 0.1 m

f = zeros(nt, 3);  % no feedforward forces

% Physical range check (ETEL datasheet: ±400 mm from center).
Y_LIMIT = 0.4;
assert(max(r(:,3)) <= Y_LIMIT && min(r(:,3)) >= -Y_LIMIT, ...
    'Y reference exceeds physical machine range [%.2f, %.2f] m. Max: %.3f, Min: %.3f', ...
    -Y_LIMIT, Y_LIMIT, max(r(:,3)), min(r(:,3)));

% Discrete LTI at operating point Y=0.3 (main.m line 218, moved before sim).
% main.m computes G AFTER sim(), but the Simulink LTI System blocks read G
% from the workspace during the run. Define it here so the first run works.
G = c2d(StageCoordinatesSystem, ts, 'zoh');

% ------------------------------------------------------------------
% 4. Run Simulink
% ------------------------------------------------------------------
% Workspace variables read by Simulink:
%   r, f        -- FromWorkspace blocks (reference and feedforward)
%   Cfb         -- feedback controller (LTI System blocks)
%   G           -- frozen LTI (LTI System blocks)
%   Y           -- operating point, may be used by Simscape model
%   cc1,cc2,ccy -- Coulomb friction gains in Simscape subsystem
%   All physical parameters (mb, mh, m1, m2, ...) -- Simscape body masses
% q, q1, q2 appear in workspace after sim() via ToWorkspace blocks.
mdl = 'gantry_2025a';
fprintf('Running Simulink model %s for %.2f s ...\n', mdl, t(end));
sim(mdl, t(end));
fprintf('Simulation complete. Samples collected: %d\n', size(q1, 1));

% ------------------------------------------------------------------
% 5. Reconstruct u applied to q1 path
% ------------------------------------------------------------------
% The Simulink q1 path runs gantrySystem.m in closed loop with Cfb.
% u_q1 = Cfb * (r - q1)   (feedforward f = 0)
% Cfb is a discrete diagonal 3x3 TF. lsim handles multi-channel TF.
%
% Note: if sim() returns q1 with fewer rows than nt (variable-step can
% produce interpolated outputs), resample r to match before computing u.
N_sim = size(q1, 1);
if N_sim ~= nt
    t_sim = linspace(0, t(end), N_sim)';
    r_sim = interp1(t, r, t_sim);
else
    t_sim = t;
    r_sim = r;
end

e_q1  = r_sim - q1;                     % (N x 3) tracking error
u_q1  = lsim(ss(Cfb), e_q1, t_sim);     % (N x 3) [F_X1, F_X2, F_Y] [N]

% ------------------------------------------------------------------
% 6. Extract scheduling variable and rename Simscape output
% ------------------------------------------------------------------
Y_trajectory = q1(:, 3);    % (N x 1) absolute Y position over time [m]
q_simscape   = q;           % rename for clarity (q = Simscape ToWorkspace output)

% ------------------------------------------------------------------
% 7. Report
% ------------------------------------------------------------------
fprintf('\nY trajectory:\n')
fprintf('  Initial Y: %.4f m\n', Y_trajectory(1))
fprintf('  Final   Y: %.4f m\n', Y_trajectory(end))
fprintf('  Max     Y: %.4f m\n', max(Y_trajectory))
fprintf('  Min     Y: %.4f m\n', min(Y_trajectory))
fprintf('  Range     : %.1f mm\n', (max(Y_trajectory) - min(Y_trajectory)) * 1e3)

% ------------------------------------------------------------------
% 8. Save
% ------------------------------------------------------------------
out_dir  = fullfile(pwd, 'Matlab-output');
out_path = fullfile(out_dir, 'lpv_sim_varying_y.mat');

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

save(out_path, 't_sim', 'fs', 'r_sim', 'u_q1', 'q1', 'q_simscape', 'Y_trajectory');

fprintf('\nSaved to: %s\n', out_path)
fprintf('Variables: t_sim (%dx1), r_sim (%dx3), u_q1 (%dx3), q1 (%dx3), ', ...
        N_sim, N_sim, N_sim, N_sim)
fprintf('q_simscape (%dx3), Y_trajectory (%dx1)\n', N_sim, N_sim)
