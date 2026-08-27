% compare_poles_gantry_vs_msd.m
% -----------------------------------------------------------------------
% Z-plane pole comparison: gantry vs Jan's MSD 2-DOF / 3-DOF examples.
%
% PURPOSE
%   Visualise why Jan's ANN routing (ANN correction -> ALL physical state
%   rows, fixed C output) works safely on MSD but causes blowup on the
%   gantry.
%
%   Key quantity: min(1 - |z|) per system.
%     MSD 2-DOF  : min(1-|z|) ~ 4.4e-3  -> all poles strictly inside unit circle
%     MSD 3-DOF  : min(1-|z|) ~ 3.8e-3  -> all poles strictly inside unit circle
%     Gantry     : min(1-|z|) = 0.0     -> 2 poles AT z=1 (rigid-body integrators)
%
%   Over nf=400 simulation steps, an ANN correction epsilon injected into
%   an integrator state grows to ~ 400*epsilon. For MSD the slowest pole
%   gives decay 0.996^400 ~ 0.17, so the correction stays bounded.
%   For gantry z=1 means the correction ACCUMULATES without any decay.
%
% DATA SOURCES
%   MSD matrices : data/mass_spring_damper/msd_2dof.mat  (Jan's ECC example)
%                  data/mass_spring_damper/msd_3dof.mat  (Jan's journal example)
%   Gantry       : re-derived from physical parameters at Y_op=0 (equilibrium),
%                  discretised at Ts = 1/4000 s (training sample rate).
%                  Matches Python gantry_linearization.py.
%
% RUN FROM PROJECT ROOT:
%   run('Matlab-scripts/compare_poles_gantry_vs_msd.m')
%
% OUTPUT
%   Matlab-scripts/Matlab-output/compare_poles_gantry_vs_msd.png

%% -----------------------------------------------------------------------
%  0. Setup
% -----------------------------------------------------------------------
close all;
out_dir = fullfile(pwd, 'Matlab-scripts', 'Matlab-output');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% -----------------------------------------------------------------------
%  1. Load MSD matrices (Jan's examples)
% -----------------------------------------------------------------------
d2 = load(fullfile(pwd, 'data', 'mass_spring_damper', 'msd_2dof.mat'));
d3 = load(fullfile(pwd, 'data', 'mass_spring_damper', 'msd_3dof.mat'));

Ad_2dof = d2.Ad;
Ad_3dof = d3.Ad;
Ts_2dof = d2.Ts;
Ts_3dof = d3.Ts;

poles_2dof = eig(Ad_2dof);
poles_3dof = eig(Ad_3dof);

%% -----------------------------------------------------------------------
%  2. Derive gantry Ad at Y_op=0, Ts=1/4000
%     Mirrors model_augmentation/systems/gantry_linearization.py
% -----------------------------------------------------------------------
% Physical parameters (identical to main_augmentation.m)
mb  = 22.8;   mh  = 10.1;  m1  = 10.2;  m2  = 10.7;
Jb  = 1.0;    Jh  = 0.05;
cg1 = 14.5;   cg2 = 20.3;  cy  = 10;
cb1 = 9;      cb2 = 9;
kb1 = 1987.5; kb2 = 1987.5;
Lb  = 0.725;  d   = 0.1;

% Input/output transformation (stage <-> logical coordinates)
P = [1,    1,     0;
     Lb/2, -Lb/2, 0;
     0,    0,     1];

% Linearise at Y_op = 0 (LFR feedback vanishes at equilibrium)
Y_op = 0;
n    = 3;

% Mass matrix at Y_op = 0
M0 = [m1+m2+mb+mh,            (m1-m2)*Lb/2 - mh*Y_op,                      0;
      (m1-m2)*Lb/2 - mh*Y_op, Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y_op^2,  -mh*d;
      0,                       -mh*d,                                          mh];

% Damping matrix (Y_op-independent)
C_damp = [cg1+cg2,        (cg1-cg2)*Lb/2,              0;
          (cg1-cg2)*Lb/2, cb1+cb2+(cg1+cg2)*Lb^2/4,    0;
          0,               0,                            cy];

% Stiffness matrix (has zeros -> integrator poles in X and Y)
K_stiff = [0,  0,       0;
           0,  kb1+kb2, 0;
           0,  0,       0];

% Continuous-time state-space (logical coordinates)
%   x = [q_logical; qdot_logical]  (6-state)
%   xdot = Ac*x + Bc_stage*u_stage
%   y    = Cc*x  (stage positions)
Ac_g  = [zeros(n),       eye(n);
         -M0\K_stiff,  -M0\C_damp];
Bc_g  = [zeros(n, n); M0\P];      % maps stage forces to state derivatives
Cc_g  = [P', zeros(n)];           % stage position = P^T * q_logical
Dc_g  = zeros(n);

sys_g_ct = ss(Ac_g, Bc_g, Cc_g, Dc_g);

% Discretise at training sample rate (4 kHz, downsampled 5x from 20 kHz)
Ts_gantry = 1/4000;
sys_g_dt  = c2d(sys_g_ct, Ts_gantry, 'zoh');
Ad_gantry = sys_g_dt.A;
poles_gantry = eig(Ad_gantry);

%% -----------------------------------------------------------------------
%  3. Compute min(1-|z|) summary
% -----------------------------------------------------------------------
min_gap_2dof   = min(1 - abs(poles_2dof));
min_gap_3dof   = min(1 - abs(poles_3dof));
min_gap_gantry = min(1 - abs(poles_gantry));

fprintf('\n--- Discrete-time pole summary ---\n');
fprintf('%-20s  Ts=%.4f s  nx=%d  min(1-|z|)=%.2e\n', ...
    'MSD 2-DOF (Jan ECC)', Ts_2dof, length(poles_2dof), min_gap_2dof);
fprintf('%-20s  Ts=%.4f s  nx=%d  min(1-|z|)=%.2e\n', ...
    'MSD 3-DOF (Jan jrnl)', Ts_3dof, length(poles_3dof), min_gap_3dof);
fprintf('%-20s  Ts=%.4f s  nx=%d  min(1-|z|)=%.2e\n', ...
    'Gantry (Y_op=0)', Ts_gantry, length(poles_gantry), min_gap_gantry);

% Amplification factor at worst pole over nf steps
nf = 400;
worst_2dof   = max(abs(poles_2dof));
worst_3dof   = max(abs(poles_3dof));
worst_gantry = max(abs(poles_gantry));
fprintf('\nAt nf=%d steps, worst-pole amplification:\n', nf);
fprintf('  MSD 2-DOF  : |z|_max=%.6f -> %.2fx\n', worst_2dof,   worst_2dof^nf);
fprintf('  MSD 3-DOF  : |z|_max=%.6f -> %.2fx\n', worst_3dof,   worst_3dof^nf);
fprintf('  Gantry     : |z|_max=%.6f -> %.2fx  (integrators: no decay)\n', worst_gantry, worst_gantry^nf);

%% -----------------------------------------------------------------------
%  4. Plot: full z-plane + zoom near z=1
% -----------------------------------------------------------------------
theta = linspace(0, 2*pi, 500);
unit_x = cos(theta);
unit_y = sin(theta);

col_2dof   = [0.00, 0.45, 0.70];   % blue
col_3dof   = [0.00, 0.62, 0.45];   % green
col_gantry = [0.84, 0.19, 0.15];   % red

fig = figure('Name', 'Z-plane: Gantry vs MSD', 'Position', [100, 80, 1100, 500]);

% ---- Left: full z-plane ----
ax1 = subplot(1, 2, 1);
hold on; grid on; axis equal;
plot(unit_x, unit_y, 'k-', 'LineWidth', 1.2);   % unit circle
plot(real(poles_2dof),   imag(poles_2dof),   'x', ...
     'Color', col_2dof,   'MarkerSize', 10, 'LineWidth', 2);
plot(real(poles_3dof),   imag(poles_3dof),   's', ...
     'Color', col_3dof,   'MarkerSize', 8,  'LineWidth', 2);
plot(real(poles_gantry), imag(poles_gantry), 'o', ...
     'Color', col_gantry, 'MarkerSize', 9,  'LineWidth', 2);
xlabel('Re(z)'); ylabel('Im(z)');
title('Full z-plane');
legend({'Unit circle', ...
        sprintf('MSD 2-DOF (Ts=%.2fs, min(1-|z|)=%.2e)', Ts_2dof, min_gap_2dof), ...
        sprintf('MSD 3-DOF (Ts=%.2fs, min(1-|z|)=%.2e)', Ts_3dof, min_gap_3dof), ...
        sprintf('Gantry    (Ts=%.4fs, min(1-|z|)=%.2e)', Ts_gantry, min_gap_gantry)}, ...
       'Location', 'southwest', 'FontSize', 9);
xlim([-1.2, 1.2]); ylim([-1.2, 1.2]);

% Shade inside unit circle
fill(unit_x, unit_y, [0.95 0.95 0.95], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
uistack(findobj(ax1, 'Type', 'line'), 'top');

% ---- Right: zoom near z=1 (critical region) ----
ax2 = subplot(1, 2, 2);
hold on; grid on; axis equal;
plot(unit_x, unit_y, 'k-', 'LineWidth', 1.2);
plot(real(poles_2dof),   imag(poles_2dof),   'x', ...
     'Color', col_2dof,   'MarkerSize', 12, 'LineWidth', 2.5);
plot(real(poles_3dof),   imag(poles_3dof),   's', ...
     'Color', col_3dof,   'MarkerSize', 10, 'LineWidth', 2.5);
plot(real(poles_gantry), imag(poles_gantry), 'o', ...
     'Color', col_gantry, 'MarkerSize', 11, 'LineWidth', 2.5);

% Annotate min(1-|z|) per system
text(0.991, 0.006, sprintf('MSD-2: min(1-|z|)=%.2e', min_gap_2dof), ...
    'Color', col_2dof, 'FontSize', 8, 'HorizontalAlignment', 'left');
text(0.993, -0.007, sprintf('MSD-3: min(1-|z|)=%.2e', min_gap_3dof), ...
    'Color', col_3dof, 'FontSize', 8, 'HorizontalAlignment', 'left');
text(0.9994, 0.014, sprintf('Gantry: z=1 exactly (x2 poles)', min_gap_gantry), ...
    'Color', col_gantry, 'FontSize', 8, 'HorizontalAlignment', 'center');

xlabel('Re(z)'); ylabel('Im(z)');
title(sprintf('Zoom near z=1  (nf=%d: MSD decays, gantry accumulates)', nf));
xlim([0.990, 1.002]); ylim([-0.020, 0.020]);

sgtitle(sprintf(['Discrete-time poles: Gantry vs MSD (Jan''s examples)\n' ...
    'Jan''s ANN routing (\\Delta x_{ANN} \\rightarrow all state rows): safe for MSD, blowup for gantry']), ...
    'FontSize', 11);

%% -----------------------------------------------------------------------
%  5. Save
% -----------------------------------------------------------------------
out_path = fullfile(out_dir, 'compare_poles_gantry_vs_msd.png');
exportgraphics(fig, out_path, 'Resolution', 150);
fprintf('\nSaved: %s\n', out_path);
