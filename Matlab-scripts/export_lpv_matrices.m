% export_lpv_matrices.m
% --------------------
% Export discrete-time LPV state-space matrices A(Y), B(Y), C, D at 50
% operating points for Python validation (Task 2.4).
%
% This script duplicates ONLY the physics setup from main.m (lines 12-36,
% 52-64, 88, 98-100, 218) and calls getss.m directly — identical to what
% main.m does at Y=0.3 but looped over a sweep of Y values.
% main.m is NOT called (it runs Simulink and generates figures).
%
% Output: Matlab-output/lpv_matrices.mat
%   A_all   (6, 6, 50)  — discrete-time state matrix at each Y
%   B_all   (6, 3, 50)  — discrete-time input matrix at each Y
%   C_all   (3, 6, 50)  — output matrix (constant, stored per Y for convenience)
%   D_all   (3, 3, 50)  — feedthrough matrix (zero)
%   Y_values (50, 1)    — Y sweep values [m]
%   det_M   (50, 1)     — det(M(Y)) at each Y [physics health check]
%
% Run from repo root (from MATLAB):
%   cd('Baseline-LPV-Augmentation')
%   run('Matlab-scripts/export_lpv_matrices.m')

addpath(genpath(fullfile(pwd, 'kamtin-fp-model', '03 Simulink gantry')))

% ------------------------------------------------------------------
% Physical parameters (from main.m lines 12-36)
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
d   = 0.1;     % Distance cross-arm to payload    [m]

% Viscous damping matrix (constant — no Y-dependence, from main.m lines 57-59)
C_damp = [           cg1 + cg2,               (cg1 - cg2) * Lb / 2,  0;
           (cg1 - cg2) * Lb / 2, cb1 + cb2 + (cg1 + cg2) * Lb^2 / 4, 0;
                               0,                                    0, cy];

% Stiffness matrix (constant, from main.m lines 62-64)
K = [0,         0, 0;
     0, kb1 + kb2, 0;
     0,         0, 0];

% Stage coordinate transform (from main.m lines 98-100)
P = [1,     1,    0;
     Lb/2, -Lb/2, 0;
     0,     0,    1];

% Sample frequency (from main.m line 164)
fs = 20e3;
ts = 1 / fs;

n = 3;  % DOF

% ------------------------------------------------------------------
% Y sweep (D-016: 50 points, linspace(0.05, 0.75))
% ------------------------------------------------------------------
% Physical range (ETEL datasheet): Y=0 at beam center, total stroke 800 mm,
% so valid range is [-0.4, 0.4] m. See docs/fp-model-structure.md.
Y_values = linspace(-0.35, 0.35, 50)';  % (50, 1) -- within physical range

nY = length(Y_values);

A_all  = zeros(6, 6, nY);
B_all  = zeros(6, 3, nY);
C_all  = zeros(3, 6, nY);
D_all  = zeros(3, 3, nY);
det_M  = zeros(nY, 1);

for k = 1:nY
    Y = Y_values(k);

    % Mass matrix (from main.m lines 52-54) — varies with Y
    M = [          m1 + m2 + mb + mh,                          (m1 - m2) * Lb / 2 - mh * Y,        0;
         (m1 - m2) * Lb / 2 - mh * Y, Jb + Jh + (m1 + m2) * Lb^2 / 4 + mh * d^2 + mh * Y^2,  -mh * d;
                                    0,                                              -mh * d,       mh];

    det_M(k) = det(M);

    % Continuous-time SS in logical coordinates (getss.m)
    [sys_logical, ~, ~, ~, ~] = getss(n, M, C_damp, K);

    % Transform to stage coordinates (main.m line 103)
    sys_stage = P.' * sys_logical * P;

    % ZOH discretization (main.m line 218)
    G = c2d(sys_stage, ts, 'zoh');

    A_all(:, :, k) = G.A;
    B_all(:, :, k) = G.B;
    C_all(:, :, k) = G.C;
    D_all(:, :, k) = G.D;
end

% ------------------------------------------------------------------
% Save
% ------------------------------------------------------------------
out_dir  = fullfile(pwd, 'Matlab-output');
out_path = fullfile(out_dir, 'lpv_matrices.mat');

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

save(out_path, 'A_all', 'B_all', 'C_all', 'D_all', 'Y_values', 'det_M', 'fs', 'ts');

fprintf('Saved %d operating points to: %s\n', nY, out_path);
fprintf('Y range: %.3f m to %.3f m\n', Y_values(1), Y_values(end));
fprintf('det(M) range: %.4f to %.4f  (all positive = M positive definite)\n', ...
        min(det_M), max(det_M));
