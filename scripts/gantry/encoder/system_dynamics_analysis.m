%% system_dynamics_analysis.m
% Determine natural frequencies and Nyquist bounds for the gantry model
% in two configurations: baseline (6-state) and MSD-augmented (8-state).
%
% Saves a JSON with f_max, f_nyquist, f_practical per configuration
% to simulations/gantry_subnet/diagnostics/system_dynamics.json.
% The Python downsampling script imports this to validate Nyquist.

clear; clc;

%% Output path
out_dir = fullfile(fileparts(mfilename('fullpath')), ...
    '..', '..', '..', 'simulations', 'gantry_subnet', 'diagnostics');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% Gantry parameters (from main_augmentation.m)
mb  = 22.8;      % Mass of the moving cross-arm (kg)
mh  = 10.1;      % Mass of the payload (Y-axis) (kg)
m1  = 10.2;      % Mass of actuator X1 (kg)
m2  = 10.7;      % Mass of actuator X2 (kg)

Jb  = 1.0;       % Rotary inertia of the cross-arm (kg.m^2)
Jh  = 0.05;      % Rotary inertia of the payload (Y-axis) (kg.m^2)

cg1 = 14.5;      % Viscous friction of actuator X1 (N/(m/s))
cg2 = 20.3;      % Viscous friction of actuator X2 (N/(m/s))
cy  = 10;        % Viscous friction of the payload (Y-axis) (N/(m/s))

cb1 = 9;         % Viscous friction of elastic joints 1 (Nm/(rad/s))
cb2 = 9;         % Viscous friction of elastic joints 2 (Nm/(rad/s))

kb1 = 1987.5;    % Stiffness of elastic joint 1 (N.m/rad)
kb2 = 1987.5;    % Stiffness of elastic joint 2 (N.m/rad)

Lb  = 0.725;     % Length of the moving cross-arm (m)
d   = 0.1;       % Distance between cross-arm and payload (m)

%% MSD parameters (from main_augmentation.m)
ma_frac  = 0.10;
ma       = ma_frac * mh;           % 1.01 kg hidden MSD mass
mh_rigid = mh - ma;                % 9.09 kg rigid part of payload
L0       = 0.10;                   % equilibrium offset (m)
fa       = 400;                    % target MSD natural frequency (Hz)
ka       = ma * (2*pi*fa)^2;       % MSD spring stiffness (N/m)
zeta_a   = 0.05;                   % damping ratio
ca       = 2 * zeta_a * sqrt(ka * ma);  % MSD damper coefficient (Ns/m)

%% Frozen operating point
Y_op = 0;

%% ==================== BASELINE (6-state) ====================
fprintf('============================================================\n');
fprintf('  BASELINE gantry at Y_op = %.2f m  (6 states)\n', Y_op);
fprintf('============================================================\n\n');

M_bl = [          m1+m2+mb+mh,                    (m1-m2)*Lb/2 - mh*Y_op,         0;
        (m1-m2)*Lb/2 - mh*Y_op,  Jb+Jh+(m1+m2)*Lb^2/4 + mh*d^2 + mh*Y_op^2,  -mh*d;
                              0,                                       -mh*d,      mh];

C_bl = [         cg1+cg2,            (cg1-cg2)*Lb/2,  0;
        (cg1-cg2)*Lb/2,  cb1+cb2+(cg1+cg2)*Lb^2/4,  0;
                      0,                          0, cy];

K_bl = [0,       0, 0;
        0, kb1+kb2, 0;
        0,       0, 0];

A_bl = [zeros(3),    eye(3);
        -M_bl\K_bl, -M_bl\C_bl];
B_bl = [zeros(3); M_bl\eye(3)];
C_bl_out = [eye(3), zeros(3)];
D_bl_out = zeros(3);

sys_bl = ss(A_bl, B_bl, C_bl_out, D_bl_out);

res_bl = analyze_system(sys_bl, 'Baseline');

%% ==================== MSD-AUGMENTED (8-state) ====================
fprintf('\n============================================================\n');
fprintf('  MSD-AUGMENTED gantry at Y_op = %.2f m  (8 states)\n', Y_op);
fprintf('  ma = %.2f kg, fa = %.0f Hz, zeta_a = %.2f\n', ma, fa, zeta_a);
fprintf('============================================================\n\n');

% 4-DOF mass matrix at frozen Y_op (linearized from gantrySystemExtended.m)
M_ext = [m1+m2+mb+mh_rigid+ma,  (m1-m2)*Lb/2 - (mh_rigid+ma)*Y_op - ma*L0,              0,      0;
         (m1-m2)*Lb/2 - (mh_rigid+ma)*Y_op - ma*L0, ...
             Jb+Jh+(m1+m2)*Lb^2/4 + (mh_rigid+ma)*d^2 + mh_rigid*Y_op^2 + ma*(Y_op+L0)^2, ...
             -(mh_rigid+ma)*d, -ma*d;
         0,  -(mh_rigid+ma)*d,  mh_rigid+ma,  ma;
         0,  -ma*d,             ma,            ma];

C_ext = [cg1+cg2,            (cg1-cg2)*Lb/2,  0,  0;
         (cg1-cg2)*Lb/2,  cb1+cb2+(cg1+cg2)*Lb^2/4,  0,  0;
         0,  0,  cy,  0;
         0,  0,   0, ca];

K_ext = [0,  0,        0,  0;
         0,  kb1+kb2,  0,  0;
         0,  0,        0,  0;
         0,  0,        0, ka];

A_ext = [zeros(4),     eye(4);
         -M_ext\K_ext, -M_ext\C_ext];
B_ext = [zeros(4,3); M_ext\[eye(3); zeros(1,3)]];
C_ext_out = [eye(4), zeros(4)];
D_ext_out = zeros(4,3);

sys_ext = ss(A_ext, B_ext, C_ext_out, D_ext_out);

res_ext = analyze_system(sys_ext, 'MSD-augmented');

%% ==================== SAVE TO JSON ====================
result = struct();
result.baseline = res_bl;
result.msd      = res_ext;
result.Y_op     = Y_op;
result.generated = datestr(now, 'yyyy-mm-dd HH:MM:SS');

json_path = fullfile(out_dir, 'system_dynamics.json');
fid = fopen(json_path, 'w');
fprintf(fid, '%s', jsonencode(result));
fclose(fid);
fprintf('\n==> Saved to %s\n', json_path);

%% ==================== LOCAL FUNCTION ====================
function res = analyze_system(sys, label)
% Analyze a CT state-space model: poles, Nyquist bounds, eigenvalues.
% Returns a struct with the key numbers (all finite, JSON-safe).

    A = sys.A;
    n = size(A, 1);

    % --- damp: natural frequencies, damping ratios, poles (all same order) ---
    fprintf('--- damp ---\n');
    [wn, zeta, p] = damp(sys);

    % --- Nyquist bounds (from non-zero poles only) ---
    wn_nonzero = wn(wn > 0);
    f_max_hz     = max(wn_nonzero) / (2*pi);
    f_nyquist_hz = 2 * f_max_hz;
    f_practical_hz = 10 * f_max_hz;

    fprintf('\n--- Nyquist conclusion (%s) ---\n', label);
    fprintf('  f_max         = %.2f Hz  (highest natural frequency)\n', f_max_hz);
    fprintf('  f_nyquist     = %.2f Hz  (2 x f_max, theoretical minimum)\n', f_nyquist_hz);
    fprintf('  f_practical   = %.2f Hz  (10 x f_max, engineering margin)\n', f_practical_hz);

    % --- Eigenvalues (using p from damp, same order as wn/zeta) ---
    fprintf('\n--- Eigenvalues of A ---\n');
    for k = 1:length(p)
        e = p(k);
        if imag(e) == 0
            if real(e) == 0
                fprintf('  lambda_%d = 0 (integrator)\n', k);
            else
                fprintf('  lambda_%d = %.4f (real, tau = %.4f s)\n', ...
                    k, real(e), -1/real(e));
            end
        else
            fprintf('  lambda_%d = %.4f %+.4fj  (f = %.2f Hz)\n', ...
                k, real(e), imag(e), abs(imag(e))/(2*pi));
        end
    end

    % --- Observability ---
    O = obsv(A, sys.C);
    obs_rank = rank(O);
    fprintf('\n--- Observability ---\n');
    fprintf('  rank(obsv) = %d / %d states\n', obs_rank, n);

    % --- Build result struct (all finite values for JSON) ---
    res = struct();
    res.n_states       = n;
    res.f_max_hz       = f_max_hz;
    res.f_nyquist_hz   = f_nyquist_hz;
    res.f_practical_hz = f_practical_hz;
    res.obs_rank       = obs_rank;

    % Pole table: use p from damp (same order as wn/zeta), only finite entries
    poles = struct('real_parts', [], 'imag_parts', [], 'freq_hz', [], 'damping', []);
    for k = 1:length(wn)
        if isfinite(wn(k)) && isfinite(zeta(k))
            poles.real_parts(end+1)  = real(p(k));
            poles.imag_parts(end+1)  = imag(p(k));
            poles.freq_hz(end+1)     = wn(k) / (2*pi);
            poles.damping(end+1)     = zeta(k);
        end
    end
    res.poles = poles;
end
