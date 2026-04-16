% export_lpv_multi_traj.m
% -----------------------
% Generate and export 6 parameter-recovery training trajectories for the
% dual-gantry LPV-LFR baseline. Each trajectory targets a specific subset
% of the trainable physical parameters by exciting distinct dynamics modes.
%
% Excitation modes covered:
%   Y-only          -- cy, mh (diagonal M0[2,2] and M1 via Y velocity/accel)
%   X symmetric     -- translational inertia/friction, mh M1 coupling at Y!=0
%   X anti-symmetric-- rotational inertia, kb1+kb2, cb1+cb2, cg1-cg2 (UNIQUE)
%   X+Y combined    -- mh M1 coupling (-mh*Y) over continuous Y range
%
% See Matlab-output/parameter-recovery/trajectories.md for full justification.
%
% Each trajectory is saved to Matlab-output/parameter-recovery/<id>.mat with
% the same variable layout as export_lpv_sim.m:
%   t_sim        (N x 1)  time vector [s]
%   fs           (1 x 1)  sample rate [Hz]
%   r_sim        (N x 3)  reference [X1, X2, Y] stage coords [m]
%   u_q1         (N x 3)  force applied to CT quasi-LPV path [F_X1,F_X2,F_Y] [N]
%   u_q          (N x 3)  force applied to Simscape path [N]
%   q1           (N x 3)  CT quasi-LPV output [X1, X2, Y] [m]  -- PRIMARY target
%   q_simscape   (N x 3)  Simscape output [X1, X2, Y] [m]
%   Y_trajectory (N x 1)  Y(t) = q1(:,3) [m]
%
% Does NOT modify any file in kamtin-fp-model/.
%
% Run from repo root:
%   cd('<path>/Baseline-LPV-Augmentation')
%   run('Matlab-scripts/export_lpv_multi_traj.m')

addpath(genpath(fullfile(pwd, 'kamtin-fp-model', '03 Simulink gantry')))

% ------------------------------------------------------------------
% 1. Physical parameters (identical to main.m lines 12-49)
% ------------------------------------------------------------------
mb  = 22.8;    % Mass of moving cross-arm            [kg]
mh  = 10.1;    % Mass of payload (Y-axis)            [kg]
m1  = 10.2;    % Mass of actuator X1                 [kg]
m2  = 10.7;    % Mass of actuator X2                 [kg]
Jb  = 1.0;     % Rotary inertia of cross-arm         [kg.m^2]
Jh  = 0.05;    % Rotary inertia of payload           [kg.m^2]
cg1 = 14.5;    % Viscous friction X1                 [N/(m/s)]
cg2 = 20.3;    % Viscous friction X2                 [N/(m/s)]
cy  = 10;      % Viscous friction Y                  [N/(m/s)]
cb1 = 9;       % Viscous friction joint 1            [Nm/(rad/s)]
cb2 = 9;       % Viscous friction joint 2            [Nm/(rad/s)]
kb1 = 1987.5;  % Stiffness joint 1                   [N.m/rad]
kb2 = 1987.5;  % Stiffness joint 2                   [N.m/rad]
Lb  = 0.725;   % Length of moving cross-arm          [m]
Lh  = 0.25;    % Length of payload                   [m]
d   = 0.1;     % Distance cross-arm to payload       [m]
cc1 = 16.8;    % Coulomb friction X1 (Simscape only) [N]
cc2 = 18.35;   % Coulomb friction X2 (Simscape only) [N]
ccy = 11.6;    % Coulomb friction Y  (Simscape only) [N]

% ------------------------------------------------------------------
% 2. Operating point, matrices, controller (identical to export_lpv_sim.m)
% ------------------------------------------------------------------
Y    = 0.3;
Y_op = Y;

M_op = [m1+m2+mb+mh,                        (m1-m2)*Lb/2 - mh*Y_op,                   0;
        (m1-m2)*Lb/2 - mh*Y_op,  Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y_op^2,          -mh*d;
        0,                                                       -mh*d,                 mh];

C_damp = [cg1+cg2,         (cg1-cg2)*Lb/2,             0;
          (cg1-cg2)*Lb/2,  cb1+cb2+(cg1+cg2)*Lb^2/4,   0;
          0,                0,                           cy];

K = [0,  0,        0;
     0,  kb1+kb2,  0;
     0,  0,        0];

n                  = 3;
sys_logical        = getss(n, M_op, C_damp, K);
P                  = [1, 1, 0; Lb/2, -Lb/2, 0; 0, 0, 1];
StageCoordinatesSystem = P.' * sys_logical * P;

fs  = 20e3;
ts  = 1 / fs;

fbw = 100;
Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
for j = 1:3
    Cfb(j,j) = ruleOfThumb(fbw, StageCoordinatesSystem(j,j), ts);
end

G = c2d(StageCoordinatesSystem, ts, 'zoh');  % frozen LTI read by Simulink

% ------------------------------------------------------------------
% 3. Trajectory definitions
% ------------------------------------------------------------------
% Each struct defines one trajectory. Fields:
%   id          string   output filename (no .mat)
%   Y_initial   [m]      Y at start of main motion (must be <= 0.3).
%                        If < 0.3, a Y settle move is prepended automatically.
%   X_sym_amp   [m]      symmetric X: X1=X2 move to +A (0 = none)
%   X_anti_amp  [m]      anti-symmetric X: X1=+A, X2=-A (0 = none)
%   Y_disp      [m]      Y displacement during main motion in negative direction
%                        (0 = Y holds at Y_initial throughout)
%   vmax_X      [m/s]    X velocity limit (used when X_sym_amp or X_anti_amp > 0)
%   amax_X      [m/s^2]  X acceleration limit
%   vmax_Y      [m/s]    Y velocity limit (settle and/or main Y move)
%   amax_Y      [m/s^2]  Y acceleration limit
%   jerkTime    [s]      jerk time (jmax = amax / jerkTime)

% T1: Y sweep, conservative -- primary cy and mh excitation.
trajs(1).id         = 'T1_Y_sweep_conservative';
trajs(1).Y_initial  = 0.3;   % no settle needed
trajs(1).X_sym_amp  = 0;
trajs(1).X_anti_amp = 0;
trajs(1).Y_disp     = 0.6;   % Y: 0.3 -> -0.3 m
trajs(1).vmax_X     = 0;
trajs(1).amax_X     = 0;
trajs(1).vmax_Y     = 1.0;   % 50% hardware max
trajs(1).amax_Y     = 10.0;  % 20% hardware max
trajs(1).jerkTime   = 0.050;

% T2: X symmetric at Y=0.3 -- translational inertia/friction + mh M1 coupling active.
% M1[0,1] = -mh*Y = -3.03 kg*m (maximum coupling in operating range).
trajs(2).id         = 'T2_X_sym_Y030';
trajs(2).Y_initial  = 0.3;   % no settle needed
trajs(2).X_sym_amp  = 0.15;  % X1=X2=0 -> +150 mm
trajs(2).X_anti_amp = 0;
trajs(2).Y_disp     = 0;     % Y held at 0.3 m
trajs(2).vmax_X     = 1.5;   % 75% hardware max
trajs(2).amax_X     = 20.0;  % 67% hardware max
trajs(2).vmax_Y     = 1.0;
trajs(2).amax_Y     = 20.0;
trajs(2).jerkTime   = 0.030;

% T3: X symmetric at Y=0 -- identical motion to T2 but M1 coupling = 0.
% T2 vs T3 contrast isolates mh from total translational mass (m1+m2+mb).
trajs(3).id         = 'T3_X_sym_Y000';
trajs(3).Y_initial  = 0.0;   % Y settle: 0.3 -> 0.0 prepended automatically
trajs(3).X_sym_amp  = 0.15;
trajs(3).X_anti_amp = 0;
trajs(3).Y_disp     = 0;     % Y held at 0.0 m during X move
trajs(3).vmax_X     = 1.5;
trajs(3).amax_X     = 20.0;
trajs(3).vmax_Y     = 1.0;
trajs(3).amax_Y     = 20.0;
trajs(3).jerkTime   = 0.030;

% T4: X anti-symmetric (pure rotation) at Y=0.2 -- UNIQUE: only trajectory
% exciting kb1+kb2 (K[1,1]). Also: Jb+Jh, cb1+cb2, cg1-cg2.
% Low dynamics: kb_sum*q_rot = 3975*0.097 = 386 N.m restoring torque is large.
% X1-X2 = 70 mm -> q_rot = 70/725 = 0.097 rad (within safe differential limit).
trajs(4).id         = 'T4_X_antisym_Y020';
trajs(4).Y_initial  = 0.2;   % Y settle: 0.3 -> 0.2 prepended automatically
trajs(4).X_sym_amp  = 0;
trajs(4).X_anti_amp = 0.035; % X1=+35 mm, X2=-35 mm -> |X1-X2|=70 mm
trajs(4).Y_disp     = 0;     % Y held at 0.2 m
trajs(4).vmax_X     = 0.5;
trajs(4).amax_X     = 8.0;
trajs(4).vmax_Y     = 1.0;
trajs(4).amax_Y     = 20.0;
trajs(4).jerkTime   = 0.040;

% T5: X symmetric + Y sweep simultaneously -- traces the full M1 coupling
% (-mh*Y) as Y varies continuously while X accelerates.
% X move (~0.2 s) finishes before Y sweep (~0.5 s); X holds while Y continues.
trajs(5).id         = 'T5_X_sym_Y_sweep';
trajs(5).Y_initial  = 0.2;   % Y settle: 0.3 -> 0.2 prepended automatically
trajs(5).X_sym_amp  = 0.10;  % X1=X2=0 -> +100 mm
trajs(5).X_anti_amp = 0;
trajs(5).Y_disp     = 0.4;   % Y: 0.2 -> -0.2 m (simultaneous with X)
trajs(5).vmax_X     = 1.0;
trajs(5).amax_X     = 15.0;
trajs(5).vmax_Y     = 1.0;
trajs(5).amax_Y     = 20.0;
trajs(5).jerkTime   = 0.035;

% T6: Y sweep, hardware-maximum dynamics -- cy vs mh regime contrast with T1.
% cy scales with velocity; mh scales with acceleration.
% cy*v/(mh*a): T1 = 0.099, T6 = 0.040 -> T1+T6 together disambiguate the two.
trajs(6).id         = 'T6_Y_sweep_aggressive';
trajs(6).Y_initial  = 0.3;   % no settle needed
trajs(6).X_sym_amp  = 0;
trajs(6).X_anti_amp = 0;
trajs(6).Y_disp     = 0.6;   % Y: 0.3 -> -0.3 m
trajs(6).vmax_X     = 0;
trajs(6).amax_X     = 0;
trajs(6).vmax_Y     = 2.0;   % hardware max
trajs(6).amax_Y     = 50.0;  % hardware max
trajs(6).jerkTime   = 0.025;

% ------------------------------------------------------------------
% 4. Run all trajectories
% ------------------------------------------------------------------
mdl     = 'gantry_2025a';
out_dir = fullfile(fileparts(mfilename('fullpath')), '..', ...
                   'Matlab-output', 'parameter-recovery');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

n_hold = round(0.5 / ts);   % 0.5 s hold periods (10000 samples at 20 kHz)

for i = 1:numel(trajs)
    sp = trajs(i);
    fprintf('=== %d/%d  %s ===\n', i, numel(trajs), sp.id);

    [r, t] = make_ref(sp, n_hold, ts);
    validate_ref(r, sp.id);

    f = zeros(size(r));  % no feedforward forces

    fprintf('  Simulating %.2f s (%d samples) ...\n', t(end), length(t));
    sim(mdl, t(end));
    fprintf('  Simulation complete. Samples collected: %d\n', size(q1, 1));

    [t_sim, r_sim, u_q1, u_q, Y_trajectory, q_simscape] = ...
        reconstruct(q1, q, r, t, Cfb);

    report_traj(q1, Y_trajectory);

    out_path = fullfile(out_dir, [sp.id, '.mat']);
    save(out_path, 't_sim', 'fs', 'r_sim', 'u_q1', 'u_q', ...
                   'q1', 'q_simscape', 'Y_trajectory');
    fprintf('  Saved: %s\n\n', out_path);
end

fprintf('Done. %d trajectories exported to:\n  %s\n', numel(trajs), out_dir);

% ======================================================================
% Local functions
% ======================================================================

function [r, t] = make_ref(sp, n_hold, ts)
% Build stage-coordinate reference r (N x 3) = [X1, X2, Y] for one trajectory.
%
% Reference phases:
%   1. Pre-hold         -- n_hold samples at [0, 0, 0.3]  (Simulink IC)
%   2. Y settle move    -- only if sp.Y_initial ~= 0.3; Y moves in negative direction
%   3. Settle hold      -- n_hold samples at [0, 0, sp.Y_initial]
%   4. Main motion      -- X and/or Y move simultaneously; shorter is padded
%   5. Post-hold        -- n_hold samples at final position

    Y0 = 0.3;  % Simulink fixed initial condition -- do not change

    % Phase 1: pre-hold
    r     = repmat([0, 0, Y0], n_hold, 1);
    Y_now = Y0;

    % Phase 2-3: Y settle (if Y_initial differs from Y0)
    if abs(sp.Y_initial - Y0) > 1e-9
        pv_s   = setpoint_1d(abs(sp.Y_initial - Y0), sp.vmax_Y, sp.amax_Y, sp.jerkTime, ts);
        n_s    = length(pv_s);
        Y_prof = Y0 - pv_s;   % always negative direction (Y_initial <= Y0)
        r      = [r; [zeros(n_s, 2), Y_prof]];
        Y_now  = sp.Y_initial;
        r      = [r; repmat([0, 0, Y_now], n_hold, 1)];
    end

    % Phase 4: main motion (X and/or Y, simultaneous; shorter padded with hold)
    n_move_X = 0;   pv_X = [];
    n_move_Y = 0;   pv_Y = [];

    amp_X = max(sp.X_sym_amp, sp.X_anti_amp);
    if amp_X > 0
        pv_X     = setpoint_1d(amp_X, sp.vmax_X, sp.amax_X, sp.jerkTime, ts);
        n_move_X = length(pv_X);
    end
    if sp.Y_disp > 0
        pv_Y     = setpoint_1d(sp.Y_disp, sp.vmax_Y, sp.amax_Y, sp.jerkTime, ts);
        n_move_Y = length(pv_Y);
    end

    n_main = max(n_move_X, n_move_Y);
    if n_main > 0
        X1 = zeros(n_main, 1);
        X2 = zeros(n_main, 1);
        Y  = Y_now * ones(n_main, 1);

        if n_move_X > 0
            xp = [pv_X; pv_X(end) * ones(n_main - n_move_X, 1)];
            if sp.X_sym_amp > 0
                X1 = xp;   X2 = xp;    % symmetric: X1 = X2 = +A
            else
                X1 = xp;   X2 = -xp;   % anti-symmetric: X1 = +A, X2 = -A
            end
        end
        if n_move_Y > 0
            yp = [pv_Y; pv_Y(end) * ones(n_main - n_move_Y, 1)];
            Y  = Y_now - yp;            % Y moves in negative direction
        end
        r = [r; [X1, X2, Y]];
    end

    % Phase 5: post-hold
    r = [r; repmat(r(end, :), n_hold, 1)];

    N = size(r, 1);
    t = ts * (0:N-1)';
end

% ----------------------------------------------------------------------

function validate_ref(r, id)
% Assert all reference positions are within hardware limits (TELICA datasheet).
    X_LIM    = 0.375;   % ±375 mm  (750 mm total stroke / 2)
    Y_LIM    = 0.400;   % ±400 mm
    DIFF_LIM = 0.100;   % 100 mm max |X1-X2| differential

    assert(max(abs(r(:,1))) <= X_LIM, ...
           '%s: X1 exceeds +/-%.0f mm', id, X_LIM*1e3);
    assert(max(abs(r(:,2))) <= X_LIM, ...
           '%s: X2 exceeds +/-%.0f mm', id, X_LIM*1e3);
    assert(max(r(:,3))  <=  Y_LIM, ...
           '%s: Y exceeds +%.0f mm', id, Y_LIM*1e3);
    assert(min(r(:,3)) >= -Y_LIM, ...
           '%s: Y exceeds -%.0f mm', id, Y_LIM*1e3);
    assert(max(abs(r(:,1) - r(:,2))) <= DIFF_LIM, ...
           '%s: |X1-X2| exceeds %.0f mm differential limit', id, DIFF_LIM*1e3);

    fprintf('  Limits OK:  X1=[%+.0f %+.0f]  X2=[%+.0f %+.0f]  Y=[%+.0f %+.0f]  |X1-X2|_max=%.0f mm\n', ...
            min(r(:,1))*1e3, max(r(:,1))*1e3, ...
            min(r(:,2))*1e3, max(r(:,2))*1e3, ...
            min(r(:,3))*1e3, max(r(:,3))*1e3, ...
            max(abs(r(:,1) - r(:,2)))*1e3);
end

% ----------------------------------------------------------------------

function [t_sim, r_sim, u_q1, u_q, Y_trajectory, q_simscape] = ...
        reconstruct(q1, q, r, t, Cfb)
% Reconstruct applied forces from simulated output and reference.
% Handles variable-step Simulink output (N_sim may differ from length(t)).
    N_sim = size(q1, 1);
    if N_sim ~= length(t)
        t_sim = linspace(0, t(end), N_sim)';
        r_sim = interp1(t, r, t_sim);
    else
        t_sim = t;
        r_sim = r;
    end
    u_q1         = lsim(ss(Cfb), r_sim - q1, t_sim);
    u_q          = lsim(ss(Cfb), r_sim - q,  t_sim);
    Y_trajectory = q1(:, 3);
    q_simscape   = q;
end

% ----------------------------------------------------------------------

function report_traj(q1, Y_trajectory)
% Print axis range statistics after simulation.
    fprintf('  X1: [%+.3f, %+.3f] m\n', min(q1(:,1)), max(q1(:,1)));
    fprintf('  X2: [%+.3f, %+.3f] m\n', min(q1(:,2)), max(q1(:,2)));
    fprintf('  Y:  [%+.3f, %+.3f] m\n', min(Y_trajectory), max(Y_trajectory));
end

% ----------------------------------------------------------------------

function pv = setpoint_1d(d, vmax, amax, jerkTime, ts)
% Call thirdOrderSetpointETEL and return the position column only.
    pvajs = thirdOrderSetpointETEL(d, vmax, amax, amax / jerkTime, Inf, ts);
    pv    = pvajs(:, 1);
end
