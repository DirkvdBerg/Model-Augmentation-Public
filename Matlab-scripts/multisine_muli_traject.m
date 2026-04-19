% multisine_muli_traject.m
% ------------------------
% Generate and export 6 parameter-recovery training trajectories with
% multisine excitation injected at the feedforward (actuator force) slot.
%
% Each trajectory defines a nominal operating point (reference positions).
% A Schroeder-phase multisine f_multisine [F_X1, F_X2, F_Y] is added on
% top at the feedforward slot after the feedback controller:
%
%   u_total = Cfb*(r - q1) + f_multisine
%
% The multisine amplitude is swept from low to high. The maximum amplitude
% that keeps the actual simulated response q1 within ETEL hardware limits
% (position, velocity, acceleration) is selected per trajectory.
%
% Controller Cfb and frozen LTI G are designed at Y = sp.Y_initial for
% each trajectory (D-039).
%
% Outputs saved to Matlab-output/parameter-recovery-multisine/<id>.mat:
%   t_sim        (N x 1)  time vector [s]
%   fs           (1 x 1)  sample rate [Hz]
%   r_sim        (N x 3)  reference [X1, X2, Y] stage coords [m]
%   u_q1         (N x 3)  feedback force Cfb*(r-q1) [F_X1,F_X2,F_Y] [N]
%   u_q          (N x 3)  feedback force on Simscape path [N]
%   f_sim        (N x 3)  multisine feedforward force [F_X1,F_X2,F_Y] [N]
%   amp_max      (1 x 1)  maximum passing RMS amplitude [N]
%   q1           (N x 3)  CT quasi-LPV output [X1, X2, Y] [m]  -- PRIMARY
%   q_simscape   (N x 3)  Simscape output [X1, X2, Y] [m]
%   Y_trajectory (N x 1)  Y(t) = q1(:,3) [m]
%
% Total actuator input: u_total = u_q1 + f_sim
%
% Does NOT modify any file in kamtin-fp-model/.
%
% Run from repo root:
%   cd('<path>/Baseline-LPV-Augmentation')
%   run('Matlab-scripts/multisine_muli_traject.m')

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
% 2. Constants shared across all trajectories
% ------------------------------------------------------------------
% C_damp and K are Y-independent (constant). M_op, Cfb, G depend on
% Y_initial and are computed per trajectory inside the loop (D-039).

C_damp = [cg1+cg2,         (cg1-cg2)*Lb/2,             0;
          (cg1-cg2)*Lb/2,  cb1+cb2+(cg1+cg2)*Lb^2/4,   0;
          0,                0,                           cy];

K = [0,  0,        0;
     0,  kb1+kb2,  0;
     0,  0,        0];

n   = 3;
P   = [1, 1, 0; Lb/2, -Lb/2, 0; 0, 0, 1];
fs  = 20e3;
ts  = 1 / fs;
fbw = 100;

% Simulink workspace variable Y must be set before each sim() call.
% It is overwritten per trajectory below.

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
                   'Matlab-output', 'parameter-recovery-multisine');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

n_hold         = round(0.5 / ts);
amp_rms_grid   = [1, 2, 5, 10, 20, 50, 100, 200];  % [N] RMS per channel

for i = 1:numel(trajs)
    sp = trajs(i);
    fprintf('=== %d/%d  %s ===\n', i, numel(trajs), sp.id);

    % -- Controller at this trajectory's operating point (D-039) -----------
    Y     = sp.Y_initial;   % Simulink workspace variable
    Y_op  = sp.Y_initial;
    M_op  = [m1+m2+mb+mh,              (m1-m2)*Lb/2 - mh*Y_op,                        0;
             (m1-m2)*Lb/2 - mh*Y_op,   Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y_op^2,   -mh*d;
             0,                         -mh*d,                                          mh];
    sys_logical            = getss(n, M_op, C_damp, K);
    StageCoordinatesSystem = P.' * sys_logical * P;
    Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
    for j = 1:3
        Cfb(j,j) = ruleOfThumb(fbw, StageCoordinatesSystem(j,j), ts);
    end
    G = c2d(StageCoordinatesSystem, ts, 'zoh');

    % -- Reference trajectory ----------------------------------------------
    [r, t] = make_ref(sp, n_hold, ts);
    validate_ref(r, sp.id);

    % -- Amplitude sweep: find maximum amp that stays within ETEL limits --
    amp_max = 0;
    for amp = amp_rms_grid
        f = generate_multisine_3ch(length(t), fs, 1, 200, amp);
        sim(mdl, t(end));
        if validate_response(q1, fs)
            amp_max = amp;
        else
            fprintf('  Amplitude %.0f N RMS exceeds ETEL limits — stopping sweep.\n', amp);
            break;
        end
    end

    if amp_max == 0
        warning('%s: no amplitude passed ETEL filter — skipping.', sp.id);
        continue;
    end
    fprintf('  amp_max = %.0f N RMS per channel\n', amp_max);

    % -- Final simulation at amp_max ---------------------------------------
    f = generate_multisine_3ch(length(t), fs, 1, 200, amp_max);
    fprintf('  Simulating %.2f s (%d samples) ...\n', t(end), length(t));
    sim(mdl, t(end));
    fprintf('  Simulation complete. Samples: %d\n', size(q1, 1));

    [t_sim, r_sim, u_q1, u_q, Y_trajectory, q_simscape, f_sim] = ...
        reconstruct(q1, q, r, t, f, Cfb);

    report_traj(q1, Y_trajectory, amp_max);

    out_path = fullfile(out_dir, [sp.id, '.mat']);
    save(out_path, 't_sim', 'fs', 'r_sim', ...
                   'u_q1', 'u_q', 'f_sim', 'amp_max', ...
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
% Assert all reference positions are within hardware limits.
% Sources:
%   X_LIM, Y_LIM : TELICA datasheet (telica-xyz-0750-0800-data.pdf) p.2/4,
%                  Dimensional Data: total stroke 750 mm (X), 800 mm (Y).
%   DIFF_LIM     : Garcia (2013) eq.(1) + Section 2.3: Theta_max = 0.1 rad,
%                  |X1-X2| = sin(Theta_max) * Lb = sin(0.1) * 0.725 m.
    Lb       = 0.725;            % cross-arm length [m] (main.m)
    X_LIM    = 0.375;            % ±375 mm  (750 mm total stroke / 2)
    Y_LIM    = 0.400;            % ±400 mm  (800 mm total stroke / 2)
    DIFF_LIM = sin(0.1) * Lb;   % 72.4 mm  (yaw limit 0.1 rad, Garcia 2013)

    assert(max(abs(r(:,1))) <= X_LIM, ...
           '%s: X1 exceeds +/-%.0f mm', id, X_LIM*1e3);
    assert(max(abs(r(:,2))) <= X_LIM, ...
           '%s: X2 exceeds +/-%.0f mm', id, X_LIM*1e3);
    assert(max(r(:,3))  <=  Y_LIM, ...
           '%s: Y exceeds +%.0f mm', id, Y_LIM*1e3);
    assert(min(r(:,3)) >= -Y_LIM, ...
           '%s: Y exceeds -%.0f mm', id, Y_LIM*1e3);
    assert(max(abs(r(:,1) - r(:,2))) <= DIFF_LIM, ...
           '%s: |X1-X2| exceeds %.1f mm yaw limit (0.1 rad, Garcia 2013)', ...
           id, DIFF_LIM*1e3);

    fprintf('  Limits OK:  X1=[%+.0f %+.0f]  X2=[%+.0f %+.0f]  Y=[%+.0f %+.0f]  |X1-X2|_max=%.1f mm\n', ...
            min(r(:,1))*1e3, max(r(:,1))*1e3, ...
            min(r(:,2))*1e3, max(r(:,2))*1e3, ...
            min(r(:,3))*1e3, max(r(:,3))*1e3, ...
            max(abs(r(:,1) - r(:,2)))*1e3);
end

% ----------------------------------------------------------------------

function [t_sim, r_sim, u_q1, u_q, Y_trajectory, q_simscape, f_sim] = ...
        reconstruct(q1, q, r, t, f, Cfb)
% Reconstruct applied forces from simulated output and reference.
% Handles variable-step Simulink output (N_sim may differ from length(t)).
    N_sim = size(q1, 1);
    if N_sim ~= length(t)
        t_sim = linspace(0, t(end), N_sim)';
        r_sim = interp1(t, r, t_sim);
        f_sim = interp1(t, f, t_sim);
    else
        t_sim = t;
        r_sim = r;
        f_sim = f;
    end
    u_q1         = lsim(ss(Cfb), r_sim - q1, t_sim);
    u_q          = lsim(ss(Cfb), r_sim - q,  t_sim);
    Y_trajectory = q1(:, 3);
    q_simscape   = q;
end

% ----------------------------------------------------------------------

function report_traj(q1, Y_trajectory, amp_max)
% Print axis range and multisine amplitude after simulation.
    fprintf('  X1: [%+.3f, %+.3f] m\n', min(q1(:,1)), max(q1(:,1)));
    fprintf('  X2: [%+.3f, %+.3f] m\n', min(q1(:,2)), max(q1(:,2)));
    fprintf('  Y:  [%+.3f, %+.3f] m\n', min(Y_trajectory), max(Y_trajectory));
    fprintf('  Multisine amp_max: %.1f N RMS per channel\n', amp_max);
end

% ----------------------------------------------------------------------

function pv = setpoint_1d(d, vmax, amax, jerkTime, ts)
% Call thirdOrderSetpointETEL and return the position column only.
    pvajs = thirdOrderSetpointETEL(d, vmax, amax, amax / jerkTime, Inf, ts);
    pv    = pvajs(:, 1);
end

% ----------------------------------------------------------------------

function f = generate_multisine_3ch(N_traj, fs, f_low, f_high, amp_rms)
% Generate 3 independent Schroeder-phase multisines tiled to trajectory length.
% Each channel gets a different random phase offset (seed = channel index).
    N_period = round(fs);          % 1 s period at fs Hz → Δf = 1 Hz, zero leakage
    f = zeros(N_traj, 3);
    for ch = 1:3
        one_period  = multisine_schroeder(N_period, fs, f_low, f_high, amp_rms, ch);
        n_tile      = ceil(N_traj / N_period);
        tiled       = repmat(one_period, n_tile, 1);
        f(:, ch)    = tiled(1:N_traj);
    end
end

% ----------------------------------------------------------------------

function sig = multisine_schroeder(N, fs, f_low, f_high, amp_rms, seed)
% Schroeder-phase multisine: minimum crest factor (CF ≈ 1.58) for given RMS.
% Frequencies are integer Hz lines → exact periodicity, zero leakage.
    rng(seed);
    freq_lines = f_low : (fs/N) : f_high;     % integer Hz steps (Δf = fs/N = 1 Hz)
    F          = length(freq_lines);
    idx        = 1:F;
    phi        = -idx .* (idx-1) * pi / F;    % Schroeder phases
    phi        = phi + 2*pi*rand(1, F);        % independent offset per channel
    t          = (0:N-1)' / fs;
    sig        = sum(cos(2*pi*freq_lines .* t + phi), 2);
    sig        = amp_rms * sig / rms(sig);     % normalise to target RMS [N]
end

% ----------------------------------------------------------------------

function ok = validate_response(q1, fs)
% Check actual simulated response against ETEL TELICA hardware limits.
% Operates on q1 (not reference r) — closed loop shapes the response.
% Sources:
%   X_LIM, Y_LIM  : TELICA datasheet p.2/4, Dimensional Data: 750 mm (X), 800 mm (Y).
%   DIFF_LIM      : Garcia (2013) eq.(1) + Section 2.3: Theta_max = 0.1 rad.
%   VEL_LIM       : TELICA datasheet p.2/4, Dynamic Performance: 2 m/s (X and Y).
%   ACC_LIM_X     : TELICA datasheet p.2/4, Dynamic Performance: 30 m/s² (X axes).
%   ACC_LIM_Y     : TELICA datasheet p.2/4, Dynamic Performance: 50 m/s² (Y axis).
    Lb        = 0.725;          % cross-arm length [m] (main.m)
    X_LIM     = 0.375;          % ±375 mm
    Y_LIM     = 0.400;          % ±400 mm
    DIFF_LIM  = sin(0.1) * Lb;  % 72.4 mm  (yaw limit 0.1 rad, Garcia 2013)
    VEL_LIM   = 2.0;            % m/s  X and Y (TELICA p.2/4)
    ACC_LIM_X = 30.0;           % m/s² X1, X2 axes (TELICA p.2/4)
    ACC_LIM_Y = 50.0;           % m/s² Y axis      (TELICA p.2/4)

    vel = diff(q1) * fs;        % (N-1 x 3)
    acc = diff(vel) * fs;       % (N-2 x 3)

    ok =    max(abs(q1(:,1)))          <= X_LIM     ...
         && max(abs(q1(:,2)))          <= X_LIM     ...
         && max(abs(q1(:,3)))          <= Y_LIM     ...
         && max(abs(q1(:,1)-q1(:,2))) <= DIFF_LIM  ...
         && max(abs(vel(:,1)))         <= VEL_LIM   ...
         && max(abs(vel(:,2)))         <= VEL_LIM   ...
         && max(abs(vel(:,3)))         <= VEL_LIM   ...
         && max(abs(acc(:,1)))         <= ACC_LIM_X ...
         && max(abs(acc(:,2)))         <= ACC_LIM_X ...
         && max(abs(acc(:,3)))         <= ACC_LIM_Y;
end
