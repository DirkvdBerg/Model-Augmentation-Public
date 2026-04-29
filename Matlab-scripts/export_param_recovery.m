% export_param_recovery.m
% -----------------------
% Unified parameter-recovery trajectory generation script.
% Replaces export_lpv_multi_traj.m (no multisine) and
% multisine_muli_traject.m (with multisine).
%
% Set USE_MULTISINE = true/false at the top. All trajectory definitions,
% physical parameters, controller design, and hardware validation are
% shared between both modes.
%
% Trajectories T1-T8:
%   T1  Y sweep, conservative         cy / mh (friction-dominant regime)
%   T2  X symmetric at Y=0.3          cg1+cg2, m_total, mh (M1 coupling active)
%   T3  X symmetric at Y=0.0          cg1+cg2, m1+m2+mb  (M1 coupling = 0)
%   T4  X anti-symmetric at Y=0.2     kb_sum, cb_sum, Jb+Jh (UNIQUE rotational)
%   T5  X symmetric + Y sweep         mh over full LPV Y range
%   T6  Y sweep, aggressive           mh / cy (inertia-dominant regime)
%   T7  X anti-symmetric + Y sweep    all 13 params + d simultaneously
%   T8  X sym + X anti + Y sweep      all 13 params + d, different operating point
%
% See docs/trajectory-design-param-recovery.md for full design rationale.
%
% Controller Cfb and frozen LTI G are designed per trajectory at Y_initial
% (D-039: operating point linearisation).
%
% Outputs saved to:
%   USE_MULTISINE=false: Matlab-output/parameter-recovery/<id>.mat
%   USE_MULTISINE=true:  Matlab-output/parameter-recovery-multisine/<id>.mat
%
% Variables saved per trajectory:
%   t_sim        (N x 1)  time vector [s]
%   fs           (1 x 1)  sample rate [Hz]
%   r_sim        (N x 3)  reference [X1, X2, Y] [m]
%   u_q1         (N x 3)  feedback force on CT LPV path [F_X1,F_X2,F_Y] [N]
%   u_q          (N x 3)  feedback force on Simscape path [N]
%   f_sim        (N x 3)  feedforward multisine force [N] (zeros if no multisine)
%   amp_max      (1 x 1)  max passing RMS amplitude [N] (NaN if no multisine)
%   q1           (N x 3)  CT quasi-LPV output [X1, X2, Y] [m]  -- PRIMARY
%   q_simscape   (N x 3)  Simscape output [X1, X2, Y] [m]
%   Y_trajectory (N x 1)  Y(t) = q1(:,3) [m]
%
% Does NOT modify any file in kamtin-fp-model/.
%
% Run from repo root:
%   cd('<path>/Baseline-LPV-Augmentation')
%   run('Matlab-scripts/export_param_recovery.m')

addpath(genpath(fullfile(pwd, 'kamtin-fp-model', '03 Simulink gantry')))

% ======================================================================
% USER FLAG
% ======================================================================
USE_MULTISINE = false;   % true  -> multisine feedforward, output to parameter-recovery-multisine/
                         % false -> zero feedforward,      output to parameter-recovery/

% ======================================================================
% 1. Physical parameters (identical to main.m lines 12-49)
% ======================================================================
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
Lh  = 0.25;    % Length of payload                   [m]  (unused in LFR model)
d   = 0.1;     % Distance cross-arm centre to payload CoM [m]
cc1 = 16.8;    % Coulomb friction X1 (Simscape only) [N]
cc2 = 18.35;   % Coulomb friction X2 (Simscape only) [N]
ccy = 11.6;    % Coulomb friction Y  (Simscape only) [N]

% ======================================================================
% 2. Constants shared across all trajectories
% ======================================================================
% C_damp and K are Y-independent. M_op, Cfb, G depend on Y_initial and
% are recomputed per trajectory inside the loop (D-039).

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
fbw = 100;     % feedback bandwidth [Hz] for ruleOfThumb controller

% ======================================================================
% 3. Trajectory definitions
% ======================================================================
% Fields:
%   id          string    output filename (no .mat)
%   Y_initial   [m]       Y at start of main motion. If < 0.3 a settle move
%                         is prepended automatically (Simulink IC is always 0.3).
%   X_sym_amp   [m]       symmetric X amplitude: X1=X2 move to +A (0 = none)
%   X_anti_amp  [m]       anti-symmetric X amplitude: X1=+A, X2=-A (0 = none)
%                         Both may be non-zero: X1 = sym+anti, X2 = sym-anti.
%   Y_disp      [m]       Y displacement during main motion, negative direction
%                         (0 = Y holds at Y_initial throughout)
%   vmax_X      [m/s]     X velocity limit
%   amax_X      [m/s^2]   X acceleration limit
%   vmax_Y      [m/s]     Y velocity limit (settle and/or main Y move)
%   amax_Y      [m/s^2]   Y acceleration limit
%   jerkTime    [s]       jerk time (jmax = amax / jerkTime)
%   ms_f_low    [Hz]      multisine lower frequency bound (used if USE_MULTISINE)
%   ms_f_high   [Hz]      maximum multisine upper bound across active modes
%   ms_modes    cellstr   active force modes:
%                         'common' -> F_X1 = F_X2
%                         'diff'   -> F_X1 = -F_X2
%                         'y'      -> F_Y
%
% Multisine design follows the system-identification lecture rules:
% periodic multisine, integer frequency lines, odd harmonics only for
% nonlinearity checks, Schroeder phases for low crest factor, enough
% frequency lines for PE order >= 13, and amplitude selected by response
% validation against ETEL constraints.

% T1: Y sweep, conservative.
% Regime: cy*v/(mh*a) = 10*1.0/(10.1*10) = 0.099 -- friction ~10% of inertia.
% Together with T6 (ratio 0.040), provides 2.5x regime contrast to break
% cy/mh collinearity. Both T1 and T6 are required.
% Garcia analog: Exp 1 (Y const velocity) + Exp 2 (Y const acceleration).
trajs(1).id         = 'T1_Y_sweep_conservative';
trajs(1).Y_initial  = 0.3;
trajs(1).X_sym_amp  = 0;
trajs(1).X_anti_amp = 0;
trajs(1).Y_disp     = 0.6;    % Y: 0.3 -> -0.3 m
trajs(1).vmax_X     = 0;
trajs(1).amax_X     = 0;
trajs(1).vmax_Y     = 1.0;    % 50% hardware max
trajs(1).amax_Y     = 10.0;   % 20% hardware max
trajs(1).jerkTime   = 0.050;
trajs(1).ms_f_low   = 1;
trajs(1).ms_f_high  = 20;     % Y-axis band: cy/mh separation
trajs(1).ms_modes   = {'y'};

% T2: X symmetric at Y=0.3.
% M1[0,1] = -mh*Y = -10.1*0.3 = -3.03 kg*m (maximum LPV coupling).
% Together with T3 (Y=0, coupling=0) isolates mh from m1+m2+mb.
% Garcia analog: Exp 3 + Exp 4.
trajs(2).id         = 'T2_X_sym_Y030';
trajs(2).Y_initial  = 0.3;
trajs(2).X_sym_amp  = 0.15;   % X1=X2=0 -> +150 mm
trajs(2).X_anti_amp = 0;
trajs(2).Y_disp     = 0;      % Y held at 0.3 m
trajs(2).vmax_X     = 1.5;    % 75% hardware max
trajs(2).amax_X     = 20.0;   % 67% hardware max
trajs(2).vmax_Y     = 1.0;
trajs(2).amax_Y     = 20.0;
trajs(2).jerkTime   = 0.030;
trajs(2).ms_f_low   = 1;
trajs(2).ms_f_high  = 100;
trajs(2).ms_modes   = {'common'};

% T3: X symmetric at Y=0.
% Identical motion to T2 but M1 coupling = -mh*0 = 0.
% T2 vs T3 contrast: mh appears in T2 but not T3 -> breaks mh/mass collinearity.
% Garcia analog: Exp 3 + Exp 4 (at different Y).
trajs(3).id         = 'T3_X_sym_Y000';
trajs(3).Y_initial  = 0.0;    % Y settle: 0.3 -> 0.0 prepended automatically
trajs(3).X_sym_amp  = 0.15;
trajs(3).X_anti_amp = 0;
trajs(3).Y_disp     = 0;      % Y held at 0.0 m during X move
trajs(3).vmax_X     = 1.5;
trajs(3).amax_X     = 20.0;
trajs(3).vmax_Y     = 1.0;
trajs(3).amax_Y     = 20.0;
trajs(3).jerkTime   = 0.030;
trajs(3).ms_f_low   = 1;
trajs(3).ms_f_high  = 100;
trajs(3).ms_modes   = {'common'};

% T4: X anti-symmetric (pure rotation) at Y=0.2.
% STRUCTURALLY IRREPLACEABLE: only trajectory exciting kb_sum, cb_sum, Jb+Jh.
% These parameters have zero gradient in every other trajectory.
% X1-X2 = 70 mm -> Theta = 70/725 = 0.097 rad (within DIFF_LIM = sin(0.1)*Lb = 72.4 mm).
% Low dynamics: kb_sum*Theta = 3975*0.097 = 386 Nm restoring torque.
% Rotational resonance ~5 Hz (omega = sqrt(kb_sum/J_eff) ~ 32 rad/s).
% ms_f_high=20: resonance-targeted to improve kb/J separation near resonance.
% Garcia analog: Exp 5 (static stiffness) + Exp 6 (const rot speed) + Exp 7 (rot inertia).
trajs(4).id         = 'T4_X_antisym_Y020';
trajs(4).Y_initial  = 0.2;    % Y settle: 0.3 -> 0.2 prepended automatically
trajs(4).X_sym_amp  = 0;
trajs(4).X_anti_amp = 0.035;  % X1=+35 mm, X2=-35 mm -> |X1-X2|=70 mm
trajs(4).Y_disp     = 0;      % Y held at 0.2 m
trajs(4).vmax_X     = 0.5;
trajs(4).amax_X     = 8.0;
trajs(4).vmax_Y     = 1.0;
trajs(4).amax_Y     = 20.0;
trajs(4).jerkTime   = 0.040;
trajs(4).ms_f_low   = 1;
trajs(4).ms_f_high  = 20;     % resonance-targeted around ~5 Hz rotational mode
trajs(4).ms_modes   = {'diff'};

% T5: X symmetric + Y sweep simultaneously.
% Traces full LPV coupling M1(Y) = -mh*Y as Y varies continuously.
% No Garcia analog: LPV-specific trajectory.
trajs(5).id         = 'T5_X_sym_Y_sweep';
trajs(5).Y_initial  = 0.2;    % Y settle: 0.3 -> 0.2 prepended automatically
trajs(5).X_sym_amp  = 0.10;   % X1=X2=0 -> +100 mm
trajs(5).X_anti_amp = 0;
trajs(5).Y_disp     = 0.4;    % Y: 0.2 -> -0.2 m (simultaneous with X)
trajs(5).vmax_X     = 1.0;
trajs(5).amax_X     = 15.0;
trajs(5).vmax_Y     = 1.0;
trajs(5).amax_Y     = 20.0;
trajs(5).jerkTime   = 0.035;
trajs(5).ms_f_low   = 1;
trajs(5).ms_f_high  = 100;
trajs(5).ms_modes   = {'common', 'y'};

% T6: Y sweep, hardware-maximum dynamics.
% Regime: cy*v/(mh*a) = 10*2.0/(10.1*50) = 0.040 -- inertia dominant.
% Together with T1 (ratio 0.099): 2.5x contrast breaks cy/mh collinearity.
% Garcia analog: Exp 1 + Exp 2.
trajs(6).id         = 'T6_Y_sweep_aggressive';
trajs(6).Y_initial  = 0.3;
trajs(6).X_sym_amp  = 0;
trajs(6).X_anti_amp = 0;
trajs(6).Y_disp     = 0.6;    % Y: 0.3 -> -0.3 m
trajs(6).vmax_X     = 0;
trajs(6).amax_X     = 0;
trajs(6).vmax_Y     = 2.0;    % hardware max
trajs(6).amax_Y     = 50.0;   % hardware max
trajs(6).jerkTime   = 0.025;
trajs(6).ms_f_low   = 1;
trajs(6).ms_f_high  = 20;     % Y-axis band: cy/mh separation
trajs(6).ms_modes   = {'y'};

% T7: X anti-symmetric + Y sweep simultaneously.
% NEW: first trajectory exciting rotational AND Y parameters simultaneously.
% All 13 trainable parameters + d have non-zero gradient in every window.
% mh*d*Yddot coupling to Theta makes d observable from position output.
% X1-X2 = 70 mm -> Theta = 0.097 rad (same constraint as T4, within DIFF_LIM).
% ms_f_high=20: resonance-targeted (same reasoning as T4).
% No Garcia analog.
trajs(7).id         = 'T7_X_antisym_Y_sweep';
trajs(7).Y_initial  = 0.3;    % no settle needed
trajs(7).X_sym_amp  = 0;
trajs(7).X_anti_amp = 0.035;  % X1=+35 mm, X2=-35 mm -> |X1-X2|=70 mm, Theta=0.097 rad
trajs(7).Y_disp     = 0.6;    % Y: 0.3 -> -0.3 m (simultaneous with X)
trajs(7).vmax_X     = 0.5;    % same constraint as T4 (rotation limit)
trajs(7).amax_X     = 8.0;
trajs(7).vmax_Y     = 1.5;
trajs(7).amax_Y     = 20.0;
trajs(7).jerkTime   = 0.040;
trajs(7).ms_f_low   = 1;
trajs(7).ms_f_high  = 20;     % resonance-targeted
trajs(7).ms_modes   = {'diff', 'y'};

% T8: X symmetric + X anti-symmetric simultaneously + Y sweep.
% NEW: second all-parameters trajectory at a different operating point from T7.
% Combined X: X1 = X_sym + X_anti = 100+20 = 120 mm
%             X2 = X_sym - X_anti = 100-20 =  80 mm
%             |X1-X2| = 2*X_anti_amp = 40 mm -> Theta = 40/725 = 0.055 rad (safe)
% X_sym excites translational parameters (cg1+cg2, m1+m2+mb).
% X_anti excites rotational parameters (kb, cb, Jb+Jh).
% Y sweep (0.2->-0.2 m) excites cy, mh, d -- different Y range than T7.
% Different operating point from T7: different Y range, smaller Theta, larger X_sym.
% No Garcia analog.
trajs(8).id         = 'T8_X_sym_anti_Y_sweep';
trajs(8).Y_initial  = 0.2;    % Y settle: 0.3 -> 0.2 prepended automatically
trajs(8).X_sym_amp  = 0.10;   % symmetric component: X1=X2 -> +100 mm
trajs(8).X_anti_amp = 0.020;  % anti-sym perturbation: X1=+20mm, X2=-20mm
                               % combined: X1->+120mm, X2->+80mm, |X1-X2|=40mm
trajs(8).Y_disp     = 0.4;    % Y: 0.2 -> -0.2 m (simultaneous)
trajs(8).vmax_X     = 1.0;
trajs(8).amax_X     = 12.0;
trajs(8).vmax_Y     = 1.5;
trajs(8).amax_Y     = 25.0;
trajs(8).jerkTime   = 0.035;
trajs(8).ms_f_low   = 1;
trajs(8).ms_f_high  = 100;    % broadband
trajs(8).ms_modes   = {'common', 'diff', 'y'};

% ======================================================================
% 4. Output directory
% ======================================================================
mdl = 'gantry_2025a';
if USE_MULTISINE
    out_subdir = 'parameter-recovery-multisine';
else
    out_subdir = 'parameter-recovery';
end
out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'Matlab-output', out_subdir);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

n_hold       = round(0.5 / ts);                         % 0.5 s hold = 10000 samples
amp_rms_grid = [1, 2, 5, 10, 20, 50, 100, 200];        % [N] RMS sweep (multisine only)
force_limits.peak = [2000, 2000, 1420];                % [N] TELICA peak force [FX1,FX2,FY]
force_limits.rms  = [916,  916,  656];                 % [N] TELICA continuous force [FX1,FX2,FY]

% ======================================================================
% 5. Run all trajectories
% ======================================================================
for i = 1:numel(trajs)
    sp = trajs(i);
    fprintf('=== %d/%d  %s ===\n', i, numel(trajs), sp.id);

    % -- Controller at this trajectory's operating point (D-039) -----------
    Y_op = sp.Y_initial;
    Y    = sp.Y_initial;   % Simulink workspace variable
    M_op = [m1+m2+mb+mh,             (m1-m2)*Lb/2 - mh*Y_op,                   0;
            (m1-m2)*Lb/2 - mh*Y_op,  Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y_op^2,  -mh*d;
            0,                        -mh*d,                                      mh];
    sys_logical        = getss(n, M_op, C_damp, K);
    StageCoordinatesSystem = P.' * sys_logical * P;
    Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
    for j = 1:3
        Cfb(j,j) = ruleOfThumb(fbw, StageCoordinatesSystem(j,j), ts);
    end
    G = c2d(StageCoordinatesSystem, ts, 'zoh');  % frozen LTI read by Simulink

    % -- Reference trajectory ----------------------------------------------
    [r, t] = make_ref(sp, n_hold, ts);
    if USE_MULTISINE
        [r, t] = pad_to_multisine_periods(r, ts, fs);
    end
    validate_ref(r, sp.id, Lb);

    % -- Feedforward force -------------------------------------------------
    if USE_MULTISINE
        % Amplitude sweep: find maximum RMS that keeps q1 and force demand within TELICA limits.
        amp_max = 0;
        for amp = amp_rms_grid
            f = generate_multisine(length(t), fs, sp, amp);
            sim(mdl, t(end));
            force_ok = validate_forces(q1, r, t, f, Cfb, force_limits);
            if validate_response(q1, fs, Lb) && force_ok
                amp_max = amp;
            else
                fprintf('  Amplitude %.0f N RMS exceeds TELICA limits — stopping sweep.\n', amp);
                break;
            end
        end

        if amp_max == 0
            warning('%s: no amplitude passed TELICA limits — skipping.', sp.id);
            continue;
        end
        fprintf('  amp_max = %.0f N RMS per channel\n', amp_max);

        % Final simulation at amp_max.
        f = generate_multisine(length(t), fs, sp, amp_max);
        fprintf('  Simulating %.2f s (%d samples) at amp_max ...\n', t(end), length(t));
        sim(mdl, t(end));
        fprintf('  Simulation complete. Samples: %d\n', size(q1, 1));
    else
        f       = zeros(length(t), 3);   % no feedforward
        amp_max = NaN;

        fprintf('  Simulating %.2f s (%d samples) ...\n', t(end), length(t));
        sim(mdl, t(end));
        fprintf('  Simulation complete. Samples: %d\n', size(q1, 1));
    end

    [t_sim, r_sim, u_q1, u_q, Y_trajectory, q_simscape, f_sim] = ...
        reconstruct(q1, q, r, t, f, Cfb);
    force_report = summarize_forces(u_q1, f_sim, force_limits);

    report_traj(q1, Y_trajectory, amp_max);
    report_forces(force_report);

    out_path = fullfile(out_dir, [sp.id, '.mat']);
    save(out_path, 't_sim', 'fs', 'r_sim', ...
                   'u_q1', 'u_q', 'f_sim', 'amp_max', ...
                   'q1', 'q_simscape', 'Y_trajectory', 'force_report');
    fprintf('  Saved: %s\n\n', out_path);
end

fprintf('Done. %d trajectories exported to:\n  %s\n', numel(trajs), out_dir);

% ======================================================================
% Local functions
% ======================================================================

function [r, t] = make_ref(sp, n_hold, ts)
% Build stage-coordinate reference r (N x 3) = [X1, X2, Y].
%
% Reference phases:
%   1. Pre-hold         -- n_hold samples at [0, 0, 0.3]  (Simulink IC)
%   2. Y settle move    -- only if sp.Y_initial ~= 0.3; Y moves negative
%   3. Settle hold      -- n_hold samples at [0, 0, sp.Y_initial]
%   4. Main motion      -- X and/or Y simultaneous; shorter padded with hold
%   5. Post-hold        -- n_hold samples at final position
%
% Supports combined X_sym_amp + X_anti_amp (T8):
%   X1 = sym_profile + anti_profile
%   X2 = sym_profile - anti_profile
% When only one is non-zero the result reduces to the pure symmetric or
% pure anti-symmetric case respectively.

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

    % Phase 4: main motion
    % Generate symmetric and anti-symmetric X profiles independently,
    % then combine. This handles the case where both are non-zero (T8).
    n_move_Y = 0;   pv_Y = [];

    % Symmetric component
    if sp.X_sym_amp > 0
        pv_sym  = setpoint_1d(sp.X_sym_amp,  sp.vmax_X, sp.amax_X, sp.jerkTime, ts);
        n_sym   = length(pv_sym);
    else
        pv_sym  = [];
        n_sym   = 0;
    end

    % Anti-symmetric component
    if sp.X_anti_amp > 0
        pv_anti = setpoint_1d(sp.X_anti_amp, sp.vmax_X, sp.amax_X, sp.jerkTime, ts);
        n_anti  = length(pv_anti);
    else
        pv_anti = [];
        n_anti  = 0;
    end

    n_move_X = max(n_sym, n_anti);

    % Y motion
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
            % Pad each component to n_move_X, then to n_main
            if n_sym > 0
                pv_sym  = [pv_sym;  pv_sym(end)  * ones(n_move_X - n_sym,  1)];
            else
                pv_sym  = zeros(n_move_X, 1);
            end
            if n_anti > 0
                pv_anti = [pv_anti; pv_anti(end) * ones(n_move_X - n_anti, 1)];
            else
                pv_anti = zeros(n_move_X, 1);
            end

            % Pad to n_main
            xp_sym  = [pv_sym;  pv_sym(end)  * ones(n_main - n_move_X, 1)];
            xp_anti = [pv_anti; pv_anti(end) * ones(n_main - n_move_X, 1)];

            X1 = xp_sym + xp_anti;   % symmetric + anti-symmetric
            X2 = xp_sym - xp_anti;   % symmetric - anti-symmetric
        end

        if n_move_Y > 0
            yp = [pv_Y; pv_Y(end) * ones(n_main - n_move_Y, 1)];
            Y  = Y_now - yp;          % Y moves in negative direction
        end

        r = [r; [X1, X2, Y]];
    end

    % Phase 5: post-hold
    r = [r; repmat(r(end, :), n_hold, 1)];

    N = size(r, 1);
    t = ts * (0:N-1)';
end

% ----------------------------------------------------------------------

function validate_ref(r, id, Lb)
% Assert all reference positions are within ETEL TELICA hardware limits.
% Sources:
%   X_LIM, Y_LIM : TELICA datasheet p.2/4 Dimensional Data:
%                  750 mm stroke (X), 800 mm stroke (Y).
%   DIFF_LIM     : Garcia (2013) eq.(1) + Section 2.3: Theta_max = 0.1 rad,
%                  |X1-X2| = sin(0.1) * Lb = 72.4 mm. More precise than 100 mm.
    X_LIM    = 0.375;            % +/-375 mm
    Y_LIM    = 0.400;            % +/-400 mm
    DIFF_LIM = sin(0.1) * Lb;   % 72.4 mm (yaw limit, Garcia 2013 eq.1)

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

    fprintf('  Ref OK: X1=[%+.0f %+.0f] X2=[%+.0f %+.0f] Y=[%+.0f %+.0f] |X1-X2|_max=%.1f mm\n', ...
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
    if isnan(amp_max)
        fprintf('  Multisine: disabled\n');
    else
        fprintf('  Multisine amp_max: %.1f N RMS per channel\n', amp_max);
    end
end

% ----------------------------------------------------------------------

function ok = validate_forces(q1, r, t, f, Cfb, force_limits)
% Check total commanded force against actuator peak and continuous limits.
% This catches unrealistic cases where the position response stays bounded
% but the controller or multisine asks for unavailable force/current.
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

    u_fb = lsim(ss(Cfb), r_sim - q1, t_sim);
    rep  = summarize_forces(u_fb, f_sim, force_limits);
    ok   = rep.ok;

    if ~ok
        fprintf('  Force check failed:\n');
        fprintf('    peak total=[%.0f %.0f %.0f] N, peak limits=[%.0f %.0f %.0f] N\n', ...
                rep.max_total, rep.peak_limits);
        fprintf('    RMS total= [%.0f %.0f %.0f] N, RMS limits= [%.0f %.0f %.0f] N\n', ...
                rep.rms_total, rep.rms_limits);
    end
end

% ----------------------------------------------------------------------

function rep = summarize_forces(u_feedback, f_feedforward, force_limits)
% Summarise peak and RMS force demand for exported metadata and reporting.
    u_total = u_feedback + f_feedforward;

    rep.peak_limits     = force_limits.peak;
    rep.rms_limits      = force_limits.rms;
    rep.max_feedforward = max(abs(f_feedforward), [], 1);
    rep.max_feedback    = max(abs(u_feedback), [], 1);
    rep.max_total       = max(abs(u_total), [], 1);
    rep.rms_feedforward = sqrt(mean(f_feedforward.^2, 1));
    rep.rms_feedback    = sqrt(mean(u_feedback.^2, 1));
    rep.rms_total       = sqrt(mean(u_total.^2, 1));
    rep.peak_ratio_total = rep.max_total ./ force_limits.peak;
    rep.rms_ratio_total  = rep.rms_total ./ force_limits.rms;
    rep.ok_peak          = all(rep.max_total <= force_limits.peak);
    rep.ok_rms           = all(rep.rms_total <= force_limits.rms);
    rep.ok               = rep.ok_peak && rep.ok_rms;
end

% ----------------------------------------------------------------------

function report_forces(rep)
% Print force demand after the final simulation.
    fprintf('  Force peaks [FX1 FX2 FY] N:\n');
    fprintf('    feedforward: [%7.1f %7.1f %7.1f]\n', rep.max_feedforward);
    fprintf('    feedback:    [%7.1f %7.1f %7.1f]\n', rep.max_feedback);
    fprintf('    total:       [%7.1f %7.1f %7.1f]  limits=[%.0f %.0f %.0f]\n', ...
            rep.max_total, rep.peak_limits);
    fprintf('  Force RMS [FX1 FX2 FY] N:\n');
    fprintf('    feedforward: [%7.1f %7.1f %7.1f]\n', rep.rms_feedforward);
    fprintf('    feedback:    [%7.1f %7.1f %7.1f]\n', rep.rms_feedback);
    fprintf('    total:       [%7.1f %7.1f %7.1f]  limits=[%.0f %.0f %.0f]\n', ...
            rep.rms_total, rep.rms_limits);
end

% ----------------------------------------------------------------------

function pv = setpoint_1d(dist, vmax, amax, jerkTime, ts)
% Call thirdOrderSetpointETEL and return the position column only.
    pvajs = thirdOrderSetpointETEL(dist, vmax, amax, amax / jerkTime, Inf, ts);
    pv    = pvajs(:, 1);
end

% ----------------------------------------------------------------------

function [r_pad, t_pad] = pad_to_multisine_periods(r, ts, fs)
% Pad the final hold so the multisine contains an integer number of 1 s
% periods. This is the lecture leakage rule in executable form: the record
% length is an integer multiple of the multisine period.
    N_period = round(fs);                  % 1 s period -> delta_f = 1 Hz
    N        = size(r, 1);
    n_period = max(2, ceil(N / N_period)); % at least 2 periods
    N_target = n_period * N_period;
    n_pad    = N_target - N;

    if n_pad > 0
        r_pad = [r; repmat(r(end, :), n_pad, 1)];
    else
        r_pad = r;
    end
    t_pad = ts * (0:size(r_pad, 1)-1)';
end

% ----------------------------------------------------------------------

function f = generate_multisine(N, fs, sp, amp_rms)
% Generate trajectory-specific force multisines [F_X1, F_X2, F_Y].
%
% Lecture checks implemented here:
%   - periodic record, N is an integer number of 1 s periods
%   - harmonic frequency grid with delta_f = 1 Hz
%   - odd harmonics only, so even-order nonlinear distortion can be observed
%     on unexcited even lines
%   - at least 7 excited lines per active mode -> PE order 2F >= 14 > 13
%   - Schroeder phases with deterministic time shifts for low crest factor
%   - trajectory-specific physical modes: common, differential, and/or Y
%   - final per-actuator RMS normalisation before the amplitude sweep
    N_period = round(fs);
    assert(mod(N, N_period) == 0, ...
           '%s: N=%d must be a multiple of N_period=%d for leakage-free multisine', ...
           sp.id, N, N_period);

    f = zeros(N, 3);
    for m = 1:numel(sp.ms_modes)
        mode = sp.ms_modes{m};
        [f_low, f_high] = mode_band(sp, mode);
        sig = multisine_schroeder_periodic(N, N_period, fs, f_low, f_high, m);

        switch mode
            case 'common'
                f(:, 1) = f(:, 1) + sig;
                f(:, 2) = f(:, 2) + sig;
            case 'diff'
                f(:, 1) = f(:, 1) + sig;
                f(:, 2) = f(:, 2) - sig;
            case 'y'
                f(:, 3) = f(:, 3) + sig;
            otherwise
                error('%s: unknown multisine mode "%s"', sp.id, mode);
        end
    end

    for ch = 1:3
        ch_rms = rms(f(:, ch));
        if ch_rms > 0
            f(:, ch) = f(:, ch) * (amp_rms / ch_rms);
        end
    end
end

% ----------------------------------------------------------------------

function [f_low, f_high] = mode_band(sp, mode)
% Frequency bands follow the lecture-based design notes:
% common X: broadband to controller bandwidth,
% differential X: rotational resonance band,
% Y: low/mid band for cy/mh separation.
    f_low = sp.ms_f_low;
    switch mode
        case 'common'
            f_high = min(100, sp.ms_f_high);
        case 'diff'
            f_high = min(20, sp.ms_f_high);
        case 'y'
            f_high = min(20, sp.ms_f_high);
        otherwise
            error('%s: unknown mode "%s"', sp.id, mode);
    end
end

% ----------------------------------------------------------------------

function sig = multisine_schroeder_periodic(N, N_period, fs, f_low, f_high, seed)
% One-second Schroeder-phase odd-harmonic multisine tiled over N samples.
    f0 = fs / N_period;
    k0 = max(1, ceil(f_low / f0));
    k1 = floor(f_high / f0);
    k  = k0:k1;
    k  = k(mod(k, 2) == 1);       % odd-only harmonic lines
    F  = numel(k);

    if F < 7
        error('Multisine band %.2f-%.2f Hz gives only %d odd lines; need >=7 for 13 parameters.', ...
              f_low, f_high, F);
    end

    idx = 1:F;
    freqs = k * f0;
    phi = -idx .* (idx - 1) * pi / F;      % Schroeder phase convention
    phi = phi + 2*pi*freqs*(seed - 1)/(7*f_high); % deterministic mode shift

    t_period = (0:N_period-1)' / fs;
    one_period = sum(cos(2*pi*t_period*freqs + phi), 2);
    one_period = one_period / rms(one_period);

    n_tile = N / N_period;
    sig = repmat(one_period, n_tile, 1);
end

% ----------------------------------------------------------------------

function ok = validate_response(q1, fs, Lb)
% Check actual simulated response q1 against ETEL TELICA hardware limits.
% Operates on q1 (not reference r) -- closed loop shapes the actual response.
% Sources:
%   X_LIM, Y_LIM  : TELICA datasheet p.2/4 Dimensional Data: 750 mm (X), 800 mm (Y).
%   DIFF_LIM      : Garcia (2013) eq.(1) + Section 2.3: Theta_max = 0.1 rad.
%   VEL_LIM       : TELICA datasheet p.2/4 Dynamic Performance: 2 m/s (X and Y).
%   ACC_LIM_X     : TELICA datasheet p.2/4 Dynamic Performance: 30 m/s^2 (X axes).
%   ACC_LIM_Y     : TELICA datasheet p.2/4 Dynamic Performance: 50 m/s^2 (Y axis).
    X_LIM     = 0.375;
    Y_LIM     = 0.400;
    DIFF_LIM  = sin(0.1) * Lb;   % 72.4 mm
    VEL_LIM   = 2.0;             % m/s
    ACC_LIM_X = 30.0;            % m/s^2
    ACC_LIM_Y = 50.0;            % m/s^2

    vel = diff(q1) * fs;         % (N-1 x 3)
    acc = diff(vel) * fs;        % (N-2 x 3)

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
