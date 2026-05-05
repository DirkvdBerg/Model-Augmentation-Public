% export_param_recovery_multisine.m
% ---------------------------------
% Parameter-recovery trajectory generation with post-controller force multisine.
%
% The reference stays clean (r = r_traj). The Schroeder-phase multisine is
% injected as feedforward force f_sim after the controller, so the plant
% receives total input u_total = u_q1 + f_sim. Best injection bands were
% determined by a diagnostic scan (multisine-analysis.md, 2026-05-05):
%   common → 50-100 Hz @ 400 N RMS
%   diff   → 100-200 Hz @ 400 N RMS
%   y      → 50-100 Hz @ 400 N RMS
%
% DIAGNOSTIC_ONLY = true : re-run band/amplitude scan, no .mat export.
% DIAGNOSTIC_ONLY = false: export using FORCE_BEST_BAND / FORCE_BEST_AMP_N.
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
% Variables saved per trajectory:
%   t_sim        (N x 1)  time vector [s]
%   fs           (1 x 1)  sample rate [Hz]
%   r_sim        (N x 3)  clean reference r_traj [X1, X2, Y] [m]
%   f_sim        (N x 3)  feedforward force multisine [N]   -- plant input component
%   u_q1         (N x 3)  feedback force = Cfb*(r_sim - q1) [N]
%   u_total      (N x 3)  total plant input = u_q1 + f_sim [N]  -- use for ID
%   q1           (N x 3)  plant output [X1, X2, Y] [m]          -- use for ID
%   Y_trajectory (N x 1)  Y(t) = q1(:,3) [m]
%   force_report (struct) force demand summary
%
% Does NOT modify any file in kamtin-fp-model/.
%
% Run from repo root:
%   cd('<path>/Baseline-LPV-Augmentation')
%   run('Matlab-scripts/export_param_recovery_multisine.m')

addpath(genpath(fullfile(pwd, 'kamtin-fp-model', '03 Simulink gantry')))

% ======================================================================
% USER FLAGS
% ======================================================================
DIAGNOSTIC_ONLY = true;  % true -> scan + print only; false -> scan then export
TRAJ_SUBSET     = 7:8;    % which trajectories to run

% Candidate force bands/amplitudes. The scan always runs first and builds
% f_sim_committed (cumulative across modes). Export uses that result directly —
% no hard-coded best params needed.
FORCE_DIAG_BANDS_HZ  = [ 5,20; 20,50; 50,100; 100,200; 200,400];
FORCE_DIAG_AMP_RMS_N = [10, 25, 50, 100, 200, 400, 800];

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
Lh  = 0.25;    % Length of payload                   [m]
d   = 0.1;     % Distance cross-arm centre to payload CoM [m]
cc1 = 16.8;    % Coulomb friction X1 (Simscape only) [N]
cc2 = 18.35;   % Coulomb friction X2 (Simscape only) [N]
ccy = 11.6;    % Coulomb friction Y  (Simscape only) [N]

% ======================================================================
% 2. Constants shared across all trajectories
% ======================================================================
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
% ms_modes for reference injection have the same spatial meaning as before:
%   'common' -> r_X1 = r_X2 = multisine  (symmetric X position perturbation)
%   'diff'   -> r_X1 = -r_X2 = multisine (anti-symmetric X, pure rotation)
%   'y'      -> r_Y = multisine           (Y position perturbation)
% ms_f_low/ms_f_high define the excited frequency band [Hz].

% T1: Y sweep, conservative.
trajs(1).id         = 'T1_Y_sweep_conservative';
trajs(1).Y_initial  = 0.3;
trajs(1).X_sym_amp  = 0;
trajs(1).X_anti_amp = 0;
trajs(1).Y_disp     = 0.6;
trajs(1).vmax_X     = 0;
trajs(1).amax_X     = 0;
trajs(1).vmax_Y     = 1.0;
trajs(1).amax_Y     = 10.0;
trajs(1).jerkTime   = 0.050;
trajs(1).ms_f_low   = 1;
trajs(1).ms_f_high  = 20;
trajs(1).ms_modes   = {'y'};

% T2: X symmetric at Y=0.3.
trajs(2).id         = 'T2_X_sym_Y030';
trajs(2).Y_initial  = 0.3;
trajs(2).X_sym_amp  = 0.15;
trajs(2).X_anti_amp = 0;
trajs(2).Y_disp     = 0;
trajs(2).vmax_X     = 1.5;
trajs(2).amax_X     = 20.0;
trajs(2).vmax_Y     = 1.0;
trajs(2).amax_Y     = 20.0;
trajs(2).jerkTime   = 0.030;
trajs(2).ms_f_low   = 1;
trajs(2).ms_f_high  = 100;
trajs(2).ms_modes   = {'common'};

% T3: X symmetric at Y=0.
trajs(3).id         = 'T3_X_sym_Y000';
trajs(3).Y_initial  = 0.0;
trajs(3).X_sym_amp  = 0.15;
trajs(3).X_anti_amp = 0;
trajs(3).Y_disp     = 0;
trajs(3).vmax_X     = 1.5;
trajs(3).amax_X     = 20.0;
trajs(3).vmax_Y     = 1.0;
trajs(3).amax_Y     = 20.0;
trajs(3).jerkTime   = 0.030;
trajs(3).ms_f_low   = 1;
trajs(3).ms_f_high  = 100;
trajs(3).ms_modes   = {'common'};

% T4: X anti-symmetric (pure rotation) at Y=0.2.
% NOTE: X_anti_amp=30 mm -> |X1-X2|=60 mm, leaving 12.4 mm yaw headroom
% for the diff multisine (up to ~4 mm RMS before hitting DIFF_LIM=72.4 mm).
trajs(4).id         = 'T4_X_antisym_Y020';
trajs(4).Y_initial  = 0.2;
trajs(4).X_sym_amp  = 0;
trajs(4).X_anti_amp = 0.030;
trajs(4).Y_disp     = 0;
trajs(4).vmax_X     = 0.5;
trajs(4).amax_X     = 8.0;
trajs(4).vmax_Y     = 1.0;
trajs(4).amax_Y     = 20.0;
trajs(4).jerkTime   = 0.040;
trajs(4).ms_f_low   = 1;
trajs(4).ms_f_high  = 20;
trajs(4).ms_modes   = {'diff'};

% T5: X symmetric + Y sweep simultaneously.
trajs(5).id         = 'T5_X_sym_Y_sweep';
trajs(5).Y_initial  = 0.2;
trajs(5).X_sym_amp  = 0.10;
trajs(5).X_anti_amp = 0;
trajs(5).Y_disp     = 0.4;
trajs(5).vmax_X     = 1.0;
trajs(5).amax_X     = 15.0;
trajs(5).vmax_Y     = 1.0;
trajs(5).amax_Y     = 20.0;
trajs(5).jerkTime   = 0.035;
trajs(5).ms_f_low   = 1;
trajs(5).ms_f_high  = 100;
trajs(5).ms_modes   = {'common', 'y'};

% T6: Y sweep, hardware-maximum dynamics.
trajs(6).id         = 'T6_Y_sweep_aggressive';
trajs(6).Y_initial  = 0.3;
trajs(6).X_sym_amp  = 0;
trajs(6).X_anti_amp = 0;
trajs(6).Y_disp     = 0.6;
trajs(6).vmax_X     = 0;
trajs(6).amax_X     = 0;
trajs(6).vmax_Y     = 1.80;
trajs(6).amax_Y     = 42.0;
trajs(6).jerkTime   = 0.025;
trajs(6).ms_f_low   = 1;
trajs(6).ms_f_high  = 20;
trajs(6).ms_modes   = {'y'};

% T7: X anti-symmetric + Y sweep simultaneously.
% NOTE: X_anti_amp=30 mm -> |X1-X2|=60 mm, same headroom as T4 (12.4 mm
% for diff multisine). DIFF_LIM applies to r_total = r_traj + r_ms.
trajs(7).id         = 'T7_X_antisym_Y_sweep';
trajs(7).Y_initial  = 0.3;
trajs(7).X_sym_amp  = 0;
trajs(7).X_anti_amp = 0.030;
trajs(7).Y_disp     = 0.6;
trajs(7).vmax_X     = 0.5;
trajs(7).amax_X     = 8.0;
trajs(7).vmax_Y     = 1.5;
trajs(7).amax_Y     = 20.0;
trajs(7).jerkTime   = 0.040;
trajs(7).ms_f_low   = 1;
trajs(7).ms_f_high  = 20;
trajs(7).ms_modes   = {'y', 'diff'};   % y first: lower survival needs full headroom; diff self-limits

% T8: X symmetric + X anti-symmetric simultaneously + Y sweep.
trajs(8).id         = 'T8_X_sym_anti_Y_sweep';
trajs(8).Y_initial  = 0.2;
trajs(8).X_sym_amp  = 0.10;
trajs(8).X_anti_amp = 0.020;
trajs(8).Y_disp     = 0.4;
trajs(8).vmax_X     = 1.0;
trajs(8).amax_X     = 8.0;
trajs(8).vmax_Y     = 1.2;
trajs(8).amax_Y     = 12.0;
trajs(8).jerkTime   = 0.035;
trajs(8).ms_f_low   = 1;
trajs(8).ms_f_high  = 100;
trajs(8).ms_modes   = {'y', 'common', 'diff'};   % y first, then common, diff last (heaviest X consumer)

% ======================================================================
% 4. Hardware limits and amplitude translation table
% ======================================================================
force_limits.peak = [2000, 2000, 1420];   % [N]    TELICA peak force  [FX1,FX2,FY]
force_limits.rms  = [916,  916,  656];    % [N]    TELICA RMS force   [FX1,FX2,FY]
ACC_LIM_X = 30.0;                         % [m/s2] TELICA X accel limit
ACC_LIM_Y = 50.0;                         % [m/s2] TELICA Y accel limit
VEL_LIM   = 2.0;                          % [m/s]  TELICA velocity limit

% Effective inertia per multisine mode.
% F_equiv = M_eff * (2*pi*f)^2 * A  [N RMS] — force controller must produce
% to track a position multisine of amplitude A [m] at frequency f [Hz].
% For reference injection all F_equiv reaches the plant (T~1 below bandwidth).
M_eff_Y      = mh;                              % Y translation
M_eff_common = m1 + m2 + mb + mh;              % symmetric X translation (total mass)
J_rot        = Jb + Jh + (m1+m2)*Lb^2/4;
M_eff_diff   = J_rot / (Lb/2)^2;               % rotation, referred to X actuator force

% Accel-limited maximum amplitude at each mode's highest frequency.
% common mode is now capped at 20 Hz (same as diff and y) — see mode_band().
A_max_X_common_m = ACC_LIM_X / (2*pi*20)^2;    % ~1.9  mm at 20 Hz
A_max_X_diff_m   = ACC_LIM_X / (2*pi*20)^2;    % ~1.9  mm at 20 Hz
A_max_Y_m        = ACC_LIM_Y / (2*pi*20)^2;    % ~3.2  mm at 20 Hz

fprintf('\n%s\n', repmat('=', 1, 82));
fprintf('REFERENCE INJECTION — AMPLITUDE TRANSLATION TABLE\n');
fprintf('F_equiv = M_eff*(2pi*f)^2*A  |  all F_equiv reaches plant (T~1 below bandwidth)\n');
fprintf('Accel limits: X=%.0f m/s^2  Y=%.0f m/s^2\n', ACC_LIM_X, ACC_LIM_Y);
fprintf('%s\n', repmat('-', 1, 82));
fprintf('%-9s  %-7s  %-10s  %-14s  %-12s  %-14s\n', ...
        'A [mm]', 'f [Hz]', 'Y [N]', 'X_common [N]', 'X_diff [N]', 'Accel [m/s^2]');
fprintf('%s\n', repmat('-', 1, 82));
for A_mm = [0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 1.9, 2.0, 3.2, 5.0]
    A = A_mm * 1e-3;
    for f_p = [10, 20, 100]
        omega   = 2*pi*f_p;
        accel   = omega^2 * A;
        F_Y     = M_eff_Y      * accel;
        F_com   = M_eff_common * accel;
        F_dif   = M_eff_diff   * accel;
        flag = '';
        if     accel > ACC_LIM_Y,  flag = ' [>Y_LIM]';
        elseif accel > ACC_LIM_X,  flag = ' [>X_LIM]';
        end
        fprintf('%-9.2f  %-7.0f  %-10.1f  %-14.1f  %-12.1f  %.1f%s\n', ...
                A_mm, f_p, F_Y, F_com, F_dif, accel, flag);
    end
end
fprintf('%s\n', repmat('-', 1, 82));
fprintf('Accel-limited max A: X_common=%.2f mm @20Hz  X_diff=%.2f mm @20Hz  Y=%.2f mm @20Hz\n', ...
        A_max_X_common_m*1e3, A_max_X_diff_m*1e3, A_max_Y_m*1e3);
fprintf('Compare: force injection 800 N * |S~0.05| = ~40 N net  <->  ~1 mm @10 Hz ref injection\n');
fprintf('%s\n\n', repmat('=', 1, 82));

% ======================================================================
% 5. Output directory
% ======================================================================
mdl       = 'gantry_2025a';
out_subdir = 'parameter-recovery-multisine';
out_dir   = fullfile(fileparts(mfilename('fullpath')), '..', 'Matlab-output', out_subdir);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

n_hold = round(0.5 / ts);   % 0.5 s hold = 10000 samples at 20 kHz

% ======================================================================
% 6. Run all trajectories
% ======================================================================
for i = TRAJ_SUBSET
    sp = trajs(i);
    fprintf('=== %d/%d  %s ===\n', i, numel(trajs), sp.id);

    % -- Controller at this trajectory's operating point (D-039) -----------
    Y_op = sp.Y_initial;
    Y    = sp.Y_initial;   % Simulink integrator IC
    M_op = [m1+m2+mb+mh,             (m1-m2)*Lb/2 - mh*Y_op,                   0;
            (m1-m2)*Lb/2 - mh*Y_op,  Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y_op^2,  -mh*d;
            0,                        -mh*d,                                      mh];
    sys_logical         = getss(n, M_op, C_damp, K);
    StageCoordinatesSystem = P.' * sys_logical * P;
    Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
    for j = 1:3
        Cfb(j,j) = ruleOfThumb(fbw, StageCoordinatesSystem(j,j), ts);
    end

    % -- Base trajectory (no multisine) ------------------------------------
    [r_traj, t_traj] = make_ref(sp, n_hold, ts);
    [r_traj, t_traj] = pad_to_multisine_periods(r_traj, ts, fs);   % leakage-free period alignment
    validate_ref(r_traj, sp.id, Lb);   % assert: base trajectory must be within limits

    % -- Baseline simulation (f=0, r=r_traj) ----------------------------------
    r = r_traj;
    t = t_traj;
    f = zeros(length(t_traj), 3);
    fprintf('  Baseline simulation (f=0) %.2f s (%d samples) ...\n', t_traj(end), length(t_traj));
    sim(mdl, t_traj(end));
    q1_base = q1;
    [t_base, ~, u_base, ~] = reconstruct(q1_base, r_traj, t_traj, Cfb);
    base_force_report = summarize_forces(u_base, zeros(size(u_base)), force_limits);
    fprintf('  Baseline complete. q1 samples=%d, u_total RMS=[%.1f %.1f %.1f] N\n', ...
            size(q1_base, 1), base_force_report.rms_total);

    % -- Cumulative force-multisine scan ------------------------------------
    % Each mode is scanned with all previously committed modes already active.
    % This ensures safety checks (force limits, response limits) reflect the
    % true combined load — fixing the single-mode isolation bug that caused
    % T7 and T8 to fail at export.
    % Score uses the committed-mode baseline so ranking measures only the
    % incremental contribution of the current trial mode (not coupling from
    % previously committed modes), fixing inflated survival ratios at m>=2.
    fprintf('  Force-multisine scan: cumulative across modes\n');
    fprintf('    %-7s %-11s %-8s %-7s %-10s %-17s %-17s %-7s\n', ...
            'mode', 'band[Hz]', 'amp[N]', 'ok', 'survive', 'dq_rms[mm]', 'u_rms[N]', 'score');

    f_sim_committed = zeros(length(t_traj), 3);  % accumulates best signal per mode

    for m = 1:numel(sp.ms_modes)
        mode_name  = sp.ms_modes{m};
        best_score = -inf;
        best_row   = '';
        best_f_mode = zeros(length(t_traj), 3);   % best force signal for this mode

        % Committed-mode baseline for scoring: reuse f=0 result for m==1;
        % run one extra sim with committed modes only for m>=2.
        if m == 1
            q1_committed = q1_base;
            t_committed  = t_base;
            u_committed  = u_base;
        else
            r = r_traj; t = t_traj; f = f_sim_committed;
            sim(mdl, t_traj(end));
            q1_committed = q1;
            [t_committed, ~, u_committed, ~] = reconstruct(q1_committed, r_traj, t_traj, Cfb);
        end

        for bi = 1:size(FORCE_DIAG_BANDS_HZ, 1)
            f_low  = FORCE_DIAG_BANDS_HZ(bi, 1);
            f_high = FORCE_DIAG_BANDS_HZ(bi, 2);
            if count_odd_bins(f_low, f_high, fs) < 7
                continue;
            end

            for amp_N = FORCE_DIAG_AMP_RMS_N
                f_trial = generate_force_one_mode(length(t_traj), fs, f_low, f_high, ...
                                                  m, mode_name, amp_N);
                % Inject committed modes + this trial together so safety
                % checks see the true combined force on the plant.
                r = r_traj;
                t = t_traj;
                f = f_sim_committed + f_trial;
                sim(mdl, t_traj(end));

                [t_sim, ~, u_q1_trial, ~] = reconstruct(q1, r_traj, t_traj, Cfb);
                f_combined  = resample_signal(f_sim_committed + f_trial, t_traj, t_sim);
                rep         = summarize_forces(u_q1_trial, f_combined, force_limits);
                ok_response = validate_response(q1, fs, Lb, false);
                ok          = rep.ok && ok_response;

                % Score: incremental contribution of this trial mode above the
                % committed-mode baseline (u_q1_trial - u_comm already removes
                % coupling from committed modes; + f_trial_sim is the injected
                % excitation that drives that increment).
                f_trial_sim  = resample_signal(f_trial,      t_traj,      t_sim);
                u_comm_sim   = resample_signal(u_committed,  t_committed, t_sim);
                q1_comm_sim  = resample_signal(q1_committed, t_committed, t_sim);
                u_delta      = (u_q1_trial - u_comm_sim) + f_trial_sim;
                diag_m       = force_multisine_metrics(q1, q1_comm_sim, f_trial_sim, ...
                                                       u_delta, mode_name);
                score = diag_m.survival * diag_m.dq_mode_rms_m * 1e3 ...
                        * sqrt(count_odd_bins(f_low, f_high, fs));
                if ~ok, score = 0; end

                row = sprintf('    %-7s %3.0f-%-6.0f %8.1f %-7s %-10.3f [%5.3f %5.3f %5.3f] [%5.1f %5.1f %5.1f] %7.3f', ...
                              mode_name, f_low, f_high, amp_N, string_ok(ok), ...
                              diag_m.survival, diag_m.dq_rms_m*1e3, diag_m.u_rms_N, score);
                fprintf('%s\n', row);

                if score > best_score
                    best_score  = score;
                    best_row    = row;
                    best_f_mode = f_trial;
                end
            end
        end

        if best_score > 0
            fprintf('  best %s:\n%s\n', mode_name, best_row);
            f_sim_committed = f_sim_committed + best_f_mode;
        else
            fprintf('  best %s: no valid candidate — mode excluded.\n', mode_name);
        end
    end

    if DIAGNOSTIC_ONLY
        fprintf('  DIAGNOSTIC_ONLY=true: skipping export for %s.\n\n', sp.id);
        continue;
    end

    % -- Export: use exactly the signals found by the cumulative scan -------
    if all(f_sim_committed(:) == 0)
        warning('%s: no mode produced a valid signal — skipping.', sp.id);
        continue;
    end

    r = r_traj;
    t = t_traj;
    f = f_sim_committed;
    fprintf('  Export simulation: %.2f s (%d samples) ...\n', t_traj(end), length(t_traj));
    sim(mdl, t_traj(end));

    report_tracking(q1, r_traj, t_traj);
    if ~validate_response(q1, fs, Lb, true) || ...
       ~validate_forces_with_ff(q1, r_traj, f_sim_committed, t_traj, Cfb, force_limits)
        warning('%s: export simulation failed validation — skipping.', sp.id);
        continue;
    end

    % -- Reconstruct, report, save -----------------------------------------
    [t_sim, r_sim, u_q1, Y_trajectory] = reconstruct(q1, r_traj, t_traj, Cfb);
    f_sim        = resample_signal(f_sim_committed, t_traj, t_sim);
    u_total      = u_q1 + f_sim;
    force_report = summarize_forces(u_q1, f_sim, force_limits);

    report_traj(q1, Y_trajectory);
    report_forces(force_report);

    out_path = fullfile(out_dir, [sp.id, '.mat']);
    save(out_path, 't_sim', 'fs', 'r_sim', 'f_sim', ...
                   'u_q1', 'u_total', 'q1', 'Y_trajectory', 'force_report');
    fprintf('  Saved: %s\n\n', out_path);
end

if DIAGNOSTIC_ONLY
    fprintf('Done. Diagnostics scanned for %d/%d trajectories. No files exported.\n', ...
            numel(TRAJ_SUBSET), numel(trajs));
else
    fprintf('Done. %d/%d trajectories exported to:\n  %s\n', ...
            numel(TRAJ_SUBSET), numel(trajs), out_dir);
end

% ======================================================================
% Local functions
% ======================================================================

function [r, t] = make_ref(sp, n_hold, ts)
% Build stage-coordinate reference r (N x 3) = [X1, X2, Y].
    r     = repmat([0, 0, sp.Y_initial], n_hold, 1);
    Y_now = sp.Y_initial;

    if sp.X_sym_amp > 0
        pv_sym = setpoint_1d(sp.X_sym_amp,  sp.vmax_X, sp.amax_X, sp.jerkTime, ts);
        n_sym  = length(pv_sym);
    else
        pv_sym = []; n_sym = 0;
    end

    if sp.X_anti_amp > 0
        pv_anti = setpoint_1d(sp.X_anti_amp, sp.vmax_X, sp.amax_X, sp.jerkTime, ts);
        n_anti  = length(pv_anti);
    else
        pv_anti = []; n_anti = 0;
    end

    n_move_X = max(n_sym, n_anti);
    n_move_Y = 0;

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
            pv_sym  = pad_vec(pv_sym,  n_move_X, n_main);
            pv_anti = pad_vec(pv_anti, n_move_X, n_main);
            X1 = pv_sym + pv_anti;
            X2 = pv_sym - pv_anti;
        end

        if n_move_Y > 0
            Y = Y_now - pad_vec(pv_Y, n_move_Y, n_main);
        end

        r = [r; [X1, X2, Y]];
    end

    r = [r; repmat(r(end, :), n_hold, 1)];
    t = ts * (0:size(r,1)-1)';
end

% ----------------------------------------------------------------------

function v = pad_vec(v, n_src, n_tgt)
% Pad column vector v (length n_src or empty) to length n_tgt by holding last value.
    if isempty(v)
        v = zeros(n_tgt, 1);
    else
        v = [v; v(end) * ones(n_src - length(v), 1)];   % pad to n_src
        v = [v; v(end) * ones(n_tgt - n_src, 1)];        % pad to n_tgt
    end
end

% ----------------------------------------------------------------------

function validate_ref(r, id, Lb)
% Assert all reference positions are within ETEL TELICA hardware limits.
% Throws on violation — used once on r_traj before the amplitude sweep.
    X_LIM    = 0.375;
    Y_LIM    = 0.400;
    DIFF_LIM = sin(0.1) * Lb;

    assert(max(abs(r(:,1))) <= X_LIM,              '%s: X1 exceeds +/-%.0f mm', id, X_LIM*1e3);
    assert(max(abs(r(:,2))) <= X_LIM,              '%s: X2 exceeds +/-%.0f mm', id, X_LIM*1e3);
    assert(max(r(:,3))       <=  Y_LIM,            '%s: Y exceeds +%.0f mm',    id, Y_LIM*1e3);
    assert(min(r(:,3))       >= -Y_LIM,            '%s: Y exceeds -%.0f mm',    id, Y_LIM*1e3);
    assert(max(abs(r(:,1) - r(:,2))) <= DIFF_LIM,  ...
           '%s: |X1-X2| exceeds %.1f mm yaw limit (0.1 rad, Garcia 2013)', id, DIFF_LIM*1e3);

    fprintf('  r_traj OK: X1=[%+.0f %+.0f] X2=[%+.0f %+.0f] Y=[%+.0f %+.0f] |X1-X2|_max=%.1f mm\n', ...
            min(r(:,1))*1e3, max(r(:,1))*1e3, ...
            min(r(:,2))*1e3, max(r(:,2))*1e3, ...
            min(r(:,3))*1e3, max(r(:,3))*1e3, ...
            max(abs(r(:,1) - r(:,2)))*1e3);
end

% ----------------------------------------------------------------------

function ok = check_ref_total(r_total, fs, Lb, verbose)
% Soft check (returns bool) of all TELICA limits on r_total = r_traj + r_ms.
% verbose (default true): print exceeded values. Pass false during sweep loops
% to suppress per-trial output.
    if nargin < 4, verbose = true; end
    X_LIM    = 0.375;  Y_LIM = 0.400;  DIFF_LIM = sin(0.1)*Lb;
    VEL_LIM  = 2.0;    ACC_LIM_X = 30.0;  ACC_LIM_Y = 50.0;

    vel = diff(r_total) * fs;
    acc = diff(vel) * fs;

    names  = {'X1 pos','X2 pos','Y pos+','Y pos-','|X1-X2|', ...
              'X1 vel','X2 vel','Y vel','X1 acc','X2 acc','Y acc'};
    vals   = [max(abs(r_total(:,1))), max(abs(r_total(:,2))), ...
               max(r_total(:,3)),    -min(r_total(:,3)), ...
               max(abs(r_total(:,1)-r_total(:,2))), ...
               max(abs(vel(:,1))), max(abs(vel(:,2))), max(abs(vel(:,3))), ...
               max(abs(acc(:,1))), max(abs(acc(:,2))), max(abs(acc(:,3)))];
    limits = [X_LIM, X_LIM, Y_LIM, Y_LIM, DIFF_LIM, ...
              VEL_LIM, VEL_LIM, VEL_LIM, ACC_LIM_X, ACC_LIM_X, ACC_LIM_Y];

    ok = all(vals <= limits);
    if ~ok && verbose
        for ii = 1:numel(vals)
            if vals(ii) > limits(ii)
                fprintf('    exceeded: %s = %.4f  limit = %.4f\n', names{ii}, vals(ii), limits(ii));
            end
        end
    end
end

% ----------------------------------------------------------------------

function [t_sim, r_sim, u_q1, Y_trajectory] = reconstruct(q1, r_total, t, Cfb)
% Reconstruct plant force from simulated output and total reference.
% u_q1 = Cfb*(r_total - q1) is the complete plant input (f=0 always).
% Handles variable-step Simulink output (N_sim may differ from length(t)).
    N_sim = size(q1, 1);
    if N_sim ~= length(t)
        t_sim = linspace(0, t(end), N_sim)';
        r_sim = interp1(t, r_total, t_sim);
    else
        t_sim = t;
        r_sim = r_total;
    end
    u_q1         = lsim(ss(Cfb), r_sim - q1, t_sim);
    Y_trajectory = q1(:, 3);
end

% ----------------------------------------------------------------------

function report_traj(q1, Y_trajectory)
% Print axis range after simulation.
    fprintf('  X1: [%+.3f, %+.3f] m\n', min(q1(:,1)), max(q1(:,1)));
    fprintf('  X2: [%+.3f, %+.3f] m\n', min(q1(:,2)), max(q1(:,2)));
    fprintf('  Y:  [%+.3f, %+.3f] m\n', min(Y_trajectory), max(Y_trajectory));
end

% ----------------------------------------------------------------------

function ok = validate_forces_with_ff(q1, r_traj, f_sim, t, Cfb, force_limits)
% Check total force (feedback + feedforward) against TELICA actuator limits.
% For force injection: total = u_q1 + f_sim where u_q1 = Cfb*(r_traj - q1).
    N_sim = size(q1, 1);
    if N_sim ~= length(t)
        t_sim = linspace(0, t(end), N_sim)';
        r_sim = interp1(t, r_traj, t_sim);
        f_sim = interp1(t, f_sim, t_sim, 'linear', 'extrap');
    else
        t_sim = t;
        r_sim = r_traj;
    end
    u_q1 = lsim(ss(Cfb), r_sim - q1, t_sim);
    rep  = summarize_forces(u_q1, f_sim, force_limits);
    ok   = rep.ok;
    if ~ok
        fprintf('  Force check failed:\n');
        fprintf('    peak total=[%.0f %.0f %.0f] N, limits=[%.0f %.0f %.0f] N\n', ...
                rep.max_total, rep.peak_limits);
        fprintf('    RMS  total=[%.0f %.0f %.0f] N, limits=[%.0f %.0f %.0f] N\n', ...
                rep.rms_total, rep.rms_limits);
    end
end

% ----------------------------------------------------------------------

function rep = summarize_forces(u_feedback, f_feedforward, force_limits)
% Summarise peak and RMS force demand. For reference injection f_feedforward=0.
    u_total = u_feedback + f_feedforward;

    rep.peak_limits      = force_limits.peak;
    rep.rms_limits       = force_limits.rms;
    rep.max_feedforward  = max(abs(f_feedforward), [], 1);
    rep.max_feedback     = max(abs(u_feedback),    [], 1);
    rep.max_total        = max(abs(u_total),        [], 1);
    rep.rms_feedforward  = sqrt(mean(f_feedforward.^2, 1));
    rep.rms_feedback     = sqrt(mean(u_feedback.^2,    1));
    rep.rms_total        = sqrt(mean(u_total.^2,        1));
    rep.peak_ratio_total = rep.max_total ./ force_limits.peak;
    rep.rms_ratio_total  = rep.rms_total ./ force_limits.rms;
    rep.ok_peak          = all(rep.max_total <= force_limits.peak);
    rep.ok_rms           = all(rep.rms_total <= force_limits.rms);
    rep.ok               = rep.ok_peak && rep.ok_rms;
end

% ----------------------------------------------------------------------

function report_forces(rep)
% Print total force demand (feedback + feedforward).
    fprintf('  Force peaks [FX1 FX2 FY] N:\n');
    fprintf('    u_total (peak): [%7.1f %7.1f %7.1f]  limits=[%.0f %.0f %.0f]\n', ...
            rep.max_total, rep.peak_limits);
    fprintf('  Force RMS [FX1 FX2 FY] N:\n');
    fprintf('    u_total (RMS):  [%7.1f %7.1f %7.1f]  limits=[%.0f %.0f %.0f]\n', ...
            rep.rms_total, rep.rms_limits);
end

% ----------------------------------------------------------------------

function pv = setpoint_1d(dist, vmax, amax, jerkTime, ts)
    pvajs = thirdOrderSetpointETEL(dist, vmax, amax, amax / jerkTime, Inf, ts);
    pv    = pvajs(:, 1);
end

% ----------------------------------------------------------------------

function [r_pad, t_pad] = pad_to_multisine_periods(r, ts, fs)
% Pad the final hold to an integer number of 1 s periods (leakage-free multisine).
    N_period = round(fs);
    N        = size(r, 1);
    n_period = max(2, ceil(N / N_period));
    N_target = n_period * N_period;
    n_pad    = N_target - N;
    if n_pad > 0
        r_pad = [r; repmat(r(end, :), n_pad, 1)];
    else
        r_pad = r;
    end
    t_pad = ts * (0:size(r_pad,1)-1)';
end

% ----------------------------------------------------------------------

function f_ms = generate_force_one_mode(N, fs, f_low, f_high, mode_idx, mode_name, amp_N)
% Generate an N-by-3 force multisine for one mode.
% amp_N is the RMS force in the scalar modal signal before channel mapping.
    N_period = round(fs);
    assert(mod(N, N_period) == 0, ...
           'N=%d must be a multiple of N_period=%d', N, N_period);

    sig = multisine_schroeder_periodic(N, N_period, fs, f_low, f_high, mode_idx);
    sig = sig * (amp_N / rms(sig));

    ramp_time = 0.100;
    N_RAMP = min(round(ramp_time * fs), floor(0.25 * N_period));
    if N_RAMP > 1
        w = 0.5 * (1 - cos(pi * (0:N_RAMP-1)' / (N_RAMP-1)));
        sig(1:N_RAMP) = sig(1:N_RAMP) .* w;
    end

    f_ms = zeros(N, 3);
    switch mode_name
        case 'common'
            f_ms(:,1) = sig;
            f_ms(:,2) = sig;
        case 'diff'
            f_ms(:,1) =  sig;
            f_ms(:,2) = -sig;
        case 'y'
            f_ms(:,3) = sig;
        otherwise
            error('unknown force multisine mode "%s"', mode_name);
    end
end

% ----------------------------------------------------------------------

function y_sim = resample_signal(y, t_src, t_sim)
% Align a scripted signal to Simulink's returned sample grid.
    if size(y, 1) == numel(t_sim)
        y_sim = y;
    else
        y_sim = interp1(t_src, y, t_sim, 'linear', 'extrap');
    end
end

% ----------------------------------------------------------------------

function diag = force_multisine_metrics(q1, q1_base, f_sim, u_total, mode_name)
% Summarise surviving net force and output response for one diagnostic run.
    N = min(size(q1, 1), size(q1_base, 1));
    dq = q1(1:N,:) - q1_base(1:N,:);
    f_sim = f_sim(1:N,:);
    u_total = u_total(1:N,:);

    diag.dq_rms_m = sqrt(mean(dq.^2, 1));
    diag.f_rms_N = sqrt(mean(f_sim.^2, 1));
    diag.u_rms_N = sqrt(mean(u_total.^2, 1));

    idx = mode_channels(mode_name);
    diag.dq_mode_rms_m = sqrt(mean(diag.dq_rms_m(idx).^2));
    f_mode_rms = sqrt(mean(diag.f_rms_N(idx).^2));
    u_mode_rms = sqrt(mean(diag.u_rms_N(idx).^2));
    diag.survival = u_mode_rms / max(f_mode_rms, 1e-12);
end

% ----------------------------------------------------------------------

function idx = mode_channels(mode_name)
    switch mode_name
        case {'common', 'diff'}
            idx = [1, 2];
        case 'y'
            idx = 3;
        otherwise
            error('unknown mode "%s"', mode_name);
    end
end

% ----------------------------------------------------------------------

function s = string_ok(ok)
    if ok
        s = 'yes';
    else
        s = 'no';
    end
end

% ----------------------------------------------------------------------

function n = count_odd_bins(f_low, f_high, fs)
% Count odd-harmonic frequency bins in [f_low, f_high] for a 1 s period.
% Used as the bandwidth factor in sweep scoring: score = amp^2 * n_bins.
    f0 = fs / round(fs);   % bin width = 1 Hz (fs=20 kHz, N_period=20000)
    k0 = max(1, ceil(f_low  / f0));
    k1 = floor(f_high / f0);
    k  = k0:k1;
    n  = sum(mod(k, 2) == 1);
end

% ----------------------------------------------------------------------

function M = mode_M_eff(mode_name, M_eff_common, M_eff_diff, M_eff_Y)
% Return effective inertia [kg or kg*m^2/m^2] for a mode — used to estimate
% peak controller force: F_est = M_eff * (2*pi*f_high)^2 * amp_rms.
    switch mode_name
        case 'common', M = M_eff_common;
        case 'diff',   M = M_eff_diff;
        case 'y',      M = M_eff_Y;
        otherwise,     M = NaN;
    end
end

% ----------------------------------------------------------------------

function report_tracking(q1, r_total, t)
% Print per-channel tracking error statistics after simulation.
% Large Y max error (>> 1 mm) indicates controller integrator windup.
    N_sim = size(q1, 1);
    if N_sim ~= length(t)
        t_sim = linspace(0, t(end), N_sim)';
        r_sim = interp1(t, r_total, t_sim);
    else
        r_sim = r_total;
    end
    e = r_sim - q1;
    fprintf('  Tracking error |r_total - q1| [mm]:\n');
    fprintf('    X1: max=%.3f  RMS=%.3f\n', max(abs(e(:,1)))*1e3, rms(e(:,1))*1e3);
    fprintf('    X2: max=%.3f  RMS=%.3f\n', max(abs(e(:,2)))*1e3, rms(e(:,2))*1e3);
    fprintf('    Y:  max=%.3f  RMS=%.3f\n', max(abs(e(:,3)))*1e3, rms(e(:,3))*1e3);
end

% ----------------------------------------------------------------------

function sig = multisine_schroeder_periodic(N, N_period, fs, f_low, f_high, seed)
% One-second Schroeder-phase odd-harmonic multisine tiled over N samples.
    f0 = fs / N_period;
    k0 = max(1, ceil(f_low / f0));
    k1 = floor(f_high / f0);
    k  = k0:k1;
    k  = k(mod(k, 2) == 1);   % odd harmonics only

    F = numel(k);
    if F < 7
        error('Band %.1f-%.1f Hz: only %d odd lines, need >=7 for 13 parameters.', ...
              f_low, f_high, F);
    end

    idx   = 1:F;
    freqs = k * f0;
    phi   = -idx .* (idx - 1) * pi / F;
    phi   = phi + 2*pi*freqs*(seed - 1)/(7*f_high);

    t_period   = (0:N_period-1)' / fs;
    one_period = sum(cos(2*pi*t_period*freqs + phi), 2);
    one_period = one_period / rms(one_period);

    sig = repmat(one_period, N/N_period, 1);
end

% ----------------------------------------------------------------------

function ok = validate_response(q1, fs, Lb, verbose)
% Check actual simulated q1 against ETEL TELICA hardware limits.
% Acceleration is NOT checked here: double-differencing q1 at 20 kHz
% amplifies ode45 sub-sample interpolation artifacts by fs^2 = 4e8,
% producing spurious spikes that have nothing to do with real machine
% acceleration. Acceleration is already checked on r_total (= r_traj +
% r_ms) in check_ref_total, where the signal is piecewise-polynomial and
% diff(diff(r))*fs^2 is exact.
    if nargin < 4, verbose = true; end
    X_LIM    = 0.375;  Y_LIM = 0.400;
    DIFF_LIM = sin(0.1) * Lb;
    VEL_LIM  = 2.0;

    vel = diff(q1) * fs;

    ok =    max(abs(q1(:,1)))          <= X_LIM    ...
         && max(abs(q1(:,2)))          <= X_LIM    ...
         && max(abs(q1(:,3)))          <= Y_LIM    ...
         && max(abs(q1(:,1)-q1(:,2))) <= DIFF_LIM ...
         && max(abs(vel(:,1)))         <= VEL_LIM  ...
         && max(abs(vel(:,2)))         <= VEL_LIM  ...
         && max(abs(vel(:,3)))         <= VEL_LIM;

    if ~ok && verbose
        names  = {'X1 pos','X2 pos','Y pos','|X1-X2|','X1 vel','X2 vel','Y vel'};
        vals   = [max(abs(q1(:,1))), max(abs(q1(:,2))), max(abs(q1(:,3))), ...
                  max(abs(q1(:,1)-q1(:,2))), max(abs(vel(:,1))), max(abs(vel(:,2))), ...
                  max(abs(vel(:,3)))];
        limits = [X_LIM, X_LIM, Y_LIM, DIFF_LIM, VEL_LIM, VEL_LIM, VEL_LIM];
        for ii = 1:numel(vals)
            if vals(ii) > limits(ii)
                fprintf('  Response exceeded: %s = %.4f  limit = %.4f\n', names{ii}, vals(ii), limits(ii));
            end
        end
    end
end
