% export_param_recovery_multisine_val_and_test.m
% -----------------------------------------------
% Validation/test trajectory generation with post-controller force multisine.
%
% Holdout counterpart to export_param_recovery_multisine.m (T1-T8 training set).
% Uses the same force-injection approach: reference stays clean (r = r_traj),
% Schroeder-phase multisine injected as feedforward force f_sim after the
% controller. Plant receives total input u_total = u_q1 + f_sim.
%
% Independent multisine realizations from training set are guaranteed by
% ms_seed_offset per trajectory (1000 for V1, 2000 for E1), so val/test
% spectral lines do not overlap with T1-T8 training lines.
%
% DIAGNOSTIC_ONLY = true : run band/amplitude scan, print results, no export.
% DIAGNOSTIC_ONLY = false: scan then export .mat files.
%
% Holdout trajectories:
%   V1  X symmetric + partial Y sweep   interpolation holdout (Y_initial=0.25)
%   E1  X sym + X anti + Y sweep        coupled test case    (Y_initial=0.10)
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
%   split        (char)   'val' or 'test'
%
% Does NOT modify any file in kamtin-fp-model/.
%
% Run from repo root:
%   cd('<path>/Baseline-LPV-Augmentation')
%   run('Matlab-scripts/export_param_recovery_multisine_val_and_test.m')

addpath(genpath(fullfile(pwd, 'kamtin-fp-model', '03 Simulink gantry')))

% ======================================================================
% USER FLAGS
% ======================================================================
DIAGNOSTIC_ONLY = true;   % true -> scan + print only; false -> scan then export
TRAJ_SUBSET     = 1:2;    % 1=V1 (val), 2=E1 (test)

% Candidate force bands/amplitudes. 5-20 Hz excluded (survival ~0.036, never
% competitive from T1-T8 scan). Full range kept otherwise until V1/E1 are
% scanned — narrow after first diagnostic run.
FORCE_DIAG_BANDS_HZ  = [20,50; 50,100; 100,200];
FORCE_DIAG_AMP_RMS_N = [100, 200, 400, 800];

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
% ms_modes define which force channels are excited:
%   'common' -> f_X1 = f_X2 = multisine  (symmetric X force perturbation)
%   'diff'   -> f_X1 = -f_X2 = multisine (anti-symmetric X, pure torque)
%   'y'      -> f_Y = multisine           (Y force perturbation)
%
% ms_seed_offset ensures spectral independence from training set (T1-T8 use
% seeds 1-3 per mode; V1 uses 1001-1003, E1 uses 2001-2003).
% y is always listed first: lower survival below bandwidth means it needs
% full headroom from the cumulative scan.

% V1: validation trajectory — X symmetric + partial Y sweep.
% Interpolation holdout between T2 (X_sym at Y=0.3) and T5 (X_sym + Y sweep
% at Y=0.2). Y_initial=0.25 is not covered by any training trajectory.
% Gentler amplitudes and dynamics than training equivalents.
trajs(1).id             = 'V1_X_sym_Y_mid_sweep';
trajs(1).split          = 'val';
trajs(1).ms_seed_offset = 1000;
trajs(1).Y_initial      = 0.25;
trajs(1).X_sym_amp      = 0.075;
trajs(1).X_anti_amp     = 0.000;
trajs(1).Y_disp         = 0.30;
trajs(1).vmax_X         = 0.8;
trajs(1).amax_X         = 12.0;
trajs(1).vmax_Y         = 0.9;
trajs(1).amax_Y         = 14.0;
trajs(1).jerkTime       = 0.040;
trajs(1).ms_modes       = {'y', 'common'};   % y first: needs full headroom

% E1: test trajectory — X symmetric + X anti-symmetric + Y sweep.
% Coupled holdout related to T8 in physics, but at a different Y region
% (Y_initial=0.10), different amplitudes, and independent multisine seed.
trajs(2).id             = 'E1_X_sym_anti_Y_low_offset_sweep';
trajs(2).split          = 'test';
trajs(2).ms_seed_offset = 2000;
trajs(2).Y_initial      = 0.10;
trajs(2).X_sym_amp      = 0.060;
trajs(2).X_anti_amp     = 0.015;
trajs(2).Y_disp         = 0.25;
trajs(2).vmax_X         = 0.7;
trajs(2).amax_X         = 10.0;
trajs(2).vmax_Y         = 0.8;
trajs(2).amax_Y         = 10.0;
trajs(2).jerkTime       = 0.045;
trajs(2).ms_modes       = {'y', 'common', 'diff'};   % y first, diff last (heaviest X consumer)

% ======================================================================
% 4. Hardware limits and amplitude translation table
% ======================================================================
force_limits.peak = [2000, 2000, 1420];   % [N]    TELICA peak force  [FX1,FX2,FY]
force_limits.rms  = [916,  916,  656];    % [N]    TELICA RMS force   [FX1,FX2,FY]
ACC_LIM_X = 30.0;                         % [m/s2] TELICA X accel limit
ACC_LIM_Y = 50.0;                         % [m/s2] TELICA Y accel limit
VEL_LIM   = 2.0;                          % [m/s]  TELICA velocity limit

% Effective inertia per multisine mode (informational only — force injection
% amplitude is in Newtons, not position; table below is for context).
M_eff_Y      = mh;
M_eff_common = m1 + m2 + mb + mh;
J_rot        = Jb + Jh + (m1+m2)*Lb^2/4;
M_eff_diff   = J_rot / (Lb/2)^2;

A_max_X_common_m = ACC_LIM_X / (2*pi*20)^2;
A_max_X_diff_m   = ACC_LIM_X / (2*pi*20)^2;
A_max_Y_m        = ACC_LIM_Y / (2*pi*20)^2;

fprintf('\n%s\n', repmat('=', 1, 82));
fprintf('FORCE INJECTION — REFERENCE TRANSLATION TABLE\n');
fprintf('F_equiv = M_eff*(2pi*f)^2*A  |  for context only; injection unit is Newtons\n');
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
fprintf('%s\n\n', repmat('=', 1, 82));

% ======================================================================
% 5. Output directory
% ======================================================================
mdl       = 'gantry_2025a';
out_subdir = 'parameter-recovery-multisine-val-test';
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
    [r_traj, t_traj] = pad_to_multisine_periods(r_traj, ts, fs);
    validate_ref(r_traj, sp.id, Lb);

    % -- Baseline simulation (f=0, r=r_traj) ------------------------------
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
    % Each mode scanned with all previously committed modes active (safety).
    % Score uses committed-mode baseline to measure only this mode's
    % incremental contribution (avoids inflated survival from coupling).
    % ms_seed_offset ensures independence from training set realizations.
    fprintf('  Force-multisine scan: cumulative across modes\n');
    fprintf('    %-7s %-11s %-8s %-7s %-10s %-17s %-17s %-7s\n', ...
            'mode', 'band[Hz]', 'amp[N]', 'ok', 'survive', 'dq_rms[mm]', 'u_rms[N]', 'score');

    f_sim_committed = zeros(length(t_traj), 3);

    for m = 1:numel(sp.ms_modes)
        mode_name  = sp.ms_modes{m};
        best_score = -inf;
        best_row   = '';
        best_f_mode = zeros(length(t_traj), 3);

        % Committed-mode baseline for scoring.
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
                % seed = m + ms_seed_offset ensures spectral independence
                % from training trajectories (which use seed = m).
                f_trial = generate_force_one_mode(length(t_traj), fs, f_low, f_high, ...
                                                  m + sp.ms_seed_offset, mode_name, amp_N);
                r = r_traj;
                t = t_traj;
                f = f_sim_committed + f_trial;
                sim(mdl, t_traj(end));

                [t_sim, ~, u_q1_trial, ~] = reconstruct(q1, r_traj, t_traj, Cfb);
                f_combined  = resample_signal(f_sim_committed + f_trial, t_traj, t_sim);
                rep         = summarize_forces(u_q1_trial, f_combined, force_limits);
                ok_response = validate_response(q1, fs, Lb, false);
                ok          = rep.ok && ok_response;

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

    split    = sp.split;
    out_path = fullfile(out_dir, [sp.id, '.mat']);
    save(out_path, 't_sim', 'fs', 'r_sim', 'f_sim', ...
                   'u_q1', 'u_total', 'q1', 'Y_trajectory', 'force_report', 'split');
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
    if isempty(v)
        v = zeros(n_tgt, 1);
    else
        v = [v; v(end) * ones(n_src - length(v), 1)];
        v = [v; v(end) * ones(n_tgt - n_src, 1)];
    end
end

% ----------------------------------------------------------------------

function validate_ref(r, id, Lb)
% Assert all reference positions are within ETEL TELICA hardware limits.
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

function [t_sim, r_sim, u_q1, Y_trajectory] = reconstruct(q1, r_total, t, Cfb)
% Reconstruct plant force from simulated output and total reference.
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
% Summarise peak and RMS force demand.
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

function f_ms = generate_force_one_mode(N, fs, f_low, f_high, seed, mode_name, amp_N)
% Generate an N-by-3 force multisine for one mode.
% seed = m + ms_seed_offset ensures independence from training realizations.
% amp_N is the RMS force [N] in the scalar modal signal before channel mapping.
    N_period = round(fs);
    assert(mod(N, N_period) == 0, ...
           'N=%d must be a multiple of N_period=%d', N, N_period);

    sig = multisine_schroeder_periodic(N, N_period, fs, f_low, f_high, seed);
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
    f_sim   = f_sim(1:N,:);
    u_total = u_total(1:N,:);

    diag.dq_rms_m = sqrt(mean(dq.^2, 1));
    diag.f_rms_N  = sqrt(mean(f_sim.^2, 1));
    diag.u_rms_N  = sqrt(mean(u_total.^2, 1));

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
    if ok, s = 'yes'; else, s = 'no'; end
end

% ----------------------------------------------------------------------

function n = count_odd_bins(f_low, f_high, fs)
% Count odd-harmonic frequency bins in [f_low, f_high] for a 1 s period.
    f0 = fs / round(fs);
    k0 = max(1, ceil(f_low  / f0));
    k1 = floor(f_high / f0);
    k  = k0:k1;
    n  = sum(mod(k, 2) == 1);
end

% ----------------------------------------------------------------------

function report_tracking(q1, r_total, t)
% Print per-channel tracking error statistics after simulation.
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
    k  = k(mod(k, 2) == 1);

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
