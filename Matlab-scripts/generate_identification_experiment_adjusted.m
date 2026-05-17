% generate_identification_experiment.m
% Generates BPTT identification trajectories with post-controller force injection.
%
% Design:
%   Band [f_low, f_high] and A_max loaded from step0_outputs.mat (pre-analysis).
%   Schroeder-phase odd-harmonic multisine injected post-controller per active mode.
%   All active modes injected simultaneously; different Schroeder seed per mode.
%
% Theory:
%   Schroeder phases:  Schroeder 1970 — minimises crest factor
%   Odd harmonics:     P&S Ch.4 §4.3.2 — PE condition, even nonlinearity detection
%   Leakage-free:      P&S Ch.2 §2.2.3 — integer periods, f0 = 1 Hz
%   Force injection:   feedback algebra U_total = S × F_sim (D-048)
%   Multi-mode seeds:  HEURISTIC — low cross-correlation, not strictly orthogonal
%
% Validation:
%   Position, velocity checked on simulated q1.
%   Acceleration checked on reference r (exact: piecewise polynomial).
%   Acceleration NOT checked on q1: ode45 at 20 kHz amplifies sub-sample
%   interpolation artifacts by fs^2 = 4e8, producing spurious spikes.
%   Forces (peak + RMS) checked on u_total = u_q1 + f_sim.
%
% Run from repo root:
%   run('Matlab-scripts/generate_identification_experiment.m')

addpath(genpath(fullfile(pwd, 'kamtin-fp-model', '03 Simulink gantry')))

% ── Pre-analysis outputs ──────────────────────────────────────────────────
step0  = load(fullfile(fileparts(mfilename('fullpath')),'..','Matlab-output','step0_outputs.mat'));
f_low  = step0.f_low;    % [Hz] lowest frequency where force survives controller
f_high = step0.f_high;   % [Hz] highest frequency with useful plant response
A_max  = step0.A_max;    % [N RMS] per mode: [common, diff, y]
fprintf('Pre-analysis: f_low=%.1f Hz  f_high=%.1f Hz  A_max=[%.0f %.0f %.0f] N\n', ...
        f_low, f_high, A_max);

% ── Physical parameters ───────────────────────────────────────────────────
mb=22.8; mh=10.1; m1=10.2; m2=10.7; Jb=1.0; Jh=0.05;
cg1=14.5; cg2=20.3; cy=10; cb1=9; cb2=9;
kb1=1987.5; kb2=1987.5; Lb=0.725; Lh=0.25; d=0.1;
cc1=16.8; cc2=18.35; ccy=11.6;

C_damp = [cg1+cg2,        (cg1-cg2)*Lb/2,            0;
          (cg1-cg2)*Lb/2, cb1+cb2+(cg1+cg2)*Lb^2/4,  0;
          0,               0,                          cy];
K  = [0,0,0; 0,kb1+kb2,0; 0,0,0];
n  = 3;  P = [1,1,0; Lb/2,-Lb/2,0; 0,0,1];
fs = 20e3;  ts = 1/fs;  fbw = 100;  mdl = 'gantry_2025a';
N_period = round(fs);       % 20000 samples = T_p = 1 s, f0 = 1 Hz
n_hold   = round(0.5/ts);  % 0.5 s settle hold at start and end of trajectory

% ── Hardware limits (TELICA spec) ─────────────────────────────────────────
lim.pos_X      = 0.375;              % [m]
lim.pos_Y      = 0.400;              % [m]
lim.diff       = sin(0.1) * Lb;     % [m] max |X1-X2| yaw limit (Garcia 2013)
lim.vel        = 2.0;               % [m/s]
lim.acc_X      = 30.0;             % [m/s^2] — checked on r only
lim.acc_Y      = 50.0;             % [m/s^2] — checked on r only
lim.force_peak = [2000, 2000, 1420]; % [N] peak [FX1,FX2,FY]
lim.force_rms  = [916,  916,  656];  % [N] RMS

% ── Mode definitions ─────────────────────────────────────────────────────
% Force direction in motor coordinates [FX1, FX2, FY].
% A_max index: 1=common, 2=diff, 3=y (matches diagnostics_system.m)
mode_def.common = struct('f_vec',[1, 1,0],'A_idx',1);
mode_def.diff   = struct('f_vec',[1,-1,0],'A_idx',2);
mode_def.y      = struct('f_vec',[0, 0,1],'A_idx',3);

% ── Trajectory definitions (T1-T8 train, V1 val, E1 test) ────────────────
% ms_modes: force injection modes active per trajectory.
% seed_offset: shifts Schroeder seed so val/test spectral lines do not overlap
%   with training lines (T1-T8 use seeds 1-3; V1 uses 1001-1003; E1 2001-2003).
% All motion parameters identical to export_param_recovery_multisine.m.
trajs(1).id='T1_Y_sweep_conservative'; trajs(1).split='train'; trajs(1).seed_offset=0;    trajs(1).Y_initial=0.3; trajs(1).X_sym_amp=0;    trajs(1).X_anti_amp=0;     trajs(1).Y_disp=0.6; trajs(1).vmax_X=0;   trajs(1).amax_X=0;    trajs(1).vmax_Y=1.00; trajs(1).amax_Y=10.0; trajs(1).jerkTime=0.050; trajs(1).ms_modes={'y'};
trajs(2).id='T2_X_sym_Y030';          trajs(2).split='train'; trajs(2).seed_offset=0;    trajs(2).Y_initial=0.3; trajs(2).X_sym_amp=0.15; trajs(2).X_anti_amp=0;     trajs(2).Y_disp=0;   trajs(2).vmax_X=1.5; trajs(2).amax_X=20.0; trajs(2).vmax_Y=1.00; trajs(2).amax_Y=20.0; trajs(2).jerkTime=0.030; trajs(2).ms_modes={'common'};
trajs(3).id='T3_X_sym_Y000';          trajs(3).split='train'; trajs(3).seed_offset=0;    trajs(3).Y_initial=0.0; trajs(3).X_sym_amp=0.15; trajs(3).X_anti_amp=0;     trajs(3).Y_disp=0;   trajs(3).vmax_X=1.5; trajs(3).amax_X=20.0; trajs(3).vmax_Y=1.00; trajs(3).amax_Y=20.0; trajs(3).jerkTime=0.030; trajs(3).ms_modes={'common'};
trajs(4).id='T4_X_antisym_Y020';      trajs(4).split='train'; trajs(4).seed_offset=0;    trajs(4).Y_initial=0.2; trajs(4).X_sym_amp=0;    trajs(4).X_anti_amp=0.030; trajs(4).Y_disp=0;   trajs(4).vmax_X=0.5; trajs(4).amax_X=8.0;  trajs(4).vmax_Y=1.00; trajs(4).amax_Y=20.0; trajs(4).jerkTime=0.040; trajs(4).ms_modes={'diff'};
trajs(5).id='T5_X_sym_Y_sweep';       trajs(5).split='train'; trajs(5).seed_offset=0;    trajs(5).Y_initial=0.2; trajs(5).X_sym_amp=0.10; trajs(5).X_anti_amp=0;     trajs(5).Y_disp=0.4; trajs(5).vmax_X=1.0; trajs(5).amax_X=15.0; trajs(5).vmax_Y=1.00; trajs(5).amax_Y=20.0; trajs(5).jerkTime=0.035; trajs(5).ms_modes={'common','y'};
trajs(6).id='T6_Y_sweep_aggressive';  trajs(6).split='train'; trajs(6).seed_offset=0;    trajs(6).Y_initial=0.3; trajs(6).X_sym_amp=0;    trajs(6).X_anti_amp=0;     trajs(6).Y_disp=0.6; trajs(6).vmax_X=0;   trajs(6).amax_X=0;    trajs(6).vmax_Y=1.80; trajs(6).amax_Y=42.0; trajs(6).jerkTime=0.025; trajs(6).ms_modes={'y'};
trajs(7).id='T7_X_antisym_Y_sweep';   trajs(7).split='train'; trajs(7).seed_offset=0;    trajs(7).Y_initial=0.3; trajs(7).X_sym_amp=0;    trajs(7).X_anti_amp=0.030; trajs(7).Y_disp=0.6; trajs(7).vmax_X=0.5; trajs(7).amax_X=8.0;  trajs(7).vmax_Y=1.50; trajs(7).amax_Y=20.0; trajs(7).jerkTime=0.040; trajs(7).ms_modes={'y','diff'};
trajs(8).id='T8_X_sym_anti_Y_sweep';  trajs(8).split='train'; trajs(8).seed_offset=0;    trajs(8).Y_initial=0.2; trajs(8).X_sym_amp=0.10; trajs(8).X_anti_amp=0.020; trajs(8).Y_disp=0.4; trajs(8).vmax_X=1.0; trajs(8).amax_X=8.0;  trajs(8).vmax_Y=1.20; trajs(8).amax_Y=12.0; trajs(8).jerkTime=0.035; trajs(8).ms_modes={'y','common','diff'};
% V1: validation — X symmetric + partial Y sweep. Y_initial=0.25 not covered by
%   any training trajectory (interpolation holdout between T2 at Y=0.3 and T5 at Y=0.2).
trajs(9).id='V1_X_sym_Y_mid_sweep';   trajs(9).split='val';   trajs(9).seed_offset=1000; trajs(9).Y_initial=0.25; trajs(9).X_sym_amp=0.075; trajs(9).X_anti_amp=0;     trajs(9).Y_disp=0.30; trajs(9).vmax_X=0.8; trajs(9).amax_X=12.0; trajs(9).vmax_Y=0.90; trajs(9).amax_Y=14.0; trajs(9).jerkTime=0.040; trajs(9).ms_modes={'y','common'};
% E1: test — X symmetric + X anti-symmetric + Y sweep. Y_initial=0.10 (different
%   Y region from all training trajectories; coupled holdout related to T8).
trajs(10).id='E1_X_sym_anti_Y_low_offset_sweep'; trajs(10).split='test'; trajs(10).seed_offset=2000; trajs(10).Y_initial=0.10; trajs(10).X_sym_amp=0.060; trajs(10).X_anti_amp=0.015; trajs(10).Y_disp=0.25; trajs(10).vmax_X=0.7; trajs(10).amax_X=10.0; trajs(10).vmax_Y=0.80; trajs(10).amax_Y=10.0; trajs(10).jerkTime=0.045; trajs(10).ms_modes={'y','common','diff'};

% ── Output directory ──────────────────────────────────────────────────────
out_dir = fullfile(fileparts(mfilename('fullpath')),'..','Matlab-output','identification-trajectories');
if ~exist(out_dir,'dir'), mkdir(out_dir); end

% ── Main loop ─────────────────────────────────────────────────────────────
for i = 1:numel(trajs)
    sp = trajs(i);
    fprintf('\n=== %d/%d  %s  [%s] ===\n', i, numel(trajs), sp.id, sp.split);

    % Controller frozen at Y_initial
    Y_op = sp.Y_initial;
    M_op = [m1+m2+mb+mh,            (m1-m2)*Lb/2-mh*Y_op,                    0;
            (m1-m2)*Lb/2-mh*Y_op,   Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y_op^2, -mh*d;
            0,                       -mh*d,                                     mh];
    sys = P.' * getss(n, M_op, C_damp, K) * P;
    Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
    for j = 1:3, Cfb(j,j) = ruleOfThumb(fbw, sys(j,j), ts); end

    % Reference trajectory padded to integer periods
    % THEORY: leakage-free condition — N must be multiple of N_period (P&S Ch.2 §2.2.3)
    [r_traj, t_traj] = make_ref(sp, n_hold, ts);
    [r_traj, t_traj] = pad_to_periods(r_traj, ts, N_period);
    validate_ref(r_traj, t_traj, sp.id, lim);   % acceleration checked here on r (exact)
    N = size(r_traj, 1);

    % Multisine force signal: all active modes summed simultaneously
    % THEORY: Schroeder 1970 phases minimise crest factor per mode
    % THEORY: odd harmonics — P&S Ch.4 §4.3.2
    % HEURISTIC: seed per mode gives low cross-correlation between modes
    % HEURISTIC: combined per-actuator RMS constraint — uncorrelated modes add in
    %   quadrature: FXj_rms = sqrt(sum_m (f_vec_m(j)*A_m)^2). Scale all modes by
    %   a single factor so the most loaded actuator stays within A_max budget.
    %   A_max already encodes the 40% headroom factor from diagnostics_system.m.
    n_modes = numel(sp.ms_modes);
    amp_vec = zeros(1, n_modes);
    fv_mat  = zeros(3, n_modes);   % rows=actuators [FX1;FX2;FY], cols=modes
    for m = 1:n_modes
        md = mode_def.(sp.ms_modes{m});
        amp_vec(m)  = A_max(md.A_idx);
        fv_mat(:,m) = abs(md.f_vec') * amp_vec(m);
    end
    actuator_rms = sqrt(sum(fv_mat.^2, 2));        % combined injection RMS per actuator
    scale = min(1, min(A_max' ./ actuator_rms));   % =1 if within budget; <1 if overloaded
    f_sim = zeros(N, 3);
    for m = 1:n_modes
        md  = mode_def.(sp.ms_modes{m});
        sig = multisine_schroeder(N, N_period, fs, f_low, f_high, m + sp.seed_offset);
        sig = sig * (amp_vec(m) * scale / rms(sig));
        f_sim = f_sim + sig * md.f_vec;
    end
    if scale < 1
        fprintf('  Injection scaled by %.3f — combined actuator load exceeded A_max.\n', scale);
    end

    % Simulation
    r = r_traj;  t = t_traj;  f = f_sim;  Y = Y_op;
    fprintf('  Simulating %.2f s (%d samples)...\n', t_traj(end), N);
    sim(mdl, t_traj(end));

    % Reconstruct u_total
    [t_sim, r_sim, u_q1] = reconstruct(q1, r_traj, t_traj, Cfb);
    f_sim_out = resample_to(f_sim, t_traj, t_sim);
    u_total   = u_q1 + f_sim_out;

    % Validate — skip trajectory if any limit violated
    if ~validate_response(q1, fs, lim) || ~validate_forces(u_total, lim)
        warning('%s: validation failed — skipping.', sp.id);
        continue
    end

    % Save
    Y_trajectory = q1(:,3);
    split = sp.split;
    save(fullfile(out_dir,[sp.id,'.mat']), ...
         't_sim','fs','r_sim','f_sim_out','u_q1','u_total','q1','Y_trajectory','split');
    fprintf('  Saved: %s.mat\n', sp.id);
end

% ════════════════════════════════════════════════════════════════════════
% Local functions
% ════════════════════════════════════════════════════════════════════════

function [r, t] = make_ref(sp, n_hold, ts)
% Build [X1, X2, Y] reference (N×3) from trajectory parameters.
    r = repmat([0, 0, sp.Y_initial], n_hold, 1);
    pv_sym = []; pv_anti = []; n_sym = 0; n_anti = 0;
    if sp.X_sym_amp  > 0, pv_sym  = sp1d(sp.X_sym_amp,  sp.vmax_X, sp.amax_X, sp.jerkTime, ts); n_sym  = numel(pv_sym);  end
    if sp.X_anti_amp > 0, pv_anti = sp1d(sp.X_anti_amp, sp.vmax_X, sp.amax_X, sp.jerkTime, ts); n_anti = numel(pv_anti); end
    n_move_X = max(n_sym, n_anti);
    n_move_Y = 0; pv_Y = [];
    if sp.Y_disp > 0, pv_Y = sp1d(sp.Y_disp, sp.vmax_Y, sp.amax_Y, sp.jerkTime, ts); n_move_Y = numel(pv_Y); end
    n_main = max(n_move_X, n_move_Y);
    if n_main > 0
        X1 = zeros(n_main,1); X2 = zeros(n_main,1); Y = sp.Y_initial*ones(n_main,1);
        if n_move_X > 0
            pv_sym  = pvpad(pv_sym,  n_sym,  n_main);
            pv_anti = pvpad(pv_anti, n_anti, n_main);
            X1 = pv_sym + pv_anti;  X2 = pv_sym - pv_anti;
        end
        if n_move_Y > 0
            Y = sp.Y_initial - pvpad(pv_Y, n_move_Y, n_main);
        end
        r = [r; [X1, X2, Y]];
    end
    r = [r; repmat(r(end,:), n_hold, 1)];
    t = ts * (0:size(r,1)-1)';
end

function v = pvpad(v, n_src, n_tgt)
    if isempty(v), v = zeros(n_tgt,1); return; end
    v = [v; v(end)*ones(n_src-numel(v),1)]; v = [v; v(end)*ones(n_tgt-n_src,1)];
end

function pv = sp1d(dist, vmax, amax, jerkTime, ts)
    pv = thirdOrderSetpointETEL(dist, vmax, amax, amax/jerkTime, Inf, ts); pv = pv(:,1);
end

function [r_pad, t_pad] = pad_to_periods(r, ts, N_period)
% Pad final hold to integer number of 1 s periods — leakage-free (P&S Ch.2 §2.2.3)
    N     = size(r,1);
    N_tgt = max(2, ceil(N/N_period)) * N_period;
    r_pad = [r; repmat(r(end,:), N_tgt-N, 1)];
    t_pad = ts * (0:size(r_pad,1)-1)';
end

function validate_ref(r, t, id, lim)
% Assert reference within hardware limits. Acceleration checked here on r
% (piecewise polynomial — derivative exact via finite difference on r).
% This is the ONLY place acceleration is validated. Never check acceleration on q1.
    ts   = t(2)-t(1);
    vel  = diff(r)/ts;
    acc  = diff(vel)/ts;
    assert(max(abs(r(:,1)))          <= lim.pos_X,  '%s: X1 position limit', id);
    assert(max(abs(r(:,2)))          <= lim.pos_X,  '%s: X2 position limit', id);
    assert(max(r(:,3))               <=  lim.pos_Y, '%s: Y+ position limit', id);
    assert(min(r(:,3))               >= -lim.pos_Y, '%s: Y- position limit', id);
    assert(max(abs(r(:,1)-r(:,2)))   <= lim.diff,   '%s: yaw |X1-X2| limit', id);
    assert(max(abs(vel(:,1:2)),[],'all') <= lim.vel,   '%s: X velocity limit', id);
    assert(max(abs(vel(:,3)))        <= lim.vel,    '%s: Y velocity limit', id);
    assert(max(abs(acc(:,1:2)),[],'all') <= lim.acc_X, '%s: X acceleration limit', id);
    assert(max(abs(acc(:,3)))        <= lim.acc_Y,  '%s: Y acceleration limit', id);
    fprintf('  r OK: X1=[%+.0f %+.0f] X2=[%+.0f %+.0f] Y=[%+.0f %+.0f] mm\n', ...
            min(r(:,1))*1e3,max(r(:,1))*1e3,min(r(:,2))*1e3,max(r(:,2))*1e3, ...
            min(r(:,3))*1e3,max(r(:,3))*1e3);
end

function [t_sim, r_sim, u_q1] = reconstruct(q1, r, t, Cfb)
% u_q1 = Cfb*(r-q1). Handles variable-step Simulink output via interpolation.
    Ns = size(q1,1);
    if Ns ~= numel(t), t_sim = linspace(0,t(end),Ns)'; r_sim = interp1(t,r,t_sim);
    else,              t_sim = t;                        r_sim = r; end
    u_q1 = lsim(ss(Cfb), r_sim - q1, t_sim);
end

function y_out = resample_to(y, t_src, t_tgt)
    if size(y,1) == numel(t_tgt), y_out = y; return; end
    y_out = interp1(t_src, y, t_tgt, 'linear', 'extrap');
end

function ok = validate_response(q1, fs, lim)
% Position and velocity on q1. Acceleration NOT checked — see header.
    vel = diff(q1)*fs;
    ok  =   max(abs(q1(:,1)))          <= lim.pos_X ...
         && max(abs(q1(:,2)))          <= lim.pos_X ...
         && max(abs(q1(:,3)))          <= lim.pos_Y ...
         && max(abs(q1(:,1)-q1(:,2))) <= lim.diff  ...
         && max(abs(vel(:,1)))         <= lim.vel   ...
         && max(abs(vel(:,2)))         <= lim.vel   ...
         && max(abs(vel(:,3)))         <= lim.vel;
    if ~ok, fprintf('  Response validation failed.\n'); end
end

function ok = validate_forces(u_total, lim)
% Peak and RMS of total actuator force u_total = u_q1 + f_sim vs TELICA limits.
    ok =   all(max(abs(u_total)) <= lim.force_peak) ...
        && all(rms(u_total)      <= lim.force_rms);
    if ~ok
        fprintf('  Force validation failed: peak=[%.0f %.0f %.0f] N  RMS=[%.0f %.0f %.0f] N\n', ...
                max(abs(u_total)), rms(u_total));
    end
end

function sig = multisine_schroeder(N, N_period, fs, f_low, f_high, seed)
% Schroeder-phase odd-harmonic multisine, tiled to N samples (N must be multiple of N_period).
% Returns unit-RMS signal; caller scales to A_max.
%
% THEORY: phi_k = -pi*k*(k-1)/F  (Schroeder 1970) minimises crest factor
% THEORY: odd harmonics — P&S Ch.4 §4.3.2
% HEURISTIC: seed phase offset decorrelates simultaneously injected modes
    assert(mod(N,N_period)==0, 'N must be a multiple of N_period');
    f0 = fs/N_period;
    k0 = ceil(f_low/f0);  if mod(k0,2)==0, k0=k0+1; end   % first odd bin >= f_low
    k1 = floor(f_high/f0); if mod(k1,2)==0, k1=k1-1; end  % last  odd bin <= f_high
    k  = k0:2:k1;
    F  = numel(k);
    assert(F >= 7, 'Only %d odd bins in [%.1f %.1f] Hz — PE condition F>=7 not met', F, f_low, f_high);

    idx      = 1:F;
    phi      = -pi * idx .* (idx-1) / F;                    % Schroeder 1970
    phi      = phi + 2*pi*(k*f0)*(seed-1)/(7*f_high);       % HEURISTIC: per-mode offset
    t_period = (0:N_period-1)' / fs;
    one_per  = sum(cos(2*pi*t_period*(k*f0) + phi), 2);
    one_per  = one_per / rms(one_per);
    sig      = repmat(one_per, N/N_period, 1);
end
