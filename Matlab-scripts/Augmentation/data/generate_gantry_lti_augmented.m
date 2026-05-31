% generate_gantry_lti_data.m
% --------------------------
% Train and validation data for the gantry LTI baseline (Phase 1).
% Y frozen at 0.3 m, X symmetric motion only, no external excitation.
% Mirrors generate_identification_experiment_without_multisine.m (T2 profile).
%
% Saved per file (all float32 except t_sim and fs):
%   u            (T×3)  plant input  [F_X1, F_X2, F_Y] [N]       — for fit()
%   y            (T×3)  plant output [X1, X2, Y]        [m]       — for fit()
%   x_logical    (T×6)  [q_logical, qdot_logical]        [m, m/s] — encoder verify
%   r_sim        (T×3)  reference [X1_ref,X2_ref,Y_ref]  [m]       — plotting
%   Y_trajectory (T×1)  Y(t) = y(:,3)                   [m]       — scheduling check
%   t_sim        (T×1)  time vector                      [s]
%   fs           scalar sample frequency = 20000         [Hz]
%   dt           scalar sample period   = 1/20000        [s]
%   split        char   'train' or 'val'
%
% Note: q_logical derived via P^{-T}*q_stage (exact); qdot_logical via
% central finite differences on q_logical at 20 kHz (smooth for these profiles).
%
% Output: data/gantry/matlab/gantry_lti_train.mat
%         data/gantry/matlab/gantry_lti_val.mat
%
% Run from project root:
%   run('Matlab-scripts/Augmentation/generate_gantry_lti_data.m')

addpath(genpath(fullfile(pwd, 'kamtin-fp-model', '03 Simulink gantry')))

% ── Physical parameters (identical to main.m) ────────────────────────────
mb=22.8; mh=10.1; m1=10.2; m2=10.7; Jb=1.0; Jh=0.05;
cg1=14.5; cg2=20.3; cy=10; cb1=9; cb2=9;
kb1=1987.5; kb2=1987.5; Lb=0.725; Lh=0.25; d=0.1;
cc1=16.8; cc2=18.35; ccy=11.6;  % Coulomb — disabled in model but expected in workspace

C_damp = [cg1+cg2,        (cg1-cg2)*Lb/2,            0;
          (cg1-cg2)*Lb/2, cb1+cb2+(cg1+cg2)*Lb^2/4,  0;
          0,               0,                          cy];
K      = [0,0,0; 0,kb1+kb2,0; 0,0,0];
n = 3;  P = [1,1,0; Lb/2,-Lb/2,0; 0,0,1];
fs = 20e3;  ts = 1/fs;  fbw = 100;  mdl = 'gantry_2025a';
N_period = round(fs);       % 20000 samples per period (f0 = 1 Hz)
n_hold   = round(0.5/ts);  % 0.5 s hold at start and end

% ── Hardware limits (TELICA spec) ─────────────────────────────────────────
lim.pos_X      = 0.375;              % [m]
lim.pos_Y      = 0.400;              % [m]
lim.diff       = sin(0.1) * Lb;     % [m] yaw limit
lim.vel        = 2.0;               % [m/s]
lim.acc_X      = 30.0;             % [m/s^2]
lim.acc_Y      = 50.0;             % [m/s^2]
lim.force_peak = [2000, 2000, 1420]; % [N]
lim.force_rms  = [916,  916,  656];  % [N RMS]

% ── Trajectories ──────────────────────────────────────────────────────────
% Both at Y=0.3 m (design operating point), X symmetric (common mode) only.
% Train: mirrors T2 from generate_identification_experiment_without_multisine.m
% Val:   smaller amplitude and speed — same regime, independent holdout
trajs(1).id='gantry_lti_train'; trajs(1).split='train'; trajs(1).Y_initial=0.3; trajs(1).X_sym_amp=0.15; trajs(1).vmax_X=1.5; trajs(1).amax_X=20.0; trajs(1).jerkTime=0.030;
trajs(2).id='gantry_lti_val';   trajs(2).split='val';   trajs(2).Y_initial=0.3; trajs(2).X_sym_amp=0.10; trajs(2).vmax_X=1.0; trajs(2).amax_X=12.0; trajs(2).jerkTime=0.040;

% ── Output directory ──────────────────────────────────────────────────────
out_dir = fullfile(pwd, 'data', 'gantry', 'matlab');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% ── Main loop ─────────────────────────────────────────────────────────────
for i = 1:numel(trajs)
    sp = trajs(i);
    fprintf('\n=== %d/%d  %s  [%s] ===\n', i, numel(trajs), sp.id, sp.split);

    % Controller and discrete LTI frozen at Y_initial
    Y_op = sp.Y_initial;
    M_op = [m1+m2+mb+mh,            (m1-m2)*Lb/2-mh*Y_op,                    0;
            (m1-m2)*Lb/2-mh*Y_op,   Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y_op^2, -mh*d;
            0,                       -mh*d,                                     mh];
    sys = P.' * getss(n, M_op, C_damp, K) * P;
    Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
    for j = 1:3, Cfb(j,j) = ruleOfThumb(fbw, sys(j,j), ts); end
    G = c2d(sys, ts, 'zoh');  % required by Simulink LTI blocks before sim()

    % Reference and padding
    [r_traj, t_traj] = make_ref(sp, n_hold, ts);
    [r_traj, t_traj] = pad_to_periods(r_traj, ts, N_period);
    validate_ref(r_traj, t_traj, sp.id, lim);
    N = size(r_traj, 1);

    % Simulation (no external injection)
    r = r_traj;  t = t_traj;  f = zeros(N,3);  Y = Y_op;
    fprintf('  Simulating %.2f s (%d samples)...\n', t_traj(end), N);
    sim(mdl, t_traj(end));

    % Reconstruct plant input: u_total = Cfb*(r - q1)  (f=0 so u_total=u_q1)
    [t_sim, r_sim, u_q1] = reconstruct(q1, r_traj, t_traj, Cfb);
    u_total = u_q1;  % f_sim = 0 throughout

    if ~validate_response(q1, fs, lim) || ~validate_forces(u_total, lim)
        warning('%s: validation failed — skipping.', sp.id);
        continue
    end

    % Derive logical state from stage positions
    % q_stage = P^T @ q_logical  =>  q_logical = P^{-T} @ q_stage  (exact)
    q_logical = ((P') \ q1')';          % (T×3)
    qdot_logical = zeros(size(q_logical));
    for j = 1:3
        qdot_logical(:,j) = gradient(q_logical(:,j), ts);  % central finite diff
    end
    x_logical = [q_logical, qdot_logical];  % (T×6)

    % Save
    u            = single(u_total);       % (T×3) plant forces  — training
    y            = single(q1);            % (T×3) stage positions — training
    x_logical    = single(x_logical);     % (T×6) full state     — encoder verify
    r_sim        = single(r_sim);         % (T×3) reference       — plotting
    Y_trajectory = single(q1(:,3));       % (T×1) Y(t)
    dt           = single(ts);            % scalar
    split        = sp.split;

    out_path = fullfile(out_dir, [sp.id, '.mat']);
    save(out_path, 'u','y','x_logical','r_sim','Y_trajectory','t_sim','fs','dt','split');
    fprintf('  Saved: %s  (%d samples, %.2f s,  Y in [%.4f, %.4f] m)\n', ...
            out_path, size(q1,1), t_sim(end), min(Y_trajectory), max(Y_trajectory));
end
fprintf('\nDone.\n');

% ── Plots ─────────────────────────────────────────────────────────────────
% Layout: 6 rows x 2 columns.  Left = train, right = val.
% Rows 1-3: positions X1, X2, Y.  Rows 4-6: forces F_X1, F_X2, F_Y.
figure('Name','Gantry LTI data — motion profiles','Position',[100 50 900 800]);
col_titles = {'Train','Val'};
row_labels = {'X1 [m]','X2 [m]','Y [m]','F_{X1} [N]','F_{X2} [N]','F_Y [N]'};

for i = 1:numel(trajs)
    d = load(fullfile(out_dir, [trajs(i).id, '.mat']));
    t = double(d.t_sim);
    signals = [double(d.y), double(d.u)];  % (T x 6): positions then forces

    for j = 1:6
        subplot(6, 2, (j-1)*2 + i);
        plot(t, signals(:,j));
        ylabel(row_labels{j});
        grid on;
        if j == 1,  title(col_titles{i}); end
        if j == 6,  xlabel('Time [s]');   end
    end
end

% ════════════════════════════════════════════════════════════════════════════
% Local functions (mirrors generate_identification_experiment_without_multisine.m)
% ════════════════════════════════════════════════════════════════════════════

function [r, t] = make_ref(sp, n_hold, ts)
    r      = repmat([0, 0, sp.Y_initial], n_hold, 1);
    pv_sym = sp1d(sp.X_sym_amp, sp.vmax_X, sp.amax_X, sp.jerkTime, ts);
    n_move = numel(pv_sym);
    r      = [r; [pv_sym, pv_sym, sp.Y_initial*ones(n_move,1)]];
    r      = [r; repmat(r(end,:), n_hold, 1)];
    t      = ts * (0:size(r,1)-1)';
end

function pv = sp1d(dist, vmax, amax, jerkTime, ts)
    pv = thirdOrderSetpointETEL(dist, vmax, amax, amax/jerkTime, Inf, ts);
    pv = pv(:,1);
end

function [r_pad, t_pad] = pad_to_periods(r, ts, N_period)
    N     = size(r,1);
    N_tgt = max(2, ceil(N/N_period)) * N_period;
    r_pad = [r; repmat(r(end,:), N_tgt-N, 1)];
    t_pad = ts * (0:size(r_pad,1)-1)';
end

function validate_ref(r, t, id, lim)
    ts  = t(2)-t(1);
    vel = diff(r)/ts;
    acc = diff(vel)/ts;
    assert(max(abs(r(:,1)))              <= lim.pos_X,  '%s: X1 pos limit', id);
    assert(max(abs(r(:,2)))              <= lim.pos_X,  '%s: X2 pos limit', id);
    assert(max(r(:,3))                   <=  lim.pos_Y, '%s: Y+ pos limit', id);
    assert(min(r(:,3))                   >= -lim.pos_Y, '%s: Y- pos limit', id);
    assert(max(abs(r(:,1)-r(:,2)))       <= lim.diff,   '%s: yaw limit',    id);
    assert(max(abs(vel(:,1:2)),[],'all') <= lim.vel,    '%s: X vel limit',  id);
    assert(max(abs(vel(:,3)))            <= lim.vel,    '%s: Y vel limit',  id);
    assert(max(abs(acc(:,1:2)),[],'all') <= lim.acc_X,  '%s: X acc limit',  id);
    assert(max(abs(acc(:,3)))            <= lim.acc_Y,  '%s: Y acc limit',  id);
    fprintf('  r OK: X1=[%+.0f %+.0f] X2=[%+.0f %+.0f] Y=[%+.0f %+.0f] mm\n', ...
            min(r(:,1))*1e3, max(r(:,1))*1e3, min(r(:,2))*1e3, max(r(:,2))*1e3, ...
            min(r(:,3))*1e3, max(r(:,3))*1e3);
end

function [t_sim, r_sim, u_q1] = reconstruct(q1, r, t, Cfb)
    Ns = size(q1,1);
    if Ns ~= numel(t)
        t_sim = linspace(0, t(end), Ns)';  r_sim = interp1(t, r, t_sim);
    else
        t_sim = t;  r_sim = r;
    end
    u_q1 = lsim(ss(Cfb), r_sim - q1, t_sim);
end

function ok = validate_response(q1, fs, lim)
    vel = diff(q1) * fs;
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
    ok =   all(max(abs(u_total)) <= lim.force_peak) ...
        && all(rms(u_total)      <= lim.force_rms);
    if ~ok
        fprintf('  Force validation failed: peak=[%.0f %.0f %.0f] N  RMS=[%.0f %.0f %.0f] N\n', ...
                max(abs(u_total)), rms(u_total));
    end
end
