% generate_gantry_lti_augmented.m
% --------------------------------
% Train and validation data for the gantry augmented system (Phase 2).
% Uses gantry_additional_state_2025a: nominal 3-DOF plant + hidden MSD on payload.
% Y frozen at 0.3 m, X symmetric motion only — same trajectories as baseline.
%
% Saved per file (all float32 except t_sim and fs):
%   u            (T×3)  plant input  [F_X1, F_X2, F_Y] [N]       — for fit()
%   y            (T×3)  plant output [X1, X2, Y]        [m]       — for fit()
%   x_logical    (T×6)  [q_logical, qdot_logical]        [m, m/s] — encoder verify (6D projection)
%   delta_a      (T×1)  hidden MSD relative displacement [m]      — diagnostic only
%   r_sim        (T×3)  reference [X1_ref,X2_ref,Y_ref]  [m]       — plotting
%   Y_trajectory (T×1)  Y(t) = y(:,3)                   [m]       — scheduling check
%   t_sim        (T×1)  time vector                      [s]
%   fs           scalar sample frequency = 20000         [Hz]
%   dt           scalar sample period   = 1/20000        [s]
%   split        char   'train' or 'val'
%
% Note on x_logical: derived from q_aug (augmented system stage positions) via
% P^{-T}*q_stage, then finite-difference velocities. This is the 6D nominal
% projection of the 8-state trajectory — not exactly the encoder output for
% mismatched data (see docs/gantry-augmentation-plan.md, Encoder verification section).
%
% Output: data/gantry/matlab/gantry_aug_train.mat
%         data/gantry/matlab/gantry_aug_val.mat
%
% Run from project root:
%   run('Matlab-scripts/Augmentation/data/generate_gantry_lti_augmented.m')

addpath(genpath(fullfile(pwd, 'kamtin-fp-model', '03 Simulink gantry')))
addpath(fullfile(pwd, 'Matlab-scripts', 'Augmentation'))

% ── Physical parameters (identical to main.m) ────────────────────────────
mb=22.8; mh=10.1; m1=10.2; m2=10.7; Jb=1.0; Jh=0.05;
cg1=14.5; cg2=20.3; cy=10; cb1=9; cb2=9;
kb1=1987.5; kb2=1987.5; Lb=0.725; Lh=0.25; d=0.1;
cc1=16.8; cc2=18.35; ccy=11.6;  % Coulomb — disabled in model but expected in workspace

% ── Hidden MSD parameters (from main_augmentation.m) ─────────────────────
% mh_total = mh_rigid + ma is conserved. Baseline uses full mh as rigid mass.
% Augmented model uses mh_rigid for the Simscape body and extended ODE.
ma_frac  = 0.10;
ma       = ma_frac * mh;           % 1.01 kg  hidden MSD mass
mh_rigid = mh - ma;               % 9.09 kg  rigid part of payload
L0       = 0.10;                   % equilibrium offset of ma in +Y direction [m]
fa       = 400;                    % target MSD natural frequency [Hz]
ka       = ma * (2*pi*fa)^2;      % MSD spring stiffness [N/m]
zeta_a   = 0.05;                   % damping ratio
ca       = 2 * zeta_a * sqrt(ka * ma);  % MSD damper coefficient [Ns/m]

C_damp = [cg1+cg2,        (cg1-cg2)*Lb/2,            0;
          (cg1-cg2)*Lb/2, cb1+cb2+(cg1+cg2)*Lb^2/4,  0;
          0,               0,                          cy];
K      = [0,0,0; 0,kb1+kb2,0; 0,0,0];
n = 3;  P = [1,1,0; Lb/2,-Lb/2,0; 0,0,1];
fs = 20e3;  ts = 1/fs;  fbw = 100;  mdl = 'gantry_additional_state_2025a';
N_period = round(fs);       % 20000 samples per period (f0 = 1 Hz)
n_hold   = round(0.5/ts);  % 0.5 s hold at start and end

% ── Hardware limits (TELICA spec) ─────────────────────────────────────────
lim.pos_X      = 0.375;
lim.pos_Y      = 0.400;
lim.diff       = sin(0.1) * Lb;
lim.vel        = 2.0;
lim.acc_X      = 30.0;
lim.acc_Y      = 50.0;
lim.force_peak = [2000, 2000, 1420];
lim.force_rms  = [916,  916,  656];

% ── Trajectories ──────────────────────────────────────────────────────────
% Same profiles as baseline — enables direct comparison baseline vs augmented.
trajs(1).id='gantry_aug_train'; trajs(1).split='train'; trajs(1).Y_initial=0.3; trajs(1).X_sym_amp=0.15; trajs(1).vmax_X=1.5; trajs(1).amax_X=20.0; trajs(1).jerkTime=0.030;
trajs(2).id='gantry_aug_val';   trajs(2).split='val';   trajs(2).Y_initial=0.3; trajs(2).X_sym_amp=0.10; trajs(2).vmax_X=1.0; trajs(2).amax_X=12.0; trajs(2).jerkTime=0.040;

% ── Output directory ──────────────────────────────────────────────────────
out_dir = fullfile(pwd, 'data', 'gantry', 'matlab');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% ── Main loop ─────────────────────────────────────────────────────────────
for i = 1:numel(trajs)
    sp = trajs(i);
    fprintf('\n=== %d/%d  %s  [%s] ===\n', i, numel(trajs), sp.id, sp.split);

    % Controller designed for nominal plant (full mh — before any swap)
    Y_op = sp.Y_initial;
    M_op = [m1+m2+mb+mh,            (m1-m2)*Lb/2-mh*Y_op,                    0;
            (m1-m2)*Lb/2-mh*Y_op,   Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y_op^2, -mh*d;
            0,                       -mh*d,                                     mh];
    sys = P.' * getss(n, M_op, C_damp, K) * P;
    Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
    for j = 1:3, Cfb(j,j) = ruleOfThumb(fbw, sys(j,j), ts); end
    G = c2d(sys, ts, 'zoh');  % must be in workspace before sim() — Simulink LTI blocks read G

    % Reference and padding
    [r_traj, t_traj] = make_ref(sp, n_hold, ts);
    [r_traj, t_traj] = pad_to_periods(r_traj, ts, N_period);
    validate_ref(r_traj, t_traj, sp.id, lim);
    N = size(r_traj, 1);

    % Swap mh to mh_rigid for augmented Simulink model, then restore after sim
    % The model reads mh from workspace for the rigid body mass (Simscape + extended ODE).
    % ma, ka, ca, L0 are already in workspace from the parameter block above.
    mh = mh_rigid;

    r = r_traj;  t = t_traj;  f = zeros(N,3);  Y = Y_op;
    fprintf('  Simulating %.2f s (%d samples)...\n', t_traj(end), N);
    sim(mdl, t_traj(end));

    mh = mh_rigid + ma;  % restore full mh for next iteration

    % Reconstruct plant input from augmented output q_aug
    % u_total = Cfb*(r - q_aug) since f=0 throughout
    [t_sim, r_sim, u_aug] = reconstruct(q_aug, r_traj, t_traj, Cfb);
    u_total = u_aug;

    if ~validate_response(q_aug, fs, lim) || ~validate_forces(u_total, lim)
        warning('%s: validation failed — skipping.', sp.id);
        continue
    end

    % Derive nominal 6D logical state from augmented stage positions
    % q_aug is the extended ODE output in stage coordinates [X1, X2, Y] (T×3).
    % q_logical = P^{-T} * q_aug_stage  (exact coordinate transform)
    % qdot_logical from central finite differences at 20 kHz
    q_logical = ((P') \ q_aug')';       % (T×3)
    qdot_logical = zeros(size(q_logical));
    for j = 1:3
        qdot_logical(:,j) = gradient(q_logical(:,j), ts);
    end
    x_logical = [q_logical, qdot_logical];  % (T×6)

    % Save
    u            = single(u_total);       % (T×3) plant forces    — training
    y            = single(q_aug);         % (T×3) stage positions — training
    x_logical    = single(x_logical);     % (T×6) 6D state projection — encoder verify
    delta_a      = single(delta_a);       % (T×1) hidden MSD displacement — diagnostic
    r_sim        = single(r_sim);         % (T×3) reference — plotting
    Y_trajectory = single(q_aug(:,3));    % (T×1) Y(t)
    dt           = single(ts);
    split        = sp.split;

    out_path = fullfile(out_dir, [sp.id, '.mat']);
    save(out_path, 'u','y','x_logical','delta_a','r_sim','Y_trajectory','t_sim','fs','dt','split');
    fprintf('  Saved: %s  (%d samples, %.2f s,  Y in [%.4f, %.4f] m,  delta_a max=%.4e m)\n', ...
            out_path, size(q_aug,1), t_sim(end), min(Y_trajectory), max(Y_trajectory), max(abs(double(delta_a))));
end
fprintf('\nDone.\n');

% ── Plots ─────────────────────────────────────────────────────────────────
% Layout: 6 rows x 2 columns.  Left = train, right = val.
% Rows 1-3: positions X1, X2, Y.  Rows 4-6: forces F_X1, F_X2, F_Y.
figure('Name','Gantry augmented data — motion profiles','Position',[100 50 900 950]);
col_titles = {'Train (aug)','Val (aug)'};
row_labels = {'X1 [m]','X2 [m]','Y [m]','F_{X1} [N]','F_{X2} [N]','F_Y [N]','\delta_a [m]'};

for i = 1:numel(trajs)
    d = load(fullfile(out_dir, [trajs(i).id, '.mat']));
    t = double(d.t_sim);
    signals = [double(d.y), double(d.u), double(d.delta_a)];  % (T x 7)

    for j = 1:7
        subplot(7, 2, (j-1)*2 + i);
        plot(t, signals(:,j));
        ylabel(row_labels{j});
        grid on;
        if j == 1,  title(col_titles{i}); end
        if j == 7,  xlabel('Time [s]');   end
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

function [t_sim, r_sim, u_aug] = reconstruct(q_aug, r, t, Cfb)
    Ns = size(q_aug,1);
    if Ns ~= numel(t)
        t_sim = linspace(0, t(end), Ns)';  r_sim = interp1(t, r, t_sim);
    else
        t_sim = t;  r_sim = r;
    end
    u_aug = lsim(ss(Cfb), r_sim - q_aug, t_sim);
end

function ok = validate_response(q_aug, fs, lim)
    vel = diff(q_aug) * fs;
    ok  =   max(abs(q_aug(:,1)))            <= lim.pos_X ...
         && max(abs(q_aug(:,2)))            <= lim.pos_X ...
         && max(abs(q_aug(:,3)))            <= lim.pos_Y ...
         && max(abs(q_aug(:,1)-q_aug(:,2))) <= lim.diff  ...
         && max(abs(vel(:,1)))              <= lim.vel   ...
         && max(abs(vel(:,2)))              <= lim.vel   ...
         && max(abs(vel(:,3)))              <= lim.vel;
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
