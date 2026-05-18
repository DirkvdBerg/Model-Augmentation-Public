% generate_identification_experiment_adjusted.m
% Selects minimum effective multisine amplitude per modal channel and plots
% final trajectories with and without multisine.
%
% Phase 1 — pass 0 for all trajectories: trajectory-only baseline.
% Phase 2 — rho sweep on single-mode trajectories only; no plots.
% Phase 3 — select minimum rho per mode where B(rho) > B_min; print table.
% Phase 4 — final pass 1 at selected rho; verify combined trajectories; plot.
%
% Run from repo root:
%   run('Matlab-scripts/generate_identification_experiment_adjusted.m')

addpath(genpath(fullfile(pwd, 'kamtin-fp-model', '03 Simulink gantry')))

% ── Design parameters ─────────────────────────────────────────────────────
% Band covers yaw resonance (~4-5 Hz across Y workspace) through controller bandwidth.
% Yaw resonance: f = (1/2pi)*sqrt((kb1+kb2)/M(Y)(2,2)), M(Y)(2,2) ~ 3.9-5.1 kg·m².
f_low   = 1;                        % [Hz]
f_high  = 100;                      % [Hz] — controller bandwidth
rho_vec = [0.01, 0.02, 0.05, 0.10]; % candidates swept in phase 2
B_min   = 1e-5;                     % [m] minimum detectable perturbation — positioning repeatability floor (10 µm)

% ── Physical parameters ───────────────────────────────────────────────────
mb=22.8; mh=10.1; m1=10.2; m2=10.7; Jb=1.0; Jh=0.05;
cg1=14.5; cg2=20.3; cy=10; cb1=9; cb2=9;
kb1=1987.5; kb2=1987.5; Lb=0.725; Lh=0.25; d=0.1;

C_damp = [cg1+cg2,        (cg1-cg2)*Lb/2,           0;
          (cg1-cg2)*Lb/2, cb1+cb2+(cg1+cg2)*Lb^2/4, 0;
          0,              0,                          cy];
K  = [0,0,0; 0,kb1+kb2,0; 0,0,0];
n  = 3;  P = [1,1,0; Lb/2,-Lb/2,0; 0,0,1];
fs = 20e3;  ts = 1/fs;  fbw = 100;  mdl = 'gantry_2025a';
N_period = round(fs);      % 20000 samples = T_p = 1 s
n_hold   = round(0.5/ts); % 0.5 s settle hold at trajectory start and end

% ── Hardware limits (TELICA spec) ─────────────────────────────────────────
lim.pos_X      = 0.375;              % [m]
lim.pos_Y      = 0.400;              % [m]
lim.diff       = sin(0.1) * Lb;     % [m] max |X1-X2|
lim.vel        = 2.0;               % [m/s]
lim.acc_X      = 30.0;             % [m/s²] — checked on r only
lim.acc_Y      = 50.0;             % [m/s²] — checked on r only
lim.force_peak = [2000, 2000, 1420]; % [N] peak [FX1,FX2,FY]
lim.force_rms  = [916,  916,  656];  % [N] RMS  [FX1,FX2,FY]

% ── Mode definitions ─────────────────────────────────────────────────────
mode_def.common = struct('f_vec', [1,  1, 0]);
mode_def.diff   = struct('f_vec', [1, -1, 0]);
mode_def.y      = struct('f_vec', [0,  0, 1]);

% ── Trajectory definitions (T1-T8 train, V1 val, E1 test) ────────────────
% seed_offset: shifts Schroeder seed so val/test spectral lines differ from train.
% Single-mode trajectories (used for rho selection): T1,T2,T3,T4,T6.
% Combined trajectories (verification only): T5,T7,T8,V1,E1.
trajs(1).id='T1_Y_sweep_conservative'; trajs(1).split='train'; trajs(1).seed_offset=0;    trajs(1).Y_initial=0.30; trajs(1).X_sym_amp=0;     trajs(1).X_anti_amp=0;     trajs(1).Y_disp=0.60; trajs(1).vmax_X=0;   trajs(1).amax_X=0;    trajs(1).vmax_Y=1.00; trajs(1).amax_Y=10.0; trajs(1).jerkTime=0.050; trajs(1).ms_modes={'y'};
trajs(2).id='T2_X_sym_Y030';          trajs(2).split='train'; trajs(2).seed_offset=0;    trajs(2).Y_initial=0.30; trajs(2).X_sym_amp=0.15;  trajs(2).X_anti_amp=0;     trajs(2).Y_disp=0;    trajs(2).vmax_X=1.5; trajs(2).amax_X=20.0; trajs(2).vmax_Y=1.00; trajs(2).amax_Y=20.0; trajs(2).jerkTime=0.030; trajs(2).ms_modes={'common'};
trajs(3).id='T3_X_sym_Y000';          trajs(3).split='train'; trajs(3).seed_offset=0;    trajs(3).Y_initial=0.00; trajs(3).X_sym_amp=0.15;  trajs(3).X_anti_amp=0;     trajs(3).Y_disp=0;    trajs(3).vmax_X=1.5; trajs(3).amax_X=20.0; trajs(3).vmax_Y=1.00; trajs(3).amax_Y=20.0; trajs(3).jerkTime=0.030; trajs(3).ms_modes={'common'};
trajs(4).id='T4_X_antisym_Y020';      trajs(4).split='train'; trajs(4).seed_offset=0;    trajs(4).Y_initial=0.20; trajs(4).X_sym_amp=0;     trajs(4).X_anti_amp=0.030; trajs(4).Y_disp=0;    trajs(4).vmax_X=0.5; trajs(4).amax_X=8.0;  trajs(4).vmax_Y=1.00; trajs(4).amax_Y=20.0; trajs(4).jerkTime=0.040; trajs(4).ms_modes={'diff'};
trajs(5).id='T5_X_sym_Y_sweep';       trajs(5).split='train'; trajs(5).seed_offset=0;    trajs(5).Y_initial=0.20; trajs(5).X_sym_amp=0.10;  trajs(5).X_anti_amp=0;     trajs(5).Y_disp=0.40; trajs(5).vmax_X=1.0; trajs(5).amax_X=15.0; trajs(5).vmax_Y=1.00; trajs(5).amax_Y=20.0; trajs(5).jerkTime=0.035; trajs(5).ms_modes={'common','y'};
trajs(6).id='T6_Y_sweep_aggressive';  trajs(6).split='train'; trajs(6).seed_offset=0;    trajs(6).Y_initial=0.30; trajs(6).X_sym_amp=0;     trajs(6).X_anti_amp=0;     trajs(6).Y_disp=0.60; trajs(6).vmax_X=0;   trajs(6).amax_X=0;    trajs(6).vmax_Y=1.80; trajs(6).amax_Y=42.0; trajs(6).jerkTime=0.025; trajs(6).ms_modes={'y'};
trajs(7).id='T7_X_antisym_Y_sweep';   trajs(7).split='train'; trajs(7).seed_offset=0;    trajs(7).Y_initial=0.30; trajs(7).X_sym_amp=0;     trajs(7).X_anti_amp=0.030; trajs(7).Y_disp=0.60; trajs(7).vmax_X=0.5; trajs(7).amax_X=8.0;  trajs(7).vmax_Y=1.50; trajs(7).amax_Y=20.0; trajs(7).jerkTime=0.040; trajs(7).ms_modes={'y','diff'};
trajs(8).id='T8_X_sym_anti_Y_sweep';  trajs(8).split='train'; trajs(8).seed_offset=0;    trajs(8).Y_initial=0.20; trajs(8).X_sym_amp=0.10;  trajs(8).X_anti_amp=0.020; trajs(8).Y_disp=0.40; trajs(8).vmax_X=1.0; trajs(8).amax_X=8.0;  trajs(8).vmax_Y=1.20; trajs(8).amax_Y=12.0; trajs(8).jerkTime=0.035; trajs(8).ms_modes={'y','common','diff'};
trajs(9).id='V1_X_sym_Y_mid_sweep';   trajs(9).split='val';   trajs(9).seed_offset=1000; trajs(9).Y_initial=0.25; trajs(9).X_sym_amp=0.075; trajs(9).X_anti_amp=0;     trajs(9).Y_disp=0.30; trajs(9).vmax_X=0.8; trajs(9).amax_X=12.0; trajs(9).vmax_Y=0.90; trajs(9).amax_Y=14.0; trajs(9).jerkTime=0.040; trajs(9).ms_modes={'y','common'};
trajs(10).id='E1_X_sym_anti_Y_low_offset_sweep'; trajs(10).split='test'; trajs(10).seed_offset=2000; trajs(10).Y_initial=0.10; trajs(10).X_sym_amp=0.060; trajs(10).X_anti_amp=0.015; trajs(10).Y_disp=0.25; trajs(10).vmax_X=0.7; trajs(10).amax_X=10.0; trajs(10).vmax_Y=0.80; trajs(10).amax_Y=10.0; trajs(10).jerkTime=0.045; trajs(10).ms_modes={'y','common','diff'};

n_traj  = numel(trajs);
n_rho   = numel(rho_vec);
out_dir = fullfile(fileparts(mfilename('fullpath')),'..','Matlab-output','identification-trajectories');
if ~exist(out_dir,'dir'), mkdir(out_dir); end

% ════════════════════════════════════════════════════════════════════════
% Phase 1 — trajectory-only baseline for all trajectories
% ════════════════════════════════════════════════════════════════════════
fprintf('=== Phase 1: trajectory-only baselines (%d trajectories) ===\n', n_traj);
td = struct();
for i = 1:n_traj
    sp   = trajs(i);
    Y_op = sp.Y_initial;
    M_op = [m1+m2+mb+mh,            (m1-m2)*Lb/2-mh*Y_op,                    0;
            (m1-m2)*Lb/2-mh*Y_op,   Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y_op^2, -mh*d;
            0,                       -mh*d,                                     mh];
    sys = P.' * getss(n, M_op, C_damp, K) * P;
    Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
    for j = 1:3, Cfb(j,j) = ruleOfThumb(fbw, sys(j,j), ts); end

    [r_traj, t_traj] = make_ref(sp, n_hold, ts);
    [r_traj, t_traj] = pad_to_periods(r_traj, ts, N_period);
    validate_ref(r_traj, t_traj, sp.id, lim);
    N = size(r_traj, 1);

    r = r_traj;  t = t_traj;  f = zeros(N, 3);  Y = Y_op;
    sim(mdl, t_traj(end));
    [t_sim0, ~, u_traj_only] = reconstruct(q1, r_traj, t_traj, Cfb);

    td(i).r_traj      = r_traj;
    td(i).t_traj      = t_traj;
    td(i).N           = N;
    td(i).Cfb         = Cfb;
    td(i).Y_op        = Y_op;
    td(i).q_nom_raw   = q1;
    td(i).u_traj_only = u_traj_only;
    td(i).t_sim0      = t_sim0;
end

% ════════════════════════════════════════════════════════════════════════
% Phase 2 — rho sweep on single-mode trajectories (no plots)
% ════════════════════════════════════════════════════════════════════════
fprintf('\n=== Phase 2: rho sweep on single-mode trajectories ===\n');
B_sweep     = nan(n_traj, n_rho);   % B(rho) [m]; NaN for combined trajectories
hw_ok_sweep = true(n_traj, n_rho);  % hardware limit flag

for i = 1:n_traj
    sp = trajs(i);
    if numel(sp.ms_modes) ~= 1, continue; end   % combined trajectories skipped here
    md = mode_def.(sp.ms_modes{1});

    for r_idx = 1:n_rho
        rho = rho_vec(r_idx);
        % Amplitude: rho fraction of trajectory modal force RMS.
        % Divide by norm(f_vec)^2 so injected modal force = rho * rms(u_traj_modal).
        % For common/diff: norm^2=2; for y: norm^2=1.
        amp = rho * rms(td(i).u_traj_only * md.f_vec') / (md.f_vec * md.f_vec');
        sig = multisine_schroeder(td(i).N, N_period, fs, f_low, f_high, 1 + sp.seed_offset);
        f_sim = sig * (amp / rms(sig)) * md.f_vec;   % N×3

        r = td(i).r_traj;  t = td(i).t_traj;  f = f_sim;  Y = td(i).Y_op;
        sim(mdl, td(i).t_traj(end));
        [t_sim, ~, u_q1] = reconstruct(q1, td(i).r_traj, td(i).t_traj, td(i).Cfb);
        f_sim_out = resample_to(f_sim, td(i).t_traj, t_sim);
        u_total   = u_q1 + f_sim_out;
        q_ms      = q1;
        q_nom     = resample_to(td(i).q_nom_raw, td(i).t_sim0, t_sim);

        B_sweep(i, r_idx)     = rms((q_ms - q_nom) * md.f_vec');
        hw_ok_sweep(i, r_idx) = all(max(abs(u_total)) <= lim.force_peak) && ...
                                 all(rms(u_total)      <= lim.force_rms);
    end
end

% ════════════════════════════════════════════════════════════════════════
% Phase 3 — select minimum rho per mode; print selection table
% ════════════════════════════════════════════════════════════════════════
mode_names   = {'common', 'diff', 'y'};
rho_selected = struct('common', NaN, 'diff', NaN, 'y', NaN);

fprintf('\n=== Phase 3: rho selection (B_min = %.2e m) ===\n', B_min);
fprintf('%-8s  %-6s  %s\n', 'mode', 'rho', 'B [m] per trajectory');
fprintf('%s\n', repmat('-', 1, 60));

for k = 1:numel(mode_names)
    mn    = mode_names{k};
    idx_m = find(arrayfun(@(ii) numel(trajs(ii).ms_modes)==1 && ...
                 strcmp(trajs(ii).ms_modes{1}, mn), 1:n_traj));

    if isempty(idx_m)
        fprintf('%-8s  no single-mode trajectories found\n', mn);
        continue;
    end

    % Find minimum rho satisfying B_min for all single-mode trajectories of this mode
    for r_idx = 1:n_rho
        if all(B_sweep(idx_m, r_idx) > B_min) && all(hw_ok_sweep(idx_m, r_idx))
            rho_selected.(mn) = rho_vec(r_idx);
            break;
        end
    end

    if isnan(rho_selected.(mn))
        rho_selected.(mn) = rho_vec(end);
        fprintf('%-8s  %-6.2f  [WARN: B_min not achieved]  ', mn, rho_selected.(mn));
    else
        fprintf('%-8s  %-6.2f  ', mn, rho_selected.(mn));
    end

    r_sel = find(rho_vec == rho_selected.(mn), 1);
    for ii = idx_m
        fprintf('%s:%.2e  ', trajs(ii).id, B_sweep(ii, r_sel));
    end
    fprintf('\n');
end

% ════════════════════════════════════════════════════════════════════════
% Phase 4 — final simulation, combined-trajectory verification, plots
% ════════════════════════════════════════════════════════════════════════
fprintf('\n=== Phase 4: final simulation and plots ===\n');
fprintf('\n%-42s  %-6s  %s\n', 'Combined trajectory', 'hw', 'B [m] per mode');
fprintf('%s\n', repmat('-', 1, 70));

for i = 1:n_traj
    sp = trajs(i);

    % Build f_sim using per-mode selected rho
    f_sim = zeros(td(i).N, 3);
    for m = 1:numel(sp.ms_modes)
        md    = mode_def.(sp.ms_modes{m});
        rho_m = rho_selected.(sp.ms_modes{m});
        amp   = rho_m * rms(td(i).u_traj_only * md.f_vec') / (md.f_vec * md.f_vec');
        sig   = multisine_schroeder(td(i).N, N_period, fs, f_low, f_high, m + sp.seed_offset);
        f_sim = f_sim + sig * (amp / rms(sig)) * md.f_vec;
    end

    r = td(i).r_traj;  t = td(i).t_traj;  f = f_sim;  Y = td(i).Y_op;
    sim(mdl, td(i).t_traj(end));
    [t_sim, r_sim, u_q1] = reconstruct(q1, td(i).r_traj, td(i).t_traj, td(i).Cfb);
    f_sim     = resample_to(f_sim, td(i).t_traj, t_sim);   % rename: precompute.py reads 'f_sim'
    u_total   = u_q1 + f_sim;
    q_ms      = q1;
    q_nom     = resample_to(td(i).q_nom_raw, td(i).t_sim0, t_sim);

    % Save — variable names must match precompute.py: mat['q1'], mat.get('f_sim'), mat['u_q1']
    q1           = q_ms;          % precompute.py reads 'q1'
    Y_trajectory = q1(:,3);
    split        = sp.split;
    save(fullfile(out_dir, [sp.id, '.mat']), ...
         't_sim', 'fs', 'r_sim', 'f_sim', 'u_q1', 'u_total', 'q1', 'Y_trajectory', 'split');
    fprintf('  Saved: %s.mat\n', sp.id);

    % Verification: combined trajectories only
    if numel(sp.ms_modes) > 1
        hw_ok  = all(max(abs(u_total)) <= lim.force_peak) && all(rms(u_total) <= lim.force_rms);
        hw_str = 'OK'; if ~hw_ok, hw_str = 'WARN'; end
        fprintf('%-42s  %-6s  ', sp.id, hw_str);
        for m = 1:numel(sp.ms_modes)
            md  = mode_def.(sp.ms_modes{m});
            B   = rms((q_ms - q_nom) * md.f_vec');
            tag = 'OK'; if B < B_min, tag = 'WARN'; end
            fprintf('B(%s)=%.2e[%s]  ', sp.ms_modes{m}, B, tag);
        end
        fprintf('\n');
    end

    rho_per_mode = cellfun(@(mn) rho_selected.(mn), sp.ms_modes);
    plot_results(t_sim, q_nom, q_ms, td(i).u_traj_only, u_total, sp, rho_per_mode, mode_def, fs, N_period);
end

% ════════════════════════════════════════════════════════════════════════
% Local functions
% ════════════════════════════════════════════════════════════════════════

function plot_results(t_sim, q_nom, q_ms, u_traj_only, u_total, sp, rho_per_mode, mode_def, fs, N_period)
% One figure per trajectory:
%   Row 1 — position overlay q_nom vs q_ms [m]
%   Row 2 — perturbation q_ms - q_nom [m]
%   Row 3 — actuator forces u_traj_only vs u_total [N]
%   Row 4+ — force PSD per active mode (one full-width row each)
    n_modes = numel(sp.ms_modes);
    n_rows  = 3 + n_modes;
    figure('Name', sprintf('%s', sp.id), 'NumberTitle', 'off');

    ax_lbl = {'X1 [m]', 'X2 [m]', 'Y [m]'};

    % Build title: one rho label per active mode
    rho_strs = arrayfun(@(k) sprintf('\\rho_{%s}=%.2f', sp.ms_modes{k}, rho_per_mode(k)), ...
                        1:n_modes, 'UniformOutput', false);
    rho_title = strjoin(rho_strs, '   ');

    % Row 1: position overlay
    for j = 1:3
        subplot(n_rows, 3, j);
        plot(t_sim, q_nom(:,j), 'b', t_sim, q_ms(:,j), 'r--', 'LineWidth', 0.8);
        ylabel(ax_lbl{j}); grid on; box off;
        if j == 1, legend('traj-only', 'traj+ms', 'Location', 'best'); end
        if j == 2, title(sprintf('%s\n%s', strrep(sp.id,'_','\_'), rho_title)); end
    end

    % Row 2: perturbation
    for j = 1:3
        subplot(n_rows, 3, 3 + j);
        plot(t_sim, q_ms(:,j) - q_nom(:,j), 'k', 'LineWidth', 0.8);
        ylabel(ax_lbl{j}); xlabel('time [s]'); grid on; box off;
    end

    % Row 3: actuator forces
    f_lbl = {'FX1 [N]', 'FX2 [N]', 'FY [N]'};
    for j = 1:3
        subplot(n_rows, 3, 6 + j);
        plot(t_sim, u_traj_only(:,j), 'b', t_sim, u_total(:,j), 'r--', 'LineWidth', 0.8);
        ylabel(f_lbl{j}); xlabel('time [s]'); grid on; box off;
        if j == 1, legend('traj-only', 'traj+ms', 'Location', 'best'); end
    end

    % Row 4+: PSD per active mode, full-width row
    win = hann(N_period);
    for m = 1:n_modes
        md = mode_def.(sp.ms_modes{m});
        [P_t, f_ax] = pwelch(u_traj_only * md.f_vec', win, N_period/2, N_period, fs, 'onesided');
        [P_o, ~]    = pwelch(u_total     * md.f_vec', win, N_period/2, N_period, fs, 'onesided');
        subplot(n_rows, 3, 9 + (m-1)*3 + (1:3));
        semilogy(f_ax, sqrt(P_t), 'b', f_ax, sqrt(P_o), 'r--', 'LineWidth', 0.8);
        xlim([0, 150]); grid on; box off;
        xlabel('Frequency [Hz]');
        ylabel(sprintf('|U_{%s}| [N/sqrt(Hz)]', sp.ms_modes{m}));
        legend('traj-only', 'traj+ms', 'Location', 'best');
    end
end

function [r, t] = make_ref(sp, n_hold, ts)
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
    N     = size(r,1);
    N_tgt = max(2, ceil(N/N_period)) * N_period;
    r_pad = [r; repmat(r(end,:), N_tgt-N, 1)];
    t_pad = ts * (0:size(r_pad,1)-1)';
end

function validate_ref(r, t, id, lim)
% Acceleration checked here on r (exact piecewise polynomial derivative).
% Never check acceleration on q1: ode45 at 20 kHz amplifies sub-sample
% interpolation artefacts by fs^2 = 4e8, producing spurious spikes.
    ts  = t(2) - t(1);
    vel = diff(r) / ts;
    acc = diff(vel) / ts;
    assert(max(abs(r(:,1)))              <= lim.pos_X,  '%s: X1 position limit', id);
    assert(max(abs(r(:,2)))              <= lim.pos_X,  '%s: X2 position limit', id);
    assert(max(r(:,3))                   <=  lim.pos_Y, '%s: Y+ position limit', id);
    assert(min(r(:,3))                   >= -lim.pos_Y, '%s: Y- position limit', id);
    assert(max(abs(r(:,1)-r(:,2)))       <= lim.diff,   '%s: yaw |X1-X2| limit', id);
    assert(max(abs(vel(:,1:2)),[],'all') <= lim.vel,    '%s: X velocity limit',  id);
    assert(max(abs(vel(:,3)))            <= lim.vel,    '%s: Y velocity limit',  id);
    assert(max(abs(acc(:,1:2)),[],'all') <= lim.acc_X,  '%s: X acceleration limit', id);
    assert(max(abs(acc(:,3)))            <= lim.acc_Y,  '%s: Y acceleration limit', id);
    fprintf('  r OK: X1=[%+.3f %+.3f] X2=[%+.3f %+.3f] Y=[%+.3f %+.3f] m\n', ...
            min(r(:,1)), max(r(:,1)), min(r(:,2)), max(r(:,2)), ...
            min(r(:,3)), max(r(:,3)));
end

function [t_sim, r_sim, u_q1] = reconstruct(q1, r, t, Cfb)
% u_q1 = Cfb*(r - q1). Handles variable-step Simulink output via interpolation.
    Ns = size(q1, 1);
    if Ns ~= numel(t), t_sim = linspace(0, t(end), Ns)'; r_sim = interp1(t, r, t_sim);
    else,              t_sim = t;                         r_sim = r; end
    u_q1 = lsim(ss(Cfb), r_sim - q1, t_sim);
end

function y_out = resample_to(y, t_src, t_tgt)
    if size(y,1) == numel(t_tgt), y_out = y; return; end
    y_out = interp1(t_src, y, t_tgt, 'linear', 'extrap');
end

function sig = multisine_schroeder(N, N_period, fs, f_low, f_high, seed)
% Schroeder-phase odd-harmonic multisine. Returns unit-RMS signal.
% THEORY: phi_k = -pi*k*(k-1)/F  (Schroeder 1970) minimises crest factor
% HEURISTIC: seed phase offset decorrelates simultaneously injected modes
    assert(mod(N, N_period) == 0, 'N must be a multiple of N_period');
    f0 = fs / N_period;
    k0 = ceil(f_low/f0);   if mod(k0,2)==0, k0 = k0+1; end
    k1 = floor(f_high/f0); if mod(k1,2)==0, k1 = k1-1; end
    k  = k0:2:k1;
    F  = numel(k);
    assert(F >= 7, 'Only %d odd bins in [%.1f %.1f] Hz — PE condition F>=7 not met', F, f_low, f_high);
    idx     = 1:F;
    phi     = -pi * idx .* (idx-1) / F;
    phi     = phi + 2*pi*(k*f0)*(seed-1) / (7*f_high);
    t_per   = (0:N_period-1)' / fs;
    one_per = sum(cos(2*pi*t_per*(k*f0) + phi), 2);
    one_per = one_per / rms(one_per);
    sig     = repmat(one_per, N/N_period, 1);
end
