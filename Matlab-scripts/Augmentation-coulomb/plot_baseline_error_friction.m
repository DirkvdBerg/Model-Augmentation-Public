function plot_baseline_error_friction()
%PLOT_BASELINE_ERROR_FRICTION  What the augmentation has to learn, with and without friction.
%
% The baseline IS the model, so its residual against the truth is exactly the
% discrepancy the augmentation is asked to account for. Five figures:
%
%   1  trajectories        T3 and TP1, friction vs frictionless, plus the
%                          difference on its own axis
%   2  T3   standstill + multisine        per-window residual RMS, full record
%   3  T6   ysweep + multisine            per-window residual RMS, full record
%   4  TP1  Telica, no multisine          per-window residual RMS, full record
%   5  TP1  error shape against the reference velocity, two consecutive moves
%
% WHY THE DIFFERENCE GETS ITS OWN AXIS IN FIGURE 1. The trajectory is set by the
% reference, order 1e-1 m. The controller rejects most of the friction force, so
% friction moves the closed-loop trajectory by order 1e-5 to 1e-4 m, i.e. 1e-3 to
% 1e-4 of the plotted range. On a shared axis that is a sub-pixel difference: the
% two traces are drawn as the same pixels and the figure would assert that
% friction does nothing. The right-hand axis carries y_fric - y_nofric in its own
% units, which is also where the Coulomb signature lives (it spikes at velocity
% reversals and sits near zero during cruise).
%
% WHY PER-WINDOW RMS AND NOT A WAVEFORM IN FIGURES 2 TO 4. The records are 12 s
% at 20 kHz. The residual is dominated by the 140 to 230 Hz multisine band, so
% 12 s of waveform plots as a band of ink. The per-window RMS is one point per
% 0.1 s window, 120 points across the record, and it is exactly the quantity the
% training loss minimises.
%
% CLOSED LOOP, AND WINDOWED. An earlier open-loop version produced a frictionless
% residual of 1.8e-04 m on T3 against a signal of 1.5e-05 m: the residual was
% TWELVE TIMES the trajectory. That is not model mismatch, it is drift. X and Y
% have K = 0, so any force error integrates without restoring force over a free
% run. The training objective does not free-run either: it wraps the known
% controller around the model (closed_loop.py: x <- f(x, u_t + u_fb)) over
% nf = 400 windows. This script does the same, and the windows are re-initialised
% from the recorded state so error never accumulates across a window boundary.
%
% LAYOUT of figures 2 to 4: rows X1 / X2 / Y, columns frictionless | friction,
% with SHARED y limits per row. Overlaying the two columns was vague: they differ
% by about an order of magnitude, so the frictionless trace collapsed onto zero
% and read as absent. Side by side on a shared scale, the size difference IS the
% figure and both amplitudes are readable.
%
% Units are metres in scientific notation throughout. mm and um were the wrong
% call: the quantities span 1e-05 to 1e-01 and a unit change between panels hides
% that.
%
% RUNTIME. Every panel needs a closed-loop RK4 replay of the full 240000-sample
% record for both the frictionless and the friction dataset, so expect a few
% minutes per case.
%
% Run:
%   matlab -batch "addpath('Matlab-scripts/Augmentation-coulomb'); plot_baseline_error_friction"

    THIS_DIR  = fileparts(mfilename('fullpath'));
    REPO_ROOT = fileparts(fileparts(THIS_DIR));
    addpath(genpath(fullfile(REPO_ROOT, 'Matlab-scripts', 'Augmentation')));
    addpath(THIS_DIR);

    FIG_DIR = fullfile(REPO_ROOT, 'scripts', 'gantry', 'meeting', ...
                       'meeting-21-09-2026', 'figures');
    if ~exist(FIG_DIR, 'dir'), mkdir(FIG_DIR); end

    MA_FRAC = 0.50;
    T_WIN   = 0.100;      % s, the training objective's horizon

    % Every record used here is generated at Y_op = 0 (gtd_build_records: T3 and
    % T6 both 0.00; gtd_build_records_telica: TP1 0.00), so ONE controller serves
    % all of them. A record at a different Y_op needs its own Cfb.
    Y_OP = 0.00;

    DS_MS = 'augmentation_ma50_b140-230_a6_z03';
    DS_TP = 'augmentation_ma50_z03_telica';

    cases = struct( ...
      'tag',  {'T3  standstill + multisine', ...
               'T6  Y sweep + multisine', ...
               'TP1  Telica jerk-limited, no multisine'}, ...
      'rec',  {'T3_standstill_Y000.mat', 'T6_ysweep_slow.mat', 'TP1_telica_y000.mat'}, ...
      'nofr', {DS_MS, DS_MS, DS_TP}, ...
      'fric', {[DS_MS '_coulomb'], [DS_MS '_coulomb'], [DS_TP '_coulomb']}, ...
      'file', {'err_rms_T3_standstill_multisine', ...
               'err_rms_T6_ysweep_multisine', ...
               'err_rms_TP1_telica_nomultisine'});

    cfg = gtd_config('augmentation', true, MA_FRAC);
    P   = [1, 1, 0; cfg.Lb/2, -cfg.Lb/2, 0; 0, 0, 1];
    ctl = controller(Y_OP, cfg);
    fb  = baseline_field(cfg);
    C   = colours();

    % figure 1: trajectories
    figure_trajectories(cases([1 3]), cfg, REPO_ROOT, FIG_DIR, C);

    % figures 2 to 4: per-window residual RMS over the full record
    tp_rms = [];
    for k = 1:numel(cases)
        r = figure_window_rms(cases(k), cfg, P, ctl, fb, T_WIN, REPO_ROOT, FIG_DIR, C);
        if k == 3, tp_rms = r; end
    end

    % figure 5: error shape against the reference velocity
    figure_shape(cases(3), cfg, P, ctl, fb, T_WIN, tp_rms, REPO_ROOT, FIG_DIR, C);
end

% ===========================================================================
% figures
% ===========================================================================

function figure_trajectories(cs, cfg, REPO_ROOT, FIG_DIR, C)
% Closed-loop trajectory of the TRUTH, friction against frictionless. No model
% rollout is involved: S.y is the recorded plant output in stage coordinates,
% so the only thing separating the two datasets is the Coulomb term.
    nc  = numel(cs);
    fig = figure('Position', [80 80 640*nc 820], 'Color', 'w');
    tl  = tiledlayout(3, nc, 'Padding', 'compact', 'TileSpacing', 'compact');
    title(tl, 'Closed-loop trajectory, friction against frictionless', ...
          'FontWeight', 'normal');

    for c = 1:nc
        A = load(recpath(REPO_ROOT, cs(c).nofr, cs(c).rec));
        B = load(recpath(REPO_ROOT, cs(c).fric, cs(c).rec));
        t = (0:size(A.y,1)-1).' * cfg.ts;
        d = double(B.y) - double(A.y);

        for r = 1:3
            ax = nexttile((r-1)*nc + c);
            yyaxis(ax, 'left');
            plot(ax, t, double(A.y(:,r)), '-', 'LineWidth', 0.8, 'Color', C.nofr);
            hold(ax, 'on');
            plot(ax, t, double(B.y(:,r)), '-', 'LineWidth', 0.8, 'Color', C.fric);
            ylabel(ax, sprintf('%s [m]', chname(r)));
            ax.YAxis(1).Color = [0 0 0];
            ytickformat(ax, '%.1e');

            yyaxis(ax, 'right');
            plot(ax, t, d(:,r), '-', 'LineWidth', 0.6, 'Color', C.diff);
            ylabel(ax, 'difference [m]');
            ax.YAxis(2).Color = C.diff;
            ytickformat(ax, '%.1e');

            set(ax, 'Box', 'off');
            if r == 1
                title(ax, cs(c).tag, 'Interpreter', 'none', 'FontWeight', 'normal');
            end
            if r == 3, xlabel(ax, 't [s]'); end
            if r == 1 && c == 1
                legend(ax, {'no friction', 'friction', 'friction - no friction'}, ...
                       'Location', 'best', 'Box', 'off', 'FontSize', 8);
            end
            fprintf('[fig1] %-40s %-3s  max|difference| %.3e m\n', ...
                    cs(c).tag, chname(r), max(abs(d(:,r))));
        end
    end
    save_fig(fig, FIG_DIR, 'trajectories_friction_vs_nofriction');
end

function rms_w = figure_window_rms(cs, cfg, P, ctl, fb, T_WIN, REPO_ROOT, FIG_DIR, C)
% Per-window residual RMS over the WHOLE record, frictionless beside friction,
% y limits shared per row. rms_w is returned for the friction dataset so
% figure 5 can pick a representative move from it.
    [tc, rmsA] = window_rms(recpath(REPO_ROOT, cs.nofr, cs.rec), cfg, P, ctl, fb, T_WIN);
    [~,  rmsB] = window_rms(recpath(REPO_ROOT, cs.fric, cs.rec), cfg, P, ctl, fb, T_WIN);
    rms_w = struct('t', tc, 'rms', rmsB);

    fprintf('\n=== %s ===\n', cs.tag);
    fprintf('  residual RMS over %d windows of %.0f ms\n', numel(tc), 1e3*T_WIN);
    for r = 1:3
        a = sqrt(mean(rmsA(:,r).^2));  b = sqrt(mean(rmsB(:,r).^2));
        fprintf('  %-3s  no friction %.4e m | friction %.4e m | ratio %.1fx\n', ...
                chname(r), a, b, b/max(a, eps));
    end

    fig = figure('Position', [80 80 1250 820], 'Color', 'w');
    tl  = tiledlayout(3, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    title(tl, sprintf('%s   baseline residual, RMS per %.0f ms window', ...
                      cs.tag, 1e3*T_WIN), 'Interpreter', 'none', 'FontWeight', 'normal');

    for r = 1:3
        lim = max([rmsA(:,r); rmsB(:,r)]) * 1.10;
        for c = 1:2
            if c == 1
                v = rmsA(:,r);  col = C.nofr;  lbl = 'no friction';
            else
                v = rmsB(:,r);  col = C.fric;  lbl = 'friction';
            end
            ax = nexttile((r-1)*2 + c);
            plot(ax, tc, v, '-', 'LineWidth', 0.9, 'Color', col);
            ylim(ax, [0 lim]); xlim(ax, [tc(1) tc(end)]);
            ytickformat(ax, '%.1e'); set(ax, 'Box', 'off');
            if c == 1, ylabel(ax, sprintf('%s RMS [m]', chname(r))); end
            if r == 3, xlabel(ax, 't [s]'); end
            if r == 1
                title(ax, sprintf('%s   (record RMS %.2e m)', lbl, sqrt(mean(v.^2))), ...
                      'FontWeight', 'normal');
            end
        end
    end
    save_fig(fig, FIG_DIR, cs.file);
end

function figure_shape(cs, cfg, P, ctl, fb, T_WIN, rms_w, REPO_ROOT, FIG_DIR, C)
% The error WAVEFORM against the reference velocity. Per-window RMS says how big
% the error is and where; it says nothing about shape. Coulomb friction switches
% on the sign of velocity, so the signature (a jump at every velocity zero
% crossing, near-flat in between) is only legible when the error is lined up with
% the reference velocity. Position alone would not show it.
%
% TP1 rather than T3: its reversals are discrete and few. Under a multisine the
% velocity crosses zero hundreds of times per second and no alignment is
% readable.
%
% TWO consecutive moves, not one. A single move gives only the two zero
% crossings at its own start and end; the pair adds the shuttle DIRECTION
% reversal between them, which is where Coulomb is most visible and a smooth
% model mismatch is not.
%
% The move is chosen by rule, not by eye: the move whose window RMS is closest to
% the MEDIAN over all moves but the first two. The first move is excluded because
% the record starts from rest, so it carries the absorber start-up transient and
% the controller's initial state on top of the friction effect, and nothing seen
% there could be attributed.
    A_path = recpath(REPO_ROOT, cs.nofr, cs.rec);
    B_path = recpath(REPO_ROOT, cs.fric, cs.rec);
    S      = load(B_path);
    r_ref  = double(S.r_sim);                 % already stage X1, X2, Y
    N      = size(r_ref, 1);
    t      = (0:N-1).' * cfg.ts;
    v_ref  = [zeros(1,3); diff(r_ref)] / cfg.ts;

    starts = move_starts(v_ref(:,3), 1e-3);
    assert(numel(starts) >= 4, 'need at least 4 moves to pick a representative pair');

    % representative move: window RMS on Y closest to the median, first two moves
    % excluded (start-up transient).
    cand = 3:numel(starts)-1;
    val  = zeros(size(cand));
    for i = 1:numel(cand)
        [~, w] = min(abs(rms_w.t - (t(starts(cand(i))) + 0.5*T_WIN)));
        val(i) = rms_w.rms(w, 3);
    end
    [~, j] = min(abs(val - median(val)));
    m      = cand(j);

    t0 = t(starts(m))   - 0.05;
    t1 = t(starts(m+1)) + 0.30;
    k0 = max(1, round(t0/cfg.ts) + 1);
    k1 = min(N, round(t1/cfg.ts) + 1);
    fprintf('\n[fig5] move %d of %d, window %.4f to %.4f s ', m, numel(starts), t(k0), t(k1));
    fprintf('(Y window RMS %.3e m, median over moves %.3e m)\n', val(j), median(val));

    [te, eA, segA] = window_wave(A_path, cfg, P, ctl, fb, T_WIN, k0, k1);
    [~,  eB, ~   ] = window_wave(B_path, cfg, P, ctl, fb, T_WIN, k0, k1);
    zc = zero_crossings(t(k0:k1), v_ref(k0:k1, :));

    fig = figure('Position', [80 80 1100 900], 'Color', 'w');
    tl  = tiledlayout(4, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
    title(tl, sprintf('%s   baseline error against the reference velocity, moves %d and %d', ...
                      cs.tag, m, m+1), 'Interpreter', 'none', 'FontWeight', 'normal');

    ax = nexttile(1);
    plot(ax, t(k0:k1), v_ref(k0:k1,1), '-', 'LineWidth', 1.0, 'Color', C.refx);
    hold(ax, 'on');
    plot(ax, t(k0:k1), v_ref(k0:k1,3), '-', 'LineWidth', 1.0, 'Color', C.refy);
    yline(ax, 0, ':', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');
    ylabel(ax, 'v_{ref} [m/s]'); set(ax, 'Box', 'off');
    xlim(ax, [t(k0) t(k1)]);
    legend(ax, {'X reference', 'Y reference'}, 'Location', 'best', ...
           'Box', 'off', 'FontSize', 8);
    mark(ax, zc, segA, t(k0), t(k1));

    for r = 1:3
        ax = nexttile(r+1);
        plot(ax, te, eA(:,r), '-', 'LineWidth', 0.7, 'Color', C.nofr);
        hold(ax, 'on');
        plot(ax, te, eB(:,r), '-', 'LineWidth', 0.7, 'Color', C.fric);
        ylabel(ax, sprintf('%s error [m]', chname(r)));
        ytickformat(ax, '%.1e'); set(ax, 'Box', 'off');
        xlim(ax, [t(k0) t(k1)]);
        mark(ax, zc, segA, t(k0), t(k1));
        if r == 1
            legend(ax, {'no friction', 'friction'}, 'Location', 'best', ...
                   'Box', 'off', 'FontSize', 8);
        end
        if r == 3, xlabel(ax, 't [s]'); end
    end
    save_fig(fig, FIG_DIR, 'err_shape_TP1_vs_setpoint');
end

function mark(ax, zc, seams, t0, t1)
% Dashed verticals at reference velocity zero crossings (where a Coulomb term
% switches), faint verticals at the window re-initialisation seams so a seam is
% never mistaken for signal.
    for s = seams(:).'
        if s > t0 && s < t1
            xline(ax, s, '-', 'Color', [0.88 0.88 0.88], 'LineWidth', 0.5, ...
                  'HandleVisibility', 'off');
        end
    end
    for z = zc(:).'
        xline(ax, z, '--', 'Color', [0.55 0.55 0.55], 'LineWidth', 0.6, ...
              'HandleVisibility', 'off');
    end
end

% ===========================================================================
% closed-loop windowed replay
% ===========================================================================

function [tc, rms_w] = window_rms(REC, cfg, P, ctl, fb, T_WIN)
% Tile CONSECUTIVE, non-overlapping windows across the whole record and return
% the residual RMS of each. Every window restarts from the recorded state, so
% this is exactly the training objective and nothing accumulates across windows.
    S  = load(REC);
    nw = round(T_WIN / cfg.ts);
    N  = size(S.y, 1);
    nb = floor((N - 1) / nw);
    tc    = zeros(nb, 1);
    rms_w = zeros(nb, 3);
    for b = 1:nb
        s = (b-1)*nw + 1;
        e = run_window(S, s, nw, fb, P, ctl, cfg.ts);
        rms_w(b,:) = sqrt(mean(e.^2, 1));
        tc(b)      = (s - 1 + nw/2) * cfg.ts;
    end
end

function [te, e_cat, seams] = window_wave(REC, cfg, P, ctl, fb, T_WIN, k0, k1)
% Same tiling, restricted to k0:k1, keeping the WAVEFORM. The window grid is
% aligned to k0 so the seams are predictable and can be drawn.
    S  = load(REC);
    nw = round(T_WIN / cfg.ts);
    nb = max(1, floor((k1 - k0 + 1) / nw));
    e_cat = zeros(nb*nw, 3);
    seams = zeros(nb, 1);
    for b = 1:nb
        s = k0 + (b-1)*nw;
        e_cat((b-1)*nw + (1:nw), :) = run_window(S, s, nw, fb, P, ctl, cfg.ts);
        seams(b) = (s - 1) * cfg.ts;
    end
    te = ((k0-1) + (0:nb*nw-1)).' * cfg.ts;
end

function e = run_window(S, s, nw, f, P, ctl, h)
% One closed-loop window. Start the BASELINE from the recorded state and step it
% with the recorded input plus the controller's correction:
%     e = y_recorded - y_model            (stage coordinates, as Cfb expects)
%     u = u_recorded + Cfb(e)
% matching closed_loop.py's  x <- f(x, u_t + u_fb).
    xl = double(S.x_logical(s,:)).';
    x  = [xl(1:3); double(S.delta_a(s)); xl(4:6); double(S.vdelta_a(s))];
    xc = zeros(size(ctl.A,1), 1);
    e  = zeros(nw, 3);
    for k = 1:nw
        y_mod  = P.' * x(1:3);                      % logical -> stage positions
        err    = double(S.y(s+k-1, :)).' - y_mod;
        e(k,:) = err.';
        u_fb   = ctl.C*xc + ctl.D*err;              % controller in STAGE coordinates
        xc     = ctl.A*xc + ctl.B*err;
        u_log  = P * (double(S.u_total(s+k-1,:)).' + u_fb);
        s1 = f(x, u_log);              s2 = f(x + (h/2)*s1, u_log);
        s3 = f(x + (h/2)*s2, u_log);   s4 = f(x + h*s3, u_log);
        x  = x + (h/6)*(s1 + 2*s2 + 2*s3 + s4);
    end
end

function f = baseline_field(cfg)
% BASELINE physics: no friction, massless absorber, mh_total preserved. The
% massless limit, not a stiffened ka, which would put a 150 kHz mode into a
% 20 kHz explicit RK4 and diverge.
    mh_total = cfg.mh;
    maB = 1e-4*mh_total;  mhB = mh_total - maB;
    kaB = maB*(2*pi*cfg.fa)^2;  caB = 2*cfg.zeta_a*sqrt(kaB*maB);
    args = {cfg.m1, cfg.m2, cfg.mb, mhB, cfg.Lb, cfg.Jb, cfg.Jh, cfg.d, ...
            cfg.cg1, cfg.cg2, cfg.cb1, cfg.cb2, cfg.cy, cfg.kb1, cfg.kb2, ...
            maB, kaB, caB, cfg.L0, 0, 0, 0, cfg.ts};     % cc = 0
    f = @(x, u) gantrySystemExtendedCoulomb(u, x, args{:});
end

function ctl = controller(Y_op, cfg)
    plant = gtd_build_plant(Y_op, cfg);
    [A, B, Cm, D] = ssdata(plant.Cfb);
    ctl = struct('A', A, 'B', B, 'C', Cm, 'D', D);
end

% ===========================================================================
% small helpers
% ===========================================================================

function s = move_starts(v, tol)
    idx = find(abs(v) > tol);
    if isempty(idx), s = []; return; end
    brk = find(diff(idx) > 1);
    s   = [idx(1); idx(brk+1)];
end

function z = zero_crossings(t, v)
% Instants where any reference velocity channel changes sign, i.e. where a
% Coulomb term switches. Crossings closer than 1 ms are merged so the axis is
% not striped.
    z = [];
    for c = 1:size(v,2)
        sg = sign(v(:,c));  sg(sg == 0) = NaN;
        d  = diff(sg);
        k  = find(d ~= 0 & ~isnan(d));
        z  = [z; t(k)];                                         %#ok<AGROW>
    end
    z = sort(z);
    if ~isempty(z), z = z([true; diff(z) > 1e-3]); end
end

function p = recpath(REPO_ROOT, ds, rec)
    p = fullfile(REPO_ROOT, 'data', 'gantry', 'matlab', 'trajectory', ds, rec);
    assert(isfile(p), 'record not found: %s', p);
end

function n = chname(r)
    names = {'X_1', 'X_2', 'Y'};
    n = names{r};
end

function C = colours()
    C = struct('nofr', [0.000 0.447 0.741], ...
               'fric', [0.850 0.325 0.098], ...
               'diff', [0.350 0.350 0.350], ...
               'refx', [0.466 0.674 0.188], ...
               'refy', [0.494 0.184 0.556]);
end

function save_fig(fig, FIG_DIR, name)
    out = fullfile(FIG_DIR, [name '.png']);
    exportgraphics(fig, out, 'Resolution', 150);
    fprintf('saved %s\n', out);
end
