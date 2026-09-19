function records = gtd_build_records(cfg)
% GTD_BUILD_RECORDS  Declarative table of the 22 records (T1-14, V1-4, E1-4).
%   records = GTD_BUILD_RECORDS(cfg) returns a 1x22 struct array. Each element
%   is pure data describing one record; no simulation logic here.
%
%   Common fields: id, split, class, Y_op, seed, p
%     class : 'standstill' | 'oscillatory' | 'aprbs'
%             (the six spec "classes" collapse to three reference SHAPES;
%              y-sweep and lissajous are just oscillatory parameterizations,
%              and E1's sinesweep is a standstill motion with a swept excitation)
%     Y_op  : frozen operating point for the controller (D-039 convention)
%     seed  : 100*track_id + index  (independent realization per record & split)
%     p     : class-specific geometry + excitation control
%             excitation : 'multisine' | 'sinesweep' | 'none'
%             amp_frac   : excitation amplitude scale vs track level (test set uses 0.80)
%
%   Ladder margins are DERIVED from cfg.lim so records and cfg cannot disagree:
%     training top (T11) = 75% of the enforced per-axis limits   (HEURISTIC, user-set)
%     test extrapolation (E3) = 90% of the enforced per-axis limits
%
%   Spec: docs/trajectory-generation-spec-draft.md sections 3, 4, 5.

    L = cfg.lim;
    base75 = struct('vmax_X',0.75*L.vel,'amax_X',0.75*L.acc_X,'vmax_Y',0.75*L.vel,'amax_Y',0.75*L.acc_Y);
    base90 = struct('vmax_X',0.90*L.vel,'amax_X',0.90*L.acc_X,'vmax_Y',0.90*L.vel,'amax_Y',0.90*L.acc_Y);
    lad = @(b,f) struct('vmax_X',f*b.vmax_X,'amax_X',f*b.amax_X,'vmax_Y',f*b.vmax_Y,'amax_Y',f*b.amax_Y);

    % Setpoint ranges (logical coords) for the APRBS class
    Xs_full = [-0.10, 0.10];  Xs_red = [-0.05, 0.05];
    Xa_off  = [0, 0];         Xa_on  = [-0.001, 0.001];   % X_anti active only where the 6mm budget allows
    Yr_full = [-0.30, 0.30];  Yr_v2  = [-0.30, -0.14];

    c = {};   % cell list, converted to struct array at the end

    % ── S1: the settling record. V2's geometry with the injected excitation OFF ──
    % WHY THIS RECORD EXISTS. V2_aprbs_Ylow is the validation jerk-limited profile, but it
    % carries the 140-230 Hz multisine, and that multisine is NOT separable after the fact:
    % gtd_run_simulation computes u_fb = lsim(Cfb, r_sim - q_with) against the output WITH
    % the multisine acting, so the CONTROLLER responded to it and u_fb is contaminated too.
    % Subtracting f_sim from u_total afterwards leaves an (u, y) pair that no simulation
    % produced. The only way to a clean jerk-limited validation record is to generate one
    % with excitation = 'none' from the start, which is what pnone() does here.
    %
    % Everything else is V2 verbatim so the two remain comparable: Y_op = -0.22 (unseen
    % interior Y), Yr_v2 = [-0.30, -0.14], jerkTime 0.035, 60% of the 75% training ladder,
    % full symmetric X range, yaw off.
    %
    % SEED is pinned to V2's value (V2 is index 16 of the shipped 22-record table), so the
    % APRBS setpoint draws are the same sequence of targets V2 drew. The realised move LIST
    % still differs, because cfg.n_hold_short is 0.4 s here against 0.1 s there and the
    % packing loop fills a fixed 10 s window, so fewer moves fit. Same draws, fewer of them.
    c{end+1} = mk('S1_settle_Ylow_noexc', 'val', 'aprbs', -0.22, ...
                  pnone(pa(lad(base75,0.60), 0.035, Xs_full, Xa_off, Yr_v2)));

    % ── Materialize. Seeds are pinned per record, not derived from position ──
    records = [c{:}];
    seeds   = [16];   % V2's index in the shipped table; see the note above
    assert(numel(seeds) == numel(records), 'seeds and records must agree in length');
    for k = 1:numel(records)
        records(k).seed = 100*cfg.track_id + seeds(k);
    end
end

% ── local builders ──────────────────────────────────────────────────────────

function r = mk(id, split, class, Y_op, p)
    r = struct('id',id, 'split',split, 'class',class, 'Y_op',Y_op, 'seed',0, 'p',p);
end

function p = ps(amp_frac)            % standstill params
    if nargin < 1, amp_frac = 1.0; end
    p = struct('excitation','multisine', 'amp_frac',amp_frac);
end

function p = psweep(band, amp_frac)  % standstill with swept-force excitation (E1)
    p = struct('excitation','sinesweep', 'amp_frac',amp_frac, 'sweep_band',band);
end

function p = po(Y_center, A_sym, f_sym, A_anti, f_anti, A_y, f_y)   % oscillatory params
    p = struct('Y_center',Y_center, 'A_sym',A_sym, 'f_sym',f_sym, ...
               'A_anti',A_anti, 'f_anti',f_anti, 'A_y',A_y, 'f_y',f_y, ...
               'excitation','multisine', 'amp_frac',1.0);
end

function p = pa(lvl, jerkTime, X_sym_range, X_anti_range, Y_range)  % aprbs params
    p = lvl;
    p.jerkTime     = jerkTime;
    p.X_sym_range  = X_sym_range;
    p.X_anti_range = X_anti_range;
    p.Y_range      = Y_range;
    p.excitation   = 'multisine';
    p.amp_frac     = 1.0;
end

function p = pnone(p)                % turn the injected excitation off (E4)
    p.excitation = 'none';
end
