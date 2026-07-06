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

    % ── T1-T5: standstill multisine at 5 frozen-Y points (>=3 for quadratic M(Y)) ─
    c{end+1} = mk('T1_standstill_Ym30','train','standstill',-0.30, ps());
    c{end+1} = mk('T2_standstill_Ym15','train','standstill',-0.15, ps());
    c{end+1} = mk('T3_standstill_Y000','train','standstill', 0.00, ps());
    c{end+1} = mk('T4_standstill_Yp15','train','standstill', 0.15, ps());
    c{end+1} = mk('T5_standstill_Yp30','train','standstill', 0.30, ps());

    % ── T6-T8: Y-sweeps (scheduling-rate coverage) ──────────────────────────
    c{end+1} = mk('T6_ysweep_slow', 'train','oscillatory', 0.00, po(0.00, 0,0,    0,0,   0.30,0.2));
    c{end+1} = mk('T7_ysweep_fast', 'train','oscillatory', 0.00, po(0.00, 0,0,    0,0,   0.30,0.7));
    c{end+1} = mk('T8_ysweep_xmix', 'train','oscillatory', 0.00, po(0.00, 0.05,1.1, 0,0, 0.30,0.35));

    % ── T9-T12: randomized jerk-limited setpoint sequences (position-level APRBS) ─
    c{end+1} = mk('T9_aprbs_30',  'train','aprbs', 0.00, pa(lad(base75,0.30), 0.050, Xs_full, Xa_off, Yr_full));
    c{end+1} = mk('T10_aprbs_60', 'train','aprbs', 0.00, pa(lad(base75,0.60), 0.035, Xs_full, Xa_off, Yr_full));
    c{end+1} = mk('T11_aprbs_100','train','aprbs', 0.00, pa(lad(base75,1.00), 0.025, Xs_full, Xa_off, Yr_full));
    c{end+1} = mk('T12_aprbs_yaw','train','aprbs', 0.00, pa(lad(base75,0.60), 0.040, Xs_red,  Xa_on,  Yr_full));

    % ── T13-T14: lissajous (multi-axis simultaneous) ────────────────────────
    c{end+1} = mk('T13_lissajous', 'train','oscillatory', 0.00, po(0.00, 0.08,1.5, 0,0,      0.25,0.4));
    c{end+1} = mk('T14_lissajous_yaw','train','oscillatory', 0.00, po(0.00, 0.06,1.3, 0.001,0.8, 0.30,0.7));

    % ── V1-V4: validation (separate generation, fresh seeds, unseen interior Y) ─
    c{end+1} = mk('V1_standstill_Yp10','val','standstill', 0.10, ps());
    c{end+1} = mk('V2_aprbs_Ylow',     'val','aprbs',     -0.22, pa(lad(base75,0.60), 0.035, Xs_full, Xa_off, Yr_v2));
    c{end+1} = mk('V3_ysweep_Yp10',    'val','oscillatory', 0.10, po(0.10, 0,0,    0,0,   0.15,0.2));
    c{end+1} = mk('V4_lissajous_Ym10', 'val','oscillatory',-0.10, po(-0.10, 0.07,1.4, 0,0, 0.20,0.5));

    % ── E1-E4: test (held out until final evaluation) ───────────────────────
    c{end+1} = mk('E1_resonance_sweep','test','standstill', 0.00, psweep([130 180], 0.80));
    c{end+1} = mk('E2_multisine_Yp22', 'test','standstill', 0.22, ps(0.80));
    c{end+1} = mk('E3_aprbs_above',    'test','aprbs',      0.00, pa(lad(base90,1.00), 0.030, Xs_full, Xa_off, Yr_full));
    c{end+1} = mk('E4_multisine_off',  'test','aprbs',      0.00, pnone(pa(lad(base75,1.00), 0.025, Xs_full, Xa_off, Yr_full)));

    % ── Assign per-record seeds and materialize the struct array ────────────
    records = [c{:}];
    for k = 1:numel(records)
        records(k).seed = 100*cfg.track_id + k;   % independent draw per record AND split
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
