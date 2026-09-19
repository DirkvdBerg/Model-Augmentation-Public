function records = gtd_build_records_telica(cfg)
% GTD_BUILD_RECORDS_TELICA  Record table for the Telica motion-profile dataset.
%   records = GTD_BUILD_RECORDS_TELICA(cfg) returns a 1x7 struct array in the
%   same shape gtd_build_records returns (id, split, class, Y_op, seed, p), so
%   the rest of the gtd_* chain consumes it unchanged.
%
%   EVERY record has p.excitation = 'none'. The reference trajectory is the only
%   excitation, which is the entire point of this dataset: the production
%   pipeline's records all carry a multisine, so a model trained on them
%   extrapolates badly on evaluation records that have none. Built with pnone()
%   the same way E4_multisine_off is, so gtd_make_multisine returns a zero force
%   and gtd_enforce_limits has nothing to scale.
%
%   class = 'telica' is a NEW reference shape, handled by
%   gtd_make_reference_telica. See D-206 and that file's header.
%
%   KINEMATICS, MEASURED from the real machine (kamtin-data/Data Telica/,
%   '06 40 mm XL 80 mm YL', all 15 operating points, this session):
%
%     axis   stroke     T_move    v_max        a_max        structure
%     X      40.000 mm  80.00 ms  1.0000 m/s   39.36 m/s^2  pure cycloid, no cruise
%     Y      80.003 mm  99.9  ms  1.5010 m/s   50.6  m/s^2  46.6 ms accel + 6.7 ms
%                                                           cruise + 46.6 ms decel
%
%   The X stroke is commanded identically on GTRX1 and GTRX2, so it is a pure
%   X_sym move and X_anti is identically zero: the machine commands no yaw.
%
%   ACCELERATION FOLLOWS THE LOGS, NOT THE DATASHEET. The ETEL datasheet
%   (ASME-YGNN-08-0750-0800W3, literature/gantry/telica-xyz-0750-0800-data.pdf)
%   rates maximum acceleration at X 30 and Y 50 m/s^2, and gtd_config's
%   lim.acc_X/lim.acc_Y are those rated figures verbatim. The logged campaign
%   exceeds them: 40.95 m/s^2 on X (37 % over) and 52.07 on Y (4 % over),
%   measured on the real position across all 15 operating points. The supervisor
%   ran these settings on the real system, so the logs are the authority and the
%   values above are the measured setpoint accelerations. Nothing trips on the
%   difference: gtd_enforce_limits.validate_response checks position, yaw,
%   velocity and force only, and lim.acc_* is used solely to derive the APRBS
%   ladder in gtd_build_records. Velocity is the binding check and 1.0 / 1.5 m/s
%   clear lim.vel = 2.0. The rated figures look like application/accuracy
%   numbers rather than ceilings: at 41 m/s^2 the machine draws well under its
%   continuous force rating on every axis. See D-206.

    % ── measured Telica kinematics, one struct so no record can disagree ────
    % MEASURED: kamtin-data/Data Telica/'06 40 mm XL 80 mm YL'. Fit residual of
    % the velocity-limited cycloidal model 9.4 um on X (0.023 % of stroke) and
    % 14.8 um on Y (0.019 %); next-best candidate (poly5 min-jerk) 51.9 um.
    % Identical at all 15 operating points.
    TEL = struct('X_stroke', 0.040,  'X_vmax', 1.0000, 'X_amax', 39.36, ...
                 'Y_stroke', 0.080,  'Y_vmax', 1.5010, 'Y_amax', 50.60);

    % MEASURED: 290 ms of standstill follows the move in the logs (444 ms
    % precedes it). A Telica log holds ONE move, so the machine's inter-move
    % period is not in the data; 0.30 s is the measured post-move standstill
    % used as the dwell. TP2, TP3, VP1, VP2 and EP1 vary it (0.25 to 0.60 s)
    % deliberately, so the dataset is not built on a single unverifiable number.
    % The dwell is the ONE timing quantity not pinned by the machine, because a
    % Telica log contains a single move. It is therefore also the only timing
    % knob that may differ between records; stroke and move duration may not.
    DWELL = 0.30;

    % Logical ranges, matching gtd_build_records' Xs_full / Yr_full / Yr_v2.
    X_full = [-0.10, 0.10];
    Y_full = [-0.30, 0.30];
    Y_low  = [-0.30, -0.14];

    % EVERY record starts at [X_sym, Y] = [0, Y_op], the controller's
    % linearisation point. This is forced, not chosen: gtd_enforce_limits runs
    % lsim from zero state on r - [0,0,Y_op], so a record starting anywhere else
    % injects a step into the closed loop at t = 0. See gtd_make_reference_telica.
    %
    % HOW Y COVERAGE IS OBTAINED WITHOUT EVER SHORTENING A MOVE. Every move is a
    % full 80 mm, so a record walks the lattice Y_op + k*0.080 and reaches a
    % range bound only if that bound lies ON its lattice. Y_op is therefore
    % offset per record:
    %     Y_op =  0.00  ->  traversal [-0.240, +0.240]
    %     Y_op = +0.06  ->  traversal [-0.260, +0.300]   (touches +0.30 exactly)
    %     Y_op = -0.06  ->  traversal [-0.300, +0.260]   (touches -0.30 exactly)
    % The set as a whole therefore covers the full [-0.30, +0.30] operating range
    % at 20 mm resolution while every individual move stays exactly the measured
    % Telica stroke and duration. Records are distinguished by Y_op, shuttle
    % DIRECTION and dwell, never by a start pose and never by stroke.

    c = {};

    % ── train ───────────────────────────────────────────────────────────────
    % TP1/TP2 share Y_op and mirror the shuttle directions, so they traverse the
    % Y range in opposite order and pair X with Y differently. The dwell differs
    % too: with both axes mirrored from the same start pose, TP2 at the SAME
    % dwell is the exact negation of TP1 (measured: identical closed-loop force
    % peak and rms to the digit), which is little new information.
    c{end+1} = mk('TP1_telica_y000',     'train',  0.00, ...
                  pt(TEL, +1, +1, X_full, Y_full, DWELL));
    c{end+1} = mk('TP2_telica_y000_rev', 'train',  0.00, ...
                  pt(TEL, -1, -1, X_full, Y_full, 0.45));
    % TP3/TP4 carry the offset lattices that reach the +0.30 and -0.30 bounds.
    % TP3 also runs the long dwell: more of the record is standstill and
    % settling, which is the regime the absorber rings down in.
    c{end+1} = mk('TP3_telica_yp06',     'train', +0.06, ...
                  pt(TEL, +1, -1, X_full, Y_full, 0.60));
    c{end+1} = mk('TP4_telica_ym06',     'train', -0.06, ...
                  pt(TEL, -1, +1, X_full, Y_full, DWELL));

    % ── validation ──────────────────────────────────────────────────────────
    % VP1: Y restricted to the low band with Y_op at its midpoint, matching the
    % V2_aprbs_Ylow convention (Y_op = midpoint of the record's Y range). -0.22
    % is on the 80 mm lattice of that band, so -0.30 and -0.14 are both reached
    % with full strokes.
    c{end+1} = mk('VP1_telica_ylow',     'val',   -0.22, ...
                  pt(TEL, +1, +1, X_full, Y_low,  0.40));
    % VP2: interior Y_op, full range, third dwell value. +0.10 mirrors the Y used
    % by V1_standstill_Yp10 and V3_ysweep_Yp10, so the validation set keeps a
    % consistent unseen-Y story across excitation types.
    c{end+1} = mk('VP2_telica_yp10',     'val',   +0.10, ...
                  pt(TEL, +1, -1, X_full, Y_full, 0.35));

    % ── test ────────────────────────────────────────────────────────────────
    c{end+1} = mk('EP1_telica_test',     'test',   0.00, ...
                  pt(TEL, -1, +1, X_full, Y_full, 0.25));

    records = [c{:}];
    for k = 1:numel(records)
        % Same rule as gtd_build_records (100*track_id + k), offset by 200 so
        % these ids cannot collide with that table's seeds. Nothing in this
        % dataset actually consumes the draw: excitation is 'none' so there is
        % no multisine realisation, and the reference is fully deterministic.
        % Kept for field parity with gtd_build_records.
        records(k).seed = 100*cfg.track_id + 200 + k;
    end
end

% ── local builders ──────────────────────────────────────────────────────────

function r = mk(id, split, Y_op, p)
    r = struct('id',id, 'split',split, 'class','telica', 'Y_op',Y_op, 'seed',0, 'p',p);
end

function p = pt(TEL, Y_dir, X_dir, X_range, Y_range, dwell)
% Telica record params: measured kinematics + which way the shuttle sets off and
% how long it dwells. Two things are deliberately NOT parameters. The START pose
% is [0, Y_op], fixed by the limiter's linearisation point. The STROKE is the
% measured machine value in TEL and is never scaled, so every move also carries
% the measured duration. pnone() sets excitation to 'none' exactly as E4 does.
    p            = TEL;
    p.Y_dir      = Y_dir;     p.X_dir = X_dir;
    p.X_range    = X_range;   p.Y_range = Y_range;
    p.dwell      = dwell;
    p.amp_frac   = 1.0;       % inert with excitation 'none'; kept for field parity
    p            = pnone(p);
end

function p = pnone(p)
% Turn the injected excitation off, the same construction E4_multisine_off uses.
    p.excitation = 'none';
end
