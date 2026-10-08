function [r, t] = gtd_make_reference(record, cfg)
% GTD_MAKE_REFERENCE  Build the [X1,X2,Y] stage reference (N x 3) for one record.
%   [r, t] = GTD_MAKE_REFERENCE(record, cfg) dispatches on record.class and
%   returns the motion reference padded to the full 12 s record. This is MOTION
%   ONLY; the injected excitation (multisine/sinesweep) is added later in the
%   multisine layer. Setpoint profiles are third-order (jerk-limited) via ETEL.
%
%   Spec: docs/trajectory-generation-spec-draft.md sections 1.4, 1.11, 3-5.

    switch record.class
        case 'standstill'
            [r, t] = ref_standstill(record, cfg);
        case 'oscillatory'
            [r, t] = ref_oscillatory(record, cfg);
        case 'aprbs'
            [r, t] = ref_aprbs(record, cfg);
        otherwise
            error('gtd_make_reference:class', 'unknown class ''%s''', record.class);
    end
    [r, t] = pad_to_record(r, cfg);
end

% ── reference shapes ────────────────────────────────────────────────────────

function [r, t] = ref_standstill(record, cfg)
% Constant hold at [0,0,Y_op]: multisine (or sweep) is the only excitation.
    row = [0, 0, record.Y_op];
    N   = cfg.n_hold + cfg.n_active + cfg.n_hold;
    r   = repmat(row, N, 1);
    t   = cfg.ts * (0:N-1)';
end

function [r, t] = ref_oscillatory(record, cfg)
% Sinusoidal sum in logical coords with a half-cosine fade so position,
% velocity and acceleration all start/end at zero. Covers y-sweep and lissajous
% (they differ only in parameter values). Adapted from make_ref_oscillatory.
    p     = record.p;
    N     = cfg.n_active;
    t_osc = (0:N-1)' * cfg.ts;
    fade  = fade_envelope(N, round(0.5/cfg.ts));

    X_sym  = fade .* (p.A_sym  * sin(2*pi*p.f_sym  * t_osc));
    X_anti = fade .* (p.A_anti * sin(2*pi*p.f_anti * t_osc));
    X1 = X_sym + X_anti;
    X2 = X_sym - X_anti;
    Y  = p.Y_center + fade .* (p.A_y * sin(2*pi*p.f_y * t_osc));

    r_hold = repmat([0, 0, p.Y_center], cfg.n_hold, 1);
    r = [r_hold; [X1, X2, Y]; r_hold];
    t = cfg.ts * (0:size(r,1)-1)';
end

function [r, t] = ref_aprbs(record, cfg)
% Randomized sequence of jerk-limited point-to-point moves (position-level
% APRBS). Setpoints drawn uniformly per record in logical coords, 0.1 s holds
% between moves, filling the 10 s active window. Reproducible via record.seed.
    p = record.p;
    rng(record.seed);

    cur  = [0, 0, record.Y_op];                 % logical [X_sym, X_anti, Y]
    body = zeros(0, 3);                          % stage rows [X1,X2,Y]
    while true
        tgt = [rand_in(p.X_sym_range), rand_in(p.X_anti_range), rand_in(p.Y_range)];
        mv  = move_between(cur, tgt, p, cfg.ts); % (K x 3) stage rows
        seg = [mv; repmat(mv(end,:), cfg.n_hold_short, 1)];   % move + 0.1 s hold
        % Never truncate mid-move: stop before a segment would overflow the
        % window, so the body always ends at rest (no velocity-jump seam).
        if size(body,1) + size(seg,1) > cfg.n_active
            break;
        end
        body = [body; seg]; %#ok<AGROW>
        cur  = tgt;
    end
    % Fill the remainder to exactly n_active with a hold at the current position.
    start_stage = logi2stage([0, 0, record.Y_op]);
    rem = cfg.n_active - size(body,1);
    if rem > 0
        body = [body; repmat(logi2stage(cur), rem, 1)];
    end

    r = [repmat(start_stage, cfg.n_hold, 1); body; repmat(body(end,:), cfg.n_hold, 1)];
    t = cfg.ts * (0:size(r,1)-1)';
end

% ── helpers ─────────────────────────────────────────────────────────────────

function mv = move_between(cur, tgt, p, ts)
% Jerk-limited multi-axis move between two logical setpoints -> stage rows.
% Each logical axis is profiled with its own limits (sym/anti use the X limits,
% Y uses the Y limits), padded to the longest, then combined to stage coords.
    pv_sym  = sp1d_signed(tgt(1)-cur(1), p.vmax_X, p.amax_X, p.jerkTime, ts);
    pv_anti = sp1d_signed(tgt(2)-cur(2), p.vmax_X, p.amax_X, p.jerkTime, ts);
    pv_Y    = sp1d_signed(tgt(3)-cur(3), p.vmax_Y, p.amax_Y, p.jerkTime, ts);

    K = max([numel(pv_sym), numel(pv_anti), numel(pv_Y), 1]);
    sym  = cur(1) + pvpad(pv_sym,  K);
    anti = cur(2) + pvpad(pv_anti, K);
    Y    = cur(3) + pvpad(pv_Y,    K);
    mv = [sym + anti, sym - anti, Y];
end

function pv = sp1d_signed(delta, vmax, amax, jerkTime, ts)
% Signed third-order position profile, 0 -> delta. Empty for a zero move.
    if delta == 0, pv = zeros(0,1); return; end
    mag = thirdOrderSetpointETEL(abs(delta), vmax, amax, amax/jerkTime, Inf, ts);
    pv  = sign(delta) * mag(:,1);
end

function v = pvpad(v, K)
% Pad a position profile to length K by holding its final value (0 if empty).
    if isempty(v), v = zeros(K,1); return; end
    v = [v; v(end)*ones(K-numel(v), 1)];
end

function fade = fade_envelope(N, n_fade)
% Half-cosine ramp up at the start and down at the end.
    fade = ones(N,1);
    ramp = 0.5 * (1 - cos(pi * (0:n_fade-1)' / n_fade));
    fade(1:n_fade)         = ramp;
    fade(end-n_fade+1:end) = flipud(ramp);
end

function s = logi2stage(q)
% Logical position [X_sym, X_anti, Y] -> stage [X1, X2, Y].
    s = [q(1)+q(2), q(1)-q(2), q(3)];
end

function x = rand_in(range)
    x = range(1) + (range(2)-range(1)) * rand;
end

function [r, t] = pad_to_record(r, cfg)
% Pad the final hold out to the full N_record (12 s) sample count.
    N = size(r,1);
    assert(N <= cfg.N_record, 'reference (%d) longer than the 12 s record (%d)', N, cfg.N_record);
    r = [r; repmat(r(end,:), cfg.N_record - N, 1)];
    t = cfg.ts * (0:size(r,1)-1)';
end
