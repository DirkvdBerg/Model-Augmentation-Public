function [r, t] = make_reference(record, cfg)
% MAKE_REFERENCE  The stage reference [X1, X2, Y] of one record (N_record x 3), checked against the machine limits.
%   Each record starts at [0, 0, Y_op] and its motion ends at rest inside the 10 s window between two 0.5 s holds.

    % the motion of each record class
    %   standstill   hold at [0, 0, Y_op]; the multisine is the only excitation
    %   oscillatory  slow sweep of Y or X-Y Lissajous path, with a half-cosine fade in and out
    %   aprbs        random point-to-point moves with S-curve profiles, each distance drawn from one of
    %                six bands between 1 mm and the range, so small and large moves occur equally often
    %   telica       operational motion profile: sine-shaped moves of 40 mm on X and 80 mm on Y, X and Y
    %                together, with the shape of the ASMPT machine's logged setpoints
    %   route        operational motion profiles of the test records: sine-shaped or S-curve moves,
    %                X and Y together or alternating
    switch record.class
        case 'standstill'
            r = repmat([0, 0, record.Y_op], cfg.n_hold + cfg.n_active + cfg.n_hold, 1);
        case 'oscillatory'
            r = ref_oscillatory(record.p, cfg);
        case 'aprbs'
            r = ref_aprbs(record, cfg);
        case {'telica', 'route'}
            r = ref_route(record, cfg);
        otherwise
            error('make_reference: unknown record class ''%s''', record.class);
    end
    N = size(r, 1);
    assert(N <= cfg.N_record, 'reference (%d) longer than the 12 s record (%d)', N, cfg.N_record);
    r = [r; repmat(r(end,:), cfg.N_record - N, 1)];
    t = cfg.ts * (0:cfg.N_record-1)';
    check_limits(r, record, cfg);
end

function check_limits(r, record, cfg)
% The reference must stay inside the position, yaw, speed and acceleration limits. E3 runs at 100 % of
% the acceleration limit and may exceed it by the margin; E6 is the machine's own move above the rated
% acceleration and skips that check
    lim = cfg.lim;  p = record.p;  id = record.file;
    vel = diff(r) / cfg.ts;
    acc = diff(vel) / cfg.ts;
    assert(max(abs(r(:,1)))          <= lim.pos_X, '%s: X1 position limit', id);
    assert(max(abs(r(:,2)))          <= lim.pos_X, '%s: X2 position limit', id);
    assert(max(abs(r(:,3)))          <= lim.pos_Y, '%s: Y position limit', id);
    assert(max(abs(r(:,1)-r(:,2)))   <= lim.diff,  '%s: yaw |X1-X2| limit', id);
    assert(max(abs(vel), [], 'all')  <= lim.vel,   '%s: velocity limit', id);
    fa = 1;
    if isfield(p, 'acc_margin') && p.acc_margin, fa = 1 + cfg.acc_margin; end
    if ~(isfield(p, 'acc_exception') && p.acc_exception)
        assert(max(abs(acc(:,1:2)), [], 'all') <= fa*lim.acc_X, '%s: X acceleration limit', id);
        assert(max(abs(acc(:,3)))              <= fa*lim.acc_Y, '%s: Y acceleration limit', id);
    end
end

% Reference shapes

function r = ref_oscillatory(p, cfg)
% Sinusoids in generalised coordinates with a half-cosine fade, so position, velocity and
% acceleration start and end at zero. Y sweeps and Lissajous paths differ only in the parameters p
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
end

function r = ref_aprbs(record, cfg)
% Point-to-point moves with S-curve profiles and 0.1 s holds. X_sym and Y each cycle through the
% distance bands 1-3, 3-10, 10-30, 30-100, 100-300 mm and 300 mm to the range in shuffled
% order, log-uniform inside a band, random sign, reflected at the range bound; X_anti is
% uniform in its range. Changing the order of the random draws changes the reference of a seed
    p = record.p;
    rng(record.seed_sp);
    strata = [1 3; 3 10; 10 30; 30 100; 100 300; 300 1e9] * 1e-3;   % [m]
    cur = [0, 0, record.Y_op];  body = zeros(0, 3);  qs = [];  qy = [];
    while true
        if isempty(qs), qs = randperm(6); end
        if isempty(qy), qy = randperm(6); end
        ds = draw_stratum(strata(qs(1), :), cur(1), p.X_sym_range);  qs(1) = [];
        dy = draw_stratum(strata(qy(1), :), cur(3), p.Y_range);      qy(1) = [];
        tgt = [cur(1) + ds, p.X_anti_range(1) + diff(p.X_anti_range) * rand, cur(3) + dy];
        mv  = move_between(cur, tgt, p, cfg.ts);
        seg = [mv; repmat(mv(end, :), cfg.n_hold_short, 1)];
        if size(body, 1) + size(seg, 1) > cfg.n_active, break; end
        body = [body; seg]; %#ok<AGROW>
        cur = tgt;
    end
    body = [body; repmat(body(end, :), cfg.n_active - size(body, 1), 1)];
    r = [repmat(gen2stage([0, 0, record.Y_op]), cfg.n_hold, 1); body; repmat(body(end, :), cfg.n_hold, 1)];
end

function r = ref_route(record, cfg)
% Moves of exactly the stroke p.X_stroke / p.Y_stroke on a lattice from [0, Y_op], each axis
% shuttling inside its range and reflecting at the bounds, each move followed by a dwell of
% p.dwell. Fields with their default for the 'telica' class:
%   p.profile    'cyc' (default; cycloidal, the measured setpoint of the machine) or 'scurve'
%                (thirdOrderSetpointETEL with jmax = amax / p.T3)
%   p.pattern    'together' (default; X and Y start each move together) or 'alternate'
%                (X-only and Y-only moves in turn, X first)
%   p.first_half true: the first move is half a stroke on each axis (long strokes)
% No X_anti: the machine commands X1 and X2 identically.
    p = record.p;
    if ~isfield(p, 'profile'), p.profile = 'cyc'; end
    alternate = isfield(p, 'pattern') && strcmp(p.pattern, 'alternate');
    first_half = isfield(p, 'first_half') && p.first_half;
    n_dwell = max(1, round(p.dwell / cfg.ts));
    x = 0;            dx = p.X_dir;                      % generalised X_sym [m]
    y = record.Y_op;  dy = p.Y_dir;                      % Y [m]
    body = zeros(0, 3);
    k = 0;
    while true
        k = k + 1;
        movX = ~alternate || mod(k, 2) == 1;
        movY = ~alternate || ~movX;
        x_next = x;  y_next = y;  fx = 1;  fy = 1;
        if k == 1 && first_half
            fx = 0.5;  fy = 0.5;
            x_next = x + dx * p.X_stroke/2;
            y_next = y + dy * p.Y_stroke/2;
            assert(x_next >= p.X_range(1) - 1e-12 && x_next <= p.X_range(2) + 1e-12 && ...
                   y_next >= p.Y_range(1) - 1e-12 && y_next <= p.Y_range(2) + 1e-12, ...
                   'ref_route: the first half stroke leaves the range');
        else
            if movX, [x_next, dx] = shuttle(x, dx, p.X_stroke, p.X_range); end
            if movY, [y_next, dy] = shuttle(y, dy, p.Y_stroke, p.Y_range); end
        end
        % every move is exactly its stroke (or half of it): a shorter move would have another
        % duration and peak velocity than the machine's setpoint
        assert(abs(abs(x_next - x) - movX*fx*p.X_stroke) < 1e-12, 'X move %.6f m', abs(x_next - x));
        assert(abs(abs(y_next - y) - movY*fy*p.Y_stroke) < 1e-12, 'Y move %.6f m', abs(y_next - y));

        sx = route_profile(x_next - x, p.X_vmax, p.X_amax, p, cfg.ts);
        sy = route_profile(y_next - y, p.Y_vmax, p.Y_amax, p, cfg.ts);
        K   = max([numel(sx), numel(sy), 1]);
        seg = [x + pvpad(sx, K), y + pvpad(sy, K)];
        seg = [seg; repmat(seg(end,:), n_dwell, 1)];
        if size(body,1) + size(seg,1) > cfg.n_active, break; end
        body = [body; xy2stage(seg)];                                    %#ok<AGROW>
        x = x_next;  y = y_next;
    end
    assert(~isempty(body), 'ref_route: not one move fits in the %.1f s active window', cfg.t_active);
    body = [body; repmat(body(end,:), cfg.n_active - size(body,1), 1)];
    r = [repmat(xy2stage([0, record.Y_op]), cfg.n_hold, 1); body; repmat(body(end,:), cfg.n_hold, 1)];
end

% Move profiles

function d = draw_stratum(st, pos, range)
% One signed distance from stratum st = [lo hi] [m], reflected at the range bound.
    room = [range(2) - pos, pos - range(1)];          % room up, room down
    hi = min(st(2), max(room));  lo = min(st(1), hi);
    m = exp(log(lo) + rand * (log(hi) - log(lo)));    % log-uniform in the stratum
    s = sign(rand - 0.5);  if s == 0, s = 1; end
    if (s > 0 && m > room(1)) || (s < 0 && m > room(2)), s = -s; end
    d = s * m;
end

function mv = move_between(cur, tgt, p, ts)
% Jerk-limited move between two generalised setpoints [X_sym, X_anti, Y], as stage rows; X_sym and
% X_anti use the X limits, Y the Y limits, each padded to the longest.
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
% Signed third-order position profile 0 -> delta; empty for a zero move.
    if delta == 0, pv = zeros(0,1); return; end
    mag = thirdOrderSetpointETEL(abs(delta), vmax, amax, amax/jerkTime, Inf, ts);
    pv  = sign(delta) * mag(:,1);
end

function s = route_profile(D, vmax, amax, p, ts)
% One axis' displacement profile 0 -> D; empty for a zero move.
    switch p.profile
        case 'cyc'
            s = cyc_profile(D, vmax, amax, ts);
        case 'scurve'
            if D == 0, s = zeros(0,1); return; end
            mag = thirdOrderSetpointETEL(abs(D), vmax, amax, amax/p.T3, Inf, ts);
            s = sign(D) * mag(:,1);
        otherwise
            error('ref_route: unknown profile ''%s''', p.profile);
    end
end

function s = cyc_profile(D, vmax, amax, ts)
% Velocity-limited cycloidal displacement 0 -> D sampled at ts, empty for a zero move:
% half-sine acceleration, constant-velocity cruise, half-sine deceleration; no cruise when the
% move is too short to reach vmax. Fitted to the machine's logged setpoints.
%   v(t) = (v/2)(1 - cos(pi t/Ta)), so a_max = pi v / (2 Ta), Ta = pi v / (2 a_max), D_acc = v Ta / 2
    if D == 0
        s = zeros(0,1);
        return
    end
    sgn = sign(D);  Dm = abs(D);

    Ta   = pi*vmax/(2*amax);
    Dacc = vmax*Ta/2;
    if 2*Dacc <= Dm
        v  = vmax;
        Tc = (Dm - 2*Dacc)/vmax;
    else
        v  = sqrt(2*Dm*amax/pi);        % peak of a pure cycloid, Dm = pi v^2 / (2 a_max)
        Ta = pi*v/(2*amax);
        Tc = 0;
    end
    Dacc = v*Ta/2;

    N  = max(1, round((2*Ta + Tc)/ts));
    tt = (1:N)'*ts;                     % starts at ts; s = 0 is the preceding hold
    s  = zeros(N,1);

    i1 = tt <= Ta;
    s(i1) = v*( tt(i1)/2 - Ta/(2*pi)*sin(pi*tt(i1)/Ta) );

    i2 = tt > Ta & tt <= Ta + Tc;
    s(i2) = Dacc + v*(tt(i2) - Ta);

    i3 = tt > Ta + Tc;
    td = min(tt(i3) - Ta - Tc, Ta);     % clamp: sub-sample rounding must not overshoot
    s(i3) = Dacc + v*Tc + v*( td/2 + Ta/(2*pi)*sin(pi*td/Ta) );

    s = sgn * s;
end

function [nxt, dir_out] = shuttle(cur, dir_in, stroke, range)
% One full stroke in dir_in, reflecting (never clamping) at the range bounds, so every move
% keeps the machine's stroke. A record from Y_op walks the lattice Y_op + k stroke; the record
% table offsets Y_op so the set as a whole reaches both range bounds.
    TOL = 1e-12;
    nxt = cur + dir_in*stroke;
    if nxt > range(2) + TOL || nxt < range(1) - TOL
        dir_out = -dir_in;
        nxt     = cur + dir_out*stroke;
    else
        dir_out = dir_in;
    end
    assert(nxt <= range(2) + TOL && nxt >= range(1) - TOL, ...
           'shuttle: a full stroke of %.4f m does not fit in [%.4f %.4f] from %.4f in either direction', ...
           stroke, range(1), range(2), cur);
end

% Helpers

function v = pvpad(v, K)
% Pad a position profile to length K by holding its final value (zeros if empty).
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

function s = gen2stage(q)
% Generalised position [X_sym, X_anti, Y] -> stage [X1, X2, Y].
    s = [q(1)+q(2), q(1)-q(2), q(3)];
end

function s = xy2stage(q)
% Generalised rows [X_sym, Y] -> stage rows [X1, X2, Y] with X1 = X2 (no yaw).
    s = [q(:,1), q(:,1), q(:,2)];
end
