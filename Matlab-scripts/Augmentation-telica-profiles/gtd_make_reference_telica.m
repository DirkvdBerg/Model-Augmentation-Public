function [r, t] = gtd_make_reference_telica(record, cfg)
% GTD_MAKE_REFERENCE_TELICA  Reference builder with the Telica 'telica' class added.
%   [r, t] = GTD_MAKE_REFERENCE_TELICA(record, cfg) handles record.class ==
%   'telica' here and DELEGATES every other class to the original, unmodified
%   gtd_make_reference. Same contract: returns the [X1,X2,Y] STAGE reference
%   (N x 3) padded to the full cfg.N_record record, motion only, no excitation.
%
%   WHY THIS FILE EXISTS. The 'telica' class needs a CYCLOIDAL (sinusoidal
%   acceleration) setpoint. The original gtd_make_reference builds point-to-point
%   moves with thirdOrderSetpointETEL, a third-order jerk-limited profile, which
%   is a different profile class: fitted against the real Telica logs it leaves
%   51.9 um rms against the cycloidal model's 9.4 um. No jerkTime value closes
%   that gap, because a trapezoidal acceleration cannot be a sine. Adding the
%   class inside gtd_make_reference.m is not permitted (nothing under
%   Matlab-scripts/Augmentation/ may be modified, D-206), so the dispatch is
%   wrapped here instead. See D-206.
%
%   THE PROFILE, measured from the machine rather than assumed. Real Telica
%   setpoints (M0) are cycloidal with a velocity limit, established on
%   kamtin-data/Data Telica/'06 40 mm XL 80 mm YL', all 15 operating points:
%     - measured acceleration tracks sin(2*pi*tau) pointwise to within 0.003 of
%       peak across the whole X move, with no acceleration plateau anywhere
%     - shape factors on X are v_max*T/D = 1.998 and a_max*T^2/D = 6.283, the
%       exact theoretical values 2 and 2*pi of a cycloidal profile
%     - fit residual 9.4 um on X (0.023 % of stroke), 14.8 um on Y (0.019 %)
%   All 15 operating points use the IDENTICAL trajectory; only the start position
%   differs. That is why this is a parametric generator and not a log replay: a
%   replay carries a single 40 x 80 mm diagonal move and cannot cover the Y
%   scheduling range the LPV model needs.

    if ~strcmp(record.class, 'telica')
        [r, t] = gtd_make_reference(record, cfg);
        return
    end

    [r, t] = ref_telica(record, cfg);
end

% ── the Telica reference shape ──────────────────────────────────────────────

function [r, t] = ref_telica(record, cfg)
% Repeated Telica point-to-point moves: X and Y start together, each runs its
% own cycloidal profile, then both dwell. Axes shuttle independently inside
% their ranges and reflect at the bounds, so the X and Y sequences go out of
% phase and the record covers a spread of (X, Y) combinations rather than one
% diagonal repeated.
    p = record.p;
    % No RNG: every record is fully deterministic. The move geometry is fixed by
    % the measured machine and nothing about it is drawn.
    n_dwell = max(1, round(p.dwell / cfg.ts));

    % THE RECORD MUST START AT THE LINEARISATION POINT [0, 0, Y_op], exactly as
    % the original ref_aprbs does (cur = [0, 0, record.Y_op]). This is not a
    % style choice. gtd_enforce_limits forms r_pert = r - [0, 0, Y_op] and runs
    % lsim from ZERO state, so any offset between the first reference sample and
    % Y_op is a step input into the closed loop at t = 0. Starting a record at
    % Y = -0.30 with Y_op = 0 injected a 0.30 m step and the pre-check returned
    % 2.19e6 N and 172 m/s, a factor 700 over limit, on a trajectory whose own
    % peak velocity is 1.5 m/s. Measured 2026-09-19.
    x = 0;            dx = p.X_dir;        % logical X_sym [m]
    y = record.Y_op;  dy = p.Y_dir;        % Y [m]
    body = zeros(0, 3);                    % stage rows [X1, X2, Y]

    while true
        [x_next, dx] = shuttle(x, dx, p.X_stroke, p.X_range);
        [y_next, dy] = shuttle(y, dy, p.Y_stroke, p.Y_range);

        % EVERY move is the full measured Telica stroke, so every move also has
        % the measured duration. This is the user's hard requirement: the
        % setpoint must match the machine exactly, so a truncated or scaled move
        % is not an acceptable degradation, it is a wrong record. Asserted here
        % rather than trusted, because the failure is silent in the .mat file.
        assert(abs(abs(x_next - x) - p.X_stroke) < 1e-12, ...
               'X move is %.6f m, not the Telica stroke %.6f m', abs(x_next-x), p.X_stroke);
        assert(abs(abs(y_next - y) - p.Y_stroke) < 1e-12, ...
               'Y move is %.6f m, not the Telica stroke %.6f m', abs(y_next-y), p.Y_stroke);

        sx = cyc_profile(x_next - x, p.X_vmax, p.X_amax, cfg.ts);
        sy = cyc_profile(y_next - y, p.Y_vmax, p.Y_amax, cfg.ts);

        % Both axes are commanded at the same instant, as on the machine (X and
        % Y motion starts were measured 0.4 ms apart). The shorter axis holds
        % its endpoint while the longer one finishes.
        K   = max([numel(sx), numel(sy), 1]);
        seg = [x + hold_pad(sx, K), y + hold_pad(sy, K)];
        seg = [seg; repmat(seg(end,:), n_dwell, 1)];                     %#ok<AGROW>

        % Never truncate mid-move: stop before a segment would overflow the
        % active window, so the body always ends at rest.
        if size(body,1) + size(seg,1) > cfg.n_active
            break
        end
        body = [body; logi2stage(seg)];                                  %#ok<AGROW>
        x = x_next;  y = y_next;
    end

    assert(~isempty(body), ...
           ['ref_telica: not one move fits in the %.1f s active window. ' ...
            'Move + dwell is %.3f s.'], cfg.t_active, (K + n_dwell)*cfg.ts);

    % Fill the remainder of the active window with a hold at the current pose.
    rem = cfg.n_active - size(body,1);
    if rem > 0
        body = [body; repmat(body(end,:), rem, 1)];
    end

    start_stage = logi2stage([0, record.Y_op]);
    r = [repmat(start_stage, cfg.n_hold, 1); body; repmat(body(end,:), cfg.n_hold, 1)];

    % Pad the final hold out to the full cfg.N_record, matching pad_to_record in
    % the original gtd_make_reference.
    N = size(r,1);
    assert(N <= cfg.N_record, 'reference (%d) longer than the record (%d)', N, cfg.N_record);
    r = [r; repmat(r(end,:), cfg.N_record - N, 1)];
    t = cfg.ts * (0:size(r,1)-1)';
end

% ── helpers ─────────────────────────────────────────────────────────────────

function s = cyc_profile(D, vmax, amax, ts)
% Velocity-limited CYCLOIDAL displacement, 0 -> D, sampled at ts. Empty for a
% zero move. Three phases: half-sine acceleration, constant-velocity cruise,
% half-sine deceleration. The cruise collapses to zero length when the move is
% too short to reach vmax, which is the pure-cycloid case.
%
% MEASURED: this is the profile the real Telica runs, see the file header and
% D-206. The closed form below is the model that was fitted to the logs.
%   phase 1   v(t) = (v/2)(1 - cos(pi t/Ta))   so   a_max = pi*v/(2*Ta)
%   hence     Ta   = pi*v/(2*a_max)   and   D_acc = v*Ta/2
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
        % Too short to reach vmax: solve Dm = pi*v^2/(2*a_max) for the peak.
        v  = sqrt(2*Dm*amax/pi);
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
% Step one FULL stroke in dir_in, REFLECTING at the range bounds.
%
% Reflection, never clamping. Clamping to the bound would produce one shorter
% move at each turnaround, and a shorter move has a different duration and a
% different peak velocity, so it is no longer the machine's setpoint. Matching
% the setpoint exactly outranks reaching the range endpoints.
%
% CONSEQUENCE, handled in gtd_build_records_telica rather than here: a record
% starting at Y_op walks the lattice Y_op + k*stroke, so it reaches the range
% endpoints only when the endpoints lie ON that lattice. From Y_op = 0 with an
% 80 mm stroke the traversal is [-0.24, +0.24], not [-0.30, +0.30]. The record
% table therefore offsets Y_op per record (0, +0.06, -0.06) so that the set as a
% whole reaches both bounds exactly and covers Y at 20 mm resolution, while
% every individual move stays a full 80 mm.
    TOL = 1e-12;
    nxt = cur + dir_in*stroke;
    if nxt > range(2) + TOL || nxt < range(1) - TOL
        dir_out = -dir_in;
        nxt     = cur + dir_out*stroke;
    else
        dir_out = dir_in;
    end
    assert(nxt <= range(2) + TOL && nxt >= range(1) - TOL, ...
           ['shuttle: a full stroke of %.4f m does not fit in range ' ...
            '[%.4f %.4f] from %.4f in either direction'], ...
           stroke, range(1), range(2), cur);
end

function v = hold_pad(v, K)
% Pad a displacement profile to length K by holding its final value.
    if isempty(v)
        v = zeros(K,1);
        return
    end
    v = [v; v(end)*ones(K-numel(v), 1)];
end

function s = logi2stage(q)
% Logical [X_sym, Y] -> stage [X1, X2, Y]. X_anti is identically zero: the real
% Telica logs GTRX1 and GTRX2 with bit-identical setpoints, so the machine
% commands no yaw and neither does this record.
    s = [q(:,1), q(:,1), q(:,2)];
end
