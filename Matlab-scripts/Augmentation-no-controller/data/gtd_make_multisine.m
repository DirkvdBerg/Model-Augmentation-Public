function ms = gtd_make_multisine(record, plant, cfg)
% GTD_MAKE_MULTISINE  Injected stage force for one record.
%   ms = GTD_MAKE_MULTISINE(record, plant, cfg) returns a struct:
%     ms.f_stage  (N x 3)  stage force [F_X1, F_X2, F_Y] to inject at plant input
%     ms.A        (1 x 3)  applied logical amplitudes [A_sym, A_anti(torque), A_Y]
%     ms.info     struct   provenance (seed, band, per-channel crest factors, ...)
%
%   Excitation type comes from record.p.excitation:
%     'multisine' : 3 independent random-phase multisines in LOGICAL channels
%                   [f_sym, f_anti, f_Y], period = full record, low-crest-factor
%                   selection (best of N), seeded + cached. CF is scored on the
%                   constrained quantity: f_sym/f_Y on their own signal (stage
%                   force is a uniform scaling), f_anti on the closed-loop yaw
%                   response. Amplitudes: A_sym/A_Y fixed (GATE-2), A_anti sized
%                   to the yaw budget. Transformed to stage via P^{-1}.
%     'sinesweep' : force chirp on f_Y over the active window (E1).
%     'none'      : zero force (E4 regression check).
%
%   Spec: docs/trajectory-generation-spec-draft.md sections 1.4-1.9, 5.

    N = cfg.N_record;

    % CHANGED (open-loop variant): open loop the excitation IS the whole input, so
    % it is designed directly in stage coordinates and must be known analytically
    % (gtd_run_simulation evaluates it at the RK4 stage times, not just on grid).
    if isfield(cfg, 'openloop') && cfg.openloop
        ms = make_openloop(record, cfg);
        return
    end

    switch record.p.excitation
        case 'none'
            ms = pack(zeros(N,3), [0 0 0], struct('excitation','none'));
            return
        case 'sinesweep'
            ms = make_sinesweep(record, cfg);
            return
    end

    % ── multisine: reuse cached unit realization if params match ────────────
    [ok, f_unit] = load_cache(record, cfg);
    if ~ok
        f_unit = select_multisine(record, plant, cfg);
        save_cache(record, cfg, f_unit);
    end

    % ── amplitudes: A_sym/A_Y fixed; A_anti sized to the yaw budget ─────────
    A_anti    = gtd_size_anti_amp(f_unit(:,2), plant, cfg);
    A         = record.p.amp_frac * [cfg.A_sym, A_anti, cfg.A_Y];
    f_logical = f_unit .* A;                        % (N x 3) .* (1 x 3)
    f_stage   = gtd_logical_to_stage(f_logical, cfg);

    ms = pack(f_stage, A, struct('excitation','multisine', 'seed',record.seed, ...
              'band',[cfg.f_low cfg.f_high], 'A_anti',A_anti));
end

% ── multisine synthesis and crest-factor selection ──────────────────────────

function f_unit = select_multisine(record, plant, cfg)
% Draw n candidates per logical channel and keep the lowest-crest-factor one.
    N     = cfg.N_record;
    df    = cfg.fs / N;
    freqs = cfg.f_low : df : cfg.f_high;
    bins  = round(freqs / df) + 1;                 % lines fall exactly on FFT bins
    H_yaw = [1 -1 0] * plant.sys_cl * ([1; -1; 0] / cfg.Lb);

    rng(record.seed);                              % reproducible per-record draw
    sym = []; anti = []; Yc = [];
    cf_sym = Inf; cf_anti = Inf; cf_Y = Inf;
    for ic = 1:cfg.n_ms_candidates
        xs = synth(bins, N);
        xa = synth(bins, N);
        xy = synth(bins, N);
        cs = max(abs(xs));                         % unit RMS -> CF = peak
        cy = max(abs(xy));
        yaw = lsim(H_yaw, xa);  ca = max(abs(yaw)) / rms(yaw);   % constrained CF for anti
        if cs < cf_sym,  sym  = xs;  cf_sym  = cs;  end
        if ca < cf_anti, anti = xa;  cf_anti = ca;  end
        if cy < cf_Y,    Yc   = xy;  cf_Y    = cy;  end
    end
    f_unit = [sym, anti, Yc];
end

function x = synth(bins, N)
% One unit-RMS random-phase multisine via IFFT (flat magnitude on the band).
    X = zeros(N,1);
    X(bins) = exp(1j * 2*pi * rand(numel(bins),1));
    x = ifft(X, 'symmetric');
    x = x / rms(x);
end

function ms = make_sinesweep(record, cfg)
% Linear force chirp on f_Y over the active window (E1), tapered at both ends.
    N = cfg.N_record;  n_a = cfg.n_active;  t_a = (0:n_a-1)' * cfg.ts;
    b   = record.p.sweep_band;  f0 = b(1);  f1 = b(2);  T = t_a(end);
    amp = record.p.amp_frac * cfg.A_Y * sqrt(2);           % peak; RMS ~ amp_frac*A_Y
    phase = 2*pi * (f0*t_a + (f1 - f0)/(2*T) * t_a.^2);    % linear instantaneous freq

    % Half-cosine fade in/out (0.5 s) so the chirp does not slam the system on/off.
    % An abrupt start/stop kicks all modes -> large onset/offset transients that bury
    % the resonance response. Same taper idea as make_ref_oscillatory.
    n_fade = round(0.5 / cfg.ts);
    win = ones(n_a, 1);
    ramp = 0.5 * (1 - cos(pi * (0:n_fade-1)' / n_fade));
    win(1:n_fade)         = ramp;
    win(end-n_fade+1:end) = flipud(ramp);

    f_logical = zeros(N, 3);
    f_logical(cfg.n_hold + (1:n_a), 3) = amp * win .* sin(phase);   % on f_Y, active window
    f_stage = gtd_logical_to_stage(f_logical, cfg);
    ms = pack(f_stage, [0, 0, amp/sqrt(2)], struct('excitation','sinesweep', 'band',b));
end

function ms = pack(f_stage, A, info)
    ms = struct('f_stage', f_stage, 'A', A, 'info', info);
end

% ── open-loop excitation (analytic in t, stage coordinates) ─────────────────

function ms = make_openloop(record, cfg)
% MAKE_OPENLOOP  The full open-loop input: no reference, no controller, no
% feedforward, so this signal IS u_total.
%
% Three INDEPENDENT random-phase multisines, one per stage channel [X1, X2, Y],
% as in generate_openloop_record.m. Independent rails excite yaw automatically,
% which is why there is no separate anti channel: the logical [f_sym, f_anti]
% split exists in the closed-loop path only so A_anti can be sized against the
% yaw budget, and open loop there is no tracking error to budget against.
%
% PERIODIC by construction with period t_record / ol_n_periods, so the lines sit
% on multiples of 1/T_p. This is Jan's construction
% (scripts/ecc_2025/msd_ndof_data_generation_dynamic.py:41-70): generate n
% periods and discard the first. Whether that transfers to this plant is open,
% because K11 = K33 = 0 means position has no periodic steady state; the record
% is generated so the question can be MEASURED rather than assumed.
%
% Returns, in addition to the closed-loop fields:
%   ms.f_half  (N x 3)  the same signal at t + ts/2, for the RK4 stage sampling
%   ms.fun     handle   t -> 1x3 stage force, for the ode45 fidelity check
%
% Amplitude is cfg.ol_A_rms in STAGE RMS. HEURISTIC, see gtd_config.

    N = cfg.N_record;  ts = cfg.ts;

    if strcmp(record.p.excitation, 'sinesweep')
        ms = make_openloop_sweep(record, cfg);
        return
    end
    assert(strcmp(record.p.excitation, 'multisine'), ...
        'gtd_make_multisine:openloop', ...
        'open-loop generation supports ''multisine'' and ''sinesweep'' only, got ''%s''', ...
        record.p.excitation);

    np = cfg.ol_n_periods;
    Np = round(N / np);
    assert(Np * np == N, 'gtd_make_multisine:period', ...
        'N_record (%d) must be an integer multiple of ol_n_periods (%d)', N, np);

    dfp  = cfg.fs / Np;                                  % line spacing = 1 / period
    bins = (round(cfg.f_low/dfp) : round(cfg.f_high/dfp))' + 1;   % lines on exact FFT bins
    fr   = (bins - 1) * dfp;                             % [Hz]
    tp   = (0:Np-1)' * ts;

    % Crest-factor selection is scored on the cheap IFFT realization; only the
    % winning PHASES are kept, and the signal itself is then rebuilt analytically
    % from those phases so that grid and midpoint values come from one formula.
    rng(record.seed);
    ph = zeros(numel(fr), 3);  cf = inf(1, 3);
    for ic = 1:cfg.n_ms_candidates
        for c = 1:3
            phc = 2*pi*rand(numel(fr), 1);
            X = zeros(Np, 1);  X(bins) = exp(1j * phc);
            x = ifft(X, 'symmetric');  x = x / rms(x);
            if max(abs(x)) < cf(c), cf(c) = max(abs(x));  ph(:, c) = phc; end
        end
    end

    up = zeros(Np, 3);  uh = zeros(Np, 3);
    for c = 1:3
        for i = 1:numel(fr)
            w = 2*pi*fr(i);
            up(:, c) = up(:, c) + cos(w * tp            + ph(i, c));
            uh(:, c) = uh(:, c) + cos(w * (tp + 0.5*ts) + ph(i, c));
        end
    end

    A  = record.p.amp_frac * cfg.ol_A_rms;
    sc = A ./ std(up, 0, 1);                             % scale fixed once, on the grid
    up = up .* sc;  uh = uh .* sc;

    % One period tiled is exact for BOTH grids: w*Np*ts = 2*pi*fr*T_p is an integer
    % multiple of 2*pi for every line, so the midpoint values repeat too.
    % (np + nd) periods are generated; gtd_run_simulation keeps the last np.
    nd = cfg.ol_n_discard;
    f_stage = repmat(up, np + nd, 1);
    f_half  = repmat(uh, np + nd, 1);

    fun = @(tq) arrayfun(@(c) sc(c) * sum(cos(2*pi*fr*tq + ph(:, c))), 1:3);

    ms = struct('f_stage', f_stage, 'f_half', f_half, 'fun', fun, 'A', A, ...
                'n_skip', nd * Np, ...
                'info', struct('excitation','multisine', 'seed',record.seed, ...
                               'band',[cfg.f_low cfg.f_high], 'n_lines',numel(fr), ...
                               'n_periods',np, 'n_discard',nd, 'T_period',Np*ts, ...
                               'crest',cf, 'coords','stage'));
end

function ms = make_openloop_sweep(record, cfg)
% Open-loop force chirp on the stage Y rail (test record OE2), tapered at both
% ends by the same half-cosine as the closed-loop make_sinesweep so the onset
% transient does not bury the resonance response.
    N = cfg.N_record;  ts = cfg.ts;
    n_a = cfg.n_active;  n0 = cfg.n_hold;
    b = record.p.sweep_band;  f0 = b(1);  f1 = b(2);
    T = (n_a - 1) * ts;
    amp = record.p.amp_frac * cfg.ol_A_rms(3) * sqrt(2);      % peak; RMS ~ amp_frac*A_Y

    n_fade = round(0.5 / cfg.ts);
    ramp = 0.5 * (1 - cos(pi * (0:n_fade-1)' / n_fade));

    chirp = @(tq) amp .* taper(tq, T, n_fade*ts) .* ...
                  sin(2*pi * (f0*tq + (f1 - f0)/(2*T) * tq.^2));
    ta = (0:n_a-1)' * ts;

    f_stage = zeros(N, 3);  f_half = zeros(N, 3);
    f_stage(n0 + (1:n_a), 3) = chirp(ta);
    f_half( n0 + (1:n_a), 3) = chirp(ta + 0.5*ts);

    fun = @(tq) [0, 0, sweep_at(tq, n0*ts, T, chirp)];

    % No discard: a chirp is not periodic, so there is no first period to drop and
    % the rectification accumulates throughout rather than settling. OE2 therefore
    % keeps its full 12 s including the onset transient. Flagged, not hidden.
    ms = struct('f_stage', f_stage, 'f_half', f_half, 'fun', fun, ...
                'A', [0, 0, amp/sqrt(2)], 'n_skip', 0, ...
                'info', struct('excitation','sinesweep', 'band',b, 'coords','stage', ...
                               'ramp',numel(ramp)*ts, 'n_discard',0));
end

function w = taper(tq, T, tf)
% Half-cosine fade of length tf at each end of [0, T].
    w = ones(size(tq));
    a = tq < tf;             w(a) = 0.5 * (1 - cos(pi * tq(a) / tf));
    b = tq > (T - tf);       w(b) = 0.5 * (1 - cos(pi * (T - tq(b)) / tf));
    w(tq < 0 | tq > T) = 0;
end

function v = sweep_at(tq, t0, T, chirp)
    tl = tq - t0;
    if tl < 0 || tl > T, v = 0; else, v = chirp(tl); end
end

% ── per-record cache (keyed by seed/band/period/operating point) ────────────

function f = cache_path(record, cfg)
    f = fullfile(cfg.out_dir, '_cache', ['ms_' record.id '.mat']);
end

function [ok, f_unit] = load_cache(record, cfg)
    ok = false;  f_unit = [];
    f = cache_path(record, cfg);
    if exist(f, 'file')
        C = load(f);
        if isfield(C,'info') && C.info.seed == record.seed && ...
           isequal(C.info.band, [cfg.f_low cfg.f_high]) && ...
           C.info.N == cfg.N_record && C.info.Yop == record.Y_op && ...
           C.info.n_cand == cfg.n_ms_candidates
            ok = true;  f_unit = C.f_unit;
        end
    end
end

function save_cache(record, cfg, f_unit)
    d = fileparts(cache_path(record, cfg));
    if ~exist(d, 'dir'), mkdir(d); end
    info = struct('seed',record.seed, 'band',[cfg.f_low cfg.f_high], ...
                  'N',cfg.N_record, 'Yop',record.Y_op, 'n_cand',cfg.n_ms_candidates);
    save(cache_path(record, cfg), 'f_unit', 'info');
end
