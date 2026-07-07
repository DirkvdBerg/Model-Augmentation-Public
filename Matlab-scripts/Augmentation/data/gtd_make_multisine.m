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
