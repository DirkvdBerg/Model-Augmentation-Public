function ms = gtd_make_multisine(record, plant, cfg)
% GTD_MAKE_MULTISINE  Injected stage force for one record.
%   ms = GTD_MAKE_MULTISINE(record, plant, cfg) returns a struct:
%     ms.f_stage  (N x 3)  stage force [F_X1, F_X2, F_Y] to inject at plant input
%     ms.A        (1 x 3)  applied generalised amplitudes [A_sym, A_anti(torque), A_Y]
%     ms.info     struct   provenance (seed, band, per-channel crest factors, ...)
%
%   Excitation type comes from record.p.excitation:
%     'multisine' : 3 independent random-phase multisines in GENERALISED channels
%                   [f_sym, f_anti, f_Y], period = full record, low-crest-factor
%                   selection (best of N), seeded. CF is scored on the
%                   constrained quantity: f_sym/f_Y on their own signal (stage
%                   force is a uniform scaling), f_anti on the closed-loop yaw
%                   response. Amplitudes: A_sym/A_Y fixed (GATE-2), A_anti sized
%                   to the yaw budget. Transformed to stage via P^{-1}.
%     'none'      : zero force (the ILC-shape and route records).
%   The lines lie on the cfg.f_step grid (0.5 Hz = every 6th DFT line of the 12 s record, so
%   the multisine is periodic in 2 s) inside [f_low, f_high].

    if strcmp(record.p.excitation, 'none')
        ms = pack(zeros(cfg.N_record,3), [0 0 0], struct('excitation','none'));
        return
    end

    [f_unit, lines] = select_multisine(record, plant, cfg);

    % ── amplitudes: A_sym/A_Y fixed; A_anti sized to the yaw budget ─────────
    A_anti    = gtd_size_anti_amp(f_unit(:,2), plant, cfg);
    A         = record.p.amp_frac * [cfg.A_sym, A_anti, cfg.A_Y];
    f_generalised = f_unit .* A;                        % (N x 3) .* (1 x 3)
    f_stage   = gtd_logical_to_stage(f_generalised, cfg);

    ms = pack(f_stage, A, struct('excitation','multisine', 'seed',record.seed, ...
              'band',[cfg.f_low cfg.f_high], 'A_anti',A_anti, ...
              'f_step',cfg.f_step, 'n_lines',numel(lines), 'bins',lines));
end

% ── multisine synthesis and crest-factor selection ──────────────────────────

function [f_unit, bins] = select_multisine(record, plant, cfg)
% Draw n candidates per generalised channel and keep the lowest-crest-factor one.
    N     = cfg.N_record;
    bins  = line_bins(cfg);
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

function bins = line_bins(cfg)
% Excited DFT bins (1-based) of freqs = f_low : f_step : f_high; each must fall exactly on a
% DFT line of the record.
    df    = cfg.fs / cfg.N_record;
    freqs = cfg.f_low : cfg.f_step : cfg.f_high;
    k     = freqs / df;
    assert(all(abs(k - round(k)) < 1e-9), 'line grid is not on the DFT grid of the record');
    bins  = round(k) + 1;
end

function x = synth(bins, N)
% One unit-RMS random-phase multisine via IFFT (flat magnitude on the band).
    X = zeros(N,1);
    X(bins) = exp(1j * 2*pi * rand(numel(bins),1));
    x = ifft(X, 'symmetric');
    x = x / rms(x);
end

function ms = pack(f_stage, A, info)
    ms = struct('f_stage', f_stage, 'A', A, 'info', info);
end
