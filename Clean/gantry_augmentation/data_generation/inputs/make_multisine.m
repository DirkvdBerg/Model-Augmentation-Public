function ms = make_multisine(record, loop, cfg)
% MAKE_MULTISINE  The injected stage force [F_X1, F_X2, F_Y] of one record (N_record x 3) and its amplitudes [A_sym, A_anti, A_Y].
%   Three independent random-phase multisines on the generalised channels [X, Theta, Y], periodic over the record.
    if strcmp(record.p.excitation, 'none')
        ms = struct('f_stage', zeros(cfg.N_record, 3), 'A', [0 0 0]);
        return
    end

    % lines on the 0.5 Hz grid inside the band, each on a DFT line of the 12 s record
    df    = cfg.fs / cfg.N_record;
    freqs = cfg.f_low : cfg.f_step : cfg.f_high;
    k     = freqs / df;
    assert(all(abs(k - round(k)) < 1e-9), 'the line grid is not on the DFT grid of the record');
    bins  = round(k) + 1;

    % per channel, the draw with the lowest crest factor out of n_ms_candidates; Theta is scored on the
    % yaw it causes in closed loop, since the yaw limit is what constrains it
    H_yaw = [1 -1 0] * loop.sys_cl * ([1; -1; 0] / cfg.Lb);
    rng(record.seed_ms);
    sym = []; anti = []; Yc = [];
    cf_sym = Inf; cf_anti = Inf; cf_Y = Inf;
    for ic = 1:cfg.n_ms_candidates
        xs = synth(bins, cfg.N_record);
        xa = synth(bins, cfg.N_record);
        xy = synth(bins, cfg.N_record);
        cs = max(abs(xs));                         % unit rms, so the crest factor is the peak
        cy = max(abs(xy));
        yaw = lsim(H_yaw, xa);  ca = max(abs(yaw)) / rms(yaw);
        if cs < cf_sym,  sym  = xs;  cf_sym  = cs;  end
        if ca < cf_anti, anti = xa;  cf_anti = ca;  end
        if cy < cf_Y,    Yc   = xy;  cf_Y    = cy;  end
    end

    % generalised forces to stage forces with P^-1: the Theta channel is a torque [N m], divided by Lb on X1 and X2
    A = record.p.amp_frac * [cfg.A_sym, cfg.A_anti, cfg.A_Y];
    f_generalised = [sym, anti, Yc] .* A;
    ms = struct('f_stage', (cfg.P \ f_generalised.').', 'A', A);
end

function x = synth(bins, N)
% One unit-rms random-phase multisine with a flat spectrum on the given DFT bins
    X = zeros(N,1);
    X(bins) = exp(1j * 2*pi * rand(numel(bins),1));
    x = ifft(X, 'symmetric');
    x = x / rms(x);
end
