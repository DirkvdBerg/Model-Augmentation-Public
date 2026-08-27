function [f, info] = generate_cached_multisine(fs, f_low, f_high, amp_rms, n_ch, n_candidates, cache_path, seed_offset)
% Generate random-phase MIMO multisine with lowest crest factor.
% Loads from cache if available with matching parameters.
%
% Inputs:
%   fs            sample rate [Hz] (period = 1/1 Hz = 1 s, N = round(fs))
%   f_low         lowest excited frequency [Hz]
%   f_high        highest excited frequency [Hz]
%   amp_rms       RMS amplitude per channel, scalar or (1 x n_ch) [N]
%   n_ch          number of channels (e.g. 3 for [F_X1, F_X2, F_Y])
%   n_candidates  number of random-phase realizations to try
%   cache_path    path to .mat file for caching
%   seed_offset   (optional, default 0) offset added to rng seed for
%                 generating independent realizations across splits
%
% Outputs:
%   f             (N_period x n_ch) one period of the best multisine
%   info          struct: crest_factor, seed, f_low, f_high, amp_rms, freq_lines
%
% Each candidate uses rng(seed_offset + candidate_index) for reproducibility.
% Crest factor = max(|f|) / rms(f), computed per channel, worst taken.

    if nargin < 8, seed_offset = 0; end

    if isscalar(amp_rms)
        amp_rms = amp_rms * ones(1, n_ch);
    end

    % Check cache
    if exist(cache_path, 'file')
        S = load(cache_path);
        if isfield(S, 'info') && S.info.fs == fs && S.info.f_low == f_low && ...
           S.info.f_high == f_high && isequal(S.info.amp_rms, amp_rms) && ...
           S.info.n_ch == n_ch && S.info.n_candidates == n_candidates && ...
           S.info.seed_offset == seed_offset
            f    = S.f;
            info = S.info;
            fprintf('Loaded cached multisine from %s  (CF = %.3f)\n', cache_path, info.crest_factor);
            return
        end
    end

    % Multisine parameters
    N      = round(fs);                      % 1-second period
    df     = fs / N;                         % frequency resolution [Hz]
    freqs  = f_low : df : f_high;            % excited frequency lines
    F      = length(freqs);
    t      = (0:N-1)' / fs;                  % time vector [s]

    fprintf('Generating %d candidates  (%d lines, %.0f-%.0f Hz, %d ch)...\n', ...
        n_candidates, F, f_low, f_high, n_ch);

    best_cf   = Inf;
    best_f    = [];
    best_seed = 0;

    for ic = 1:n_candidates
        rng(seed_offset + ic);
        f_cand = zeros(N, n_ch);

        for ch = 1:n_ch
            phi = 2 * pi * rand(1, F);       % random phases
            sig = sum(cos(2*pi*freqs .* t + phi), 2);
            f_cand(:, ch) = amp_rms(ch) * sig / rms(sig);
        end

        % Crest factor: worst channel
        cf_per_ch = max(abs(f_cand)) ./ rms(f_cand);
        cf = max(cf_per_ch);

        if cf < best_cf
            best_cf   = cf;
            best_f    = f_cand;
            best_seed = ic;
        end
    end

    f = best_f;
    info.crest_factor = best_cf;
    info.seed         = best_seed;
    info.fs           = fs;
    info.f_low        = f_low;
    info.f_high       = f_high;
    info.amp_rms      = amp_rms;
    info.n_ch         = n_ch;
    info.n_candidates = n_candidates;
    info.seed_offset  = seed_offset;
    info.N_period     = N;
    info.freq_lines   = freqs;
    info.n_lines      = F;

    % Save cache
    save(cache_path, 'f', 'info');
    fprintf('Best seed = %d,  CF = %.3f,  saved to %s\n', best_seed, best_cf, cache_path);
end
