function v = encoder_noise(cfg, axis_seeds)
% ENCODER_NOISE  One draw of the encoder noise on the stage axes [X1, X2, Y] (N_record x 3) [m].
%   Independent coloured noise per axis: a second-order low-pass on a white floor, sized so that at standstill
%   the simulated servo error matches the noise level and spectral shape of the ASMPT gantry, without its resonances
%   Each record and axis has a fixed seed; the noise-free version of a record is the same simulation without this noise
    N = cfg.N_record;
    k = (1:N/2 - 1)';                          % DFT bins without DC and Nyquist
    fk = k / cfg.t_record;
    v = zeros(N, 3);
    for ax = 1:3
        [b, a] = butter(2, cfg.noise_fc(ax) / (cfg.fs / 2), 'low');
        phi = cfg.noise_g(ax)^2 * abs(freqz(b, a, fk, cfg.fs)).^2 + cfg.noise_floor(ax)^2;
        s = RandStream('mt19937ar', 'Seed', axis_seeds(ax));
        Z = randn(s, numel(k), 2);
        w = complex(Z(:, 1), Z(:, 2)) / sqrt(2);   % complex white noise, E|w|^2 = 1
        % E|V_k|^2 = N fs phi(f_k) / 2 gives var(v) = (fs/N) sum_k phi(f_k), the discrete integral of phi (Parseval)
        V = zeros(N, 1);
        V(k + 1) = sqrt(N * cfg.fs / 2) * sqrt(phi) .* w;
        v(:, ax) = ifft(V, 'symmetric');
    end
end
