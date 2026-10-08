function [v, seeds] = tdg_noise_draw(nz, seed)
% TDG_NOISE_DRAW  One realisation of the three independent encoder noises, N x 3 [m] (D-225).
%   [v, seeds] = TDG_NOISE_DRAW(nz, seed) draws each stage axis from its own generator:
%     seed_j = tdg_seed('<seed>|axis<j>')    (the record's noise seed, one stream per axis)
%     V_j,k  = sqrt(N fs / 2) sqrt(phi_j(f_k)) w_j,k,   w_j,k ~ CN(0, 1)
%     v_j    = ifft(V_j, 'symmetric')        (DC and Nyquist bins zero)
%   THEORY: for a real sequence, E|V_k|^2 = N fs phi(f_k) / 2 gives
%   var(v) = (fs/N) sum_k phi(f_k), the discrete form of int phi df (Parseval).
%   A circular FIR shaping filter applied to unit white noise, written in the frequency
%   domain; no cross-spectra, so the axes are independent (D-225 amendment item 2). The
%   realisation is periodic over the 12 s record; the loop starts at rest inside the hold.
    N  = nz.N;  fs = nz.fs;
    n  = numel(nz.k);
    v  = zeros(N, 3);
    seeds = zeros(1, 3);
    for ax = 1:3
        seeds(ax) = tdg_seed(sprintf('%.0f|axis%d', seed, ax));
        s = RandStream('mt19937ar', 'Seed', seeds(ax));   % own stream: no global RNG state
        Z = randn(s, n, 2);
        w = complex(Z(:, 1), Z(:, 2)) / sqrt(2);          % E|w|^2 = 1
        V = zeros(N, 1);
        V(nz.k + 1) = sqrt(N * fs / 2) * sqrt(nz.phi(:, ax)) .* w;
        v(:, ax) = ifft(V, 'symmetric');
    end
end
