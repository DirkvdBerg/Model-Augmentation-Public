function nz = tdg_noise_psd(cfg)
% TDG_NOISE_PSD  Encoder-noise spectra, one per stage axis.
%   nz = TDG_NOISE_PSD(cfg) returns, on the DFT grid of the 12 s record (bins k = 1 .. N/2 - 1,
%   f_k = k / T, up to just below the 10 kHz Nyquist frequency), the one-sided auto-spectra phi
%   (n x 3, m^2/Hz) of three INDEPENDENT encoder noises [v_X1, v_X2, v_Y], a low-order model per axis:
%     phi_j(f) = g_j^2 |H_j(f)|^2 + floor_j^2,   [b, a] = butter(2, fc_j / (fs/2), 'low')
%   with cfg.noise_fc, cfg.noise_g, cfg.noise_floor, the same for every record. Only the source
%   hash of the measured spectrum is read from cfg.noise_target (it is stored in the record).
    T  = load(cfg.noise_target);
    N  = cfg.N_record;  Trec = cfg.t_record;
    k  = (1:N/2 - 1)';
    fk = k / Trec;
    phi = zeros(numel(fk), 3);
    for ax = 1:3
        [b, a] = butter(2, cfg.noise_fc(ax) / (cfg.fs / 2), 'low');
        H2 = abs(freqz(b, a, fk, cfg.fs)).^2;
        phi(:, ax) = cfg.noise_g(ax)^2 * H2 + cfg.noise_floor(ax)^2;
    end
    nz = struct('k', k, 'f', fk, 'phi', phi, 'N', N, 'fs', cfg.fs, 'node', 'encoder', ...
                'model', 'lowpass', 'fc_Hz', cfg.noise_fc, 'g', cfg.noise_g, 'floor', cfg.noise_floor, ...
                'target_sha256', T.source_sha256, ...
                'method', 'D-238: per-axis white noise through butter(2, fc) low-pass plus a white floor');
end
