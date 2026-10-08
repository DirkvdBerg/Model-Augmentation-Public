function nz = tdg_noise_psd(Y_op, cfg, K)
% TDG_NOISE_PSD  Encoder-noise spectra, one per stage axis, for a record at Y_op (D-225).
%   nz = TDG_NOISE_PSD(Y_op, cfg, K) returns, on the DFT grid of the 12 s record (bins
%   k = 1 .. N/2 - 1, f_k = k / T, up to just below the 10 kHz Nyquist frequency), the
%   one-sided auto-spectra phi (n x 3, m^2/Hz) of three INDEPENDENT encoder noises
%   [v_X1, v_X2, v_Y] (D-225 amendment: one generator per axis, no cross-spectra):
%
%     THEORY (D-225 amendment): the simulated measured error is e = -S v with
%       S = (I + G K)^-1 (output sensitivity). To map the shape of the measured error,
%       each axis's measured auto-spectrum is divided by its own |S_jj|^2:
%         phi_j(f) = Phi_e,jj(f) / |S_jj(f)|^2     for f >= cfg.noise_corner
%         phi_j(f) = Phi_e,jj(f)                   below it (uncorrected; stated limit)
%     G: tdg_truth_lin(Y_op, cfg, cfg.friction_gain), the truth at rest INCLUDING the tanh
%        friction's slope (the measured noise was recorded at standstill), ZOH at ts
%     K: K1 (the noise is a sensor property; K2 records get the K1-calibrated noise)
%     Phi_e: measured Telica standstill error (noise/noise_target.mat, Welch bins 9.77 Hz),
%        linear interpolation between bins (HEURISTIC, as before)
%   Per-axis correction ignores the loop's cross-coupling (S_jk, j ~= k), so the match is
%   close, not exact (D-225 amendment item 3); check_encoder_noise reports how close.
%   Band edges: below the first nonzero Welch bin (9.77 Hz) held at that bin (HEURISTIC);
%   DC and the Nyquist bin zero.
%
%   D-238 (cfg.noise_model = 'lowpass', the default for new data): a low-order model per axis,
%     phi_j(f) = g_j^2 |H_j(f)|^2 + floor_j^2,   [b, a] = butter(2, fc_j / (fs/2), 'low')
%   with cfg.noise_fc, cfg.noise_g, cfg.noise_floor (fitted, NOISE-INJECTION.md section 10).
%   The same for every record: Y_op and K are not used (kept in the signature for the callers).
    T  = load(cfg.noise_target);
    N  = cfg.N_record;  Trec = cfg.t_record;
    k  = (1:N/2 - 1)';
    fk = k / Trec;
    if strcmp(cfg.noise_model, 'lowpass')
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
        return
    end
    fw = T.f(:);  Pw = T.P;                                  % 1025 x 3 x 3
    fe = min(max(fk, fw(2)), fw(end));                       % hold below the first bin
    j  = discretize(fe, fw);
    j(fe == fw(end)) = numel(fw) - 1;
    a  = (fe - fw(j)) ./ (fw(j + 1) - fw(j));
    pe = zeros(numel(fk), 3);                                % measured auto-spectra
    for ax = 1:3
        p = real(Pw(:, ax, ax));
        pe(:, ax) = (1 - a) .* p(j) + a .* p(j + 1);
    end

    Gd = c2d(tdg_truth_lin(Y_op, cfg, cfg.friction_gain), cfg.ts, 'zoh');
    S  = feedback(eye(3), Gd * K);                           % e = -S v
    hi = fk >= cfg.noise_corner;
    H  = freqresp(S, 2 * pi * fk(hi));                       % 3 x 3 x nhi
    S2 = zeros(nnz(hi), 3, 3);                               % |S_jk|^2
    for r = 1:3
        for c = 1:3
            S2(:, r, c) = abs(squeeze(H(r, c, :))).^2;
        end
    end
    phi = pe;
    switch cfg.noise_correction
        case 'diag'
            for ax = 1:3
                phi(hi, ax) = pe(hi, ax) ./ S2(:, ax, ax);
            end
        otherwise
            error('tdg_noise_psd: unknown noise_correction %s', cfg.noise_correction);
    end

    nz = struct('k', k, 'f', fk, 'phi', phi, 'phi_meas', pe, 'S2', S2, 'hi', hi, ...
                'Y_op', Y_op, 'N', N, 'fs', cfg.fs, 'corner', cfg.noise_corner, ...
                'correction', cfg.noise_correction, 'friction_gain', cfg.friction_gain, ...
                'hold_below', fw(2), 'node', 'encoder', 'target_sha256', T.source_sha256);
end
