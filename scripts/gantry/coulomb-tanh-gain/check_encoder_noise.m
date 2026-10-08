function check_encoder_noise(parts, rebuild)
%CHECK_ENCODER_NOISE  D-225 (amended) checks of the encoder noise in the thesis generator.
%
%   check_encoder_noise(parts, rebuild)
%     parts    'A', 'B' or 'AB'                                              (default 'AB')
%     rebuild  true = run tdg_make_model first (part A only)                  (default true)
%
% (A) implementation: the generator model with encoder noise (EncoderSum on Gain4 -> Sum3) equals
%     the replica cl_sim with v_enc, bitwise (max |q_sim - q_replica| < 1e-10 m), on the TR-T1 inputs
%     (r, f, Y_op of the stored record) with the noise draw of TR-T1's own seed, and the plumbing
%     u_plant = u_total (= lsim(K1, r - y) + f, y = q + v) to 1e-6 N; also the noise-free twin.
%     Needs Simulink (about 3.5 GB free).
% (B) shape (D-225 amendment item 5): standstill at Y 0 (r = [0 0 0], f = 0), friction ON at
%     cfg.friction_gain, one 12 s draw from tdg_noise_psd / tdg_noise_draw, replica. Per axis, the
%     simulated measured error e = r - (q + v) after the hold, band by band against the measured
%     spectrum: the ratios in the bands above the corner lie within +-cfg.noise_shape_tol of their
%     common level (the ratio over everything above the corner). Reported beside it: the ratio below
%     the corner (the stated limit), the common level, and the ANALYTIC band ratios of the implemented
%     per-axis correction (sum_k |S_jk|^2 phi_k against Phi_e,jj) and of the uncorrected injection.
%     Saves outputs/noise_shape_Y0.mat for plot_noise_shape.py.
% Criteria stated in D-225 (amendment) before the first run.

    if nargin < 1 || isempty(parts), parts = 'AB'; end
    if nargin < 2 || isempty(rebuild), rebuild = true; end
    here = fileparts(mfilename('fullpath'));
    repo = fileparts(fileparts(fileparts(here)));
    addpath(fullfile(repo, 'Thesis-writeup', 'Code', 'Data'));
    cfg = tdg_config();
    addpath(here);
    base = {cfg.m1, cfg.m2, cfg.mb, cfg.mh_rigid, cfg.Lb, cfg.Jb, cfg.Jh, cfg.d, ...
            cfg.cg1, cfg.cg2, cfg.cb1, cfg.cb2, cfg.cy, cfg.kb1, cfg.kb2, cfg.ma, cfg.ka, cfg.ca, cfg.L0};
    odeT = @(x, u) gantrySystemExtendedTanh(u, x, base{:}, cfg.cc1, cfg.cc2, cfg.ccy, cfg.friction_gain);
    plant0 = gtd_build_plant(0, cfg, 1);
    [Ac, Bc, Cc, Dc] = ssdata(plant0.Cfb);
    ctl = struct('A', Ac, 'B', Bc, 'C', Cc, 'D', Dc);
    fprintf('friction gain %g s/m, corner %g Hz, correction %s, tolerance %g\n', cfg.friction_gain, ...
            cfg.noise_corner, cfg.noise_correction, cfg.noise_shape_tol);

    if contains(parts, 'A')
        Simulink.fileGenControl('set', 'CacheFolder', cfg.cache_dir, 'CodeGenFolder', cfg.cache_dir);
        if rebuild, tdg_make_model(); end
        cfg = tdg_init(false);
        R = load(fullfile(repo, 'Thesis-writeup', 'Data', 'Coulomb-and-MSD', 'Training', 'TR-T1_r1.mat'), ...
                 'r_sim', 't_sim', 'f_sim', 'meta');
        plant = gtd_build_plant(R.meta.Y_op, cfg, R.meta.controller.gain);
        nzA = tdg_noise_psd(R.meta.Y_op, cfg, plant0.Cfb);
        v = tdg_noise_draw(nzA, R.meta.noise_seed);
        ok = true;
        for vers = {'noisy', 'noise-free'}
            vv = v;  if strcmp(vers{1}, 'noise-free'), vv = zeros(size(v)); end
            out = gtd_run_simulation([], R.r_sim, R.t_sim, R.f_sim, plant, cfg, vv);
            S = struct('r_sim', R.r_sim, 'f_sim', R.f_sim, 'v_enc', vv);
            q = cl_sim(S, odeT, R.meta.Y_op, ctl, cfg.P, cfg.ts);
            dq = max(abs(out.q_with - q), [], 1);
            plumb = max(abs(out.u_plant - out.u_total), [], 'all');
            pass = all(dq < 1e-10) && plumb <= 1e-6;  ok = ok && pass;
            fprintf(['(A) %-10s max|q_sim - q_replica| %s m | plumbing |u_plant - u_total| %.2e N | ' ...
                     'noise std %s nm | %s\n'], vers{1}, num2str(dq, '%.2e '), plumb, ...
                    mat2str(1e9 * std(vv), 3), ternary(pass, 'PASS', 'FAIL'));
        end
        fprintf('(A) %s\n', ternary(ok, 'PASS', 'FAIL'));
        close_system(cfg.mdl, 0);
    end

    if contains(parts, 'B')
        N = cfg.N_record;  fs = cfg.fs;
        nz = tdg_noise_psd(0, cfg, plant0.Cfb);
        [v, seeds] = tdg_noise_draw(nz, 20260928);           % any seed: a shape check
        S = struct('r_sim', zeros(N, 3), 'f_sim', zeros(N, 3), 'v_enc', v);
        t0 = tic;
        q = cl_sim(S, odeT, 0, ctl, cfg.P, cfg.ts);
        e = -(q + v);
        keep = cfg.n_hold + 1:N;
        fprintf('(B) replica %.0f s; axis seeds %s; injected v rms %s nm\n', toc(t0), mat2str(seeds), ...
                mat2str(1e9 * rms(v), 4));
        % band powers of the simulated error (periodogram of the kept part) and of the measured target
        ek = e(keep, :);  Nk = size(ek, 1);
        E = fft(ek);  fe = (0:Nk - 1)' * fs / Nk;
        Pe = 2 * abs(E(1:floor(Nk / 2), :)).^2 / (Nk * fs);  fe = fe(1:floor(Nk / 2));
        dfe = fs / Nk;  dfm = nz.f(2) - nz.f(1);
        fc = cfg.noise_corner;
        bands = [10 fc; fc 100; 100 300; 300 1000; 1000 9990];
        % analytic simulated error of the implemented injection and of the uncorrected one
        pred = nz.phi_meas;  unc = nz.phi_meas;
        for j = 1:3
            pred(nz.hi, j) = sum(squeeze(nz.S2(:, j, :)) .* nz.phi(nz.hi, :), 2);
            unc(nz.hi, j)  = sum(squeeze(nz.S2(:, j, :)) .* nz.phi_meas(nz.hi, :), 2);
        end
        % the alternative (not implemented): still independent axes, but the three auto-spectra
        % solved per frequency from sum_k |S_jk|^2 phi_k = Phi_e,jj (nonnegative least squares where
        % the plain solve goes negative), so the coupling is accounted for
        cpl = nz.phi_meas;  nneg = 0;  hidx = find(nz.hi);
        for i = 1:numel(hidx)
            A = squeeze(nz.S2(i, :, :));  b = nz.phi_meas(hidx(i), :).';
            x = A \ b;
            if any(x < 0), x = lsqnonneg(A, b); nneg = nneg + 1; end
            cpl(hidx(i), :) = (A * x).';
        end
        ratio = @(ps, fsim, dfs, b) sqrt(sum(ps(fsim >= b(1) & fsim < b(2), :), 1) * dfs ./ ...
                    (sum(nz.phi_meas(nz.f >= b(1) & nz.f < b(2), :), 1) * dfm));
        above = [fc 9990];
        common = ratio(Pe, fe, dfe, above);
        fprintf('    band [Hz]        simulated / measured rms    analytic (implemented)    analytic (uncorrected)    analytic (coupled alternative)\n');
        dev = zeros(size(bands, 1) - 1, 3);
        for b = 1:size(bands, 1)
            rs = ratio(Pe, fe, dfe, bands(b, :));
            ra = ratio(pred, nz.f, dfm, bands(b, :));
            ru = ratio(unc, nz.f, dfm, bands(b, :));
            rc = ratio(cpl, nz.f, dfm, bands(b, :));
            if bands(b, 2) <= fc                              % no analytic below the corner (S2 not computed)
                ra = nan(1, 3);  ru = ra;  rc = ra;
            end
            fprintf('    %5g to %5g   %-28s %-25s %-25s %s\n', bands(b, 1), bands(b, 2), mat2str(rs, 3), ...
                    mat2str(ra, 3), mat2str(ru, 3), mat2str(rc, 3));
            if b > 1, dev(b - 1, :) = rs ./ common - 1; end
        end
        pass = all(abs(dev(:)) <= cfg.noise_shape_tol);
        fprintf('(B) common level above %g Hz %s; max |band / common - 1| %s -> %s\n', fc, ...
                mat2str(common, 3), mat2str(max(abs(dev), [], 1), 3), ternary(pass, 'PASS', 'FAIL'));
        fprintf('    coupled alternative: nonnegativity active at %d of %d bins above the corner\n', nneg, numel(hidx));
        fprintf('    total simulated error %s nm against measured %s nm\n', mat2str(1e9 * rms(ek), 4), ...
                mat2str(1e9 * sqrt(sum(nz.phi_meas, 1) * dfm), 4));
        f_meas = nz.f;  phi_meas = nz.phi_meas;                         %#ok<NASGU>
        save(fullfile(here, 'outputs', 'noise_shape_Y0.mat'), 'ek', 'fs', 'f_meas', 'phi_meas', 'fc', '-v7');
    end
end

function out = ternary(c, a, b)
    if c, out = a; else, out = b; end
end
