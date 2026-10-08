function check_noise_calibration(Yset, n_real)
% CHECK_NOISE_CALIBRATION  C11 and C2 of IMPLEMENTATION.md (NOISE-INJECTION.md check 2).
%
% Stated before the run:
%   C11 tdg_truth_lin equals the finite-difference Jacobian of the vendored ODE
%       (gantrySystemExtendedCoulomb, cc = 0) at rest: PASS iff max relative entry
%       error <= 1e-6 at every Y tested.
%   (C13, the synthesis check, moved to check_noise_draw.m: its first criterion here, one
%   draw within 5 % of int Phi_d df, was wrong and failed; see that file's header. The
%   single-draw variance error is still printed below, for information.)
%   C2  friction off (cc = 0), standstill at each Y_op the manifest uses, no multisine,
%       K1, noise on: the simulated servo error e = r - y, Welch (Hann 2048 periodic,
%       50 %, 20 kHz, as the measurement), first 1 s discarded, summed over the Welch
%       bins 19.53 to 292.97 Hz, gives 3.370 / 3.696 / 1.086 nm rms (X1 / X2 / Y)
%       within 2 %, pooled over n_real realisations of 12 s. PASS iff every Y, every axis.
%   Also reported: per-sample sigma of d, rms of d below 300 Hz, the peak of Phi_d and
%   its frequency, cond(T), and the margin of the A_prod multisine over Phi_d in B_tr.
%   Figure: checks/fig_noise_calibration.png (simulated vs measured error PSD).
    here = fileparts(fileparts(mfilename('fullpath')));                % scripts/gantry/thesis-data-verification
    addpath(fullfile(fileparts(fileparts(fileparts(here))), 'Thesis-writeup', 'Code', 'Data'));   % the generator
    cfg = tdg_init(false);
    if nargin < 1 || isempty(Yset)
        man = tdg_manifest(cfg);
        Yset = unique([man.Y_op]);
    end
    if nargin < 2, n_real = 2; end
    T = load(cfg.noise_target);
    target = T.rms_band_m(:)';
    fprintf('target band rms (measured, %.2f to %.2f Hz): %s nm\n', T.band_lo, T.band_hi, mat2str(1e9*target, 4));

    % == C11 ==
    ok11 = true;
    for Y = [min(Yset), 0, max(Yset)]
        sysT = tdg_truth_lin(Y, cfg);
        [Afd, Bfd] = fd_jacobian(Y, cfg);
        eA = max(abs(sysT.A - Afd), [], 'all') / max(abs(sysT.A), [], 'all');
        Bl = sysT.B / cfg.P;                                   % back to logical input
        eB = max(abs(Bl - Bfd), [], 'all') / max(abs(Bl), [], 'all');
        ok11 = ok11 && eA <= 1e-6 && eB <= 1e-6;
        fprintf('C11 Y = %+.3f: rel err A %.2e, B %.2e\n', Y, eA, eB);
    end
    fprintf('C11 truth linearisation = ODE Jacobian -> %s\n', pf(ok11));

    % == C2 ==
    c0 = cfg;  c0.cc1 = 0; c0.cc2 = 0; c0.ccy = 0;              % friction off (check only)
    N = cfg.N_record;  t = (0:N-1)' * cfg.ts;
    win = hann(2048, 'periodic');
    ok2 = true;  rows = {};
    Pacc_all = [];
    for iy = 1:numel(Yset)
        Y = Yset(iy);
        plant = gtd_build_plant(Y, cfg, 1);                     % K1, pre-check plant at Y
        nz = tdg_noise_psd(Y, cfg, plant.Cfb);
        dfk = 1 / cfg.t_record;
        var_int = [sum(squeeze(real(nz.Phi(1,1,:)))), sum(squeeze(real(nz.Phi(2,2,:)))), sum(squeeze(real(nz.Phi(3,3,:))))] * dfk;
        in300 = nz.f <= 300;
        var300 = [sum(squeeze(real(nz.Phi(1,1,in300)))), sum(squeeze(real(nz.Phi(2,2,in300)))), sum(squeeze(real(nz.Phi(3,3,in300))))] * dfk;
        r = repmat([0, 0, Y], N, 1);
        Pacc = 0;  nseg = 0;
        for m = 1:n_real
            seed = 900000 + 1000*iy + m;                        % check seeds, not record seeds
            d = tdg_noise_draw(nz, seed);
            e13 = abs(var(d) ./ var_int - 1);
            rec = struct('id', 'noisecal');
            o = gtd_run_simulation(rec, r, t, zeros(N, 3), plant, c0, d);
            e = r - o.q_with;
            e = e(round(1/cfg.ts)+1:end, :);                    % discard the first 1 s
            [Pxx, fw] = pwelch(e, win, 1024, 2048, cfg.fs);     % one-sided, m^2/Hz
            Pacc = Pacc + Pxx;  nseg = nseg + 1;
            if m == 1
                fprintf('  Y %+.3f draw 1: std d %s N (int Phi_d: %s), rms d below 300 Hz %s N, var err %s\n', Y, ...
                        mat2str(std(d), 3), mat2str(sqrt(var_int), 3), mat2str(sqrt(var300), 3), mat2str(e13, 2));
            end
        end
        Pxx = Pacc / nseg;
        band = fw >= T.band_lo - 1e-9 & fw <= T.band_hi + 1e-9;
        rmsb = sqrt(sum(Pxx(band, :), 1) * (fw(2) - fw(1)));
        rel = rmsb ./ target - 1;
        ok = all(abs(rel) <= cfg.noise_tol);
        ok2 = ok2 && ok;
        % Phi_d features and multisine margin in B_tr (EXCITATION-VALIDATION section 19 item 1)
        dY = squeeze(real(nz.Phi(3,3,:)));  dX1 = squeeze(real(nz.Phi(1,1,:)));
        [pkY, iY] = max(dY);  [pkX, iX] = max(dX1);
        inB = nz.f >= cfg.f_low & nz.f <= cfg.f_high;
        Ams = [cfg.A_sym, cfg.A_anti, cfg.A_Y];  B = cfg.f_high - cfg.f_low;
        Pms_stage = [ (Ams(1)/2)^2 + (Ams(2)/cfg.Lb)^2, (Ams(1)/2)^2 + (Ams(2)/cfg.Lb)^2, Ams(3)^2 ] / B;   % N^2/Hz per stage axis
        marg = 10*log10(Pms_stage ./ max([dX1(inB), squeeze(real(nz.Phi(2,2,inB))), dY(inB)], [], 1));
        fprintf('C2  Y %+.3f: band rms %s nm, rel %s -> %s | Phi_d peak X1 %.2e N^2/Hz at %.1f Hz, Y %.2e at %.1f Hz | max cond(T) %.1e | multisine over noise in B_tr >= %s dB\n', ...
                Y, mat2str(1e9*rmsb, 4), mat2str(rel, 2), pf(ok), pkX, nz.f(iX), pkY, nz.f(iY), max(nz.condH), mat2str(round(marg, 1)));
        rows(end+1, :) = {Y, rmsb, rel, sqrt(var_int), sqrt(var300)}; %#ok<AGROW>
        if isempty(Pacc_all), Pacc_all = zeros(size(Pxx)); end
        Pacc_all = Pacc_all + Pxx;
    end
    fprintf('C2  noise calibration, %d Y_op values, %d x 12 s each, tol %.0f %% -> %s\n', numel(Yset), n_real, 100*cfg.noise_tol, pf(ok2));

    Psim = Pacc_all / numel(Yset);  fsim = fw;  %#ok<NASGU>
    save(fullfile(here, 'checks', 'noise_calibration_spectra.mat'), 'Psim', 'fsim', 'Yset');
    save(fullfile(here, 'checks', 'noise_calibration_result.mat'), 'rows', 'Yset', 'target', 'n_real');
    if ~usejava('jvm')
        fprintf('no JVM: figure skipped; checks/plot_noise_calibration.py draws it from noise_calibration_spectra.mat\n');
        return
    end
    % figure: pooled simulated PSD vs measured
    Pm = zeros(numel(T.f), 3);
    for i = 1:3, Pm(:, i) = real(T.P(:, i, i)); end
    fig = figure('Visible', 'off', 'Position', [100 100 1100 380]);
    lab = {'X1', 'X2', 'Y'};
    for i = 1:3
        subplot(1, 3, i);
        loglog(T.f(2:end), sqrt(Pm(2:end, i)), 'k', fw(2:end), sqrt(Pacc_all(2:end, i)/numel(Yset)), 'r');
        xline([T.band_lo 300], ':'); xlim([5 2000]); grid on;
        title(sprintf('%s servo error', lab{i})); xlabel('f [Hz]'); ylabel('ASD [m/\surdHz]');
        if i == 1, legend('measured (Telica, 145 logs)', 'simulated (friction off, mean over Y_{op})', 'Location', 'southwest'); end
    end
    exportgraphics(fig, fullfile(here, 'checks', 'fig_noise_calibration.png'), 'Resolution', 110);
    save(fullfile(here, 'checks', 'noise_calibration_result.mat'), 'rows', 'Yset', 'target', 'n_real');
end

function [A, B] = fd_jacobian(Y, cfg)
% Central-difference Jacobian of the vendored ODE at rest, cc = 0, mh = mh_rigid.
    x0 = [0; 0; Y; 0; 0; 0; 0; 0];  u0 = zeros(3, 1);
    f = @(x, u) gantrySystemExtendedCoulomb(u, x, cfg.m1, cfg.m2, cfg.mb, cfg.mh_rigid, cfg.Lb, ...
            cfg.Jb, cfg.Jh, cfg.d, cfg.cg1, cfg.cg2, cfg.cb1, cfg.cb2, cfg.cy, cfg.kb1, cfg.kb2, ...
            cfg.ma, cfg.ka, cfg.ca, cfg.L0, 0, 0, 0, cfg.ts);
    h = 1e-6;
    A = zeros(8);  B = zeros(8, 3);
    for j = 1:8
        e = zeros(8, 1); e(j) = h;
        A(:, j) = (f(x0 + e, u0) - f(x0 - e, u0)) / (2*h);
    end
    for j = 1:3
        e = zeros(3, 1); e(j) = h;
        B(:, j) = (f(x0, u0 + e) - f(x0, u0 - e)) / (2*h);
    end
end

function s = pf(ok)
    if ok, s = 'PASS'; else, s = 'FAIL'; end
end
