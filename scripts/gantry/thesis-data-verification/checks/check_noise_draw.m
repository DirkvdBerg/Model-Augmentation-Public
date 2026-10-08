function check_noise_draw()
% CHECK_NOISE_DRAW  C13 of IMPLEMENTATION.md, revised: the noise synthesis has the
% intended covariance. No simulation.
%
% History: C13 was first stated inside check_noise_calibration.m as "one draw's
% variance within 5 % of int Phi_d df (chi-square spread of ~4800 bins under 3 %)" and
% FAILED (logs/C2_C11_C13_noise.log: up to 8.9 % on Y). The criterion was wrong, not
% the draw: it assumed a flat spectrum, while Phi_d is concentrated (Y: a line at the
% 150 Hz anti-resonance; X: the 300 to 400 Hz roll-off band), so one draw has few
% effective degrees of freedom. The single-draw spread is sigma_rel = sqrt(sum Phi^2) /
% sum Phi per axis (complex Gaussian bins, var |w|^2 = 1).
%
% Stated before the run (revised criterion):
%   for Y_op in {-0.39, 0, +0.35}, M = 40 draws (seeds 950000 + ...):
%   (a) mean over draws of var(d_i) / int Phi_ii within 1 +- 3 sigma_rel / sqrt(M), per axis
%   (b) the draw-to-draw std of that ratio within [0.6, 1.5] x sigma_rel, per axis
%       (HEURISTIC band: the sample std of 40 draws scatters about 11 % around sigma_rel,
%       so [0.6, 1.5] admits a correct draw and catches a factor-2 scaling error)
%   (c) the sample covariance of d (pooled over draws) matches int Re(Phi) off-diagonal:
%       correlation coefficient within 0.05 of the target, per pair
%   PASS iff all.
    here = fileparts(fileparts(mfilename('fullpath')));                % scripts/gantry/thesis-data-verification
    addpath(fullfile(fileparts(fileparts(fileparts(here))), 'Thesis-writeup', 'Code', 'Data'));   % the generator
    cfg = tdg_config();
    K1 = gtd_build_plant(0, cfg, 1).Cfb;
    M = 40;  ok = true;
    for Y = [-0.39, 0, 0.35]
        nz = tdg_noise_psd(Y, cfg, K1);
        df = 1 / cfg.t_record;
        C = real(sum(nz.Phi, 3)) * df;                 % target covariance [N^2]
        Pd = zeros(numel(nz.k), 3);
        for i = 1:3, Pd(:, i) = squeeze(real(nz.Phi(i, i, :))); end
        srel = sqrt(sum(Pd.^2, 1)) ./ sum(Pd, 1);
        ratio = zeros(M, 3);  Cs = zeros(3);
        for m = 1:M
            d = tdg_noise_draw(nz, 950000 + round(1000*Y) + m);
            ratio(m, :) = var(d) ./ diag(C)';
            Cs = Cs + cov(d) / M;
        end
        mr = mean(ratio);  sr = std(ratio);
        okA = all(abs(mr - 1) <= 3 * srel / sqrt(M));
        okB = all(sr >= 0.6 * srel & sr <= 1.5 * srel);
        rt = C ./ sqrt(diag(C) * diag(C)');  rs = Cs ./ sqrt(diag(Cs) * diag(Cs)');
        okC = max(abs(rt - rs), [], 'all') <= 0.05;
        ok = ok && okA && okB && okC;
        fprintf('Y %+.2f: target std %s N | predicted single-draw spread %s | mean ratio %s (a %s) | draw spread %s (b %s) | corr target [%.3f %.3f %.3f] sample [%.3f %.3f %.3f] (c %s)\n', ...
                Y, mat2str(sqrt(diag(C))', 3), mat2str(srel, 2), mat2str(mr, 4), pf(okA), mat2str(sr, 2), pf(okB), ...
                rt(1,2), rt(1,3), rt(2,3), rs(1,2), rs(1,3), rs(2,3), pf(okC));
        % where the force variance sits
        f = nz.f;
        share = @(sel) sum(Pd(sel, :), 1) ./ sum(Pd, 1);
        fprintf('         force variance share: below 100 Hz %s, 100-300 Hz %s, above 300 Hz (roll-off) %s; Y within 145-155 Hz %s\n', ...
                mat2str(share(f < 100), 2), mat2str(share(f >= 100 & f <= 300), 2), mat2str(share(f > 300), 2), ...
                mat2str(share(f >= 145 & f <= 155), 2));
    end
    fprintf('C13 noise synthesis covariance (revised criterion) -> %s\n', pf(ok));
end

function s = pf(ok)
    if ok, s = 'PASS'; else, s = 'FAIL'; end
end
