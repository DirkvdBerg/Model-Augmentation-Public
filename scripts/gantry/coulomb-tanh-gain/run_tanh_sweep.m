function run_tanh_sweep(versions, gains, rec)
%RUN_TANH_SWEEP  D-223: tanh friction gain sweep on the thesis truth, outside Simulink.
%
%   run_tanh_sweep(versions, gains, rec)
%     versions  cell, subset of {'noise-free', 'noisy'}      (default both)
%     gains     tanh gains g [s/m]; [] = gate only           (default [100 300 1e3 3e3 1e4])
%     rec       {split folder, file stem}                    (default {'Training', 'TR-T1_r1'})
%
% THE LOOP. The recorded loop of Thesis-writeup/Code/Data/model/gantry_tdg_2025a.slx, read from
% the model file (D-223): Reference1, Feedforward1 and NoiseForce are From Workspace blocks at
% sample time ts, K1 is the discrete LTI block `Cfb`, Sum3 forms r - q, Sum2 adds the feedforward,
% NoiseSum adds d, Gain3 = P maps stage to logical force, the Extended ODE chart calls
% gantrySystemExtendedCoulomb, the Integrator starts at [0; 0; Y; 0; 0; 0; 0; 0], Gain4 = P.' gives
% q, and the solver is fixed-step ode4 at ts. The input is therefore held over each RK4 step and
% the controller output uses q at the start of the step. gtd_run_simulation sets mh = mh_rigid
% for the run, so the ODE gets mh_rigid here too.
%
% THE GATE. With the vendored Karnopp ODE this replay must reproduce the stored y (noisy version:
% with the stored d_in). Pass: max |q_replay - y| < 1e-10 m (HEURISTIC: a tenth of a nanometre,
% ten times below the smallest calibrated noise level 1.1 nm). A failed gate stops the sweep.
%
% OUTPUT. outputs/summary_<stem>.txt (everything printed), outputs/sweep_<stem>_<version>.mat
% (metrics, and the servo error of every run as single for the figure script).
%
% Run (from this folder, Git Bash):  bash tools/run_matlab.sh <name> "run_tanh_sweep"

    if nargin < 1 || isempty(versions), versions = {'noise-free', 'noisy'}; end
    if nargin < 2, gains = [100 300 1e3 3e3 1e4]; end
    if nargin < 3 || isempty(rec), rec = {'Training', 'TR-T1_r1'}; end

    here = fileparts(mfilename('fullpath'));
    repo = fileparts(fileparts(fileparts(here)));
    addpath(fullfile(repo, 'Thesis-writeup', 'Code', 'Data'));
    cfg = tdg_config();                       % puts the generator and its vendor/ on the path
    addpath(here);
    out = fullfile(here, 'outputs');
    if ~exist(out, 'dir'), mkdir(out); end
    fid = fopen(fullfile(out, sprintf('summary_%s.txt', rec{2})), 'a');
    cleanup = onCleanup(@() fclose(fid));
    say = @(varargin) say_both(fid, varargin{:});

    FOLDER = struct('noise_free', 'Coulomb-and-MSD-noise-free', 'noisy', 'Coulomb-and-MSD');
    P = cfg.P;  h = cfg.ts;  fs = cfg.fs;
    cc = [cfg.cc1; cfg.cc2; cfg.ccy];
    base = {cfg.m1, cfg.m2, cfg.mb, cfg.mh_rigid, cfg.Lb, cfg.Jb, cfg.Jh, cfg.d, ...
            cfg.cg1, cfg.cg2, cfg.cb1, cfg.cb2, cfg.cy, cfg.kb1, cfg.kb2, ...
            cfg.ma, cfg.ka, cfg.ca, cfg.L0, cfg.cc1, cfg.cc2, cfg.ccy};
    odeK = @(x, u) gantrySystemExtendedCoulomb(u, x, base{:}, cfg.ts);   % ts unused since D-209

    % Measured machine standstill level and calibrated noise level (NOISE-INJECTION.md, D-218).
    MEAS_NM = [9.8 10.4 6.3];  CAL_NM = [3.370 3.696 1.086];

    say('\n==== %s  %s  gains %s ====\n', datestr(now, 31), rec{2}, mat2str(gains));
    for iv = 1:numel(versions)
        ver = versions{iv};
        fpath = fullfile(repo, 'Thesis-writeup', 'Data', FOLDER.(strrep(ver, '-', '_')), ...
                         rec{1}, [rec{2} '.mat']);
        S = load(fpath, 'y', 'r_sim', 'f_sim', 'u_fb', 'd_in', 'meta');
        c = S.meta.controller;
        assert(strcmp(c.name, 'K1') && c.design_Y == 0 && c.gain == 1, 'not K1: %s', c.name);
        plant = gtd_build_plant(S.meta.Y_op, cfg, c.gain);
        [Ac, Bc, Cc, Dc] = ssdata(plant.Cfb);
        ctl = struct('A', Ac, 'B', Bc, 'C', Cc, 'D', Dc);
        Y0 = S.meta.Y_op;
        mov = any(abs([zeros(1, 3); diff(S.r_sim)]) > 0, 2);     % reference moving, any axis
        % A record without moves (standstill multisine) compares the runs over the whole record
        % (D-223 amendment); the dwell metrics then have nothing to score.
        cmp = mov;  if ~any(mov), cmp = true(size(mov)); end
        say('\n== %s  (%s, Y_op %+.2f, |d| max %.3g N)\n', ver, fpath, Y0, max(abs(S.d_in(:))));

        % == gate: Karnopp replay against the stored record ==
        t0 = tic;
        [qK, ufK, vK] = cl_sim(S, odeK, Y0, ctl, P, h);
        dq  = max(abs(qK - S.y), [], 1);
        duf = max(abs(ufK - S.u_fb), [], 1) ./ max(abs(S.u_fb), [], 1);
        pass = all(dq < 1e-10);
        say('gate  Karnopp replay vs stored y: max|dq| %s m, u_fb rel %s  (%.0f s)  %s\n', ...
            num2str(dq, '%.2e '), num2str(duf, '%.1e '), toc(t0), ternary(pass, 'PASS', 'FAIL'));
        if ~pass
            [~, k1] = max(any(abs(qK - S.y) > 1e-10, 2));
            say('      first sample above 1e-10 m: k = %d (t = %.4f s)\n', k1, (k1 - 1) * h);
            say('      gate failed: sweep skipped for this version\n');
            continue
        end

        runs = struct('name', {}, 'g', {}, 'e', {}, 'v', {});
        runs(1) = struct('name', 'Karnopp', 'g', NaN, 'e', S.r_sim - qK, 'v', vK);
        for ig = 1:numel(gains)
            g = gains(ig);
            odeT = @(x, u) gantrySystemExtendedTanh(u, x, base{:}, g);
            t0 = tic;
            [q, ~, v] = cl_sim(S, odeT, Y0, ctl, P, h);
            runs(end + 1) = struct('name', sprintf('tanh g=%g', g), 'g', g, ...
                                   'e', S.r_sim - q, 'v', v); %#ok<AGROW>
            say('run   %-14s %.0f s\n', runs(end).name, toc(t0));
        end

        % == metrics ==
        dw = dwells(mov, round(0.1 * fs));
        say('dwells scored: %d (>= 0.1 s, preceded by a move); moving %.1f %% of the record\n', ...
            size(dw, 1), 100 * mean(mov));
        lh = lambda_h(base, P, cc, gains, S.r_sim(:, 3), h);
        eK = runs(1).e;
        Mc = cell(1, numel(runs));
        for ir = 1:numel(runs)
            m = score(runs(ir).e, runs(ir).v, cmp, dw, fs);
            m.lambda_h = NaN;  m.vs_karnopp = zeros(1, 3);  m.vs_next = nan(1, 3);
            if ir > 1
                m.lambda_h = lh(gains == runs(ir).g);
                m.vs_karnopp = relrms(runs(ir).e(cmp, :) - eK(cmp, :), eK(cmp, :));
                if ir < numel(runs)
                    m.vs_next = relrms(runs(ir).e(cmp, :) - runs(ir + 1).e(cmp, :), ...
                                       runs(ir + 1).e(cmp, :));
                end
            end
            Mc{ir} = m;
        end
        M = [Mc{:}];

        say('\n%-14s | %-26s | %-20s | %-20s | %-26s | %-20s | %-20s | %6s %5s\n', 'run', ...
            'dwell 2nd-half rms [nm]', 'Q4/Q3 max', 'reversals/s dwell', 'moving rms [um]', ...
            'moving vs Karnopp', 'moving vs next g', 'lam*h', 'p2');
        for ir = 1:numel(runs)
            m = M(ir);
            say('%-14s | %8.3g %8.3g %8.3g | %6.2f %6.2f %6.2f | %6.1f %6.1f %6.1f | %8.4g %8.4g %8.4g | %6.3f %6.3f %6.3f | %6.3f %6.3f %6.3f | %6.3f %5d\n', ...
                runs(ir).name, 1e9 * m.dwell_rms, m.decay, m.rev_rate, 1e6 * m.mov_rms, ...
                m.vs_karnopp, m.vs_next, m.lambda_h, m.p2);
        end
        say('reference levels [nm]: measured machine standstill %s, calibrated noise %s\n', ...
            mat2str(MEAS_NM), mat2str(CAL_NM));

        if ~any(mov)
            say('no moves: the moving columns are over the whole record; dwell columns empty\n');
        end
        if strcmp(ver, 'noise-free') && numel(runs) > 1 && ~isempty(dw)
            ok_a = arrayfun(@(m) all(1e9 * m.dwell_rms < MEAS_NM) && all(m.decay <= 1), M);
            ok_b = arrayfun(@(m) all(m.vs_next < 0.05), M);
            ok_c = arrayfun(@(m) m.lambda_h < 2.785 && all(m.p2 == 0), M);
            say('criterion (D-223)  a standstill  b converged  c numerics\n');
            for ir = 2:numel(runs)
                say('  %-14s %d %d %d\n', runs(ir).name, ok_a(ir), ok_b(ir), ok_c(ir));
            end
            pick = find(ok_a & ok_b & ok_c & (1:numel(runs)) > 1, 1);
            if isempty(pick)
                say('recommended gain: none meets a, b and c\n');
            else
                say('recommended gain: %g\n', runs(pick).g);
            end
        end

        % == save: metrics and the servo errors as single, for the figure ==
        E = single(cat(3, runs.e));                               % N x 3 x runs
        names = {runs.name};  gvec = [runs.g];                    %#ok<NASGU>
        metrics = M;  gate_dq = dq;  gate_duf = duf;              %#ok<NASGU>
        save(fullfile(out, sprintf('sweep_%s_%s.mat', rec{2}, ver)), 'E', 'names', 'gvec', ...
             'metrics', 'mov', 'dw', 'fs', 'gate_dq', 'gate_duf', 'MEAS_NM', 'CAL_NM', '-v7');
    end
end

% ================================================================================================

function dw = dwells(mov, nmin)
% [start, stop] of every run of reference standstill at least nmin samples long that follows a
% move (the initial hold has no move before it and is not scored).
    d  = diff([0; ~mov(:); 0]);
    st = find(d == 1);  en = find(d == -1) - 1;
    keep = (en - st + 1) >= nmin & st > 1;
    dw = [st(keep), en(keep)];
end

function m = score(e, v, mov, dw, fs)
    sq = zeros(1, 3);  n2 = 0;  decay = zeros(1, 3);  rev = zeros(1, 3);
    for i = 1:size(dw, 1)
        a = dw(i, 1);  b = dw(i, 2);  n = b - a + 1;
        h2 = a + floor(n / 2) : b;
        q3 = a + floor(n / 2) : a + floor(3 * n / 4) - 1;
        q4 = a + floor(3 * n / 4) : b;
        sq = sq + sum(e(h2, :) .^ 2, 1);  n2 = n2 + numel(h2);
        r3 = rms(e(q3, :), 1);  r4 = rms(e(q4, :), 1);
        % HEURISTIC: below 1 pm (a thousandth of the smallest noise level) a dwell counts as
        % settled; the ratio of two round-off levels means nothing.
        ratio = r4 ./ r3;  ratio(r3 < 1e-12) = 0;
        decay = max(decay, ratio);
        sv = sign(v(h2, :));
        rev = rev + sum(sv(2:end, :) ~= sv(1:end - 1, :) & sv(2:end, :) ~= 0 & sv(1:end - 1, :) ~= 0, 1);
    end
    m.dwell_rms = sqrt(sq / max(n2, 1));
    m.decay = decay;
    m.rev_rate = rev / max(n2 / fs, eps);
    m.mov_rms = rms(e(mov, :), 1);
    % period-2 alternation of the velocity sign on three consecutive steps: numerical ringing
    s = sign(v);  fl = s(2:end, :) == -s(1:end - 1, :) & s(2:end, :) ~= 0;
    m.p2 = sum(fl(3:end, :) & fl(2:end - 1, :) & fl(1:end - 2, :), 'all');
end

function lh = lambda_h(base, P, cc, gains, Yr, h)
% max |lambda| h of the tanh friction slope, -G diag(cc g), with G the stage force -> stage
% acceleration map read off the frictionless ODE (g = 0 gives F = 0 exactly), at the lowest,
% middle and highest Y of the record. THEORY: classical RK4 is stable on the negative real axis
% for |lambda h| < 2.785.
    lh = zeros(size(gains));
    for Y = [min(Yr), mean(Yr), max(Yr)]
        x0 = [0; 0; Y; 0; 0; 0; 0; 0];
        f0 = gantrySystemExtendedTanh(zeros(3, 1), x0, base{:}, 0);
        Bl = zeros(8, 3);
        for j = 1:3
            ej = zeros(3, 1);  ej(j) = 1;
            Bl(:, j) = gantrySystemExtendedTanh(ej, x0, base{:}, 0) - f0;
        end
        G = P.' * Bl(5:7, :) * P;            % stage force -> stage acceleration
        for ig = 1:numel(gains)
            lh(ig) = max(lh(ig), max(abs(eig(-G * diag(cc * gains(ig))))) * h);
        end
    end
end

function r = relrms(a, b)
    r = rms(a, 1) ./ max(rms(b, 1), eps);
end

function out = ternary(c, a, b)
    if c, out = a; else, out = b; end
end

function say_both(fid, varargin)
    fprintf(varargin{:});
    fprintf(fid, varargin{:});
end
