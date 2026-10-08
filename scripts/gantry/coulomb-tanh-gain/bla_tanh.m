function bla_tanh(which, gate_only)
%BLA_TANH  Band re-check of D-224: the closed-loop BLA of the truth at the five training Y (cases B6 to
% B10 of scripts/gantry/excitation-closed-loop/matlab/bla_truth.m), with the tanh friction law.
%
%   bla_tanh(which, gate_only)
%     which      cell of case ids, default {'B6','B7','B8','B9','B10'}
%     gate_only  true = only the gate below
%
% EVERYTHING AS bla_truth.m (read it for the reasons): K1 designed at Y = 0, reference constant
% [0 0 Y_op], production level H = [240 87 180] in the logical channels [sym, anti, Y], zippered
% random-phase multisines on 1 Hz to 1 kHz, period 2 s, 1 transient + 2 measured periods, M = 10,
% seed rng(1000*o + m) with o the case's row in bla_truth's table (B6 = 7 ... B10 = 11), so the
% injections are the same signals bla_truth used. The ONLY change: the loop is cl_sim (the D-223
% replica of the generator's recorded loop) with gantrySystemExtendedTanh at cfg.friction_gain,
% instead of Simulink with the Karnopp ODE.
%
% GATE (stated before the run): with the Karnopp ODE, realisation 1 of B10 reproduces the stored
% outputs/j2_bla/B10.mat Yk of the excitation folder to max relative 1e-9 (HEURISTIC; the D-223
% gate on the generator's records was bitwise).
%
% Writes outputs/band_tanh_g<gain>/j2_bla/B*.mat (the schema j2_bla.py reads); nothing in the excitation
% folder is written.

    if nargin < 1 || isempty(which), which = {'B6', 'B7', 'B8', 'B9', 'B10'}; end
    if nargin < 2 || isempty(gate_only), gate_only = false; end
    here = fileparts(mfilename('fullpath'));
    repo = fileparts(fileparts(fileparts(here)));
    addpath(fullfile(repo, 'Thesis-writeup', 'Code', 'Data'));
    cfg = tdg_config();
    addpath(here);
    out_dir = fullfile(here, 'outputs', sprintf('band_tanh_g%g', cfg.friction_gain), 'j2_bla');   % one folder per gain
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    ref_dir = fullfile(repo, 'scripts', 'gantry', 'excitation-closed-loop', 'outputs', 'j2_bla');

    H = [240, 87, 180];  Np = round(2.0 / cfg.ts);  NPER = 3;  M = 10;  band = [1 1000];
    OPS = {'B6', -0.30, 7;  'B7', -0.15, 8;  'B8', 0.15, 9;  'B9', 0.30, 10;  'B10', 0.00, 11};
    base = {cfg.m1, cfg.m2, cfg.mb, cfg.mh_rigid, cfg.Lb, cfg.Jb, cfg.Jh, cfg.d, ...
            cfg.cg1, cfg.cg2, cfg.cb1, cfg.cb2, cfg.cy, cfg.kb1, cfg.kb2, ...
            cfg.ma, cfg.ka, cfg.ca, cfg.L0, cfg.cc1, cfg.cc2, cfg.ccy};
    odeK = @(x, u) gantrySystemExtendedCoulomb(u, x, base{:}, cfg.ts);
    odeT = @(x, u) gantrySystemExtendedTanh(u, x, base{:}, cfg.friction_gain);
    kall = round(band(1) * Np * cfg.ts) : round(band(2) * Np * cfg.ts);
    N = NPER * Np;
    t = (0:N - 1)' * cfg.ts;

    % == gate: Karnopp replay of B10 realisation 1 against the stored Simulink run ==
    R = load(fullfile(ref_dir, 'B10.mat'), 'Yk', 'kall');
    assert(isequal(R.kall(:), kall(:)), 'DFT lines differ from the stored B10');
    [Yk1, ~, ~, ~] = one_realisation(cfg, odeK, 0.00, 11, 1, H, Np, NPER, kall, t);
    ref = squeeze(R.Yk(1, :, :, :));
    rel = max(abs(Yk1(:) - ref(:))) / max(abs(ref(:)));
    fprintf('gate  B10 m=1 Karnopp replica vs stored Simulink Yk: max rel %.2e  %s\n', rel, ...
            ternary(rel < 1e-9, 'PASS', 'FAIL'));
    if rel >= 1e-9 || gate_only, return; end

    fprintf('tanh friction, g = %g s/m\n', cfg.friction_gain);
    for o = 1:size(OPS, 1)
        [id, yop, orow] = OPS{o, :};
        if ~any(strcmp(which, id)), continue; end
        fout = fullfile(out_dir, [id '.mat']);
        assert(~isfile(fout), 'refusing to overwrite %s', fout);
        tic;
        Yk = zeros(M, NPER - 1, numel(kall), 3);  Uk = Yk;  Fk = Yk;
        stick = zeros(M, 3);  vrms = zeros(M, 3);  nonper = zeros(M, 3);
        for m = 1:M
            [Yk(m, :, :, :), Uk(m, :, :, :), Fk(m, :, :, :), st] = ...
                one_realisation(cfg, odeT, yop, orow, m, H, Np, NPER, kall, t);
            stick(m, :) = st.stick;  vrms(m, :) = st.vrms;  nonper(m, :) = st.nonper;
            fprintf('  %s m=%d: |v|<1mm/s %s  rms v %s  nonperiodic %s  (%.0f s)\n', id, m, ...
                    mat2str(st.stick, 3), mat2str(st.vrms, 3), mat2str(st.nonper, 3), toc);
        end
        f_lines = kall / (Np * cfg.ts);                                         %#ok<NASGU>
        meta = struct('id', id, 'Y_op', yop, 'level', H, 'band', band, 'cc_on', true, ...
                      'Np', Np, 'NPER', NPER, 'M', M, 'ts', cfg.ts, 'K_design_Y', 0, ...
                      'friction', sprintf('tanh g = %g (D-224), replica cl_sim', cfg.friction_gain)); %#ok<NASGU>
        save(fout, 'Yk', 'Uk', 'Fk', 'kall', 'f_lines', 'stick', 'vrms', 'nonper', 'meta');
        fprintf('%s saved (%.0f s)\n', id, toc);
    end
end

function [Yk, Uk, Fk, st] = one_realisation(cfg, ode, yop, orow, m, H, Np, NPER, kall, t)
% One realisation exactly as bla_truth.m builds, simulates and transforms it.
    N = numel(t);
    rng(1000 * orow + m);
    flog = zeros(Np, 3);
    for ch = 1:3
        kc = kall(mod(kall - kall(1), 3) == ch - 1);
        X = zeros(Np, 1);
        X(kc + 1) = exp(1j * 2 * pi * rand(numel(kc), 1));
        x = ifft(X, 'symmetric');
        flog(:, ch) = H(ch) * x / rms(x);
    end
    flog = repmat(flog, NPER, 1);
    fst = gtd_logical_to_stage(flog, cfg);
    r = repmat([0, 0, yop], N, 1);
    plant = gtd_build_plant(yop, cfg, 1);                 % Cfb = K1 (design Y = 0) at any yop
    [Ac, Bc, Cc, Dc] = ssdata(plant.Cfb);
    ctl = struct('A', Ac, 'B', Bc, 'C', Cc, 'D', Dc);
    S = struct('r_sim', r, 'f_sim', fst, 'd_in', zeros(N, 3));
    y = cl_sim(S, ode, yop, ctl, cfg.P, cfg.ts);
    ufb = lsim(plant.Cfb, r - y);                         % as bla_truth / gtd_run_simulation
    ut = ufb + fst;
    Yk = zeros(NPER - 1, numel(kall), 3);  Uk = Yk;  Fk = Yk;
    for p = 2:NPER
        idx = (p - 1) * Np + (1:Np);
        Yf = fft(y(idx, :) - r(idx, :));  Uf = fft(ut(idx, :));  Ff = fft(flog(idx, :));
        Yk(p - 1, :, :) = Yf(kall + 1, :);
        Uk(p - 1, :, :) = Uf(kall + 1, :);
        Fk(p - 1, :, :) = Ff(kall + 1, :);
    end
    v = zeros(N, 3);
    for j = 1:3, v(:, j) = gradient(y(:, j), cfg.ts); end
    meas = Np + 1:N;
    st.stick = mean(abs(v(meas, :)) < 1e-3, 1);            % reporting proxy, as bla_truth
    st.vrms = rms(v(meas, :), 1);
    st.nonper = rms(y(2 * Np + 1:3 * Np, :) - y(Np + 1:2 * Np, :), 1) ./ rms(y(meas, :) - r(meas, :), 1);
end

function out = ternary(c, a, b)
    if c, out = a; else, out = b; end
end
