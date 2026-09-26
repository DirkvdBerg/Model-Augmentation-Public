function ref_planned()
% REF_PLANNED  J9 of EXCITATION-VALIDATION.md (section 17): the 18 training references of
% DATA-DESIGN.md 7.1, motion only, no simulation, built with the generator's own shape functions:
%   standstill and sinusoidal paths (TR-S, TR-Y, TR-L)  gtd_make_reference, class standstill / oscillatory
%   ILC-shape cycloids (TR-T)                            gtd_make_reference_telica, per-record level and dwell
%   S-curve moves (TR-P)                                 thirdOrderSetpointETEL, log-strata distances (5.6)
% The generator does not yet have log strata or per-record ILC kinematics (DATA-DESIGN.md implementation
% items 4 and 7), so those two are assembled here from the same building blocks.
% Writes scripts/gantry/excitation-closed-loop/outputs/j9_refs.mat: R (18 x N4 x 3, stage [X1 X2 Y]) at
% 4 kHz (every 5th sample of the 20 kHz reference), ids, Y_op, dt. Nothing else is written.
    here = fileparts(mfilename('fullpath'));
    ex   = fileparts(here);
    repo = fileparts(fileparts(fileparts(ex)));
    addpath(fullfile(repo, 'Matlab-scripts', 'Augmentation', 'data'));
    addpath(fullfile(repo, 'Matlab-scripts', 'Augmentation-telica-profiles'));
    addpath(fullfile(repo, 'kamtin-fp-model', '03 Simulink gantry', 'functions'));   % read only
    cfg = gtd_config('augmentation', true, 0.50);

    AMAX = [30, 50];  VMAX = 2;               % datasheet maximum X / Y [m/s^2], velocity [m/s] (DATA-DESIGN.md 0)
    lvl  = @(L) struct('vmax_X', L*VMAX, 'amax_X', L*AMAX(1), 'vmax_Y', L*VMAX, 'amax_Y', L*AMAX(2));
    c = {};
    for y = [-0.30, -0.15, 0, 0.15, 0.30]
        c{end+1} = {sprintf('TR-S_Y%+.2f', y), 'std', y, struct('excitation', 'none')}; %#ok<AGROW>
    end
    fy = [0.2, 0.5, 0.75];
    for k = 1:3
        c{end+1} = {sprintf('TR-Y%d', k), 'osc', 0, po(0, 0, 0, 0, 0, 0.30, fy(k))}; %#ok<AGROW>
    end
    %            id       level      T3 [s]  X_sym range     X_anti range      seed
    P = {'TR-P1', lvl(0.25), 0.036, [-0.28 0.28], [0 0],          9001;
         'TR-P2', lvl(0.50), 0.030, [-0.28 0.28], [0 0],          9002;
         'TR-P3', lvl(0.75), 0.012, [-0.28 0.28], [0 0],          9003;
         'TR-P4', lvl(0.50), 0.025, [-0.14 0.14], [-0.001 0.001], 9004};
    for k = 1:size(P, 1)
        c{end+1} = {P{k, 1}, 'p2p', 0, P(k, 2:end)}; %#ok<AGROW>
    end
    c{end+1} = {'TR-L1', 'osc', 0, po(0, 0.28, 0.85, 0, 0, 0.30, 0.35)};
    c{end+1} = {'TR-L2', 'osc', 0, po(0, 0.08, 1.5, 0.001, 0.8, 0.25, 0.7)};
    %            id       Y_op   level  Y_dir X_dir dwell [s]
    T = {'TR-T1', 0.00, 0.75, +1, +1, 0.30;
         'TR-T2', 0.00, 0.45, -1, -1, 0.45;
         'TR-T3', 0.06, 0.75, +1, -1, 0.60;
         'TR-T4', -0.06, 0.45, -1, +1, 0.30};
    for k = 1:size(T, 1)
        L = T{k, 3};
        % stroke 40 / 80 mm (measured, DATA-DESIGN.md 5.7), a = L x max; the velocity cap is set above the
        % pure-cycloid peak so the 7.1 peaks (0.76 / 1.38 at 0.75, 0.59 / 1.07 at 0.45) are reached
        p = struct('X_stroke', 0.040, 'X_vmax', VMAX, 'X_amax', L*AMAX(1), ...
                   'Y_stroke', 0.080, 'Y_vmax', VMAX, 'Y_amax', L*AMAX(2), ...
                   'Y_dir', T{k, 4}, 'X_dir', T{k, 5}, 'X_range', [-0.10 0.10], 'Y_range', [-0.30 0.30], ...
                   'dwell', T{k, 6}, 'amp_frac', 1.0, 'excitation', 'none');
        c{end+1} = {T{k, 1}, 'ilc', T{k, 2}, p}; %#ok<AGROW>
    end

    n = numel(c);  N4 = numel(1:5:cfg.N_record);
    R = zeros(n, N4, 3);  ids = cell(n, 1);  Yop = zeros(n, 1);
    for k = 1:n
        [id, kind, y0, p] = c{k}{:};
        switch kind
            case 'std'
                r = gtd_make_reference(struct('class', 'standstill', 'Y_op', y0, 'seed', 0, 'p', p), cfg);
            case 'osc'
                r = gtd_make_reference(struct('class', 'oscillatory', 'Y_op', y0, 'seed', 0, 'p', p), cfg);
            case 'ilc'
                r = gtd_make_reference_telica(struct('class', 'telica', 'Y_op', y0, 'seed', 0, 'p', p), cfg);
            case 'p2p'
                r = ref_logstrata(y0, p{:}, cfg);
        end
        assert(size(r, 1) == cfg.N_record, '%s: %d samples', id, size(r, 1));
        v = diff(r) / cfg.ts;  a = diff(v) / cfg.ts;  j = diff(a) / cfg.ts;
        lg = [(r(:, 1) + r(:, 2)) / 2, r(:, 3)];                 % logical X_sym, Y
        vl = diff(lg) / cfg.ts;  al = diff(vl) / cfg.ts;  jl = diff(al) / cfg.ts;
        fprintf('%-10s peak v X/Y %.3f / %.3f m/s, a %.2f / %.2f m/s^2, j %.0f / %.0f m/s^3, X_anti %.4f m\n', ...
                id, max(abs(vl)), max(abs(al)), max(abs(jl)), max(abs(r(:, 1) - r(:, 2))) / 2);
        R(k, :, :) = r(1:5:end, :);  ids{k} = id;  Yop(k) = y0;
    end
    dt = 5 * cfg.ts;
    save(fullfile(ex, 'outputs', 'j9_refs.mat'), 'R', 'ids', 'Yop', 'dt');
    fprintf('saved outputs/j9_refs.mat (%d records, %d samples at %.0f Hz)\n', n, N4, 1/dt);
end

function p = po(Y_center, A_sym, f_sym, A_anti, f_anti, A_y, f_y)
% oscillatory parameters, as gtd_build_records.m
    p = struct('Y_center', Y_center, 'A_sym', A_sym, 'f_sym', f_sym, 'A_anti', A_anti, 'f_anti', f_anti, ...
               'A_y', A_y, 'f_y', f_y, 'excitation', 'none', 'amp_frac', 1.0);
end

function r = ref_logstrata(Y_op, lv, T3, Xs_range, Xa_range, seed, cfg)
% S-curve point-to-point moves as ref_aprbs of gtd_make_reference.m (same move profile, 0.1 s holds, never
% truncated mid-move, 0.5 s holds at both ends), with distances from the log strata of DATA-DESIGN.md 5.6.
% HEURISTIC (5.6 leaves the details open): each axis cycles through the six strata in shuffled order,
% log-uniform inside a stratum, random sign, reflected at the range bound; the top stratum runs up to the
% largest move that fits.
    rng(seed);
    strata = [1 3; 3 10; 10 30; 30 100; 100 300; 300 1e9] * 1e-3;
    Y_range = [-0.30 0.30];
    cur = [0, 0, Y_op];  body = zeros(0, 3);  qs = [];  qy = [];
    while true
        if isempty(qs), qs = randperm(6); end
        if isempty(qy), qy = randperm(6); end
        ds = draw(strata(qs(1), :), cur(1), Xs_range);  qs(1) = [];
        dy = draw(strata(qy(1), :), cur(3), Y_range);   qy(1) = [];
        tgt = [cur(1) + ds, Xa_range(1) + diff(Xa_range) * rand, cur(3) + dy];
        mv  = move_between(cur, tgt, lv, T3, cfg.ts);
        seg = [mv; repmat(mv(end, :), cfg.n_hold_short, 1)];
        if size(body, 1) + size(seg, 1) > cfg.n_active, break; end
        body = [body; seg]; %#ok<AGROW>
        cur = tgt;
    end
    body = [body; repmat(body(end, :), cfg.n_active - size(body, 1), 1)];
    r = [repmat([0, 0, Y_op], cfg.n_hold, 1); body];
    r = [r; repmat(r(end, :), cfg.N_record - size(r, 1), 1)];
end

function d = draw(st, pos, range)
    room = [range(2) - pos, pos - range(1)];          % room up, room down
    hi = min(st(2), max(room));  lo = min(st(1), hi);
    m = exp(log(lo) + rand * (log(hi) - log(lo)));    % log-uniform in the stratum
    s = sign(rand - 0.5);  if s == 0, s = 1; end
    if (s > 0 && m > room(1)) || (s < 0 && m > room(2)), s = -s; end   % reflect at the bound
    d = s * m;
end

function mv = move_between(cur, tgt, p, T3, ts)
% gtd_make_reference.m move_between: each logical axis its own third-order profile, padded to the longest
    pv = {sp1d(tgt(1) - cur(1), p.vmax_X, p.amax_X, T3, ts), sp1d(tgt(2) - cur(2), p.vmax_X, p.amax_X, T3, ts), ...
          sp1d(tgt(3) - cur(3), p.vmax_Y, p.amax_Y, T3, ts)};
    K = max([cellfun(@numel, pv), 1]);
    q = zeros(K, 3);
    for i = 1:3
        x = pv{i};
        if isempty(x), x = zeros(K, 1); else, x = [x; x(end) * ones(K - numel(x), 1)]; end %#ok<AGROW>
        q(:, i) = cur(i) + x;
    end
    mv = [q(:, 1) + q(:, 2), q(:, 1) - q(:, 2), q(:, 3)];
end

function pv = sp1d(delta, vmax, amax, T3, ts)
    if delta == 0, pv = zeros(0, 1); return; end
    mag = thirdOrderSetpointETEL(abs(delta), vmax, amax, amax / T3, Inf, ts);
    pv  = sign(delta) * mag(:, 1);
end
