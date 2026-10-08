function man = tdg_manifest(cfg)
% TDG_MANIFEST  Every T_AF record of the first campaign (DATA-DESIGN.md 7.1 to 7.9).
%   man = TDG_MANIFEST(cfg) returns a struct array, one element per simulation of the
%   NOISY version (48 = 18 training + 6 validation + 12 test + 12 FRF, DATA-DESIGN 7.9).
%   The noise-free version holds the same 48 records with d = 0 and everything else
%   identical (handoff reading, SPEC.md). Each record's noise floor is the difference
%   between its two versions (user decision 2026-09-27), so the 4 noise-only twins of the
%   earlier design are no longer in this table (the 4 generated ones are kept as a
%   cross-check in scripts/gantry/thesis-data-verification/noise-twins/).
%   ONE realisation of training and validation (user decision 2026-09-27; the earlier
%   design had three replicate realisations). Records keep the suffix _r1 and their seed
%   keys, so the files generated before this change stay valid unchanged.
%   Replaces scripts/gantry/thesis-data-verification/vendor/original/gtd_build_records.m and gtd_build_records_telica.m (their
%   22 + 7 records are the "current equivalents" named in the 7.1 to 7.4 tables).
%
%   Fields: file, rec, set, split, block, real, twin, class, Y_op, K_gain, p, desc,
%           seed_ms (multisine phases), seed_sp (setpoint draws), seed_noise, free.
%   Every number below is traced in SPEC.md; readings not fixed by the design are
%   marked READING there and here.
    L  = @(x) struct('vmax_X', x*cfg.lim.vel, 'amax_X', x*cfg.lim.acc_X, ...
                     'vmax_Y', x*cfg.lim.vel, 'amax_Y', x*cfg.lim.acc_Y);    % level x max (section 0)
    c  = {};

    % ── 7.1 training, 18 records, one realisation ────────────────────────────
    for r = 1                                        % user decision 2026-09-27 (was 1:3)
        Ys = [-0.30, -0.15, 0, 0.15, 0.30];
        for k = 1:5
            c{end+1} = mk('train', sprintf('TR-S%d', k), r, 'standstill', Ys(k), ps(), ...
                          sprintf('standstill Y %+.2f, A_prod multisine', Ys(k))); %#ok<AGROW>
        end
        fy = [0.2, 0.5, 0.75];
        for k = 1:3
            c{end+1} = mk('train', sprintf('TR-Y%d', k), r, 'oscillatory', 0, po(0, 0,0, 0,0, 0.30, fy(k)), ...
                          sprintf('Y sweep 0 +- 0.30 at %.2f Hz, A_prod', fy(k))); %#ok<AGROW>
        end
        %            id       level     T3     X_sym range     X_anti range
        P = {'TR-P1', L(0.25), 0.036, [-0.28 0.28], [0 0];
             'TR-P2', L(0.50), 0.030, [-0.28 0.28], [0 0];
             'TR-P3', L(0.75), 0.012, [-0.28 0.28], [0 0];
             'TR-P4', L(0.50), 0.025, [-0.14 0.14], [-0.001 0.001]};
        for k = 1:4
            c{end+1} = mk('train', P{k,1}, r, 'aprbs', 0, pa(P{k,2}, P{k,3}, P{k,4}, P{k,5}, [-0.30 0.30]), ...
                          sprintf('S-curve moves, log strata, T3 %.0f ms, A_prod', 1e3*P{k,3})); %#ok<AGROW>
        end
        c{end+1} = mk('train', 'TR-L1', r, 'oscillatory', 0, po(0, 0.28, 0.85, 0, 0, 0.30, 0.35), ...
                      'Lissajous Y 0.30 @0.35 Hz, X 0.28 @0.85 Hz, A_prod'); %#ok<AGROW>
        c{end+1} = mk('train', 'TR-L2', r, 'oscillatory', 0, po(0, 0.08, 1.5, 0.001, 0.8, 0.25, 0.7), ...
                      'Lissajous Y 0.25 @0.7 Hz, X 0.08 @1.5 Hz, X_anti 1 mm @0.8 Hz, A_prod'); %#ok<AGROW>
        %            id       Y_op   level Ydir Xdir dwell
        T = {'TR-T1',  0.00, 0.75, +1, +1, 0.30;
             'TR-T2',  0.00, 0.45, -1, -1, 0.45;
             'TR-T3',  0.06, 0.75, +1, -1, 0.60;
             'TR-T4', -0.06, 0.45, -1, +1, 0.30};
        for k = 1:4
            c{end+1} = mk('train', T{k,1}, r, 'telica', T{k,2}, ...
                          pilc(T{k,3}, T{k,4}, T{k,5}, T{k,6}, [-0.30 0.30]), ...
                          sprintf('ILC-shape sine shuttle, lattice Y_op %+.2f, %.0f %% max, dwell %.2f s, no multisine', ...
                                  T{k,2}, 100*T{k,3}, T{k,6})); %#ok<AGROW>
        end
    end

    % ── 7.2 validation, 6 records, one realisation ───────────────────────────
    for r = 1                                        % user decision 2026-09-27 (was 1:3)
        c{end+1} = mk('val', 'VA-S1', r, 'standstill', 0.10, ps(), 'standstill Y +0.10, A_prod'); %#ok<AGROW>
        c{end+1} = mk('val', 'VA-Y1', r, 'oscillatory', 0.05, po(0.05, 0,0, 0,0, 0.25, 0.5), ...
                      'Y sweep +0.05 +- 0.25 at 0.5 Hz, A_prod'); %#ok<AGROW>
        c{end+1} = mk('val', 'VA-P1', r, 'aprbs', 0, pa(L(0.50), 0.030, [-0.28 0.28], [0 0], [-0.30 0.30]), ...
                      'S-curve moves, log strata, 50 %, T3 30 ms, new draw, A_prod'); %#ok<AGROW>
        c{end+1} = mk('val', 'VA-L1', r, 'oscillatory', -0.05, po(-0.05, 0.08, 1.5, 0, 0, 0.20, 0.7), ...
                      'Lissajous centre -0.05, Y 0.20 @0.7 Hz, X 0.08 @1.5 Hz, A_prod'); %#ok<AGROW>
        % READING (SPEC.md): directions and dwell of the current equivalents VP1, VP2
        c{end+1} = mk('val', 'VA-T1', r, 'telica', 0.04, pilc(0.75, +1, +1, 0.40, [-0.30 0.30]), ...
                      'ILC-shape, held-out lattice Y_op +0.04, 75 % max, no multisine'); %#ok<AGROW>
        c{end+1} = mk('val', 'VA-T2', r, 'telica', -0.04, pilc(0.45, +1, -1, 0.35, [-0.30 0.30]), ...
                      'ILC-shape, held-out lattice Y_op -0.04, 45 % max, no multisine'); %#ok<AGROW>
    end

    % ── 7.3 / 7.4 test, 12 records (7.4b pair definitions) ───────────────────
    Yl = -0.01;                                          % lattice residue 0.07 (7.4b)
    I3 = pilc(0.60, +1, +1, 0.30, [-0.20 0.20]);         % start [0, -0.01], first move +X +Y
    c{end+1} = mk('test', 'I1', 1, 'aprbs', 0, pa(L(0.50), 0.030, [-0.28 0.28], [0 0], [-0.30 0.30]), ...
                  'as TR-P2: S-curve moves, 50 %, T3 30 ms, new phases, noise, setpoints');
    c{end+1} = mk('test', 'I2', 1, 'standstill', -0.225, ps(), 'standstill Y -0.225, A_prod');
    c{end+1} = mk('test', 'I3', 1, 'telica', Yl, I3, 'ASMPT sine route, X and Y together, 60 % max, Y inside +-0.20');
    c{end+1} = mk('test', 'I4', 1, 'route', Yl, proute(I3, 'cyc', 'alternate', false), ...
                  'ASMPT sine, alternating X-only 40 mm and Y-only 80 mm moves, 60 % max');
    c{end+1} = mk('test', 'I5', 1, 'route', Yl, proute(pscurve(I3, 0.60), 'scurve', 'together', false), ...
                  'ASMPT S-curve route (I3 sequence), 60 % max, vmax 1.5, T3 25 ms');
    c{end+1} = mk('test', 'I6', 1, 'route', 0, plong(1.5), 'ASMPT sine long strokes 0.50 / 0.56 m, 60 % max, v cap 1.5');
    c{end+1} = mk('test', 'E1', 1, 'standstill', -0.39, ps(), 'standstill Y -0.39, A_prod');
    E2 = I3;  E2.Y_range = [-0.39 0.39];
    c{end+1} = mk('test', 'E2', 1, 'telica', Yl, E2, 'I3 route shuttling to -0.33 / +0.39');
    E3 = proute(pscurve(I3, 1.00), 'scurve', 'together', false);  E3.acc_margin = true;
    c{end+1} = mk('test', 'E3', 1, 'route', Yl, E3, 'I5 at 100 % max (30 / 50 m/s^2), T3 25 ms');
    c{end+1} = mk('test', 'E4', 1, 'route', 0, plong(1.95), 'I6 with v cap 1.95 m/s');
    c{end+1} = mk('test', 'E5', 1, 'telica', Yl, I3, 'I3 under K2-g+ (K1 gains +20 %)');
    c{end}.K_gain = cfg.K2_gain;
    E6 = I3;                                             % D-210 measured kinematics, exact
    E6.X_vmax = 1.000562817;  E6.X_amax = 39.31412412;
    E6.Y_vmax = 1.499812523;  E6.Y_amax = 50.65906898;
    E6.acc_exception = true;                             % section 10 item 11: named exception
    c{end+1} = mk('test', 'E6', 1, 'telica', Yl, E6, 'measured ILC move (D-210), exact, on the I3 lattice');

    % ── 7.6 FRF references, 3 Y x 4 realisations ─────────────────────────────
    Yf = [0.15, 0.225, 0.35];
    for k = 1:3
        for r = 1:4
            c{end+1} = mk('test', sprintf('TF-%d', k), r, 'standstill', Yf(k), ps(), ...
                          sprintf('FRF reference, standstill Y %+.3f, periodic B_tr multisine (6 periods of 2 s)', Yf(k))); %#ok<AGROW>
        end
    end

    % (noise-only twins removed 2026-09-27: the floor comes from each record's noise-free
    % twin; the twin field and its seed key stay, so a twin could be added back here)

    man = [c{:}];
    blk = {'train','training'; 'val','validation'};
    for k = 1:numel(man)
        e = man(k);
        man(k).file = sprintf('%s_r%d', e.rec, e.real);
        if e.twin > 0, man(k).file = sprintf('%s_nt%d', man(k).file, e.twin); end
        man(k).split = e.set;
        man(k).free  = e.twin == 0;
        base = sprintf('TAF|%s|%s|r%d', e.set, e.rec, e.real);
        man(k).seed_ms    = tdg_seed([base '|nt0|ms']);          % shared by the noise twin
        man(k).seed_sp    = tdg_seed([base '|nt0|sp']);
        man(k).seed_noise = tdg_seed(sprintf('%s|nt%d|noise', base, e.twin));
        j = find(strcmp(blk(:,1), e.set));
        if ~isempty(j), man(k).block = blk{j, 2};
        elseif startsWith(e.rec, 'TF'), man(k).block = 'frf';
        elseif e.twin > 0, man(k).block = 'noise-twin';
        elseif startsWith(e.rec, 'I'), man(k).block = 'test-I';
        else, man(k).block = 'test-E';
        end
    end

    % collision-free seeds (section 10 item 8): distinct keys must give distinct seeds
    keys = {};  vals = [];
    for k = 1:numel(man)
        base = sprintf('TAF|%s|%s|r%d', man(k).set, man(k).rec, man(k).real);
        keys = [keys, {[base '|nt0|ms'], [base '|nt0|sp'], sprintf('%s|nt%d|noise', base, man(k).twin)}]; %#ok<AGROW>
        vals = [vals, man(k).seed_ms, man(k).seed_sp, man(k).seed_noise]; %#ok<AGROW>
    end
    [uk, ia] = unique(keys);
    assert(numel(unique(vals(ia))) == numel(uk), 'tdg_manifest: seed collision among %d keys', numel(uk));
    assert(numel(unique({man.file})) == numel(man), 'tdg_manifest: duplicate file names');
end

% ── builders ────────────────────────────────────────────────────────────────

function e = mk(set, rec, real, class, Y_op, p, desc)
    e = struct('file','', 'rec',rec, 'set',set, 'split','', 'block','', 'real',real, 'twin',0, ...
               'class',class, 'Y_op',Y_op, 'K_gain',1, 'p',p, 'desc',desc, ...
               'seed_ms',NaN, 'seed_sp',NaN, 'seed_noise',NaN, 'free',true);
end

function p = ps()                                    % standstill, A_prod multisine
    p = struct('excitation','multisine', 'amp_frac',1.0);
end

function p = po(Y_center, A_sym, f_sym, A_anti, f_anti, A_y, f_y)   % gtd_build_records po
    p = struct('Y_center',Y_center, 'A_sym',A_sym, 'f_sym',f_sym, 'A_anti',A_anti, 'f_anti',f_anti, ...
               'A_y',A_y, 'f_y',f_y, 'excitation','multisine', 'amp_frac',1.0);
end

function p = pa(lvl, jerkTime, Xs, Xa, Yr)          % S-curve moves, log-strata distances
    p = lvl;
    p.jerkTime = jerkTime;  p.X_sym_range = Xs;  p.X_anti_range = Xa;  p.Y_range = Yr;
    p.distances = 'logstrata';
    p.excitation = 'multisine';  p.amp_frac = 1.0;
end

function p = pilc(level, Ydir, Xdir, dwell, Yr)
% ILC-shape sine moves: measured strokes 40 / 80 mm (D-210), a = level x max,
% velocity cap 2 m/s above the pure-cycloid peak (not binding), as ref_planned.m.
    p = struct('X_stroke',0.040, 'X_vmax',2.0, 'X_amax',level*30, ...
               'Y_stroke',0.080, 'Y_vmax',2.0, 'Y_amax',level*50, ...
               'Y_dir',Ydir, 'X_dir',Xdir, 'X_range',[-0.10 0.10], 'Y_range',Yr, ...
               'dwell',dwell, 'amp_frac',1.0, 'excitation','none');
end

function p = pscurve(p, level)                       % I5 / E3: S-curve, vmax 1.5, T3 25 ms (7.4b)
    p.X_vmax = 1.5;  p.Y_vmax = 1.5;
    p.X_amax = level*30;  p.Y_amax = level*50;
    p.T3 = 0.025;
end

function p = proute(p, profile, pattern, first_half)
    p.profile = profile;  p.pattern = pattern;  p.first_half = first_half;
    if ~isfield(p, 'T3'), p.T3 = NaN; end
end

function p = plong(vcap)
% I6 / E4 (7.4b): start [0, 0], half stroke to (-0.25, -0.28), full strokes 0.50 / 0.56 m,
% a = 60 % max, velocity cap vcap on both axes; dwell 0.30 s READING (SPEC.md).
    p = struct('X_stroke',0.50, 'X_vmax',vcap, 'X_amax',0.60*30, ...
               'Y_stroke',0.56, 'Y_vmax',vcap, 'Y_amax',0.60*50, ...
               'Y_dir',-1, 'X_dir',-1, 'X_range',[-0.25 0.25], 'Y_range',[-0.28 0.28], ...
               'dwell',0.30, 'amp_frac',1.0, 'excitation','none');
    p = proute(p, 'cyc', 'together', true);
end
