function recs = record_table(cfg)
% RECORD_TABLE  The 42 records: 18 training, 6 validation, 11 test and 7 test records repeated with the multisine.
%   One struct per record with its motion parameters p, the operating point Y_op and its seeds.
    L  = @(x) struct('vmax_X', x*cfg.lim.vel, 'amax_X', x*cfg.lim.acc_X, ...
                     'vmax_Y', x*cfg.lim.vel, 'amax_Y', x*cfg.lim.acc_Y);    % fraction x of the limits
    c  = {};

    % Training: standstill at five Y, Y sweeps, random S-curve moves, Lissajous paths and the
    % machine's sine-shaped moves
    Ys = [-0.30, -0.15, 0, 0.15, 0.30];
    for k = 1:5
        c{end+1} = mk('train', 'training', sprintf('TR-S%d', k), 'standstill', Ys(k), ps(), ...
                      sprintf('standstill Y %+.2f, A_prod multisine', Ys(k))); %#ok<AGROW>
    end
    fy = [0.2, 0.5, 0.75];
    for k = 1:3
        c{end+1} = mk('train', 'training', sprintf('TR-Y%d', k), 'oscillatory', 0, po(0, 0,0, 0,0, 0.30, fy(k)), ...
                      sprintf('Y sweep 0 +- 0.30 at %.2f Hz, A_prod', fy(k))); %#ok<AGROW>
    end
    %            name     level     jerk time  X_sym range     X_anti range
    P = {'TR-P1', L(0.25), 0.036, [-0.28 0.28], [0 0];
         'TR-P2', L(0.50), 0.030, [-0.28 0.28], [0 0];
         'TR-P3', L(0.75), 0.012, [-0.28 0.28], [0 0];
         'TR-P4', L(0.50), 0.025, [-0.14 0.14], [-0.001 0.001]};
    for k = 1:4
        c{end+1} = mk('train', 'training', P{k,1}, 'aprbs', 0, pa(P{k,2}, P{k,3}, P{k,4}, P{k,5}, [-0.30 0.30]), ...
                      sprintf('S-curve moves, log strata, T3 %.0f ms, A_prod', 1e3*P{k,3})); %#ok<AGROW>
    end
    c{end+1} = mk('train', 'training', 'TR-L1', 'oscillatory', 0, po(0, 0.28, 0.85, 0, 0, 0.30, 0.35), ...
                  'Lissajous Y 0.30 @0.35 Hz, X 0.28 @0.85 Hz, A_prod');
    c{end+1} = mk('train', 'training', 'TR-L2', 'oscillatory', 0, po(0, 0.08, 1.5, 0.001, 0.8, 0.25, 0.7), ...
                  'Lissajous Y 0.25 @0.7 Hz, X 0.08 @1.5 Hz, X_anti 1 mm @0.8 Hz, A_prod');
    %            name     Y_op   level Ydir Xdir dwell
    T = {'TR-T1',  0.00, 0.75, +1, +1, 0.30;
         'TR-T2',  0.00, 0.45, -1, -1, 0.45;
         'TR-T3',  0.06, 0.75, +1, -1, 0.60;
         'TR-T4', -0.06, 0.45, -1, +1, 0.30};
    for k = 1:4
        c{end+1} = mk('train', 'training', T{k,1}, 'telica', T{k,2}, ...
                      pilc(T{k,3}, T{k,4}, T{k,5}, T{k,6}, [-0.30 0.30]), ...
                      sprintf('ILC-shape sine shuttle, lattice Y_op %+.2f, %.0f %% max, dwell %.2f s, no multisine', ...
                              T{k,2}, 100*T{k,3}, T{k,6})); %#ok<AGROW>
    end

    % Validation
    c{end+1} = mk('val', 'validation', 'VA-S1', 'standstill', 0.10, ps(), 'standstill Y +0.10, A_prod');
    c{end+1} = mk('val', 'validation', 'VA-Y1', 'oscillatory', 0.05, po(0.05, 0,0, 0,0, 0.25, 0.5), ...
                  'Y sweep +0.05 +- 0.25 at 0.5 Hz, A_prod');
    c{end+1} = mk('val', 'validation', 'VA-P1', 'aprbs', 0, pa(L(0.50), 0.030, [-0.28 0.28], [0 0], [-0.30 0.30]), ...
                  'S-curve moves, log strata, 50 %, T3 30 ms, new draw, A_prod');
    c{end+1} = mk('val', 'validation', 'VA-L1', 'oscillatory', -0.05, po(-0.05, 0.08, 1.5, 0, 0, 0.20, 0.7), ...
                  'Lissajous centre -0.05, Y 0.20 @0.7 Hz, X 0.08 @1.5 Hz, A_prod');
    c{end+1} = mk('val', 'validation', 'VA-T1', 'telica', 0.04, pilc(0.75, +1, +1, 0.40, [-0.30 0.30]), ...
                  'ILC-shape, held-out lattice Y_op +0.04, 75 % max, no multisine');
    c{end+1} = mk('val', 'validation', 'VA-T2', 'telica', -0.04, pilc(0.45, +1, -1, 0.35, [-0.30 0.30]), ...
                  'ILC-shape, held-out lattice Y_op -0.04, 45 % max, no multisine');

    % Test: I inside the training range (interpolation), E outside it (extrapolation)
    Yl = -0.01;                                          % Y_op on the lattice of the machine's moves
    I3 = pilc(0.60, +1, +1, 0.30, [-0.20 0.20]);
    c{end+1} = mk('test', 'test-I', 'I1', 'aprbs', 0, pa(L(0.50), 0.030, [-0.28 0.28], [0 0], [-0.30 0.30]), ...
                  'as TR-P2: S-curve moves, 50 %, T3 30 ms, new phases, noise, setpoints');
    c{end+1} = mk('test', 'test-I', 'I2', 'standstill', -0.225, ps(), 'standstill Y -0.225, A_prod');
    c{end+1} = mk('test', 'test-I', 'I3', 'telica', Yl, I3, 'ASMPT sine route, X and Y together, 60 % max, Y inside +-0.20');
    c{end+1} = mk('test', 'test-I', 'I4', 'route', Yl, proute(I3, 'cyc', 'alternate', false), ...
                  'ASMPT sine, alternating X-only 40 mm and Y-only 80 mm moves, 60 % max');
    c{end+1} = mk('test', 'test-I', 'I5', 'route', Yl, proute(pscurve(I3, 0.60), 'scurve', 'together', false), ...
                  'ASMPT S-curve route (I3 sequence), 60 % max, vmax 1.5, T3 25 ms');
    c{end+1} = mk('test', 'test-I', 'I6', 'route', 0, plong(1.5), 'ASMPT sine long strokes 0.50 / 0.56 m, 60 % max, v cap 1.5');
    c{end+1} = mk('test', 'test-E', 'E1', 'standstill', -0.39, ps(), 'standstill Y -0.39, A_prod');
    E2 = I3;  E2.Y_range = [-0.39 0.39];
    c{end+1} = mk('test', 'test-E', 'E2', 'telica', Yl, E2, 'I3 route shuttling to -0.33 / +0.39');
    E3 = proute(pscurve(I3, 1.00), 'scurve', 'together', false);
    E3.acc_margin = true;                                % designed exactly at the acceleration limit, see cfg.acc_margin
    c{end+1} = mk('test', 'test-E', 'E3', 'route', Yl, E3, 'I5 at 100 % max (30 / 50 m/s^2), T3 25 ms');
    c{end+1} = mk('test', 'test-E', 'E4', 'route', 0, plong(1.95), 'I6 with v cap 1.95 m/s');
    E6 = I3;                                             % the measured move of the machine, exact
    E6.X_vmax = 1.000562817;  E6.X_amax = 39.31412412;
    E6.Y_vmax = 1.499812523;  E6.Y_amax = 50.65906898;
    E6.acc_exception = true;                             % above the rated acceleration, as the machine itself moves
    c{end+1} = mk('test', 'test-E', 'E6', 'telica', Yl, E6, 'measured ILC move, exact, on the I3 lattice');

    % The motion-only test records repeated with the multisine, for evaluation only. E records come first,
    % because I3M, I5M and I6M use the same multisine scale as their E partners
    rep = {'E2', {};  'E3', {};  'E4', {};  'E6', {};
           'I3', {'E2M_r1', 'E6M_r1'};  'I5', {'E3M_r1'};  'I6', {'E4M_r1'}};
    names = cellfun(@(x) x.rec, c, 'UniformOutput', false);
    for k = 1:size(rep, 1)
        e = c{strcmp(names, rep{k, 1})};
        e.rec = [rep{k, 1} 'M'];
        e.block = 'test-M';
        e.p.excitation = 'multisine';
        e.scale_from = rep{k, 2};
        e.desc = [e.desc ', + A_prod multisine'];
        c{end+1} = e; %#ok<AGROW>
    end

    % File names and seeds. The seed keys are fixed text: changing a key changes the record's data
    recs = [c{:}];
    for k = 1:numel(recs)
        e = recs(k);
        recs(k).file = [e.rec '_r1'];
        key = sprintf('TAF|%s|%s|r1|nt0', e.set, e.rec);
        recs(k).seed_ms    = seed_of([key '|ms']);       % multisine phases
        recs(k).seed_sp    = seed_of([key '|sp']);       % random moves
        recs(k).seed_noise = seed_of([key '|noise']);    % encoder noise
        recs(k).seed_axes  = arrayfun(@(ax) seed_of(sprintf('%.0f|axis%d', recs(k).seed_noise, ax)), 1:3);
    end
    assert(numel(unique({recs.file})) == numel(recs), 'record_table: duplicate file names');
    seeds = [recs.seed_ms, recs.seed_sp, recs.seed_noise];
    assert(numel(unique(seeds)) == numel(seeds), 'record_table: two records share a seed');
end

% Record builders

function e = mk(set, block, rec, class, Y_op, p, desc)
    e = struct('file','', 'rec',rec, 'set',set, 'block',block, 'class',class, 'Y_op',Y_op, 'p',p, 'desc',desc, ...
               'seed_ms',NaN, 'seed_sp',NaN, 'seed_noise',NaN, 'seed_axes',NaN(1,3), 'scale_from',{{}});
end

function p = ps()                                    % standstill with the multisine
    p = struct('excitation','multisine', 'amp_frac',1.0);
end

function p = po(Y_center, A_sym, f_sym, A_anti, f_anti, A_y, f_y)   % Y sweep or Lissajous path with the multisine
    p = struct('Y_center',Y_center, 'A_sym',A_sym, 'f_sym',f_sym, 'A_anti',A_anti, 'f_anti',f_anti, ...
               'A_y',A_y, 'f_y',f_y, 'excitation','multisine', 'amp_frac',1.0);
end

function p = pa(lvl, jerkTime, Xs, Xa, Yr)          % random S-curve moves with the multisine
    p = lvl;
    p.jerkTime = jerkTime;  p.X_sym_range = Xs;  p.X_anti_range = Xa;  p.Y_range = Yr;
    p.distances = 'logstrata';
    p.excitation = 'multisine';  p.amp_frac = 1.0;
end

function p = pilc(level, Ydir, Xdir, dwell, Yr)
% The machine's sine-shaped moves: the measured strokes 40 / 80 mm, acceleration level x the limit,
% velocity cap 2 m/s, no multisine
    p = struct('X_stroke',0.040, 'X_vmax',2.0, 'X_amax',level*30, ...
               'Y_stroke',0.080, 'Y_vmax',2.0, 'Y_amax',level*50, ...
               'Y_dir',Ydir, 'X_dir',Xdir, 'X_range',[-0.10 0.10], 'Y_range',Yr, ...
               'dwell',dwell, 'amp_frac',1.0, 'excitation','none');
end

function p = pscurve(p, level)                       % S-curve version: vmax 1.5 m/s, jerk time 25 ms
    p.X_vmax = 1.5;  p.Y_vmax = 1.5;
    p.X_amax = level*30;  p.Y_amax = level*50;
    p.T3 = 0.025;
end

function p = proute(p, profile, pattern, first_half)
    p.profile = profile;  p.pattern = pattern;  p.first_half = first_half;
    if ~isfield(p, 'T3'), p.T3 = NaN; end
end

function p = plong(vcap)
% Long strokes 0.50 / 0.56 m from [0, 0] with a first half stroke, 60 % of the acceleration limit,
% velocity cap vcap, dwell 0.30 s
    p = struct('X_stroke',0.50, 'X_vmax',vcap, 'X_amax',0.60*30, ...
               'Y_stroke',0.56, 'Y_vmax',vcap, 'Y_amax',0.60*50, ...
               'Y_dir',-1, 'X_dir',-1, 'X_range',[-0.25 0.25], 'Y_range',[-0.28 0.28], ...
               'dwell',0.30, 'amp_frac',1.0, 'excitation','none');
    p = proute(p, 'cyc', 'together', true);
end

function seed = seed_of(key)
% A 32-bit seed from a text key: the FNV-1a hash (Fowler, Noll, Vo), offset basis 2166136261, prime 16777619
    b = uint64(unicode2native(char(key), 'UTF-8'));
    h = uint64(2166136261);
    for i = 1:numel(b)
        h = bitxor(h, b(i));
        h = mod(h * uint64(16777619), uint64(4294967296));
    end
    seed = double(h);
end
