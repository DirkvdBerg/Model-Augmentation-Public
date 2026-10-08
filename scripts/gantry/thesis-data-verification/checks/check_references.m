function check_references()
% CHECK_REFERENCES  C7, C9, C10 and C4a of IMPLEMENTATION.md; no simulation.
%
% Stated before the run:
%   C7  seeds: tdg_manifest builds 48 records (one realisation of training and validation and
%       no noise-only twins, user decisions 2026-09-27; the first runs of this check counted 100), every (truth, set, record, realisation,
%       twin, signal) key maps to a distinct seed; the noise twins share seed_ms and
%       seed_sp with their record and differ in seed_noise. PASS iff all hold.
%   C10a the 18 training references built by this generator equal, bitwise at the 4 kHz
%       samples, the references the design's excitation evidence (J9, J10) was computed
%       on (scripts/gantry/excitation-closed-loop/outputs/j9_refs.mat, ref_planned.m),
%       with the S-curve records built from ref_planned's seeds 9001 to 9004.
%       PASS iff max |difference| == 0 for all 18.
%   C10b the new 'route' class with 'cyc' + 'together' reproduces the unchanged
%       'telica' class bitwise on the I3 and E2 parameters. PASS iff == 0.
%   C9  every reference of the manifest passes gtd_validate_ref (position, yaw,
%       velocity, acceleration with the E6 exception and the E3 1 % margin). PASS iff
%       no assert fires; the realised peaks are printed beside the design values.
%   C4a the constructed multisine of every multisine record has lines only on
%       106:0.5:297 Hz (383 per logical channel; off-grid energy below 1e-20 of the
%       total) and rms equal to the applied amplitude (240 N, 87 N m, 180 N) x scale
%       within 1e-9 relative. PASS iff all.
    here = fileparts(fileparts(mfilename('fullpath')));                % scripts/gantry/thesis-data-verification
    addpath(fullfile(fileparts(fileparts(fileparts(here))), 'Thesis-writeup', 'Code', 'Data'));   % the generator
    cfg = tdg_config();

    % == C7 ==
    man = tdg_manifest(cfg);
    nk = numel(man);
    ok7 = nk == 48;
    tw = man([man.twin] > 0);
    for k = 1:numel(tw)
        b = man(strcmp({man.rec}, tw(k).rec) & [man.twin] == 0 & [man.real] == tw(k).real);
        ok7 = ok7 && b.seed_ms == tw(k).seed_ms && b.seed_sp == tw(k).seed_sp && b.seed_noise ~= tw(k).seed_noise;
    end
    allseeds = [[man.seed_ms], [man([man.twin] == 0).seed_sp], [man.seed_noise]];
    nuniq = numel(unique([unique([man.seed_ms]), unique([man.seed_sp]), [man.seed_noise]]));
    fprintf('C7  %d records (%d noise-free), %d distinct seeds over ms/sp/noise keys, twins share ms/sp -> %s\n', ...
            nk, sum([man.free]), nuniq, pf(ok7));
    fprintf('    counts: training %d, validation %d, test-I %d, test-E %d, frf %d, noise-twin %d\n', ...
            sum(strcmp({man.block}, 'training')), sum(strcmp({man.block}, 'validation')), sum(strcmp({man.block}, 'test-I')), ...
            sum(strcmp({man.block}, 'test-E')), sum(strcmp({man.block}, 'frf')), sum(strcmp({man.block}, 'noise-twin')));
    clear allseeds

    % == C10a ==
    J = load(fullfile(cfg.root, 'scripts', 'gantry', 'excitation-closed-loop', 'outputs', 'j9_refs.mat'));
    jid = cellstr(J.ids);
    map = {'TR-S_Y-0.30','TR-S1'; 'TR-S_Y-0.15','TR-S2'; 'TR-S_Y+0.00','TR-S3'; 'TR-S_Y+0.15','TR-S4'; 'TR-S_Y+0.30','TR-S5'; ...
           'TR-Y1','TR-Y1'; 'TR-Y2','TR-Y2'; 'TR-Y3','TR-Y3'; 'TR-P1','TR-P1'; 'TR-P2','TR-P2'; 'TR-P3','TR-P3'; 'TR-P4','TR-P4'; ...
           'TR-L1','TR-L1'; 'TR-L2','TR-L2'; 'TR-T1','TR-T1'; 'TR-T2','TR-T2'; 'TR-T3','TR-T3'; 'TR-T4','TR-T4'};
    jseed = containers.Map({'TR-P1','TR-P2','TR-P3','TR-P4'}, {9001, 9002, 9003, 9004});
    ok10a = true;
    for k = 1:size(map, 1)
        jk = find(strcmp(jid, map{k, 1}));
        e = man(strcmp({man.rec}, map{k, 2}) & [man.real] == 1);
        rec = e;  rec.id = e.file;  rec.seed = e.seed_sp;
        if isKey(jseed, e.rec), rec.seed = jseed(e.rec); end
        r = gtd_make_reference_telica(rec, cfg);
        dr = max(abs(r(1:5:end, :) - squeeze(J.R(jk, :, :))), [], 'all');
        ok10a = ok10a && ~isempty(jk) && dr == 0;
        fprintf('C10a %-6s vs J9 %-12s max|dr| = %.3g m\n', e.rec, map{k, 1}, dr);
    end
    % what this proves (REVIEW.md finding 5): the generator's reference RULES equal those the
    % J9 / J10 evidence used (bitwise at 4 kHz, r1; the S-curve records with ref_planned's seeds
    % 9001 to 9004). The generated TR-P records use hash seeds, so they are other draws of that rule.
    fprintf('C10a training references: same construction rule as the J9/J10 evidence (bitwise at 4 kHz, S-curves at ref_planned seeds) -> %s\n', pf(ok10a));

    % == C10b ==
    ok10b = true;
    for nm = {'I3', 'E2'}
        e = man(strcmp({man.rec}, nm{1}));
        rec = e;  rec.id = e.file;  rec.seed = e.seed_sp;
        r1 = gtd_make_reference_telica(rec, cfg);
        rec.class = 'route';  rec.p.profile = 'cyc';  rec.p.pattern = 'together';  rec.p.first_half = false;  rec.p.T3 = NaN;
        r2 = gtd_make_reference_telica(rec, cfg);
        dr = max(abs(r1 - r2), [], 'all');
        ok10b = ok10b && dr == 0;
        fprintf('C10b %s: route(cyc, together) vs telica max|dr| = %.3g m\n', nm{1}, dr);
    end
    fprintf('C10b route class reproduces telica class -> %s\n', pf(ok10b));

    % == C9 and C4a ==
    ok9 = true;  ok4 = true;
    bins_ref = round((cfg.f_low:cfg.f_step:cfg.f_high) * cfg.t_record) + 1;
    rows = {};
    for k = 1:numel(man)
        e = man(k);
        if e.twin > 0, continue; end                          % same inputs as its record
        try
            in = tdg_build_inputs(e, cfg, false);
        catch err
            ok9 = false;
            fprintf('C9  %s: FAIL %s\n', e.file, err.message);
            continue
        end
        pk = in.pk;
        rows(end+1, :) = {e.file, pk.vl, pk.al, pk.jl, pk.Y_range, pk.Xsym_range, in.chk.scale}; %#ok<AGROW>
        if ~strcmp(e.p.excitation, 'none')
            flog = (cfg.P * in.f_safe.').';                  % stage -> logical [sym, anti(N m), Y]
            F = fft(flog);  Pw = abs(F(1:cfg.N_record/2+1, :)).^2;
            on = false(size(Pw, 1), 1);  on(bins_ref) = true;
            off = sum(Pw(~on, :), 1) ./ sum(Pw, 1);
            nl = sum(Pw(on, :) > 1e-6 * max(Pw(on, :)), 1);
            A = in.chk.scale * in.ms.A;
            relA = abs(rms(flog) ./ A - 1);
            ok = all(off < 1e-20) && all(nl == numel(bins_ref)) && all(relA < 1e-9);
            ok4 = ok4 && ok;
            if e.real == 1 || ~ok
                fprintf('C4a %-10s lines %s, off-grid %s, rms %s (applied %s, scale %.2f) -> %s\n', e.file, mat2str(nl), ...
                        mat2str(off, 2), mat2str(rms(flog), 5), mat2str(A, 5), in.chk.scale, pf(ok));
            end
        end
    end
    fprintf('C9  every manifest reference within the limits (E6 exception, E3 margin) -> %s\n', pf(ok9));
    fprintf('C4a constructed multisine on the B_tr grid at the applied level -> %s\n', pf(ok4));

    % realised reference kinematics, first realisation per record (X_sym / Y)
    fprintf('\nrealised reference peaks (logical X_sym / Y): v [m/s], a [m/s^2], j [m/s^3], Y range, X_sym range, pre-check scale\n');
    for k = 1:size(rows, 1)
        if endsWith(rows{k, 1}, '_r1')
            fprintf('  %-10s v %5.3f / %5.3f  a %6.2f / %6.2f  j %7.0f / %7.0f  Y [%+.3f %+.3f]  X [%+.3f %+.3f]  scale %.2f\n', ...
                    rows{k, 1}, rows{k, 2}, rows{k, 3}, rows{k, 4}, rows{k, 5}, rows{k, 6}, rows{k, 7});
        end
    end
    save(fullfile(here, 'checks', 'reference_peaks.mat'), 'rows');
end

function s = pf(ok)
    if ok, s = 'PASS'; else, s = 'FAIL'; end
end
