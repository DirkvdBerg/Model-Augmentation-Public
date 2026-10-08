function check_controller_gate()
% CHECK_CONTROLLER_GATE  C6 of IMPLEMENTATION.md (DATA-DESIGN 5.11 and section 10 item 12).
%
% Stated before the run:
%   Plant: T_AF linearised at rest, friction off (tdg_truth_lin: absorber included,
%   ma at Y + L0), ZOH at 20 kHz, over Y = -0.39 : 0.01 : +0.39 m (the full test grid).
%   Controllers: K1 (designed once at Y = 0) and K2-g+ = 1.2 K1 (E5).
%   PASS iff the discrete closed loop is stable at every Y for both (max |pole| < 1).
%   Margins reported per stage loop, broken at the plant input one loop at a time with
%   the other two closed (getLoopTransfer): minimum gain margin, minimum phase margin
%   and the first crossover. Flagged, not failed, below GM 6 dB or PM 30 deg
%   (Skogestad and Postlethwaite, Multivariable Feedback Control, 2nd ed., 2005,
%   section 2.4.3: "typically we require GM > 2 and PM > 30 deg"); the design gives no
%   number, so the flag is for the user.
%   Friction is not in a linear gate: the post-simulation check and the records test it.
    here = fileparts(fileparts(mfilename('fullpath')));                % scripts/gantry/thesis-data-verification
    addpath(fullfile(fileparts(fileparts(fileparts(here))), 'Thesis-writeup', 'Code', 'Data'));   % the generator
    cfg = tdg_config();
    K1 = gtd_build_plant(0, cfg, 1).Cfb;
    Ks = {K1, cfg.K2_gain * K1};  names = {'K1', 'K2-g+'};
    Ygrid = -0.39:0.01:0.39;
    lab = {'X1', 'X2', 'Y'};
    ok = true;  res = struct();
    for c = 1:2
        K = Ks{c};
        rho = zeros(numel(Ygrid), 1);  GM = inf(numel(Ygrid), 3);  PM = inf(numel(Ygrid), 3);  fc = nan(numel(Ygrid), 3);
        for iy = 1:numel(Ygrid)
            G = c2d(tdg_truth_lin(Ygrid(iy), cfg), cfg.ts, 'zoh');
            CL = feedback(G, K);
            rho(iy) = max(abs(pole(CL)));
            AP = AnalysisPoint('u', 3);
            T = feedback(G * AP * K, eye(3));
            for j = 1:3
                Lj = getLoopTransfer(T, sprintf('u(%d)', j), -1);
                S = allmargin(Lj);
                gm = S.GainMargin(S.GainMargin > 0);  pm = abs(S.PhaseMargin);
                if ~isempty(gm), GM(iy, j) = min(abs(20*log10(gm))); end   % nearest gain change to instability
                if ~isempty(pm), PM(iy, j) = min(pm); end
                if ~isempty(S.PMFrequency), fc(iy, j) = min(S.PMFrequency) / (2*pi); end
            end
        end
        stable = all(rho < 1);
        ok = ok && stable;
        GMabs = GM;
        fprintf('%s: closed loop stable at all %d Y points: %s (max |pole| %.6f at Y %+.2f)\n', names{c}, numel(Ygrid), ...
                pf(stable), max(rho), Ygrid(find(rho == max(rho), 1)));
        for j = 1:3
            [gmin, ig] = min(GMabs(:, j));  [pmin, ip] = min(PM(:, j));
            flag = '';
            if gmin < 6 || pmin < 30, flag = '  <-- below GM 6 dB / PM 30 deg'; end
            fprintf('   loop %-2s: min |GM| %5.2f dB (Y %+.2f), min PM %5.1f deg (Y %+.2f), first crossover %5.1f to %5.1f Hz%s\n', ...
                    lab{j}, gmin, Ygrid(ig), pmin, Ygrid(ip), min(fc(:, j)), max(fc(:, j)), flag);
        end
        res.(matlab.lang.makeValidName(names{c})) = struct('Y', Ygrid, 'rho', rho, 'GM_dB', GM, 'PM_deg', PM, 'fc_Hz', fc);
    end
    fprintf('C6  controller gate (stability of K1 and K2-g+ on T_AF, Y -0.39 to +0.39) -> %s\n', pf(ok));
    save(fullfile(here, 'checks', 'controller_gate_result.mat'), 'res');
end

function s = pf(ok)
    if ok, s = 'PASS'; else, s = 'FAIL'; end
end
