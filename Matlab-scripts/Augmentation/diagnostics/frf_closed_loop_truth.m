% frf_closed_loop_truth.m
% Analytic closed-loop FRF of the 8-state TRUTH gantry, overlaid on the
% nonparametric estimates from the records the dataset already contains.
%
% WHY THIS FILE EXISTS. Every excitation band this project has used was designed
% on the PLANT G. The multisine is injected as an exogenous force BEFORE the
% controller, so the force that actually reaches the plant is S*f_ms, and S
% notches exactly where |G| peaks. This script produces the three quantities of
% the loop, analytically and from data, so the design can be moved onto S*f_ms.
%
% THE LOOP. u_fb = Cfb*(r - y),  u_total = f_ms + u_fb,  y = G*u_total, so
%     u_total = S f_ms,      S  = (I + Cfb G)^-1      (input sensitivity)
%     y       = G S f_ms,    G S                      (closed-loop disturbance response)
% with G, Cfb, S all in STAGE coordinates: gtd_save_record.m stores y = q (stage),
% f_sim = f_stage and u_fb = lsim(Cfb, r - q), and gtd_build_plant designs Cfb on
% sys = P.' * G_phys * P, which is the stage plant.
%
% WHAT THE DATA CAN AND CANNOT ESTIMATE. On a standstill record only channel 3 of
% f_sim is driven, so every signal is a deterministic linear function of one scalar
% source. Therefore
%     tfestimate(f_sim_3, u_total_3) = S(3,3)                exact
%     tfestimate(f_sim_3, y_3)       = (G S)(3,3)            exact
%     tfestimate(u_total_3, y_3)     = (G S)(3,3) / S(3,3)   NOT G(3,3)
% because u_total is an output of the loop, not an independent input. The third
% quantity is plotted as G_eff and compared against that ratio; the true G(3,3) is
% drawn alongside so the difference between them is visible rather than assumed.
%
% Run from the repo root:
%   run('Matlab-scripts/Augmentation/diagnostics/frf_closed_loop_truth.m')

clearvars; close all; clc

% gtd_config lives in ../data and adds the rest of the paths itself.
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'data'));

% -- settings ---------------------------------------------------------------
DATA_NAME = 'augmentation_ma50_b140-230_a6_z03';
RECORDS   = {'E1_resonance_sweep', 'T1_standstill_Ym30', 'T3_standstill_Y000', ...
             'T4_standstill_Yp15', 'T5_standstill_Yp30', 'E2_multisine_Yp22'};
MA_FRAC         = 0.50;   % as generated: generate_trajectory_data.m MA_FRAC
ZETA_A_OVERRIDE = 0.03;   % as generated: generate_trajectory_data.m ZETA_A_OVERRIDE
NPERSEG   = 8192;         % 2.4414 Hz resolution at 20 kHz; matches the handoff estimates
F_LO      = 100;          % [Hz] plot band
F_HI      = 300;
F_CMP     = [150, 260];   % [Hz] band the acceptance criterion is evaluated over
COH_MIN   = 0.99;         % coherence floor, reported only (see the mask note below)
EXC_DB    = 20;           % [dB] a bin counts as excited within this of the peak f_sim PSD
SAVE_FIGS = true;

% -- configuration, with the two generator overrides reapplied --------------
cfg = gtd_config('augmentation', true, MA_FRAC);
if ~isempty(ZETA_A_OVERRIDE)
    cfg.zeta_a = ZETA_A_OVERRIDE;
    % THEORY: gtd_config.m, c = 2*zeta*sqrt(k*m); recomputed, never scaled.
    cfg.ca     = 2*cfg.zeta_a*sqrt(cfg.ka*cfg.ma);
end
root     = cfg.root;
data_dir = fullfile(root, 'data', 'gantry', 'matlab', 'trajectory', DATA_NAME);
fig_dir  = fullfile(root, 'scripts', 'gantry', 'meeting', 'meeting-11-09-2026', 'figures');
if SAVE_FIGS && ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
assert(exist(data_dir, 'dir') == 7, 'Dataset folder not found: %s', data_dir);

fn_free = cfg.fa * sqrt(1 + cfg.ma/cfg.mh_rigid);   % free-free two-mass root [Hz]
fprintf('Dataset      : %s\n', DATA_NAME);
fprintf('ma_frac %.2f  zeta_a %.3f  ca %.3f Ns/m  fa %g Hz  coupled mode %.2f Hz\n\n', ...
        cfg.ma_frac, cfg.zeta_a, cfg.ca, cfg.fa, fn_free);

% -- frequency grid: exactly the Welch bin grid -----------------------------
f_all = (0:NPERSEG/2)' * cfg.fs / NPERSEG;
sel   = f_all >= F_LO & f_all <= F_HI;
f     = f_all(sel);
w     = 2*pi*f;
cmp   = f >= F_CMP(1) & f <= F_CMP(2);
win   = hann(NPERSEG, 'periodic');
nov   = NPERSEG/2;

summary = struct([]);

for ir = 1:numel(RECORDS)
    name = RECORDS{ir};
    R = load(fullfile(data_dir, [name '.mat']));
    fs = double(R.fs);
    assert(abs(fs - cfg.fs) < 1e-9, 'Record rate %g differs from cfg.fs %g', fs, cfg.fs);

    % -- active window: the samples where the exogenous force is actually on -
    fsim = double(R.f_sim);  utot = double(R.u_total);  yq = double(R.y);
    da   = double(R.delta_a);  rsim = double(R.r_sim);
    on   = find(any(abs(fsim) > 0, 2));
    idx  = on(1):on(end);

    % How many exogenous channels are actually driven decides which estimator is
    % valid. E1 drives Y alone; the T and E multisine records drive all three, so
    % a SISO H1 from f_3 alone is biased by the other two and the full 3-input
    % estimator is required.
    e_ch  = sum(fsim(idx,:).^2, 1);
    drive = e_ch > 1e-12 * max(e_ch);
    n_in  = nnz(drive);

    Y_op = median(rsim(idx,3));       % the operating point the controller was built at
    Y_sp = max(rsim(idx,3)) - min(rsim(idx,3));
    if Y_sp > 1e-4
        warning('%s: reference Y moves by %.3g m over the window; frozen-Y FRF is approximate.', ...
                name, Y_sp);
    end

    % -- nonparametric estimates --------------------------------------------
    % Outputs stacked as [u_total(3); y(3); delta_a]. With n_in driven inputs the
    % H1 solution is H.' = Sxx \ Sxy, using MATLAB's cpsd convention
    % Pxy = E[conj(X) Y], which is the same convention tfestimate inverts.
    Xin  = fsim(idx, drive);
    Yout = [utot(idx,:), yq(idx,:), da(idx)];
    [Hm, coh_mult] = mimo_h1(Xin, Yout, win, nov, NPERSEG, fs);
    Hm = Hm(sel,:,:);  coh_mult = coh_mult(sel,:);

    iy = find(find(drive) == 3);      % column of Hm belonging to the Y force
    assert(~isempty(iy), '%s: the Y channel of f_sim is not driven.', name);

    S_col  = squeeze(Hm(:,1:3,iy));   % u_total over f_Y   (3 outputs)
    SG_col = squeeze(Hm(:,4:6,iy));   % y over f_Y
    S_m    = S_col(:,3);
    SG_m   = SG_col(:,3);
    Df_m   = squeeze(Hm(:,7,iy));
    coh_S  = coh_mult(:,3);           % multiple coherence of u_total_3
    coh_SG = coh_mult(:,6);           % multiple coherence of y_3

    if n_in >= 3
        % The full 3x3 S is invertible, so the TRUE plant is recoverable:
        % G = (G S) S^-1, with no closed-loop bias left in it.
        G_m = zeros(numel(f),1);  Du_m = zeros(numel(f),1);
        for k = 1:numel(f)
            Sk  = squeeze(Hm(k,1:3,:));
            GSk = squeeze(Hm(k,4:6,:));
            Gk  = GSk / Sk;
            G_m(k)  = Gk(3,3);
            Dk = squeeze(Hm(k,7,:)).' / Sk;
            Du_m(k) = Dk(3);
        end
        g_label = 'G_{33} (MIMO-recovered)';
    else
        % One driven input: S and G*S are exact, but the plant is only reachable
        % as the ratio (G S)_{33}/S_{33}, which is not G_{33} (see the header).
        G_m  = SG_m ./ S_m;
        Du_m = Df_m ./ S_m;
        g_label = 'G_{eff} = (GS)_{33}/S_{33}';
    end

    Pf = pwelch(fsim(idx,3), win, nov, NPERSEG, fs);  Pf = Pf(sel);
    Pu = pwelch(utot(idx,3), win, nov, NPERSEG, fs);  Pu = Pu(sel);
    coh_G = coh_SG;                    % the recovered G inherits the y_3 coherence

    % -- analytic loop at this operating point ------------------------------
    [Gd, Cfb] = truth_loop(Y_op, cfg);
    Hg = freqresp(Gd,  w);       % 4x3xN : [X1; X2; Y; delta_a] over stage forces
    Hc = freqresp(Cfb, w);       % 3x3xN
    nf = numel(w);  I3 = eye(3);
    S_a = zeros(nf,1); SG_a = zeros(nf,1); G_a = zeros(nf,1);
    Df_a = zeros(nf,1); Du_a = zeros(nf,1);
    for k = 1:nf
        Gp = Hg(1:3,:,k);  Gda = Hg(4,:,k);
        Sk = (I3 + Hc(:,:,k)*Gp) \ I3;      % S = (I + Cfb G)^-1
        GS = Gp*Sk;  DS = Gda*Sk;
        S_a(k) = Sk(3,3);  SG_a(k) = GS(3,3);  G_a(k) = Gp(3,3);
        Df_a(k) = DS(3);   Du_a(k) = Gda(3);
    end
    if n_in >= 3
        Gcmp_a = G_a;                % the MIMO recovery targets the plant itself
    else
        Gcmp_a = SG_a ./ S_a;        % the single-input ratio targets (GS)/S
    end

    % -- agreement test (acceptance criterion, section 10 of the handoff) ----
    % The criterion asks for agreement "at every bin from 150 to 260 Hz". The
    % multisine has no lines above about 228 Hz, so above that the record carries
    % only leakage and float32 rounding and the measured curve estimates nothing.
    % COHERENCE CANNOT DETECT THIS: the simulation is noiseless, so the multiple
    % coherence stays above 0.99 well outside the excitation. The informative
    % bins are therefore defined from the measured EXCITATION level instead.
    % HEURISTIC: a bin counts as excited when its f_sim PSD is within EXC_DB of
    % the in-band maximum. 20 dB is a factor 100 in power, far above the float32
    % storage floor (the out-of-band bins sit 100+ dB down) and far below the
    % in-band ripple of a flat multisine.
    ok   = 10*log10(Pf) > max(10*log10(Pf)) - EXC_DB;
    mask = cmp & ok;
    [dev_S,  sc_S ] = agreement(S_m,  S_a,    coh_S,  mask, COH_MIN);
    [dev_SG, sc_SG] = agreement(SG_m, SG_a,   coh_SG, mask, COH_MIN);
    [dev_G,  sc_G ] = agreement(G_m,  Gcmp_a, coh_G,  mask, COH_MIN);
    dev_S_all  = 20*log10(abs(S_m(cmp)))  - 20*log10(abs(S_a(cmp)));
    dev_SG_all = 20*log10(abs(SG_m(cmp))) - 20*log10(abs(SG_a(cmp)));
    dev_G_all  = 20*log10(abs(G_m(cmp)))  - 20*log10(abs(Gcmp_a(cmp)));

    f_ok = f(mask);
    fprintf('-- %-22s  Y_op = %+.3f m   inputs driven: %d ------------\n', name, Y_op, n_in);
    if any(mask)
        fprintf('   informative bins: %.1f to %.1f Hz (%d bins, f_sim PSD within %g dB of peak)\n', ...
                f_ok(1), f_ok(end), nnz(mask), EXC_DB);
    else
        fprintf('   informative bins: none in %g-%g Hz\n', F_CMP(1), F_CMP(2));
    end
    fprintf('   quantity  max|dev|   at [Hz]  median|dev|   scatter   max|dev| over %g-%g Hz\n', ...
            F_CMP(1), F_CMP(2));
    report('S',   dev_S,  sc_S,  dev_S_all,  f_ok);
    report('G*S', dev_SG, sc_SG, dev_SG_all, f_ok);
    report('G',   dev_G,  sc_G,  dev_G_all,  f_ok);

    % Peaks are searched around the mode only; over the whole 100-300 Hz window
    % the compliance at 100 Hz can exceed the resonance and the report is useless.
    pk   = f >= 180 & f <= 280;
    fpk  = f(pk);
    [~, kp]  = max(abs(SG_a(pk)));  [~, kpm] = max(abs(SG_m(pk)));
    [~, kg]  = max(abs(Gcmp_a(pk))); [~, kgm] = max(abs(G_m(pk)));
    fprintf('   peak |G*S| : %.2f Hz analytic, %.2f Hz measured\n', fpk(kp), fpk(kpm));
    fprintf('   peak |G|   : %.2f Hz analytic, %.2f Hz measured\n', fpk(kg), fpk(kgm));
    probe = [150 200 212 232 250];
    fprintf('   |S| :');
    for pf = probe
        [~, kk] = min(abs(f - pf));
        fprintf('  %gHz %.3f/%.3f', pf, abs(S_m(kk)), abs(S_a(kk)));
    end
    fprintf('   (measured/analytic)\n\n');

    s.name = name; s.Y_op = Y_op; s.n_in = n_in;
    s.dev = [max(abs(dev_S)), max(abs(dev_SG)), max(abs(dev_G))];
    s.sc  = [sc_S, sc_SG, sc_G];
    s.f_peak_SG = fpk(kp); s.f_peak_G = fpk(kg);
    if isempty(summary), summary = s; else, summary(end+1) = s; end %#ok<SAGROW>

    % -- figures ------------------------------------------------------------
    ttl = sprintf('%s   Y_{op} = %+.3f m   (ma\\_frac %.2f, \\zeta_a %.3f)', ...
                  strrep(name,'_','\_'), Y_op, cfg.ma_frac, cfg.zeta_a);
    fig1 = loop_figure(f, S_m, S_a, SG_m, SG_a, G_m, Gcmp_a, G_a, g_label, ...
                       coh_S, coh_SG, coh_G, ok, f_ok, dev_S, dev_SG, dev_G, ...
                       fn_free, cfg.fa, F_CMP, ttl);
    fig2 = absorber_figure(f, Df_m, Df_a, Du_m, Du_a, Pf, Pu, fn_free, cfg.fa, ttl);
    if SAVE_FIGS
        save_fig(fig1, fullfile(fig_dir, sprintf('frf_cl_truth_%s', short(name))));
        save_fig(fig2, fullfile(fig_dir, sprintf('frf_cl_absorber_%s', short(name))));
    end
end

% -- cross-record summary: is S Y-dependent? --------------------------------
fprintf('== summary ========================================================\n');
fprintf('%-22s %8s %5s %9s %9s %9s %9s\n', 'record', 'Y_op', 'n_in', 'dev_S', 'dev_GS', 'dev_G', 'f_pk_GS');
for k = 1:numel(summary)
    fprintf('%-22s %+8.3f %5d %9.3f %9.3f %9.3f %9.2f\n', summary(k).name, summary(k).Y_op, ...
            summary(k).n_in, summary(k).dev(1), summary(k).dev(2), summary(k).dev(3), ...
            summary(k).f_peak_SG);
end
fprintf('dev columns are max |measured - analytic| in dB over the informative bins.\n');

% ===========================================================================
function [Gd, Cfb] = truth_loop(Y_op, cfg)
% 8-state truth plant in STAGE coordinates, discretised at the record rate, and
% the controller the record was generated with.
    [A, B] = truth_AB(Y_op, cfg);
    G_l = ss(A, B, [eye(4), zeros(4,4)], zeros(4,3));   % logical force -> [X;Th;Y;delta_a]
    P   = cfg.P;
    % q_stage = P.' q_logical and f_logical = P f_stage (gtd_logical_to_stage.m),
    % so G_stage = P.' G_logical P. delta_a is a scalar relative coordinate: its
    % row takes the input transform only.
    G_s = [P.' * G_l(1:3,:) * P ; G_l(4,:) * P];
    % Cfb is discrete (ruleOfThumb ends in c2d tustin at ts), so the plant must be
    % discretised at the SAME rate before the loop is formed, or the notch depth
    % will not match the records.
    Gd  = c2d(G_s, cfg.ts, 'zoh');
    pl  = gtd_build_plant(Y_op, cfg);
    Cfb = pl.Cfb;
end

function [A, B] = truth_AB(Y, cfg)
% THEORY: gantrySystemExtended.m, the 4x4 M, C4, K4 and the state-space assembly,
% frozen at x = [.., Y, delta_a = 0, ..]. mh is the RIGID payload mass, as that
% function requires.
    m1=cfg.m1; m2=cfg.m2; mb=cfg.mb; mh=cfg.mh_rigid; ma=cfg.ma; L0=cfg.L0;
    Lb=cfg.Lb; Jb=cfg.Jb; Jh=cfg.Jh; d=cfg.d;
    cg1=cfg.cg1; cg2=cfg.cg2; cb1=cfg.cb1; cb2=cfg.cb2; cy=cfg.cy;
    kb1=cfg.kb1; kb2=cfg.kb2; ka=cfg.ka; ca=cfg.ca;
    da = 0;

    M = [ m1+m2+mb+mh+ma, (m1-m2)*Lb/2 - (mh+ma)*Y - ma*L0 - ma*da, 0, 0;
          (m1-m2)*Lb/2 - (mh+ma)*Y - ma*L0 - ma*da, ...
              Jb+Jh+(m1+m2)*Lb^2/4 + (mh+ma)*d^2 + mh*Y^2 + ma*(Y+L0+da)^2, -(mh+ma)*d, -ma*d;
          0, -(mh+ma)*d, mh+ma, ma;
          0, -ma*d,      ma,    ma];

    C4 = [ cg1+cg2,        (cg1-cg2)*Lb/2,            0,  0;
           (cg1-cg2)*Lb/2, cb1+cb2+(cg1+cg2)*Lb^2/4,  0,  0;
           0,              0,                         cy, 0;
           0,              0,                         0,  ca];

    K4 = [0, 0,       0, 0;
          0, kb1+kb2, 0, 0;
          0, 0,       0, 0;
          0, 0,       0, ka];

    A = [zeros(4), eye(4); -M\K4, -M\C4];
    B = [zeros(4,3); M \ [eye(3); zeros(1,3)]];
end

function [H, coh] = mimo_h1(X, Y, win, nov, nfft, fs)
% Multi-input H1 estimate and multiple coherence.
%   X   (N x p) inputs, Y (N x q) outputs
%   H   (nf x q x p) with Y(f) = H(f) X(f)
%   coh (nf x q) multiple coherence per output
% MATLAB's cpsd convention is Pxy = E[conj(X) Y], so Sxy = Sxx * H.' and the
% solution is H.' = Sxx \ Sxy. That is the same convention tfestimate inverts,
% so a one-input call reproduces tfestimate exactly.
    p = size(X,2);  q = size(Y,2);
    Sxx = [];  Sxy = [];
    for i = 1:p
        for j = 1:p
            Pij = cpsd(X(:,i), X(:,j), win, nov, nfft, fs);
            if isempty(Sxx), Sxx = zeros(numel(Pij), p, p); end
            Sxx(:,i,j) = Pij; %#ok<AGROW>
        end
        for k = 1:q
            Pik = cpsd(X(:,i), Y(:,k), win, nov, nfft, fs);
            if isempty(Sxy), Sxy = zeros(numel(Pik), p, q); end
            Sxy(:,i,k) = Pik; %#ok<AGROW>
        end
    end
    Syy = zeros(size(Sxx,1), q);
    for k = 1:q
        Syy(:,k) = pwelch(Y(:,k), win, nov, nfft, fs);
    end

    nf  = size(Sxx,1);
    H   = zeros(nf, q, p);
    coh = zeros(nf, q);
    for n = 1:nf
        A = reshape(Sxx(n,:,:), p, p);
        B = reshape(Sxy(n,:,:), p, q);
        Ht = A \ B;                      % p x q, equals H.'
        H(n,:,:) = Ht.';
        % Multiple coherence: the share of each output's power explained by X.
        coh(n,:) = real(sum(conj(B) .* Ht, 1)) ./ max(Syy(n,:), realmin);
    end
    coh = min(max(coh, 0), 1);
end

function [dev, scat] = agreement(H_meas, H_anal, coh, mask, coh_min)
% dev  = measured minus analytic in dB, over the informative bins.
% scat = the measured estimate's own roughness in dB (see the HEURISTIC note).
    if ~any(mask)
        dev = NaN;
    else
        dev = 20*log10(abs(H_meas(mask))) - 20*log10(abs(H_anal(mask)));
    end
    mdB  = 20*log10(abs(H_meas));
    good = coh > coh_min;
    if nnz(good) < 8
        scat = NaN;
    else
        resid = mdB(good) - medfilt1(mdB(good), 5);
        scat  = std(resid(3:end-2));      % drop the filter's edge samples
    end
end

function report(label, dev, scat, dev_all, f_ok)
    [mx, km] = max(abs(dev));
    fprintf('   %-9s %8.3f %9.1f %12.3f %9.3f %14.2f\n', ...
            label, mx, f_ok(km), median(abs(dev)), scat, max(abs(dev_all)));
end

function s = short(name)
    p = strsplit(name, '_');  s = p{1};
end

function save_fig(fig, base)
    exportgraphics(fig, [base '.png'], 'Resolution', 200);
    exportgraphics(fig, [base '.pdf'], 'ContentType', 'vector');
end

function fig = loop_figure(f, S_m,S_a, SG_m,SG_a, G_m,Gc_a, G_a, g_label, ...
                           cS,cSG,cG, ok, f_ok, dS,dSG,dG, fn, fa, F_CMP, ttl)
    fig = figure('Visible','off','Position',[50 50 1250 950]);
    tiledlayout(4, 2, 'TileSpacing','compact', 'Padding','compact');

    if isequal(Gc_a, G_a), G_extra = []; else, G_extra = G_a; end
    rows = { g_label,      G_m,  Gc_a, G_extra
             'S_{33}',     S_m,  S_a,  []
             '(GS)_{33}',  SG_m, SG_a, [] };
    for r = 1:3
        nexttile; hold on
        plot(f, 20*log10(abs(rows{r,2})), 'b', 'LineWidth', 1.2)
        plot(f, 20*log10(abs(rows{r,3})), 'r--', 'LineWidth', 1.1)
        if ~isempty(rows{r,4})
            plot(f, 20*log10(abs(rows{r,4})), 'k:', 'LineWidth', 1.0)
        end
        marks(fn, fa, F_CMP); grid on; xlim([f(1) f(end)])
        ylabel('|.| [dB]'); title(rows{r,1})
        if r == 1
            legend({'measured','analytic','analytic G_{33}'}, 'Location','best', 'FontSize', 7)
        end

        % Phase is drawn over the EXCITED bins only, and both curves are
        % unwrapped on that subset with a common branch. Unwrapping across the
        % out-of-band region, where the measured curve is leakage, walks the two
        % curves onto different branches and invents a disagreement that the
        % complex values do not have.
        nexttile; hold on
        pm = rad2deg(unwrap(angle(rows{r,2}(ok))));
        pa = rad2deg(unwrap(angle(rows{r,3}(ok))));
        pm = pm - 360*round((pm(1) - pa(1))/360);
        plot(f(ok), pm, 'b', 'LineWidth', 1.2)
        plot(f(ok), pa, 'r--', 'LineWidth', 1.1)
        marks(fn, fa, F_CMP); grid on; xlim([f(1) f(end)])
        ylabel('phase [deg]'); title([rows{r,1} '  phase (excited bins)'])
    end

    nexttile; hold on
    plot(f, cG, 'k', 'LineWidth', 1.0)
    plot(f, cS,  'b', 'LineWidth', 1.0)
    plot(f, cSG, 'r', 'LineWidth', 1.0)
    marks(fn, fa, F_CMP); grid on; xlim([f(1) f(end)]); ylim([0 1.02])
    xlabel('frequency [Hz]'); ylabel('\gamma^2'); title('coherence')
    legend({'G','f \rightarrow u','f \rightarrow y'}, ...
           'Location','southwest', 'FontSize', 7)

    nexttile; hold on
    fc = f_ok;
    plot(fc, dG,  'k', 'LineWidth', 1.0)
    plot(fc, dS,  'b', 'LineWidth', 1.0)
    plot(fc, dSG, 'r', 'LineWidth', 1.0)
    yline(0, 'k:'); grid on; xlim([f(1) f(end)])
    xlabel('frequency [Hz]'); ylabel('meas - anal [dB]'); title('deviation')

    sgtitle(ttl)
end

function fig = absorber_figure(f, Df_m,Df_a, Du_m,Du_a, Pf, Pu, fn, fa, ttl)
    fig = figure('Visible','off','Position',[80 50 1250 720]);
    tiledlayout(2, 2, 'TileSpacing','compact', 'Padding','compact');

    nexttile; hold on
    plot(f, 20*log10(abs(Du_m)), 'b', 'LineWidth', 1.2)
    plot(f, 20*log10(abs(Du_a)), 'r--', 'LineWidth', 1.1)
    marks(fn, fa, []); grid on; xlim([f(1) f(end)])
    ylabel('|\delta_a / u_{total}| [dB]'); title('absorber over plant input')
    legend({'measured','analytic'}, 'Location','best', 'FontSize', 7)

    nexttile; hold on
    plot(f, 20*log10(abs(Df_m)), 'b', 'LineWidth', 1.2)
    plot(f, 20*log10(abs(Df_a)), 'r--', 'LineWidth', 1.1)
    marks(fn, fa, []); grid on; xlim([f(1) f(end)])
    ylabel('|\delta_a / f_{sim}| [dB]'); title('absorber over exogenous force')

    nexttile([1 2]); hold on
    plot(f, 10*log10(Pf), 'Color',[0.4 0.4 0.4], 'LineWidth', 1.2)
    plot(f, 10*log10(Pu), 'm', 'LineWidth', 1.2)
    marks(fn, fa, []); grid on; xlim([f(1) f(end)])
    xlabel('frequency [Hz]'); ylabel('PSD [dB N^2/Hz]')
    title('exogenous force vs the force that reaches the plant')
    legend({'f_{sim}','u_{total} = S f_{sim}'}, 'Location','best', 'FontSize', 8)

    sgtitle(ttl)
end

function marks(fn, fa, F_CMP)
    xline(fn, 'k:',  sprintf('%.1f Hz', fn), 'LineWidth', 1.0, 'FontSize', 7);
    xline(fa, 'k-.', sprintf('%.0f Hz', fa), 'LineWidth', 0.8, 'FontSize', 7);
    if ~isempty(F_CMP)
        xline(F_CMP(1), 'Color',[0.7 0.7 0.7]); xline(F_CMP(2), 'Color',[0.7 0.7 0.7]);
    end
end
