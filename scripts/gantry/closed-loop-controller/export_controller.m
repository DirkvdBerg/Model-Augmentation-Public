function export_controller()
% EXPORT_CONTROLLER  Export MATLAB's Cfb in DOUBLE precision for the exactness test.
%
%   Writes controller_export.mat next to this file, holding everything needed to check
%   the closed-form controller of controller-in-derivation.tex against the object MATLAB
%   actually builds and simulates:
%
%     per channel   num{j}, den{j}   tf coefficients of C_j(z)
%                   kappa(j)         normalisation gain, ruleOfThumb.m:11
%                   sysjj(j)         sys(j,j) evaluated at i*2*pi*fbw, the only place the
%                                    plant enters the controller design
%     realisation   A, B, C, D       ss(Cfb), which is what lsim actually runs
%     time domain   e_test, u_test   a deterministic error signal and lsim's response
%
%   No Simulink, no plant simulation, no record is read or written. Runs in seconds.
%
%   Everything is saved in double. The generated records store u_fb, y and r_sim as
%   single (gtd_save_record.m), which caps the existing check at about 1e-9 relative;
%   this export removes that cap so the comparison is limited by arithmetic only.
%
%   Y_op = 0.10 matches V1_standstill_Yp10 (gtd_build_records.m).

    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(fileparts(here)));      % repo root
    addpath(fullfile(root, 'Matlab-scripts', 'Augmentation', 'data'));

    Y_op = 0.10;
    cfg   = gtd_config('augmentation', true, 0.10);    % same call the generator makes
    plant = gtd_build_plant(Y_op, cfg);
    Cfb   = plant.Cfb;
    ts    = cfg.ts;

    fprintf('Cfb: %s, %d states, ts = %.12g s, fbw = %g Hz\n', ...
        class(Cfb), size(Cfb.A, 1), ts, cfg.fbw);

    % ── per-channel transfer function coefficients ──────────────────────────
    Ctf = tf(Cfb);
    num = cell(3, 1);  den = cell(3, 1);
    for j = 1:3
        [nj, dj] = tfdata(Ctf(j, j), 'v');
        num{j} = nj(:).';  den{j} = dj(:).';
    end

    % ── the two scalars the design depends on, per channel ──────────────────
    % Rebuild sys exactly as gtd_build_plant.m:18-22 does, then evaluate at the bandwidth.
    m1=cfg.m1; m2=cfg.m2; mb=cfg.mb; mh=cfg.mh; Jb=cfg.Jb; Jh=cfg.Jh;
    Lb=cfg.Lb; d=cfg.d; P=cfg.P;
    M_op = [m1+m2+mb+mh,           (m1-m2)*Lb/2 - mh*Y_op,                       0;
            (m1-m2)*Lb/2 - mh*Y_op, Jb+Jh + (m1+m2)*Lb^2/4 + mh*d^2 + mh*Y_op^2, -mh*d;
            0,                      -mh*d,                                        mh];
    sys = P.' * getss(cfg.n, M_op, cfg.C_damp, cfg.K) * P;

    wb    = 2*pi*cfg.fbw;
    Fr    = freqresp(sys, wb);
    sysjj = [Fr(1,1); Fr(2,2); Fr(3,3)];

    % kappa recovered the same way ruleOfThumb.m does, for cross-checking eq. (6)
    s = tf('s');
    Cnorm = ((s+2*pi*cfg.fbw/6)/s) * ((s+2*pi*cfg.fbw/3)/(s+2*pi*cfg.fbw*3)) * ...
            (2*pi*10*cfg.fbw/(s+2*pi*10*cfg.fbw));
    cw    = freqresp(Cnorm, wb);
    kappa = zeros(3,1);
    for j = 1:3
        kappa(j) = 1/abs(sysjj(j) * cw);
    end

    % ── deterministic test signal ───────────────────────────────────────────
    % Three segments, so every pole of the controller is excited rather than only the
    % band a given record happens to occupy:
    %   0.0-0.5 s  step of 1e-4 m          drives the integrator (pole at z = 1)
    %   0.5-1.5 s  multisine 1-500 Hz      covers crossover, the absorber band and roll-off
    %   1.5-2.0 s  decaying sine at 150 Hz absorber frequency, at tracking-error amplitude
    T  = 2.0;  N = round(T/ts);  t = (0:N-1).'*ts;
    e_test = zeros(N, 3);

    i1 = t < 0.5;
    e_test(i1, :) = 1e-4;

    i2 = (t >= 0.5) & (t < 1.5);
    t2 = t(i2) - 0.5;
    fr = (1:1:500).';
    rng(0);
    ph = 2*pi*rand(numel(fr), 3);
    for c = 1:3
        raw = sum(cos(2*pi*t2*fr.' + ph(:, c).'), 2);
        e_test(i2, c) = 1e-5 * raw / std(raw);        % 10 um rms, a realistic tracking error
    end

    i3 = t >= 1.5;
    t3 = t(i3) - 1.5;
    for c = 1:3
        e_test(i3, c) = 2e-5 * exp(-t3/0.1) .* sin(2*pi*150*t3 + c);
    end

    u_test = lsim(Cfb, e_test);                        % the reference response, double

    % ── the SAME controller at the TRAINING rate ────────────────────────────
    % Every check above runs at the record rate, 20 kHz, because that is what MATLAB
    % produced. The training loop steps Cfb at cfg.ts_new = 1/4000 s and re-discretises it
    % in Python (p2_rate_compare.build_cfb_at), so until this export existed the object the
    % loop actually steps was never compared against MATLAB at all. Same construction as
    % ruleOfThumb.m, same kappa (kappa is a frequency-response scalar and does NOT depend
    % on ts), only c2d's sample time differs.
    %
    % Dc IS rate dependent: tustin maps z = inf to the finite s = 2/ts, so
    % Dc_jj = kappa_j*Cnorm(2/ts) and the 4 kHz value is ~2.83x the 20 kHz one. That factor
    % is the thing this export lets Python check rather than assert.
    ts_train = 1/4000;
    num4k = cell(3, 1);  den4k = cell(3, 1);  chan = cell(1, 3);
    for j = 1:3
        Cj = c2d(kappa(j)*Cnorm, ts_train, 'tustin');
        [nj, dj] = tfdata(Cj, 'v');
        num4k{j} = nj(:).';  den4k{j} = dj(:).';
        chan{j} = Cj;
    end
    C4ss = ss(blkdiag(chan{:}));                       % the 3-in 3-out diagonal controller
    fprintf('4 kHz: %d states, diag(D) = [%.6e %.6e %.6e] N/m\n', ...
        size(C4ss.A, 1), diag(C4ss.D));

    % ── save ────────────────────────────────────────────────────────────────
    S = struct();
    S.num = num;            S.den = den;
    S.num4k = num4k;        S.den4k = den4k;
    S.ts_train = ts_train;
    S.A4k = C4ss.A;         S.B4k = C4ss.B;
    S.C4k = C4ss.C;         S.D4k = C4ss.D;
    S.kappa = kappa;        S.sysjj = sysjj;
    S.A = Cfb.A;            S.B = Cfb.B;
    S.C = Cfb.C;            S.D = Cfb.D;
    S.e_test = e_test;      S.u_test = u_test;
    S.ts = ts;              S.fbw = cfg.fbw;      S.Y_op = Y_op;
    S.matlab_version = version;
    S.note = 'double precision; u_test = lsim(ss(Cfb), e_test) from rest';

    out = fullfile(here, 'controller_export.mat');
    save(out, '-struct', 'S');
    fprintf('wrote %s\n', out);
    fprintf('kappa = [%.10e %.10e %.10e]\n', kappa);
    fprintf('|sys_jj(i wb)| = [%.10e %.10e %.10e]\n', abs(sysjj));
    fprintf('u_test rms = [%.6e %.6e %.6e] N\n', std(u_test));
end
