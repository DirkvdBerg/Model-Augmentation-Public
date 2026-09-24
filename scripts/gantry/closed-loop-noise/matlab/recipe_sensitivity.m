function recipe_sensitivity()
% RECIPE_SENSITIVITY  How to validate closed-loop noise math with sensitivities in MATLAB (G6).
%
% Toolboxes: Control System (ss, feedback, c2d, covar, freqresp) and Signal Processing (pwelch,
% cpsd for the data side). No Robust Control (no loopsens) and no System Identification needed.
%
% The loop (Ljung and Forssell 1999, Eqs. 13/15):  y_meas = q + v,  e = r - y_meas,  u = K e.
%   S = (I + G K)^-1 = feedback(eye(3), G*K)         (output sensitivity, MIMO, stage axes)
%   noise part of the error:   e_v = -S v,        Phi_e = S Phi_v S^H
%   noise part of the current: u_v = -K S v,      true position gets -T v
% Consequences checked below, with numbers:
%   1. measured Phi_e is ALREADY shaped by S. Injecting it in a simulated loop applies S twice.
%   2. the source to inject is Phi_v = S^-1 Phi_e S^-H (MIMO; the per-axis |S_ii|^2 form is
%      not enough on this machine, G2).
%   3. covar gives the exact steady-state error covariance of the loop under a shaped noise, so
%      it closes the loop analytically; lsim + pwelch closes it by simulation.
%
% Inputs: outputs/recipe_inputs/telica_loop.mat (closure/export_recipe_inputs.py): identified
% Telica controller and plant per position, the source model H_v (VAR 512), and a VAR fit of the
% MEASURED error spectrum (to show the S-twice effect). Output: outputs/recipe/recipe.mat and a
% printed table. Also exports S_sim of the simulated (Garcia + ruleOfThumb) loop for G5(c).
    here = fileparts(mfilename('fullpath'));
    cn   = fileparts(here);
    outd = fullfile(cn, 'outputs', 'recipe');
    if ~exist(outd, 'dir'), mkdir(outd); end
    L  = load(fullfile(cn, 'outputs', 'recipe_inputs', 'telica_loop.mat'));
    Ts = L.Ts;

    % ==== the noise models as state-space systems (v = Hv w, w unit white per sample) =========
    Hv = var2ss(L.A_tanh,  L.Sig_tanh,  Ts);    % the SOURCE (de-shaped, G3)
    He = var2ss(L.A_stuck, L.Sig_stuck, Ts);    % a fit of the MEASURED error spectrum Phi_e
    fprintf('H_v: %d states, H_e: %d states\n', order(Hv), order(He));

    % ==== Telica loop: identified controller + identified plant (tanh reading), 15 positions ==
    K = ss(L.Ac, L.Bc, L.Cc, L.Dc, Ts);          % e [m] -> F [N]
    P = numel(L.w);
    var_e = zeros(1, 3);  var_twice = zeros(1, 3);  var_src = diag(covar(Hv, eye(3)))';
    Smax = zeros(P, 3);
    for p = 1:P
        G = ss(squeeze(L.Ap_tanh(p, :, :)), squeeze(L.Bp_tanh(p, :, :)), ...
               squeeze(L.Cp_tanh(p, :, :)), zeros(3), Ts);
        S = feedback(eye(3), G*K);               % (I + G K)^-1
        assert(isstable(S), 'closed loop unstable at position %d', p);
        var_e     = var_e     + L.w(p) * diag(covar(S*Hv, eye(3)))';   % inject the SOURCE
        var_twice = var_twice + L.w(p) * diag(covar(S*He, eye(3)))';   % inject the MEASURED Phi_e
        [mag, ~] = bode(S, 2*pi*logspace(0, log10(0.49/Ts), 400));
        for j = 1:3, Smax(p, j) = max(squeeze(mag(j, j, :))); end
    end
    rms_e = 1e9 * sqrt(var_e);  rms_twice = 1e9 * sqrt(var_twice);  rms_src = 1e9 * sqrt(var_src);
    fprintf('\nTelica loop (covar, position mix of the measured pool)          X1      X2      Y  [nm rms]\n');
    fprintf('  measured standstill error                                  %7.2f %7.2f %7.2f\n', L.rms_meas_nm);
    fprintf('  injected source v (H_v, de-shaped)                         %7.2f %7.2f %7.2f\n', rms_src);
    fprintf('  error with v injected at the encoder node  S Phi_v S^H     %7.2f %7.2f %7.2f\n', rms_e);
    fprintf('  error if the MEASURED Phi_e were injected  S Phi_e S^H     %7.2f %7.2f %7.2f\n', rms_twice);
    fprintf('  peak |S_ii| over positions                                 %7.2f %7.2f %7.2f\n', max(Smax));

    % ==== simulated loop (Garcia plant + hidden absorber, ruleOfThumb Cfb): S_sim for G5(c) ====
    addpath(fullfile(here, 'gen'));
    cfg = gtd_config('augmentation', true, 0.50);
    cfg.zeta_a = 0.03;  cfg.ca = 2*cfg.zeta_a*sqrt(cfg.ka*cfg.ma);   % the production knobs
    ids  = {'T1_standstill_Ym30','T2_standstill_Ym15','T3_standstill_Y000','T4_standstill_Yp15', ...
            'T5_standstill_Yp30','V1_standstill_Yp10','E1_resonance_sweep','E2_multisine_Yp22'};
    Yops = [-0.30, -0.15, 0.00, 0.15, 0.30, 0.10, 0.00, 0.22];
    SS = struct('id', {}, 'Y_op', {}, 'A', {}, 'B', {}, 'C', {}, 'D', {}, 'var_e', {}, ...
                'KA', {}, 'KB', {}, 'KC', {}, 'KD', {});
    fprintf('\nSimulated loop (frictionless 8-state linearisation, c2d zoh, Cfb)   X1      X2      Y  [nm rms]\n');
    for k = 1:numel(ids)
        plant = gtd_build_plant(Yops(k), cfg);
        Gd = c2d(linearise_ode(cfg, Yops(k)), Ts, 'zoh');
        Ssim = feedback(eye(3), Gd*plant.Cfb);
        assert(isstable(Ssim), 'simulated loop unstable at %s', ids{k});
        ve = diag(covar(Ssim*Hv, eye(3)))';
        SS(k) = struct('id', ids{k}, 'Y_op', Yops(k), 'A', Ssim.A, 'B', Ssim.B, 'C', Ssim.C, ...
                       'D', Ssim.D, 'var_e', ve, 'KA', plant.Cfb.A, 'KB', plant.Cfb.B, ...
                       'KC', plant.Cfb.C, 'KD', plant.Cfb.D);
        fprintf('  %-22s S_sim Phi_v S_sim^H                  %7.2f %7.2f %7.2f\n', ids{k}, 1e9*sqrt(ve));
    end
    % controllers of the Telica-profile records (G5(d) integrator check)
    recs = gtd_build_records_telica(cfg);
    TP = struct('id', {}, 'Y_op', {}, 'KA', {}, 'KB', {}, 'KC', {}, 'KD', {});
    for k = 1:numel(recs)
        plant = gtd_build_plant(recs(k).Y_op, cfg);
        TP(k) = struct('id', recs(k).id, 'Y_op', recs(k).Y_op, 'KA', plant.Cfb.A, 'KB', plant.Cfb.B, ...
                       'KC', plant.Cfb.C, 'KD', plant.Cfb.D);
    end
    save(fullfile(outd, 'recipe.mat'), 'rms_e', 'rms_twice', 'rms_src', 'Smax', 'SS', 'TP');
    fprintf('saved %s\n', fullfile(outd, 'recipe.mat'));
end

function H = var2ss(A, Sig, Ts)
% VAR(p) v_t = sum_i A_i v_{t-i} + L w_t as ss: state [v_{t-1}; ...; v_{t-p}], output v_t.
    p = size(A, 1);  m = size(A, 2);
    Ab = zeros(m, m*p);
    for i = 1:p, Ab(:, (i-1)*m+1:i*m) = squeeze(A(i, :, :)); end
    Lc = chol(Sig, 'lower');
    Acomp = [Ab; eye(m*(p-1)), zeros(m*(p-1), m)];
    H = ss(Acomp, [Lc; zeros(m*(p-1), m)], Ab, Lc, Ts);
end

function G = linearise_ode(cfg, Y_op)
% Frictionless (cc = 0) linearisation of the vendored gantrySystemExtendedCoulomb at rest at Y_op,
% by central differences on the ODE the simulation integrates. Input: stage force (Gain3 = P maps
% it to logical); output: stage position (Gain4 = P.').
    x0 = [0; 0; Y_op; 0; 0; 0; 0; 0];  u0 = zeros(3, 1);
    f = @(x, u) gantrySystemExtendedCoulomb(u, x, cfg.m1, cfg.m2, cfg.mb, cfg.mh_rigid, cfg.Lb, ...
            cfg.Jb, cfg.Jh, cfg.d, cfg.cg1, cfg.cg2, cfg.cb1, cfg.cb2, cfg.cy, cfg.kb1, cfg.kb2, ...
            cfg.ma, cfg.ka, cfg.ca, cfg.L0, 0, 0, 0, cfg.ts);
    h = 1e-7;
    A = zeros(8);  B = zeros(8, 3);
    for i = 1:8, e = zeros(8, 1); e(i) = h; A(:, i) = (f(x0 + e, u0) - f(x0 - e, u0)) / (2*h); end
    for i = 1:3, e = zeros(3, 1); e(i) = h; B(:, i) = (f(x0, u0 + e) - f(x0, u0 - e)) / (2*h); end
    G = ss(A, B*cfg.P, [cfg.P.' zeros(3, 5)], zeros(3));
end
