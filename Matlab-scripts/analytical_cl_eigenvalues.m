% analytical_cl_eigenvalues.m
% Closed-loop eigenvalue verification for step response diagnostics.
%
% Constructs the identical plant and controller used in
% generate_identification_experiment.m, forms the discrete-time closed-loop,
% converts poles back to the continuous-time s-domain, and extracts τ and fres.
%
% Purpose:
%   Cross-check the nonparametric step response results.
%   If Tsettle from the step response matches 4·τ from the dominant eigenvalue,
%   the step response diagnostic is trustworthy.
%   If they disagree, the eigenvalue result is the ground truth.
%
% Note on aliasing:
%   The z → s conversion s = log(z)/ts produces aliased copies at
%   s + j·k·(2π/ts) for integer k. Poles with |Re(s)| >> 2π·fbw are
%   computational artefacts and are filtered out below.
%
% Run from repo root:
%   run('Matlab-scripts/analytical_cl_eigenvalues.m')

addpath(genpath(fullfile(pwd, 'kamtin-fp-model', '03 Simulink gantry')))

% ── Physical parameters (identical to generate_identification_experiment.m) ──
mb=22.8; mh=10.1; m1=10.2; m2=10.7; Jb=1.0; Jh=0.05;
cg1=14.5; cg2=20.3; cy=10; cb1=9; cb2=9;
kb1=1987.5; kb2=1987.5; Lb=0.725; Lh=0.25; d=0.1;

C_damp = [cg1+cg2,         (cg1-cg2)*Lb/2,             0;
          (cg1-cg2)*Lb/2,  cb1+cb2+(cg1+cg2)*Lb^2/4,   0;
          0,                0,                           cy];
K_stiff = [0,0,0; 0,kb1+kb2,0; 0,0,0];   % named K_stiff to avoid clash with ctrl gain
n   = 3;
P   = [1,1,0; Lb/2,-Lb/2,0; 0,0,1];
fs  = 20e3;  ts = 1/fs;  fbw = 100;

Y_vals      = [-0.4, 0.0, 0.4];

% ── Summary accumulators ──────────────────────────────────────────────────────
all_tau  = [];
all_fres = [];

fprintf('\n%s\n', repmat('=', 1, 72));
fprintf('Closed-loop eigenvalue analysis\n');
fprintf('Plant: modal coordinates  |  Controller: ruleOfThumb, fbw=%d rad/s (%.1f Hz)\n', fbw, fbw/(2*pi));
fprintf('%s\n', repmat('=', 1, 72));

for i = 1:numel(Y_vals)
    Y0 = Y_vals(i);

    % ── Plant at this operating point — continuous time, modal coordinates ────
    M_op = [m1+m2+mb+mh,            (m1-m2)*Lb/2-mh*Y0,                    0;
            (m1-m2)*Lb/2-mh*Y0,     Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y0^2, -mh*d;
            0,                       -mh*d,                                   mh];
    sys_ct = P.' * getss(n, M_op, C_damp, K_stiff) * P;   % continuous-time 3×3

    % ── Diagonal controller — same construction as generate_identification_experiment.m
    % ruleOfThumb takes the continuous-time modal plant and ts; returns discrete tf.
    Cfb_test = ruleOfThumb(fbw, sys_ct(1,1), ts);
    if Cfb_test.Ts ~= 0
        % ruleOfThumb returned discrete controller → discretise plant to match
        sys_for_fb = c2d(sys_ct, ts, 'zoh');
        Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)), ts);
    else
        % ruleOfThumb returned continuous controller → keep plant continuous
        sys_for_fb = sys_ct;
        Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
    end
    for j = 1:3
        Cfb(j,j) = ruleOfThumb(fbw, sys_ct(j,j), ts);
    end

    % ── Closed-loop: disturbance at plant input → modal output ────────────────
    % feedback(G, K) = G*(I+KG)^{-1}  — disturbance-to-output transfer function.
    % This is the same path the multisine and step force will excite.
    sys_cl   = feedback(sys_for_fb, Cfb);
    poles_z  = pole(sys_cl);    % discrete z-domain (or continuous s if Cfb is CT)

    % Convert discrete z-domain poles to continuous s-domain
    if sys_cl.Ts ~= 0
        % Discrete: s = log(z) / ts  (exact ZOH inverse)
        poles_s = log(poles_z) / ts;
    else
        poles_s = poles_z;
    end

    % ── Filter and sort ───────────────────────────────────────────────────────
    % Keep only poles that are:
    %   (a) stable:             Re(s) < 0
    %   (b) not integrators:    Re(s) < -1e-4
    %   (c) in principal strip: |Im(s)| < π/ts
    %       The z → s mapping produces aliased copies at s + j·k·(2π/ts).
    %       Restricting to the principal strip [-π/ts, π/ts] removes them.
    %   (d) not ZOH artifacts:  Re(s) > -20·fbw
    %       Discrete controller poles from ZOH realisation appear at large
    %       negative Re(s) >> fbw. They are not physical dynamics and inflate
    %       tau_fastest and Fs_new if left in.
    keep = real(poles_s) < -1e-4 ...
         & abs(imag(poles_s)) < pi/ts ...
         & real(poles_s) > -20*fbw;
    poles_s = poles_s(keep);

    % Sort by slowest tau first (least negative Re(s))
    [~, idx] = sort(real(poles_s), 'descend');
    poles_s  = poles_s(idx);

    % Drop conjugates: keep Im(s) >= 0
    poles_s = poles_s(imag(poles_s) >= -1e-6);

    fprintf('\n  Y0 = %+.1f m\n', Y0);
    fprintf('  %-26s  %-10s  %-10s  %-8s\n', 'Pole s [rad/s]', 'tau [s]', 'fres [Hz]', 'zeta');
    fprintf('  %s\n', repmat('-', 1, 60));

    for k = 1:numel(poles_s)
        s   = poles_s(k);
        tau = -1 / real(s);
        all_tau(end+1) = tau; %#ok<AGROW>

        if abs(imag(s)) > 1   % oscillatory: |Im(s)| > 1 rad/s
            fres = abs(imag(s)) / (2*pi);
            zeta = -real(s) / abs(s);
            all_fres(end+1) = fres; %#ok<AGROW>
            fprintf('  %+8.2f %+8.2fj         %8.5f    %8.3f    %.3f\n', ...
                    real(s), imag(s), tau, fres, zeta);
        else
            fprintf('  %+8.2f                  %8.5f    (real)    ---\n', ...
                    real(s), tau);
        end
    end
end

% ── Worst-case summary ────────────────────────────────────────────────────────
if isempty(all_tau)
    error('all_tau is empty — no poles survived the filter. Inspect poles_s at each Y0.');
end
tau_slowest = max(all_tau);
tau_fastest = min(all_tau);
if ~isempty(all_fres)
    fres_max = max(all_fres);
else
    fres_max = NaN;
    fprintf('\n  [NOTE] No oscillatory poles found — controller damps all resonances.\n');
end

fprintf('\n%s\n', repmat('=', 1, 72));
fprintf('Worst-case summary (across all Y0)\n');
fprintf('%s\n', repmat('-', 1, 72));
fprintf('  tau_slowest  = %.5f s   (→ 10·tau = %.4f s)\n', tau_slowest, 10*tau_slowest);
fprintf('  tau_fastest  = %.5f s\n', tau_fastest);
if ~isnan(fres_max)
    fprintf('  fres_max     = %.3f Hz\n', fres_max);
else
    fprintf('  fres_max     = N/A (no visible oscillation)\n');
end

% ── Candidate design values ───────────────────────────────────────────────────
% These should match the step response outputs from diagnostics_step.m.
% If they differ by more than ~30%, investigate which mode or Y0 is responsible.
T_ms_candidate   = 10 * tau_slowest;   % multisine period lower bound [s]
Df_candidate     = 1 / T_ms_candidate; % frequency resolution [Hz]
f_tau            = 1 / (2*pi*tau_fastest);
if ~isnan(fres_max)
    fmax_candidate = max(fres_max, f_tau);
else
    fmax_candidate = f_tau;
end
Fs_new_candidate = 10 * fmax_candidate;
% fbw is in rad/s — convert to Hz for the multisine band lower bound.
% Below fbw, the sensitivity S ≈ 0 and post-controller injection is suppressed.
f_low = fbw / (2*pi);   % [Hz]  THEORY: S(jω) ≈ 0 for ω < fbw (controller bandwidth)

fprintf('\n  Candidate design values (analytical ground truth):\n');
fprintf('  T_ms   = %.4f s     (multisine period — NOT the BPTT segment length)\n', T_ms_candidate);
fprintf('  Df     = %.4f Hz\n', Df_candidate);
fprintf('  Fs_new = %.1f Hz    (lower bound; round up to standard rate)\n', Fs_new_candidate);
fprintf('  f_low  = %.1f Hz    (controller bandwidth fbw=%.0f rad/s — by design)\n', f_low, fbw);
if ~isnan(fres_max)
    fprintf('  f_high = %.1f Hz    (fres_max — highest oscillatory pole)\n', fres_max);
else
    fprintf('  f_high = %.1f Hz    (1/tau_fastest — no oscillation found)\n', f_tau);
end
fprintf('%s\n', repmat('=', 1, 72));
