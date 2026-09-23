% make_ref_g2  MATLAB references for gate G2 (TR-007). Run with the watchdog:
%   matlab -sd telica-real/friction/matlab -batch make_ref_g2
% Reads  telica-real/outputs/g2_friction/ref_inputs.mat   (written by friction/check_friction.py prep)
% Writes telica-real/outputs/g2_friction/ref_outputs.mat
%   R2: gantry6Coulomb (6-state copy, friction verbatim)           -> y_R2, held_R2, stuck0_R2
%   R1: gantrySystemExtendedCoulomb.m (UNMODIFIED, read from its own folder) at two absorber
%       masses ma1, ma2                                             -> x8_ma1, x8_ma2
% Same fixed-step RK4 with zero-order-hold input as the torch block (Gantry_State_Block._rk4).

here = fileparts(mfilename('fullpath'));
tr   = fileparts(fileparts(here));                 % telica-real
repo = fileparts(tr);
addpath(fullfile(repo, 'Matlab-scripts', 'Augmentation-coulomb'));   % read-only use
S = load(fullfile(tr, 'outputs', 'g2_friction', 'ref_inputs.mat'));

p = S.p;  h = S.h;  U = S.u_stage;  N = size(U, 1);
Lb = p.Lb;
P = [1, 1, 0; Lb/2, -Lb/2, 0; 0, 0, 1];
fprintf('[make_ref_g2] N = %d steps, h = %g s, cc = [%g %g %g]\n', N, h, p.cc1, p.cc2, p.ccy);

% ================ R2: 6-state copy ================
f6 = @(u, x) gantry6Coulomb(u, x, p.m1, p.m2, p.mb, p.mh, Lb, p.Jb, p.Jh, p.d, ...
                           p.cg1, p.cg2, p.cb1, p.cb2, p.cy, p.kb1, p.kb2, p.cc1, p.cc2, p.ccy);
x = S.x0(:);
x_R2 = zeros(N, 6);  held_R2 = zeros(N, 3);  stuck0_R2 = zeros(N, 3);
tic
for k = 1:N
    x_R2(k, :) = x.';
    u = P * U(k, :).';
    [d1, hk, s0] = f6(u, x);
    held_R2(k, :) = hk.';  stuck0_R2(k, :) = s0.';
    k1 = h * d1;
    k2 = h * f6(u, x + k1/2);
    k3 = h * f6(u, x + k2/2);
    k4 = h * f6(u, x + k3);
    x = x + (k1 + 2*k2 + 2*k3 + k4) / 6;
end
fprintf('[make_ref_g2] R2 done in %.1f s\n', toc);

% ================ R1: the original file at two absorber masses ================
mas = [S.ma1, S.ma2];
x8 = cell(1, 2);
for j = 1:2
    ma = mas(j);
    ka = ma * (2*pi*S.f_abs)^2;               % absorber at f_abs Hz
    ca = 2 * S.zeta_abs * sqrt(ka * ma);
    f8 = @(u, x) gantrySystemExtendedCoulomb(u, x, p.m1, p.m2, p.mb, p.mh, Lb, p.Jb, p.Jh, ...
                    p.d, p.cg1, p.cg2, p.cb1, p.cb2, p.cy, p.kb1, p.kb2, ma, ka, ca, 0.0, ...
                    p.cc1, p.cc2, p.ccy, h);
    x = [S.x0(1:3); 0; S.x0(4:6); 0];
    X8 = zeros(N, 8);
    tic
    for k = 1:N
        X8(k, :) = x.';
        u = P * U(k, :).';
        k1 = h * f8(u, x);
        k2 = h * f8(u, x + k1/2);
        k3 = h * f8(u, x + k2/2);
        k4 = h * f8(u, x + k3);
        x = x + (k1 + 2*k2 + 2*k3 + k4) / 6;
    end
    x8{j} = X8;
    fprintf('[make_ref_g2] R1 ma = %g done in %.1f s\n', ma, toc);
end
x8_ma1 = x8{1};  x8_ma2 = x8{2};
save(fullfile(tr, 'outputs', 'g2_friction', 'ref_outputs.mat'), ...
     'x_R2', 'held_R2', 'stuck0_R2', 'x8_ma1', 'x8_ma2', '-v7');
fprintf('[make_ref_g2] saved ref_outputs.mat (%s)\n', version);
