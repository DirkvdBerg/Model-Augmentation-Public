%% PBH_observability_test_MSD.m
% Frozen-point PBH observability test for the extended gantry (8 states).
% Sweeps over scheduling parameters Y and delta_a, checks whether the
% hidden MSD states are observable from encoder measurements [X, Theta, Y].

clear; clc;

%% -----------------------------------------------------------------------
%  PARAMETERS (same as additional_state_lagrangian.m)
%  -----------------------------------------------------------------------
mb  = 22.8;   m1  = 10.2;   m2  = 10.7;
Jb  = 1.0;    Jh  = 0.05;
Lb  = 0.725;  d   = 0.1;
cg1 = 14.5;   cg2 = 20.3;   cy = 10;
cb1 = 9;      cb2 = 9;
kb1 = 1987.5; kb2 = 1987.5;

mh_total = 10.1;
ma  = 0.10 * mh_total;       % 1.01 kg
mh  = mh_total - ma;         % 9.09 kg (rigid part)
ka  = 500;   ca = 2;  L0 = 0.10;

%% -----------------------------------------------------------------------
%  SWEEP GRID
%  -----------------------------------------------------------------------
Y_vals      = linspace(0, 0.6, 50);
delta_a_vals = linspace(-0.03, 0.03, 21);

% Output matrix: measure encoder positions [X, Theta, Y]
C_out = [eye(3), zeros(3,5)];   % 3x8

% Storage
sigma_min_all = nan(length(Y_vals), length(delta_a_vals));

%% -----------------------------------------------------------------------
%  FROZEN-POINT PBH SWEEP
%  -----------------------------------------------------------------------
n = 8;  % state dimension

for iY = 1:length(Y_vals)
    for iD = 1:length(delta_a_vals)
        Y_op = Y_vals(iY);
        da_op = delta_a_vals(iD);

        % Build A at this operating point
        A = build_A(Y_op, da_op, m1, m2, mb, mh, Lb, Jb, Jh, d, ...
                    cg1, cg2, cb1, cb2, cy, kb1, kb2, ma, ka, ca, L0);

        % PBH test at each eigenvalue — find worst case
        lambdas = eig(A);
        sigma_worst = inf;
        for k = 1:n
            PBH = [A - lambdas(k)*eye(n); C_out];
            sigma_worst = min(sigma_worst, min(svd(PBH)));
        end
        sigma_min_all(iY, iD) = sigma_worst;
    end
end

%% -----------------------------------------------------------------------
%  RESULTS
%  -----------------------------------------------------------------------
fprintf('Min sigma across entire grid: %.4e\n', min(sigma_min_all, [], 'all'));
fprintf('Max sigma across entire grid: %.4e\n', max(sigma_min_all, [], 'all'));

if min(sigma_min_all, [], 'all') > max(size(A)) * eps(norm(A,1))
    fprintf('PASS: System is observable at all frozen operating points.\n');
else
    fprintf('WARN: Observability may be lost at some operating points.\n');
end

% Heatmap
figure;
imagesc(delta_a_vals, Y_vals, sigma_min_all);
set(gca, 'YDir', 'normal');
colorbar;
xlabel('\delta_a (m)');
ylabel('Y (m)');
title('PBH min singular value across operating range');

%% =======================================================================
%  LOCAL FUNCTION
%  =======================================================================
function A = build_A(Y, delta_a, m1, m2, mb, mh, Lb, Jb, Jh, d, ...
                     cg1, cg2, cb1, cb2, cy, kb1, kb2, ma, ka, ca, L0)
% Build the 8x8 state matrix at a frozen operating point (Y, delta_a).
% Replicates the matrices from gantrySystemExtended.m.

    M = [m1+m2+mb+mh+ma,  (m1-m2)*Lb/2-(mh+ma)*Y-ma*L0-ma*delta_a,  0,        0;
         (m1-m2)*Lb/2-(mh+ma)*Y-ma*L0-ma*delta_a, ...
             Jb+Jh+(m1+m2)*Lb^2/4+(mh+ma)*d^2+mh*Y^2+ma*(Y+L0+delta_a)^2, ...
             -(mh+ma)*d, -ma*d;
         0,  -(mh+ma)*d,  mh+ma,  ma;
         0,  -ma*d,       ma,     ma];

    C4 = [cg1+cg2,          (cg1-cg2)*Lb/2,                0,  0;
          (cg1-cg2)*Lb/2,   cb1+cb2+(cg1+cg2)*Lb^2/4,      0,  0;
          0,                0,                              cy,  0;
          0,                0,                               0, ca];

    K4 = [0, 0,       0,  0;
          0, kb1+kb2, 0,  0;
          0, 0,       0,  0;
          0, 0,       0, ka];

    A = [zeros(4),  eye(4);
         -M\K4,    -M\C4];
end
