function dxdt = gantrySystemOA(u, x, m1, m2, mb, mh, Lb, Jb, Jh, d, cg1, cg2, cb1, cb2, cy, kb1, kb2, ...
                               oa_Ma0, oa_Ma1, oa_Ma2, oa_Ca0, oa_Ka0, oa_Ka1, oa_Ka2, ...
                               oa_m, oa_w, oa_b0, oa_b1)
% gantrySystemOA  Baseline gantry ODE PLUS the constructed orthogonal addition (OA-003).
%
% Copy of kamtin-fp-model/03 Simulink gantry/functions/gantrySystem.m. The baseline lines are
% verbatim (mass, damping and stiffness matrices, backslash for A, pinv for B). The ONLY change
% is the addition, which is zero when every oa_* argument is zero (check_oa_noop.m proves the
% result is then bit-identical to gantrySystem.m).
%
% The addition (derivation: ../DERIVATION.md, decision OA-003):
%   stateless   M_a(Y) = oa_Ma0 + Y*oa_Ma1 + Y^2*oa_Ma2   (oa_Ma2 added by OA-018)
%               C_a    = oa_Ca0
%               K_a(Y) = oa_Ka0 + Y*oa_Ka1 + Y^2*oa_Ka2
%   absorbers   N = numel(oa_m) lossless tuned masses, direction b_i(Y) = oa_b0(:,i) + Y*oa_b1(:,i),
%               natural frequency oa_w(i) [rad/s], mass oa_m(i) [kg]. MASS-CONSERVING: the
%               absorber masses are part of the baseline rigid mass (as mh_rigid + ma = mh in
%               gantrySystemExtended.m), so only their motion RELATIVE to the carrier is added.
%   Lagrangian result (frozen b, no Coriolis, the convention gantrySystem uses for M(Y)):
%     (M + M_a - SUM m_i b_i b_i') qdd = u - (C + C_a) qd - (K + K_a) q + SUM m_i w_i^2 d_i b_i
%     d_i''                            = -b_i' qdd - w_i^2 d_i
%   so relative to the baseline M qdd + C qd + K q = u the added generalised force is
%     f_a = -M_a qdd - C_a qd - K_a q - SUM m_i b_i d_i''.
%
% State  x = [X; Theta; Y; dX; dTheta; dY; d_1..d_N; dd_1..dd_N]
% Input  u = [F_X; F_Theta; F_Y]  (logical-coordinate forces)
%
% An unused absorber slot has oa_m(i) = 0 and oa_b0(:,i) = oa_b1(:,i) = 0; its two states then
% have derivative exactly zero and stay at their zero initial condition.

    Y = x(3); % This works because Y is defined the same in stage & logical coordinates.

    % Mass Matrix (verbatim)
    M = [          m1 + m2 + mb + mh,                          (m1 - m2) * Lb / 2 - mh * Y,        0;
         (m1 - m2) * Lb / 2 - mh * Y, Jb + Jh + (m1 + m2) * Lb^2 / 4 + mh * d^2 + mh * Y^2,  -mh * d;
                                   0,                                              -mh * d,       mh];

    % Viscous Damping Matrix (verbatim)
    C = [           cg1 + cg2,               (cg1 - cg2) * Lb / 2,  0;
         (cg1 - cg2) * Lb / 2, cb1 + cb2 + (cg1 + cg2) * Lb^2 / 4,  0;
                            0,                                  0, cy];

    % Stiffness Matrix (verbatim)
    K = [0,         0, 0;
         0, kb1 + kb2, 0;
         0,         0, 0];

    % ==== the addition
    N  = numel(oa_m);
    Mr = M + (oa_Ma0 + Y * oa_Ma1 + Y^2 * oa_Ma2);
    for i = 1:N
        bi = oa_b0(:, i) + Y * oa_b1(:, i);
        Mr = Mr - oa_m(i) * (bi * bi.');
    end
    Ct = C + oa_Ca0;
    Kt = K + (oa_Ka0 + Y * oa_Ka1 + Y^2 * oa_Ka2);

    fabs = zeros(3, 1);
    for i = 1:N
        bi = oa_b0(:, i) + Y * oa_b1(:, i);
        fabs = fabs + (oa_m(i) * oa_w(i)^2 * x(6 + i)) * bi;
    end

    % Baseline structure, with the carrier's matrices in place of M, C, K.
    A = [zeros(3)   eye(3);
         -Mr\Kt    -Mr\Ct];
    B = [zeros(3); pinv(Mr)];

    dxq  = A*x(1:6) + B*u + [zeros(3,1); pinv(Mr) * fabs];
    qdd  = dxq(4:6);

    dda = zeros(N, 1);
    for i = 1:N
        bi = oa_b0(:, i) + Y * oa_b1(:, i);
        dda(i) = -(bi.' * qdd) - oa_w(i)^2 * x(6 + i);
    end

    dxdt = [dxq; x(6 + N + (1:N)); dda];
end
