function dxdt = gantrySystemCubic(u,x,m1,m2,mb,mh,Lb,Jb,Jh,d,cg1,cg2,cb1,cb2,cy,kb1,kb2,k3)
% gantrySystemCubic  6-state baseline gantry ODE PLUS a cubic spring on X and Y.
%
% Copy of kamtin-fp-model/03 Simulink gantry/functions/gantrySystem.m with ONE
% addition: a cubic restoring force on the two translational coordinates whose
% stiffness is a structural zero. Everything else is verbatim, including the
% nonlinear mass matrix, the damping matrix, the Theta spring and the use of
% pinv rather than backslash for B.
%
%   k3 = 0   -> EXACTLY the original function (see check_cubic_noop.m)
%   k3 > 0   -> hardening spring, F = -k3*q^3, on X and Y
%
% WHY IT IS NOT A STIFFNESS-MATRIX ENTRY. T4 could put its linear k_xy straight
% into K because a linear spring is linear in the state. This file computes
% dxdt = A*x + B*u with A = [0 I; -M\K, -M\C], and a cubic term cannot be
% represented there. It enters as an added generalised force instead.
%
% THE THETA SPRING IS UNTOUCHED. K(2,2) = kb1 + kb2 is real physics on the
% actual machine. The cubic occupies the X and Y slots, which are currently
% structural zeros, i.e. the same physical slot T4 filled with linear k_xy.
%
% WHY THIS EXISTS (D-135, scripts/gantry/discrepancy-ladder/PLAN.md). The
% current discrepancy, a hidden 150 Hz absorber, is about 1e-9 of the sim-RMS
% metric while the untrained baseline already scores 1.66e-4 m, so the ANN's
% best possible gain is near zero and training has a strictly negative expected
% value. Every completed run selected epoch 0. This replaces that discrepancy
% with a STATIC one that is large in the metric and visible inside a 400-sample
% training window, and then asks whether the network learns it.
%
% TRUTH ONLY. The Python baseline (model_augmentation/systems/gantry_ss.py,
% Gantry_State_Block) does NOT get this spring, and must not. The mismatch IS
% the learning target. This is the one place the design differs from T4 on
% purpose: T4 put its spring in both truth and model so they agreed, because it
% was asking a question about poles. Putting it in both here would leave
% nothing to learn.
%
% NOT A MODIFICATION OF ANY EXISTING PIPELINE. New file in a new folder.
% Nothing under Matlab-scripts/Augmentation/ is touched, and kamtin-fp-model/
% is read only and is only ever copied from.
%
% k3 value: see derive_k3.m. It is sized by the FREE-RUN DEGRADATION it induces,
% not by excitation preservation, because the spring is truth-only so the whole
% spring force is a mismatch force acting on an integrating axis. On the Y axis
% mh = 10.1 kg and cy = 10 Ns/m give a mismatch-force-to-sim-RMS gain of about
% 0.64 m per newton over a 12 s record. Candidate range 1 to 5 N/m^3.
%
% State  x = [X; Theta; Y; dX; dTheta; dY]
% Input  u = [F_X; F_Theta; F_Y]  (logical-coordinate forces)
%
    X = x(1); % cubic spring acts on X, whose stiffness is a structural zero
    Y = x(3); % This works because Y is defined the same in stage & logical coordinates.

    % Mass Matrix
    M = [          m1 + m2 + mb + mh,                          (m1 - m2) * Lb / 2 - mh * Y,        0;
         (m1 - m2) * Lb / 2 - mh * Y, Jb + Jh + (m1 + m2) * Lb^2 / 4 + mh * d^2 + mh * Y^2,  -mh * d;
                                   0,                                              -mh * d,       mh];

    % Viscous Damping Matrix
    C = [           cg1 + cg2,               (cg1 - cg2) * Lb / 2,  0;
         (cg1 - cg2) * Lb / 2, cb1 + cb2 + (cg1 + cg2) * Lb^2 / 4,  0;
                            0,                                  0, cy];

    % Stiffness Matrix -- UNCHANGED. X and Y stay zero here on purpose; the
    % cubic cannot live in a matrix that multiplies x linearly.
    K = [0,         0, 0;
         0, kb1 + kb2, 0;
         0,         0, 0];

    A = [zeros(3)   eye(3);
         -M\K    -M\C];
    B = [zeros(3); pinv(M)];

    C = [eye(3), zeros(3)];

    % ------------------------------------------------------------------
    % THE ONLY CHANGE versus the original: cubic hardening spring to ground
    % on the two zero-stiffness axes. Vanishes identically at k3 = 0 and at
    % the origin, and is restoring for k3 > 0.
    % pinv, not backslash, to match how B is built two lines above.
    % ------------------------------------------------------------------
    f_nl = [-k3 * X^3;
                     0;
            -k3 * Y^3];

    dxdt = A*x + B*u + [zeros(3,1); pinv(M) * f_nl];
end
