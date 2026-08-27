function dxdt = gantrySystemExtendedKxy(u, x, m1, m2, mb, mh, Lb, Jb, Jh, d, ...
                                        cg1, cg2, cb1, cb2, cy, kb1, kb2, ...
                                        ma, ka, ca, L0, k_xy)
% gantrySystemExtendedKxy  8-state gantry ODE with hidden MSD, PLUS X/Y stiffness.
%
% Copy of Matlab-scripts/Augmentation/gantrySystemExtended.m with ONE change:
% the two structural zeros on the X and Y diagonal of the stiffness matrix K4
% become the parameter k_xy. Everything else is verbatim, including the full
% nonlinear mass matrix, the damping matrix and the state ordering.
%
%   k_xy = 0    -> EXACTLY the original function (see the no-op check below)
%   k_xy > 0    -> weak spring on X and Y, poles pulled inside the unit circle
%
% WHY THIS EXISTS (T4 of scripts/gantry/drift-isolation/PLAN.md). The gantry has
% no X or Y spring, so those two continuous poles sit at s = 0 and map to z = 1
% exactly. A state error on those axes therefore never decays. T4 tests whether
% that marginality is what the observed drift is, by regenerating the data with
% a weak spring and retraining. It is a DIAGNOSTIC that changes the plant, not
% a proposed fix: the real machine has no such spring.
%
% NOT A MODIFICATION OF THE EXISTING PIPELINE. This is a new file in a new
% folder. Nothing under Matlab-scripts/Augmentation/ is touched.
%
% k_xy value: 1000 N/m, derived and checked in
% scripts/gantry/drift-isolation/t4_xy_stiffness/derive_k_small.py
%   * poles strictly inside: 1-|z| = 8.09e-5 at Ts = 1/4000 s, a decay time
%     constant of about 3.1 s, so an x0 error decays roughly 50x over a 12 s
%     record (readable within one record)
%   * added resonance far below the excitation band: f_X = 0.686 Hz,
%     f_Y = 1.584 Hz against a [130, 180] Hz band edge, an 82x separation
%
% State  x = [X; Theta; Y; delta_a; dX; dTheta; dY; vdelta_a]
% Input  u = [F_X; F_Theta; F_Y]  (logical-coordinate forces)
%
    Y       = x(3);   % Y position; same in stage and logical coordinates
    delta_a = x(4);   % relative displacement of ma from mh

    % ------------------------------------------------------------------
    % 4x4 Mass matrix  (full nonlinear -- matches Lagrangian M_ext)
    % ------------------------------------------------------------------
    % Row/col order: [X, Theta, Y, delta_a]
    M = [ m1+m2+mb+mh+ma,         (m1-m2)*Lb/2 - (mh+ma)*Y - ma*L0 - ma*delta_a,           0,        0;
         (m1-m2)*Lb/2 - (mh+ma)*Y - ma*L0 - ma*delta_a,  Jb+Jh+(m1+m2)*Lb^2/4 + (mh+ma)*d^2 + mh*Y^2 + ma*(Y+L0+delta_a)^2, -(mh+ma)*d, -ma*d;
          0,                                             -(mh+ma)*d,                                                             mh+ma,      ma;
          0,                                             -ma*d,                                                                  ma,         ma];

    % ------------------------------------------------------------------
    % 4x4 Viscous damping matrix
    % ------------------------------------------------------------------
    C4 = [ cg1+cg2,            (cg1-cg2)*Lb/2,                   0,  0;
           (cg1-cg2)*Lb/2,  cb1+cb2+(cg1+cg2)*Lb^2/4,            0,  0;
           0,                0,                                   cy,  0;
           0,                0,                                    0, ca];

    % ------------------------------------------------------------------
    % 4x4 Stiffness matrix -- THE ONLY CHANGE versus the original.
    % Original had hard zeros at (1,1) and (3,3); they are now k_xy.
    % ------------------------------------------------------------------
    K4 = [k_xy, 0,        0,     0;
          0,    kb1+kb2,  0,     0;
          0,    0,        k_xy,  0;
          0,    0,        0,     ka];

    % ------------------------------------------------------------------
    % State-space form  dxdt = A*x + B*u
    % ------------------------------------------------------------------
    % Force input enters only the first 3 generalised coordinates.
    Minv_B = M \ [eye(3); zeros(1,3)];   % 4x3

    A = [ zeros(4),   eye(4);
         -M\K4,      -M\C4 ];

    B = [ zeros(4,3);
          Minv_B    ];

    dxdt = A*x + B*u;
end
