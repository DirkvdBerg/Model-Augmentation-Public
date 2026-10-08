function dxdt = gantrySystemExtendedTanh(u, x, m1, m2, mb, mh, Lb, Jb, Jh, d, ...
                                     cg1, cg2, cb1, cb2, cy, kb1, kb2, ...
                                     ma, ka, ca, L0, cc1, cc2, ccy, g)
% gantrySystemExtendedTanh  8-state gantry ODE + hidden MSD + tanh-smoothed Coulomb friction.
%
% Copy of Thesis-writeup/Code/Data/vendor/gantrySystemExtendedCoulomb.m (D-223). The mass,
% damping, stiffness and state-space sections are copied VERBATIM (lines 95-132 of the vendored
% file, pasted by script; in the comments one dash became a comma and separator lines use '=').
% The ONLY change is the friction law, and the last argument, which is the tanh gain g [s/m]
% instead of the unused ts:
%
%   Karnopp (vendored):  stuck if |v_i| < V_BRK, stuck force solved, else cc_i*sign(v_i)
%   this file:           F_i = cc_i * tanh(g * v_i)          on each STAGE rail [X1, X2, Y]
%
% THEORY: ASMPT supervisor 2026-09-28, "F_i = cc_i tanh(100 v_i), play with the gain 100, higher
% is more accurate"; the same smoothed Coulomb law as D-116 (tanh(v/v0), v0 = 1/g). For
% |v_i| >> 1/g it equals Garcia's cc_i*sign(v_i) (garcia2013 Fig. 2, cc identified sliding).
% For |v_i| << 1/g it is a viscous damper of slope cc_i*g, which is what removes the set-valued
% stick and with it the standstill limit cycle; D-209's objection (Coulomb force in the damping
% matrix) is accepted for this experiment and stated in D-223.
%
% Frame exactly as the vendored file: friction acts on the rails, v_stage = P.' * qdot_logical,
% generalized force P * F_stage. The absorber DOF carries no friction.
%
% State  x = [X; Theta; Y; delta_a; dX; dTheta; dY; vdelta_a]
% Input  u = [F_X; F_Theta; F_Y]  (logical-coordinate forces)
    Y       = x(3);   % Y position; same in stage and logical coordinates
    delta_a = x(4);   % relative displacement of ma from mh

    % ==================================================================
    % 4x4 Mass matrix  (full nonlinear, matches Lagrangian M_ext)
    % ==================================================================
    % Row/col order: [X, Theta, Y, delta_a]
    M = [ m1+m2+mb+mh+ma,         (m1-m2)*Lb/2 - (mh+ma)*Y - ma*L0 - ma*delta_a,           0,        0;
         (m1-m2)*Lb/2 - (mh+ma)*Y - ma*L0 - ma*delta_a,  Jb+Jh+(m1+m2)*Lb^2/4 + (mh+ma)*d^2 + mh*Y^2 + ma*(Y+L0+delta_a)^2, -(mh+ma)*d, -ma*d;
          0,                                             -(mh+ma)*d,                                                             mh+ma,      ma;
          0,                                             -ma*d,                                                                  ma,         ma];

    % ==================================================================
    % 4x4 Viscous damping matrix
    % ==================================================================
    C4 = [ cg1+cg2,            (cg1-cg2)*Lb/2,                   0,  0;
           (cg1-cg2)*Lb/2,  cb1+cb2+(cg1+cg2)*Lb^2/4,            0,  0;
           0,                0,                                   cy,  0;
           0,                0,                                    0, ca];

    % ==================================================================
    % 4x4 Stiffness matrix
    % ==================================================================
    K4 = [0,  0,        0,  0;
          0,  kb1+kb2,  0,  0;
          0,  0,        0,  0;
          0,  0,        0, ka];

    % ==================================================================
    % State-space form  (identical to the original file)
    % ==================================================================
    Minv_B = M \ [eye(3); zeros(1,3)];   % 4x3

    A = [ zeros(4),   eye(4);
         -M\K4,      -M\C4 ];

    B = [ zeros(4,3);
          Minv_B    ];

    % ==================================================================
    % Coulomb friction, tanh-smoothed (the only change against the vendored file)
    % ==================================================================
    P  = [1,     1,      0;
          Lb/2, -Lb/2,   0;
          0,     0,      1];
    cc = [cc1; cc2; ccy];

    v_stage = P.' * x(5:7);                % 3x1  [dX1; dX2; dY]
    F       = cc .* tanh(g * v_stage);     % THEORY: see header (ASMPT rule, D-116 form)

    % ==================================================================
    % dxdt = A*x + B*u_eff   (u_eff = u - P*F_stage), as in the vendored file
    % ==================================================================
    dxdt = A*x + B*(u - P*F);
end

