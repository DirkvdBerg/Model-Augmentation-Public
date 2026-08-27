function dxdt = gantrySystemExtendedCoulomb(u, x, m1, m2, mb, mh, Lb, Jb, Jh, d, ...
                                        cg1, cg2, cb1, cb2, cy, kb1, kb2, ...
                                        ma, ka, ca, L0, cc1, cc2, ccy, ts)
% gantrySystemExtendedCoulomb  8-state gantry ODE + hidden MSD + Coulomb friction
%                              with a set-valued (Karnopp) stick state.
%
% Copy of Matlab-scripts/Augmentation/gantrySystemExtended.m. The ONLY change is
% that dry friction is subtracted from the applied force before the (unchanged)
% state-space assembly:  dxdt = A*x + B*(u - P*F_stage).
%
% THE FRICTION LAW
% ----------------
% THEORY: garcia2013 (garcia2013_gantry-decoupling-control.pdf, Fig. 2) models
% dry friction as one force per PHYSICAL actuator, cc1*sign(dX1), cc2*sign(dX2),
% ccy*sign(dY), identified as cc1 = 16.8 N, cc2 = 18.35 N, ccy = 11.6 N by
% displacing each axis at CONSTANT VELOCITY. Those are sliding experiments, so
% the paper measures the SLIP force and never the value at rest.
%
% Coulomb's law is set-valued at zero velocity. It is NOT "F = 0 when v = 0":
%
%   |v_i| > 0                          F_i = cc_i*sign(v_i)        (slip)
%   |v_i| = 0 and |F_applied,i| <= cc_i  F_i = F_applied,i         (stick)
%   |v_i| = 0 and |F_applied,i| >  cc_i  breaks away, slips        (slip)
%
% The slip branch is EXACTLY Garcia and is unchanged from the previous version of
% this file. Only the middle line is added, and it introduces NO new physical
% parameter: classical Coulomb friction has a single coefficient per contact and
% is already set-valued at rest. The earlier hard-sign version used sign(0) = 0,
% which asserts a stage at rest has zero friction and can be moved by an
% arbitrarily small force. That is not an approximation of a rail, it is the
% opposite of one.
%
% WHY IT WAS CHANGED (measured, not preference; see D-138)
% -------------------------------------------------------
% With sign(0) = 0 the vector field is discontinuous: crossing v = 0 flips the
% force by 2*cc ~ 35 N. Measured consequence, perturbing dX by 1e-12 m/s and
% integrating 3 s (scripts/gantry/coulomb-offset/diag_sign_floor.py,
% diag_karnopp.py):
%
%   hard sign   perturbation gain  X 1.07e+06   Theta 2.38e+06   Y 4.59e+06
%   Karnopp     perturbation gain  X 1.47e+00   Theta 1.44e-03   Y 1.53e-03
%   frictionless (reference)       X 1.43e+00
%
% Hard sign amplified round-off by a factor of a million, which put a ~1e-6 m
% floor under every open-loop replay measurement on this dataset. The stick state
% removes it and restores the sensitivity of the frictionless system. The result
% moves by 4% when V_EPS is swept over two decades, so it does not rest on the
% threshold.
%
% Formally: with hard sign the equation is a differential inclusion, not an ODE,
% and its solution on the switching surface is defined in the Filippov sense. The
% Filippov solution IS the stick solution, so this is not a departure from Garcia
% but the correct solution of the law Garcia wrote down.
%
% SOLVING FOR THE STUCK FORCES
% ----------------------------
% A stuck rail's friction is whatever holds its acceleration at zero, so it is
% SOLVED, not evaluated. In stage coordinates, with G mapping stage friction to
% stage acceleration:  a_stage = a_free_stage - G*F_stage.  Setting a_stage = 0
% on the stuck rails and F_i = cc_i*sign(v_i) on the sliding ones gives one 3x3
% system. If a solved force exceeds its rail's cc the rail cannot hold it, so it
% is moved to the sliding set and the system is re-solved. With three rails that
% active-set loop is exact, not an approximation.
%
% Written with fixed-size arrays and explicit loops (no logical indexing) so it
% stays codegen-compatible inside the Simulink chart.
%
% FRAME, and the one way to get it wrong. Friction acts on the physical RAILS,
% not the logical coordinates. The state is logical [X; Theta; Y; delta_a; ...],
% so friction is built in STAGE coordinates and projected back with the same P
% the rest of the pipeline uses:
%   P            = [1, 1, 0; Lb/2, -Lb/2, 0; 0, 0, 1]
%   P.' * qdot   maps LOGICAL velocity -> STAGE velocity [dX1; dX2; dY]
%   P   * F      maps STAGE force       -> LOGICAL generalized force
% Applying cc directly to [dX; dTheta; dY] is wrong and looks plausible.
%
% NOTE on delta_a: the absorber is an internal payload DOF, not a rail, so it
% carries no friction. F_stage is 3x1 and enters only the first three
% generalized coordinates, exactly where u already enters.
%
% Extra parameters beyond the original 19:
%   cc1, cc2, ccy  Coulomb friction of X1, X2 and the Y payload [N]
%   ts             integrator step [s], used ONLY to size the stick band V_EPS
%
% With cc1 = cc2 = ccy = 0 this function reproduces gantrySystemExtended EXACTLY:
% V_EPS is then 0, no rail can be stuck, F is identically zero and u_eff = u.
% check_coulomb_noop.m is the gate.
%
% State  x = [X; Theta; Y; delta_a; dX; dTheta; dY; vdelta_a]
% Input  u = [F_X; F_Theta; F_Y]  (logical-coordinate forces)
%
    Y       = x(3);   % Y position; same in stage and logical coordinates
    delta_a = x(4);   % relative displacement of ma from mh

    % ------------------------------------------------------------------
    % 4x4 Mass matrix  (full nonlinear — matches Lagrangian M_ext)
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
    % 4x4 Stiffness matrix
    % ------------------------------------------------------------------
    K4 = [0,  0,        0,  0;
          0,  kb1+kb2,  0,  0;
          0,  0,        0,  0;
          0,  0,        0, ka];

    % ------------------------------------------------------------------
    % State-space form  (identical to the original file)
    % ------------------------------------------------------------------
    Minv_B = M \ [eye(3); zeros(1,3)];   % 4x3

    A = [ zeros(4),   eye(4);
         -M\K4,      -M\C4 ];

    B = [ zeros(4,3);
          Minv_B    ];

    % ------------------------------------------------------------------
    % Coulomb friction, set-valued at v = 0  (the only addition)
    % ------------------------------------------------------------------
    P  = [1,     1,      0;
          Lb/2, -Lb/2,   0;
          0,     0,      1];
    cc = [cc1; cc2; ccy];

    % HEURISTIC: V_EPS = (cc1+cc2)/m_total * ts, the velocity dry friction can
    % remove in ONE integrator step. Below it a rail would be arrested within the
    % step regardless, so treating it as stuck matches the integrator's own
    % resolution. It is a numerical detection band, not a physical parameter, and
    % the perturbation gain moves 4% when it is swept over two decades
    % (diag_karnopp.py: 1.448 / 1.472 / 1.510 at V_EPS/10, V_EPS, V_EPS*10).
    % At cc = 0 this is exactly 0, which is what makes the no-op gate hold.
    m_total = m1 + m2 + mb + mh + ma;
    v_eps   = (cc1 + cc2) / m_total * ts;

    dx_free      = A*x + B*u;              % frictionless derivative
    a_free       = dx_free(5:8);
    MP           = Minv_B * P;             % 4x3: stage friction -> logical accel
    G            = P.' * MP(1:3, :);       % 3x3: stage friction -> stage accel
    a_free_stage = P.' * a_free(1:3);      % 3x1
    v_stage      = P.' * x(5:7);           % 3x1  [dX1; dX2; dY]

    stuck = false(3,1);
    for i = 1:3
        stuck(i) = abs(v_stage(i)) < v_eps;
    end

    % Slip force ASSIGNED to each rail. For a rail that is sliding from the
    % start this is Garcia's cc*sign(v). For a rail that BREAKS AWAY it is
    % cc*sign(F_required), the sign of the force it could not hold, and NOT
    % cc*sign(v): a stuck rail has |v| < V_EPS, so sign(v) there is set by a
    % near-zero velocity and is arbitrary, and sign(0) = 0 would remove the
    % friction entirely at the exact moment it should saturate. Getting this
    % wrong is invisible to a "does it accelerate" test, because zero friction
    % also accelerates; gate A6 now compares against the frictionless
    % acceleration to catch it.
    Fassign = zeros(3,1);
    for i = 1:3
        if ~stuck(i)
            % THEORY: garcia2013 Fig. 2 -- slip force cc_i*sign(v_i)
            Fassign(i) = cc(i) * sign(v_stage(i));
        end
    end

    F = zeros(3,1);
    for iter = 1:4                          % active set: 3 rails, converges fast
        % One fixed-size 3x3 system. Stuck rails get the zero-acceleration row
        % from G; sliding rails get an identity row pinning them to their
        % assigned slip force. Building it this way avoids variable-size
        % indexing, which keeps the chart codegen-compatible.
        Amat = zeros(3,3);
        bvec = zeros(3,1);
        for i = 1:3
            if stuck(i)
                Amat(i,:) = G(i,:);
                bvec(i)   = a_free_stage(i);
            else
                Amat(i,i) = 1;
                bvec(i)   = Fassign(i);
            end
        end
        F = Amat \ bvec;

        % A stuck rail that needs more than cc cannot hold: it breaks away, and
        % its force saturates at cc in the direction of the force it could not
        % hold.
        broke = false;
        for i = 1:3
            if stuck(i) && abs(F(i)) > cc(i)
                Fassign(i) = cc(i) * sign(F(i));
                stuck(i)   = false;
                broke      = true;
            end
        end
        if ~broke
            break
        end
    end

    % ------------------------------------------------------------------
    % dxdt = A*x + B*u_eff   (u_eff = u - P*F_stage)
    % ------------------------------------------------------------------
    dxdt = A*x + B*(u - P*F);
end
