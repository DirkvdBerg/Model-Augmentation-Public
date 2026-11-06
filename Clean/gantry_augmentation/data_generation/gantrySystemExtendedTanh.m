function dxdt = gantrySystemExtendedTanh(u, x, m1, m2, mb, mh, Lb, Jb, Jh, d, ...
                                     cg1, cg2, cb1, cb2, cy, kb1, kb2, ...
                                     ma, ka, ca, L0, cc1, cc2, ccy, g)
% GANTRYSYSTEMEXTENDEDTANH  Jasper's gantrySystem.m with a hidden absorber on the payload and tanh Coulomb friction.
%   State x = [X; Theta; Y; delta_a; dX; dTheta; dY; ddelta_a], input u = [F_X; F_Theta; F_Y] (generalised forces).
    Y       = x(3);   % Y position, the same in stage and generalised coordinates
    delta_a = x(4);   % displacement of the absorber mass ma relative to the payload mh

    % Mass, damping and stiffness of Jasper's gantrySystem.m, with the absorber as a fourth degree of
    % freedom [X, Theta, Y, delta_a]; mh is the rigid part of the payload, ma sits L0 further along +Y
    M = [ m1+m2+mb+mh+ma,         (m1-m2)*Lb/2 - (mh+ma)*Y - ma*L0 - ma*delta_a,           0,        0;
         (m1-m2)*Lb/2 - (mh+ma)*Y - ma*L0 - ma*delta_a,  Jb+Jh+(m1+m2)*Lb^2/4 + (mh+ma)*d^2 + mh*Y^2 + ma*(Y+L0+delta_a)^2, -(mh+ma)*d, -ma*d;
          0,                                             -(mh+ma)*d,                                                             mh+ma,      ma;
          0,                                             -ma*d,                                                                  ma,         ma];

    C4 = [ cg1+cg2,            (cg1-cg2)*Lb/2,                   0,  0;
           (cg1-cg2)*Lb/2,  cb1+cb2+(cg1+cg2)*Lb^2/4,            0,  0;
           0,                0,                                   cy,  0;
           0,                0,                                    0, ca];

    K4 = [0,  0,        0,  0;
          0,  kb1+kb2,  0,  0;
          0,  0,        0,  0;
          0,  0,        0, ka];

    Minv_B = M \ [eye(3); zeros(1,3)];   % the forces act on X, Theta and Y, not on the absorber
    A = [ zeros(4),   eye(4);
         -M\K4,      -M\C4 ];
    B = [ zeros(4,3);
          Minv_B    ];

    % Coulomb friction on each actuator [X1, X2, Y], F_i = cc_i tanh(g v_i): Garcia's cc_i sign(v_i) for
    % |v_i| >> 1/g, and a smooth damper near standstill, so the stage settles without a limit cycle
    P  = [1,     1,      0;
          Lb/2, -Lb/2,   0;
          0,     0,      1];
    cc = [cc1; cc2; ccy];
    v_stage = P.' * x(5:7);                % stage velocities [dX1; dX2; dY]
    F       = cc .* tanh(g * v_stage);

    dxdt = A*x + B*(u - P*F);              % the friction opposes the generalised forces through P
end
