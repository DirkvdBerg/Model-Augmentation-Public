function [dxdt, held, stuck0] = gantry6Coulomb(u, x, m1, m2, mb, mh, Lb, Jb, Jh, d, ...
                                                cg1, cg2, cb1, cb2, cy, kb1, kb2, cc1, cc2, ccy)
% gantry6Coulomb  6-state gantry ODE + Karnopp Coulomb friction (TR-007 reference R2).
%
% Copy of Matlab-scripts/Augmentation-coulomb/gantrySystemExtendedCoulomb.m with the hidden
% absorber DOF (delta_a, ma, ka, ca, L0) DELETED, i.e. the ma = 0 case, which the original
% cannot be called with (its 4x4 mass matrix is singular at ma = 0). The friction block below
% is VERBATIM from the original (lines 137-243), except that the second and third outputs
% return the stuck flags for the census. The original file is not modified.
%
% State  x = [X; Theta; Y; dX; dTheta; dY]      Input u = [F_X; F_Theta; F_Y] (logical)

    Y = x(3);

    M = [ m1+m2+mb+mh,            (m1-m2)*Lb/2 - mh*Y,                               0;
          (m1-m2)*Lb/2 - mh*Y,     Jb+Jh+(m1+m2)*Lb^2/4 + mh*d^2 + mh*Y^2,          -mh*d;
          0,                       -mh*d,                                             mh];

    C3 = [ cg1+cg2,            (cg1-cg2)*Lb/2,               0;
           (cg1-cg2)*Lb/2,  cb1+cb2+(cg1+cg2)*Lb^2/4,        0;
           0,                0,                              cy];

    K3 = [0, 0,       0;
          0, kb1+kb2, 0;
          0, 0,       0];

    Minv_B = M \ eye(3);                   % 3x3

    A = [ zeros(3),   eye(3);
         -M\K3,      -M\C3 ];
    B = [ zeros(3,3);
          Minv_B    ];

    % ==== friction block, VERBATIM from gantrySystemExtendedCoulomb.m:137-243 ====
    P  = [1,     1,      0;
          Lb/2, -Lb/2,   0;
          0,     0,      1];
    cc = [cc1; cc2; ccy];

    V_BRK = 2.25e-3;   % [m/s] THEORY: lee2020feeddrive Table 5 p. 2839

    dx_free      = A*x + B*u;
    a_free       = dx_free(4:6);            % (5:8 in the 8-state original)
    MP           = Minv_B * P;
    G            = P.' * MP(1:3, :);
    a_free_stage = P.' * a_free(1:3);
    v_stage      = P.' * x(4:6);            % (5:7 in the 8-state original)

    stuck = false(3,1);
    for i = 1:3
        stuck(i) = abs(v_stage(i)) < V_BRK;
    end
    stuck0 = stuck;

    Fassign = zeros(3,1);
    for i = 1:3
        if ~stuck(i)
            Fassign(i) = cc(i) * sign(v_stage(i));
        end
    end

    F = zeros(3,1);
    for iter = 1:4
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
    held = stuck;

    dxdt = A*x + B*(u - P*F);
end
