function dxdt = gantrySystemExtended(u, x, m1, m2, mb, mh, Lb, Jb, Jh, d, ...
                                        cg1, cg2, cb1, cb2, cy, kb1, kb2, ...
                                        ma, ka, ca, L0)
% gantrySystemExtended  8-state gantry ODE with hidden MSD extra state.
%
% State  x = [X; Theta; Y; delta_a; dX; dTheta; dY; vdelta_a]
% Input  u = [F_X; F_Theta; F_Y]  (logical-coordinate forces, same as gantrySystem)
%
% Parameters:
%   m1..kb2   -- same 15 parameters as gantrySystem (mh here is mh_rigid,
%                i.e. total payload minus ma)
%   ma        -- hidden MSD mass (kg)
%   ka        -- MSD spring stiffness (N/m)
%   ca        -- MSD damper coefficient (Ns/m)
%   L0        -- equilibrium offset of ma from mh in +Y direction (m)
%
% Mass split convention (Option A):
%   mh_total = mh + ma  (conserved w.r.t. baseline)
%   Baseline gantrySystem uses mh_total as rigid mass.
%   This function uses mh (rigid) + ma (MSD), caller must pass mh_rigid.
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
