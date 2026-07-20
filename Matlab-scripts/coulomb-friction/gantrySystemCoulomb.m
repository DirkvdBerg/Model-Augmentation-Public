function dxdt = gantrySystemCoulomb(baseFn, u, x, Lb, cc)
% gantrySystemCoulomb  Add Coulomb (dry) friction to any gantry EOM base.
%
%   dxdt = gantrySystemCoulomb(baseFn, u, x, Lb, cc)
%
% Coulomb friction is a generalized force, so it is added as an EFFECTIVE
% input: the base EOM is evaluated at (u - F_c) instead of u. This is the
% exact structure of the commented force vector in the FP model
% (kamtin-fp-model/03 Simulink gantry/main.m) and Garcia 2013 Fig. 2:
%
%   f = [F1 + F2 - cc1*sign(dX1) - cc2*sign(dX2);
%        (F1 - F2 - cc1*sign(dX1) + cc2*sign(dX2))*Lb/2;
%        Fy - ccy*sign(dY)]
%
% which is exactly  u_logical - P*(cc .* sign(P' * qdot)).
%
% Inputs
%   baseFn : function handle @(u_eff, x) -> dxdt of the friction-free EOM
%            (e.g. a closure over gantrySystemCoriolisCentripetal or
%            gantrySystem with all physical parameters bound). Keeping the
%            base as a handle makes this wrapper base-agnostic: the same
%            Coulomb term F_c is used whether or not the base includes the
%            Coriolis-centripetal forces.
%   u      : (3x1) applied generalized force in LOGICAL coordinates [F_X; F_Th; F_Y]
%   x      : (6x1) state [X; Theta; Y; dX; dTheta; dY] in logical coordinates
%   Lb     : () cross-arm length [m] (defines the stage<->logical transform P)
%   cc     : (3x1) Coulomb force magnitudes [cc1; cc2; ccy] [N] (>= 0)
%
% Output
%   dxdt   : (6x1) state derivative
%
% Coordinate transforms (same P as main.m):
%   P = [1 1 0; Lb/2 -Lb/2 0; 0 0 1]
%   stage positions/velocities = P' * logical      (P' maps logical -> stage)
%   logical generalized force  = P  * stage force  (P  maps stage  -> logical)
%
% Note: hard sign() is used here to match the Simscape Signum blocks for the
% MATLAB cross-check. The Python LPV-LFR implementation uses a smoothed
% tanh(v/v0) surrogate for differentiable BPTT (see docs/decisions.md D-116).

    P = [1,     1,      0;
         Lb/2, -Lb/2,   0;
         0,     0,      1];

    qdot     = x(4:6);
    v_stage  = P.' * qdot;                    % [dX1; dX2; dY] stage velocities
    % THEORY: garcia2013 -- per-actuator Coulomb force F_c,i = cc_i*sign(v_i)
    Fc_stage = cc(:) .* sign(v_stage);        % (3x1) stage-frame Coulomb force [N]
    Fc_log   = P * Fc_stage;                   % (3x1) logical generalized force

    dxdt = baseFn(u - Fc_log, x);
end
