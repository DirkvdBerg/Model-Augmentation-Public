function [A_anti, yaw_peak] = gtd_size_anti_amp(f_anti_unit, plant, cfg)
% GTD_SIZE_ANTI_AMP  Size the anti-symmetric amplitude to the yaw budget.
%   [A_anti, yaw_peak] = GTD_SIZE_ANTI_AMP(f_anti_unit, plant, cfg) returns the
%   anti-symmetric logical amplitude A_anti (a yaw TORQUE, N*m; see D-080) such
%   that the multisine-induced peak |X1-X2| equals cfg.yaw_budget.
%
%   f_anti_unit : (N x 1) unit-RMS anti-symmetric logical multisine.
%   Uses the SISO closed-loop transfer from anti torque to yaw displacement,
%   built from the same P^{-1} anti column verified in gtd_check_transform:
%       H_yaw = [1 -1 0] * sys_cl * ([1;-1;0]/Lb)
%   The loop is linear, so A_anti = budget / peak(response to unit torque) is exact.

    H_yaw    = [1 -1 0] * plant.sys_cl * ([1; -1; 0] / cfg.Lb);
    yaw_unit = lsim(H_yaw, f_anti_unit);       % yaw displacement per unit torque
    yaw_peak = max(abs(yaw_unit));
    A_anti   = cfg.yaw_budget / yaw_peak;      % torque [N*m] giving exactly the budget peak
end
