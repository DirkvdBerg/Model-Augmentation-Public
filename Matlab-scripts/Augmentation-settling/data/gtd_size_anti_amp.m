function [A_anti, yaw_at_target] = gtd_size_anti_amp(f_anti_unit, plant, cfg)
% GTD_SIZE_ANTI_AMP  Anti-symmetric amplitude: modest fixed level, yaw-budget capped.
%   [A_anti, yaw_at_target] = GTD_SIZE_ANTI_AMP(f_anti_unit, plant, cfg) returns
%   the anti-symmetric logical amplitude A_anti (a yaw TORQUE, N*m; see D-080).
%
%   A_anti = min( cfg.A_anti , budget-limited amplitude )
%   i.e. use the modest fixed target cfg.A_anti, and only scale DOWN if it would
%   drive the multisine-induced peak |X1-X2| past cfg.yaw_budget. The budget is a
%   CEILING, not a target: filling it at 130-180 Hz would demand kilonewtons of
%   force (inertia ~ omega^2) and does not help activate the Y-axis MSD (D-084).
%
%   f_anti_unit : (N x 1) unit-RMS anti-symmetric logical multisine.
%   Yaw transfer uses the same P^{-1} anti column verified in gtd_check_transform:
%       H_yaw = [1 -1 0] * sys_cl * ([1;-1;0]/Lb)

    H_yaw     = [1 -1 0] * plant.sys_cl * ([1; -1; 0] / cfg.Lb);
    yaw_unit  = lsim(H_yaw, f_anti_unit);          % yaw per unit-RMS torque
    yaw_peak  = max(abs(yaw_unit));                % [m] per unit torque
    A_cap     = cfg.yaw_budget / yaw_peak;         % torque that would exactly fill the budget
    A_anti    = min(cfg.A_anti, A_cap);            % modest target, capped by the budget
    yaw_at_target = A_anti * yaw_peak;             % actual multisine-induced yaw peak [m]
end
