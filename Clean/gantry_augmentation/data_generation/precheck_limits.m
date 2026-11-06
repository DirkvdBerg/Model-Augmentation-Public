function scale = precheck_limits(loop, r, f_stage, cfg)
% PRECHECK_LIMITS  The largest multisine scale (1, 0.9, ..., 0) at which the linear closed loop stays inside the machine limits.
%   Motion and multisine responses are simulated separately and added (superposition), so each scale costs no new simulation.
%   The limit check after the simulation of the truth, with friction and absorber, has the final say.
    r_eq   = [0, 0, loop.Y_op];
    r_pert = r - r_eq;
    q0    = lsim(loop.T_cl, r_pert);            % position from the motion alone, around the operating point
    u0_fb = lsim(loop.Cfb,  r_pert - q0);       % force from the motion alone
    q_ms  = lsim(loop.sys_cl, f_stage);         % position from the multisine alone
    u_ms  = lsim(loop.Cfb, -q_ms) + f_stage;    % force from the multisine alone

    for scale = [1, 0.9:-0.1:0.1, 0]
        if within_limits(q0 + scale*q_ms + r_eq, u0_fb + scale*u_ms, cfg)
            return
        end
    end
    error('precheck_limits: the motion alone violates the limits');
end

function ok = within_limits(q, u, cfg)
% Position, yaw, speed, peak force and rms force inside the limits
    lim = cfg.lim;
    vel = diff(q) * cfg.fs;
    ok = all(max(abs(q)) <= [lim.pos_X, lim.pos_X, lim.pos_Y]) ...
      && max(abs(q(:,1) - q(:,2))) <= lim.diff ...
      && all(max(abs(vel)) <= lim.vel) ...
      && all(max(abs(u)) <= lim.force_peak) ...
      && all(rms(u) <= lim.force_rms);
end
