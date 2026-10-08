function [f_safe, info] = gtd_enforce_limits(plant, r, f_stage, cfg)
% GTD_ENFORCE_LIMITS  Linear closed-loop limit pre-check + proportional scale-down.
%   [f_safe, info] = GTD_ENFORCE_LIMITS(plant, r, f_stage, cfg) checks the total
%   (trajectory + multisine) response and force against the enforced limits using
%   the LINEAR closed loop (lsim superposition, no Simulink), and scales the
%   multisine down proportionally until the limits pass. Mirrors the safety loop
%   in generate_oscillatory_multisine_data.m (lines 342-368).
%
%   Returns the scaled stage force f_safe and an info struct (scale, force peak/
%   RMS, yaw). The hard 6 mm yaw and force limits are enforced here; the Simulink
%   run then uses f_safe.

    lim = cfg.lim;  fs = cfg.fs;
    r_eq   = [0, 0, plant.Y_op];
    r_pert = r - r_eq;

    q0    = lsim(plant.T_cl,  r_pert);          % trajectory-only position (perturbation)
    u0_fb = lsim(plant.Cfb,   r_pert - q0);     % trajectory-only force
    q_ms  = lsim(plant.sys_cl, f_stage);        % multisine-only position (perturbation)
    u_ms  = lsim(plant.Cfb,  -q_ms) + f_stage;

    scale = NaN;
    for s = [1, 0.9:-0.1:0.1, 0]
        q_tot = q0 + s*q_ms + r_eq;
        u_tot = u0_fb + s*u_ms;
        if validate_response(q_tot, fs, lim) && validate_forces(u_tot, lim)
            scale = s;  break
        end
    end
    assert(~isnan(scale), '%s: even the trajectory alone violates limits', inputname(1));

    f_safe = scale * f_stage;
    q_tot  = q0 + scale*q_ms + r_eq;
    u_tot  = u0_fb + scale*u_ms;
    info = struct('scale',scale, ...
                  'force_peak',max(abs(u_tot)), 'force_rms',rms(u_tot), ...
                  'yaw_mm',1e3*max(abs(q_tot(:,1)-q_tot(:,2))));
end

% ── validators (copied from generate_oscillatory_multisine_data.m) ──────────

function ok = validate_response(q, fs, lim)
    vel = diff(q)*fs;
    ok  =  max(abs(q(:,1)))           <= lim.pos_X ...
        && max(abs(q(:,2)))           <= lim.pos_X ...
        && max(abs(q(:,3)))           <= lim.pos_Y ...
        && max(abs(q(:,1)-q(:,2)))    <= lim.diff  ...
        && max(abs(vel(:,1)))         <= lim.vel   ...
        && max(abs(vel(:,2)))         <= lim.vel   ...
        && max(abs(vel(:,3)))         <= lim.vel;
end

function ok = validate_forces(u_total, lim)
    ok =  all(max(abs(u_total)) <= lim.force_peak) ...
       && all(rms(u_total)      <= lim.force_rms);
end
