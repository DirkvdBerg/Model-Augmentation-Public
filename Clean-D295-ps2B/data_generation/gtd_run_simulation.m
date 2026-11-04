function out = gtd_run_simulation(r, t, f_stage, plant, cfg, v)
% GTD_RUN_SIMULATION  Simulate one record version in closed loop with the truth.
%   out = GTD_RUN_SIMULATION(r, t, f_stage, plant, cfg, v) runs the recorded loop with the
%   controller plant.Cfb, reference r, injected force f_stage and encoder noise v (N x 3, stage
%   coordinates [m]; zeros for the noise-free version), in Simulink (cfg.mdl) or, with
%   cfg.simulator = 'replica', in plain MATLAB (tdg_replica.m). The controller sees y = q + v.
%   Returned struct: t_sim, r_sim, f_ms, q_with (true stage position), y_meas (= q + v),
%   u_fb = lsim(Cfb, r - y), u_total = u_fb + f, da_with (absorber state), v, u_plant (the force
%   logged at the plant input, equal to u_total).
    assert(isequal(size(v), [numel(t), 3]), 'noise is %d x %d, the record has %d samples', size(v,1), size(v,2), numel(t));
    if strcmp(cfg.simulator, 'replica')
        [q_aug, da, u_plant] = tdg_replica(r, f_stage, v, plant, cfg);
    else
        [q_aug, da, u_plant] = run_simulink(r, t, f_stage, plant, cfg, v);
    end
    assert(size(q_aug, 1) == numel(t), 'the simulation returned %d samples, the record has %d', size(q_aug, 1), numel(t));

    y_meas  = q_aug + v;
    u_fb    = lsim(plant.Cfb, r - y_meas);
    u_total = u_fb + f_stage;
    out = struct('t_sim',t, 'r_sim',r, 'f_ms',f_stage, 'q_with',q_aug, 'y_meas',y_meas, ...
                 'u_fb',u_fb, 'u_total',u_total, 'da_with',da, 'v',v, 'u_plant',u_plant);
end

function [q_aug, da, u_plant] = run_simulink(r, t, f, plant, cfg, v)
% One Simulink run. The model reads its inputs from the base workspace. The payload mass is the
% rigid part, so the hidden absorber mass ma enters only through the absorber state.
    names = {'mb','m1','m2','Jb','Jh','cg1','cg2','cy','cb1','cb2','kb1','kb2','Lb','d', ...
             'cc1','cc2','ccy','ma','L0','ka','ca','P','ts'};
    for k = 1:numel(names), assignin('base', names{k}, cfg.(names{k})); end
    assignin('base', 'mh', cfg.mh_rigid);
    assignin('base', 'g_fric', cfg.friction_gain);              % gain of the tanh friction in the chart
    assignin('base', 'Cfb', plant.Cfb);
    assignin('base', 'Y', plant.Y_op);                          % integrator initial state [0;0;Y;0;...]
    assignin('base', 'r', r);
    assignin('base', 't', t);
    assignin('base', 'f', f);
    assignin('base', 'v_in', [t(:), v]);
    evalin('base', 'clear q_aug delta_a_ode u_plant');          % no stale outputs
    evalin('base', sprintf('sim(''%s'', %.12g);', cfg.mdl, t(end)));
    q_aug   = evalin('base', 'q_aug');                          % true stage position
    da      = evalin('base', 'delta_a_ode');                    % absorber state of the ODE
    u_plant = evalin('base', 'u_plant');
end
