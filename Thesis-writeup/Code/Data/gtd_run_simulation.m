function out = gtd_run_simulation(record, r, t, f_stage, plant, cfg, v) %#ok<INUSL>
% GTD_RUN_SIMULATION  Run the Simulink model for one record (the only impure fn).
%   out = GTD_RUN_SIMULATION(record, r, t, f_stage, plant, cfg, v) simulates the
%   T_AF truth (cfg.mdl) in closed loop with the record's controller plant.Cfb,
%   reference r, injected force f_stage and encoder noise v (N x 3, stage, [m], D-225).
%
%   Simulink resolves variables from the BASE workspace, so all model inputs are
%   pushed to base with assignin and the run is launched with evalin (the proven
%   interface of the copied generator).
%
%   Returned struct: t_sim, r_sim, f_ms, q_with (stage, TRUE position), y_meas (= q + v,
%   the measured output the controller saw), u_fb, u_total, da_with, v (the encoder
%   noise), u_plant (the force logged at the plant input), sim_s (wall time).
%
%   THESIS-DATA changes against scripts/gantry/thesis-data-verification/vendor/original/gtd_run_simulation.m:
%     - encoder noise v pushed as v_in = [t, v] (D-225; D-218's noise force d is gone);
%       v = [] or zeros gives the noise-free twin. The controller sees y = q + v, so
%       u_fb = lsim(Cfb, r - y) and u_total = u_ff + u_fb is the plant force.
%     - ONE run per record: the second run "without multisine" fed only a printed
%       delta_a diagnostic, never a saved field; dropped for runtime.
%     - u_plant read back for the plumbing check u_plant = u_total (D-225).
%     - g_fric (cfg.friction_gain) pushed for the tanh friction law of the chart (D-224).
%     - cfg.simulator = 'replica' steps the loop with tdg_replica.m instead of Simulink (D-231).

    push_params(cfg);
    assignin('base', 'Cfb',    plant.Cfb);
    assignin('base', 'Cfb_ss', plant.Cfb);
    assignin('base', 'G',      plant.G);
    assignin('base', 'sys',    plant.sys);
    assignin('base', 'sys_cl', plant.sys_cl);
    assignin('base', 'T_cl',   plant.T_cl);
    assignin('base', 'r', r);
    assignin('base', 't', t);
    assignin('base', 'Y', plant.Y_op);          % Integrator IC [0;0;Y;0;...]: the record start

    if nargin < 7 || isempty(v), v = zeros(numel(t), 3); end
    v = double(v);
    assert(isequal(size(v), [numel(t), 3]), 'noise length %d x %d, record %d', size(v,1), size(v,2), numel(t));
    assignin('base', 'v_in', [t(:), v]);

    tend = t(end);
    tic;
    if strcmp(cfg.simulator, 'replica')                  % D-231: the same loop outside Simulink
        [q_aug, da, u_plant] = tdg_replica(r, f_stage, v, plant, cfg);
    else
        [q_aug, da, u_plant] = run_once(f_stage, cfg, tend);
    end
    sim_s = toc;
    assert(size(q_aug, 1) == numel(t), 'output length %d vs %d samples', size(q_aug, 1), numel(t));
    t_sim = t;  r_sim = r;  f_ms = f_stage;

    y_meas  = q_aug + v;                               % D-225: the signal the controller saw
    u_fb    = lsim(plant.Cfb, r_sim - y_meas);
    u_total = u_fb + f_ms;

    out = struct('t_sim',t_sim, 'r_sim',r_sim, 'f_ms',f_ms, 'q_with',q_aug, 'y_meas',y_meas, ...
                 'u_fb',u_fb, 'u_total',u_total, 'da_with',da, 'v',v, ...
                 'u_plant',u_plant, 'sim_s',sim_s);
end

% ── helpers ─────────────────────────────────────────────────────────────────

function [q_aug, da, u_plant] = run_once(f, cfg, tend)
% One Simulink run with force f. Swaps the payload mass to the rigid part so the
% hidden MSD (ma) enters only through the extra state, then restores it.
    assignin('base', 'f', f);
    evalin('base', 'clear q_aug delta_a_ode u_plant');     % no stale outputs
    assignin('base', 'mh', cfg.mh_rigid);
    evalin('base', sprintf('sim(''%s'', %.12g);', cfg.mdl, tend));
    assignin('base', 'mh', cfg.mh);
    % q_aug is the stage output of the 'Extended ODE' loop, delta_a_ode its own absorber
    % state (see the original file for why not the Simscape 'delta_a').
    q_aug = evalin('base', 'q_aug');
    da    = evalin('base', 'delta_a_ode');
    if evalin('base', 'exist(''u_plant'', ''var'')')
        u_plant = evalin('base', 'u_plant');
    else
        u_plant = [];                                        % the original model has no log
    end
end

function push_params(cfg)
    names = {'mb','mh','m1','m2','Jb','Jh','cg1','cg2','cy','cb1','cb2', ...
             'kb1','kb2','Lb','Lh','d','cc1','cc2','ccy','n','P','C_damp','K','fs','ts','fbw'};
    for k = 1:numel(names), assignin('base', names{k}, cfg.(names{k})); end
    msd = {'ma_frac','ma','mh_rigid','L0','fa','ka','zeta_a','ca'};
    for k = 1:numel(msd), assignin('base', msd{k}, cfg.(msd{k})); end
    assignin('base', 'mh_original', cfg.mh);
    % THESIS-DATA (D-224): the tanh friction gain, a Parameter of the Extended ODE chart
    assignin('base', 'g_fric', cfg.friction_gain);
end
