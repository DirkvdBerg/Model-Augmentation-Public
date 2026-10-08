function out = gtd_run_simulation(record, r, t, f_stage, plant, cfg)
% GTD_RUN_SIMULATION  Run the Simulink model for one record (the only impure fn).
%   out = GTD_RUN_SIMULATION(record, r, t, f_stage, plant, cfg) simulates the
%   gantry model with (and, for the MSD model, without) the injected multisine,
%   and returns the reconstructed signals.
%
%   Simulink resolves variables from the BASE workspace, so all model inputs are
%   pushed to base with assignin and the run is launched with evalin. This is the
%   proven interface of generate_oscillatory_multisine_data.m; all base-workspace
%   contact is contained in this one function.
%
%   Returned struct: t_sim, r_sim, f_ms, q_with, u_fb, u_total, and (MSD only)
%   da_with, q_without, da_without.

    % ── push constant parameters and the frozen plant to the base workspace ──
    push_params(cfg);
    assignin('base', 'Cfb',    plant.Cfb);
    assignin('base', 'Cfb_ss', plant.Cfb);
    assignin('base', 'G',      plant.G);
    assignin('base', 'sys',    plant.sys);
    assignin('base', 'sys_cl', plant.sys_cl);
    assignin('base', 'T_cl',   plant.T_cl);
    assignin('base', 'r', r);
    assignin('base', 't', t);
    assignin('base', 'Y', plant.Y_op);

    tend = t(end);

    % ── WITH multisine ──────────────────────────────────────────────────────
    [q_aug, da] = run_once(f_stage, cfg, tend);
    [t_sim, r_sim, f_ms, q_with, da_with] = resample_sim(q_aug, da, r, f_stage, t);

    u_fb    = lsim(plant.Cfb, r_sim - q_with);
    u_total = u_fb + f_ms;

    out = struct('t_sim',t_sim, 'r_sim',r_sim, 'f_ms',f_ms, ...
                 'q_with',q_with, 'u_fb',u_fb, 'u_total',u_total);

    % ── WITHOUT multisine (informativeness baseline; MSD model only) ─────────
    if cfg.use_msd
        out.da_with = da_with;
        [q_aug0, da0] = run_once(zeros(size(f_stage)), cfg, tend);
        [~, ~, ~, q_without, da_without] = resample_sim(q_aug0, da0, r, zeros(size(f_stage)), t);
        out.q_without  = q_without;
        out.da_without = da_without;
    end
end

% ── helpers ─────────────────────────────────────────────────────────────────

function [q_aug, da] = run_once(f, cfg, tend)
% One Simulink run with force f. Swaps the payload mass to the rigid part so the
% hidden MSD (ma) enters only through the extra state, then restores it.
    assignin('base', 'f', f);
    if cfg.use_msd, assignin('base', 'mh', cfg.mh_rigid); end
    evalin('base', sprintf('sim(''%s'', %.12g);', cfg.mdl, tend));
    if cfg.use_msd, assignin('base', 'mh', cfg.mh); end
    if cfg.use_msd
        % q_aug is the output of the 'Extended ODE' block (gantrySystemExtended.m,
        % 8-state), whose own absorber state Simulink logs as delta_a_ode. The plain
        % 'delta_a' variable is outport 4 of the Simscape Multibody 'Single H-gantry'
        % subsystem, i.e. a DIFFERENT plant (it retains the Coriolis/centrifugal terms
        % that gantrySystemExtended drops by freezing M), so pairing it with q_aug made
        % the saved record internally inconsistent. Read the ODE's own state instead.
        % gtd_save_record derives vdelta_a as gradient(da, ts), so it follows this fix.
        q_aug = evalin('base', 'q_aug');
        da    = evalin('base', 'delta_a_ode');
    else
        q_aug = evalin('base', 'q1');
        da    = zeros(size(q_aug,1), 1);
    end
end

function push_params(cfg)
    names = {'mb','mh','m1','m2','Jb','Jh','cg1','cg2','cy','cb1','cb2', ...
             'kb1','kb2','Lb','Lh','d','cc1','cc2','ccy','n','P','C_damp','K','fs','ts','fbw'};
    for k = 1:numel(names), assignin('base', names{k}, cfg.(names{k})); end
    if cfg.use_msd
        msd = {'ma_frac','ma','mh_rigid','L0','fa','ka','zeta_a','ca'};
        for k = 1:numel(msd), assignin('base', msd{k}, cfg.(msd{k})); end
        assignin('base', 'mh_original', cfg.mh);   % full payload mass (model MATLAB Function refs it)
    end
end

function [t_sim, r_sim, f_sim, q_sim, da_sim] = resample_sim(q_aug, delta_a, r, f, t)
% Handle variable-step Simulink output via interpolation onto a uniform grid.
    Ns = size(q_aug, 1);
    if Ns ~= numel(t)
        t_sim = linspace(0, t(end), Ns)';
        r_sim = interp1(t, r, t_sim);
        f_sim = interp1(t, f, t_sim);
    else
        t_sim = t;  r_sim = r;  f_sim = f;
    end
    q_sim  = q_aug;
    da_sim = delta_a;
end
