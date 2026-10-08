function check_tanh_model(rebuild, rec)
%CHECK_TANH_MODEL  D-224 check: the generator model with the tanh chart reproduces the D-223 replica.
%
%   check_tanh_model(rebuild, rec)
%     rebuild  true = run tdg_make_model first (rebuilds model/gantry_tdg_2025a.slx; the Karnopp
%              build is in git and in outputs/model_backup/)                 (default true)
%     rec      {split folder, file stem}                        (default {'Training', 'TR-T1_r1'})
%
% For each version (noise-free, noisy) the stored record's inputs (r_sim, t_sim, f_sim, d_in,
% Y_op, controller) drive (1) the Simulink generator through gtd_run_simulation, exactly as
% generate_thesis_data does, and (2) cl_sim with gantrySystemExtendedTanh at cfg.friction_gain.
% Pass (stated in D-224 before the run): max |q_sim - q_replica| < 1e-10 m, and the plumbing
% u_plant = u_total + d to round-off. Needs Simulink (about 2 to 3.5 GB RAM).

    if nargin < 1 || isempty(rebuild), rebuild = true; end
    if nargin < 2 || isempty(rec), rec = {'Training', 'TR-T1_r1'}; end
    here = fileparts(mfilename('fullpath'));
    repo = fileparts(fileparts(fileparts(here)));
    addpath(fullfile(repo, 'Thesis-writeup', 'Code', 'Data'));
    cfg0 = tdg_config();
    Simulink.fileGenControl('set', 'CacheFolder', cfg0.cache_dir, 'CodeGenFolder', cfg0.cache_dir);
    if rebuild, tdg_make_model(); end
    cfg = tdg_init(false);
    addpath(here);
    fprintf('friction gain %g s/m\n', cfg.friction_gain);
    base = {cfg.m1, cfg.m2, cfg.mb, cfg.mh_rigid, cfg.Lb, cfg.Jb, cfg.Jh, cfg.d, ...
            cfg.cg1, cfg.cg2, cfg.cb1, cfg.cb2, cfg.cy, cfg.kb1, cfg.kb2, ...
            cfg.ma, cfg.ka, cfg.ca, cfg.L0, cfg.cc1, cfg.cc2, cfg.ccy};
    ode = @(x, u) gantrySystemExtendedTanh(u, x, base{:}, cfg.friction_gain);
    FOLDER = struct('noise_free', 'Coulomb-and-MSD-noise-free', 'noisy', 'Coulomb-and-MSD');
    allpass = true;
    for ver = {'noise-free', 'noisy'}
        v = ver{1};
        S = load(fullfile(repo, 'Thesis-writeup', 'Data', FOLDER.(strrep(v, '-', '_')), rec{1}, ...
                          [rec{2} '.mat']), 'r_sim', 't_sim', 'f_sim', 'd_in', 'meta');
        c = S.meta.controller;
        plant = gtd_build_plant(S.meta.Y_op, cfg, c.gain);
        t0 = tic;
        out = gtd_run_simulation([], S.r_sim, S.t_sim, S.f_sim, plant, cfg, S.d_in);
        ts_sim = toc(t0);
        [Ac, Bc, Cc, Dc] = ssdata(plant.Cfb);
        ctl = struct('A', Ac, 'B', Bc, 'C', Cc, 'D', Dc);
        t0 = tic;
        q = cl_sim(S, ode, S.meta.Y_op, ctl, cfg.P, cfg.ts);
        dq = max(abs(out.q_with - q), [], 1);
        plumb = max(abs(out.u_plant - (out.u_total + S.d_in)), [], 'all');
        pass = all(dq < 1e-10);
        allpass = allpass && pass;
        fprintf(['%-10s Simulink %.0f s, replica %.0f s | max|q_sim - q_replica| %s m | ' ...
                 'plumbing %.2e N | %s\n'], v, ts_sim, toc(t0), num2str(dq, '%.2e '), plumb, ...
                ternary(pass, 'PASS', 'FAIL'));
    end
    fprintf('D-224 model check: %s\n', ternary(allpass, 'PASS', 'FAIL'));
    close_system(cfg.mdl, 0);
end

function out = ternary(c, a, b)
    if c, out = a; else, out = b; end
end
