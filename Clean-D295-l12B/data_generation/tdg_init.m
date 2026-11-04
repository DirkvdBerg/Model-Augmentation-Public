function cfg = tdg_init()
% TDG_INIT  Config + Simulink cache folder + the generator model loaded.
    cfg = tdg_config();
    if ~exist(cfg.cache_dir, 'dir'), mkdir(cfg.cache_dir); end
    % Windows 260-character build limit: the repo path is long, so Simulink's build
    % files go to a short temp folder (they are regenerated on demand).
    Simulink.fileGenControl('set', 'CacheFolder', cfg.cache_dir, 'CodeGenFolder', cfg.cache_dir);
    f = fullfile(cfg.mdl_dir, [cfg.mdl '.slx']);
    assert(isfile(f), 'model %s missing', f);
    if ~bdIsLoaded(cfg.mdl), load_system(f); end
    set_param(cfg.mdl, 'SolverType', 'Fixed-step', 'Solver', 'ode4', 'FixedStep', sprintf('%.12g', cfg.ts));
end
