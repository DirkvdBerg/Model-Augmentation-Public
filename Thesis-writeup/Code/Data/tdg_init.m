function cfg = tdg_init(load_orig)
% TDG_INIT  Config + Simulink cache folder + the generator model loaded (and the
% original model when load_orig is true, for the equivalence checks only).
    cfg = tdg_config();
    if ~exist(cfg.cache_dir, 'dir'), mkdir(cfg.cache_dir); end
    % Windows 260-character build limit: the repo path is long, so Simulink's build
    % files go to a short temp folder (they are regenerated on demand).
    Simulink.fileGenControl('set', 'CacheFolder', cfg.cache_dir, 'CodeGenFolder', cfg.cache_dir);
    f = fullfile(cfg.mdl_dir, [cfg.mdl '.slx']);
    assert(isfile(f), 'model %s missing: run tdg_make_model first', f);
    if ~bdIsLoaded(cfg.mdl), load_system(f); end
    set_param(cfg.mdl, 'SolverType', 'Fixed-step', 'Solver', 'ode4', 'FixedStep', sprintf('%.12g', cfg.ts));
    if nargin > 0 && load_orig && ~bdIsLoaded(cfg.mdl_orig)
        % the original model's other loops need gantrySystem.m and gantrySystemCoriolisCentripetal.m,
        % which live in scripts/gantry/thesis-data-verification/vendor/ (check C1 adds them)
        load_system(fullfile(cfg.here, 'vendor', [cfg.mdl_orig '.slx']));
        set_param(cfg.mdl_orig, 'SolverType', 'Fixed-step', 'Solver', 'ode4', 'FixedStep', sprintf('%.12g', cfg.ts));
    end
end
