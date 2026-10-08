function cfg = tdg_init_paths()
% TDG_INIT_PATHS  Put the generator on the path and set the short Simulink cache
% folder, without loading any model (used before the model exists).
    here = fileparts(mfilename('fullpath'));
    addpath(here);
    cfg = tdg_config();
    if ~exist(cfg.cache_dir, 'dir'), mkdir(cfg.cache_dir); end
    Simulink.fileGenControl('set', 'CacheFolder', cfg.cache_dir, 'CodeGenFolder', cfg.cache_dir);
end
