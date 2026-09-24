function cfg = attach_noise(cfg, mode, noise_dir, id, T)
% ATTACH_NOISE  (CN-009) put the record's encoder noise into cfg for gtd_run_simulation.
%   mode 'off' / 'off64': v = 0 (no field -> gtd_run_simulation injects zeros).
%   mode 'noise': v from noise_dir/<id>.mat (written by model/make_noise_files.py), length T.
    cfg.noise_v = [];  cfg.noise_seed = NaN;  cfg.noise_model = 'none';
    if strcmp(mode, 'noise')
        N = load(fullfile(noise_dir, [id '.mat']));
        assert(size(N.v, 1) == T, 'noise file %s has %d samples, record %d', id, size(N.v, 1), T);
        cfg.noise_v = double(N.v);
        cfg.noise_seed = double(N.seed);
        cfg.noise_model = sprintf('%s VAR, H_v sha256 %s, scale %g', char(N.reading), char(N.hv_sha256), double(N.scale));
        fprintf('  encoder noise: seed %d, rms [%.2f %.2f %.2f] nm\n', cfg.noise_seed, 1e9*std(cfg.noise_v));
    end
end
