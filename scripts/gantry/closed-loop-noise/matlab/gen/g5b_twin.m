function g5b_twin()
% G5 (b) (CN-009): the noisy twin of the production dataset, both halves into ONE new folder.
    here = fileparts(mfilename('fullpath'));
    cn   = fileparts(fileparts(here));
    repo = fileparts(fileparts(fileparts(cn)));
    out  = fullfile(repo, 'data', 'gantry', 'matlab', 'trajectory', ...
                    'augmentation_ma50_z03_b140-230_a6all_telica_coulomb_noise_ss');
    noise_dir = fullfile(cn, 'matlab', 'noise_ss');
    cn_gen_multisine('noise', {}, out, noise_dir);
    cn_gen_telica('noise', {}, out, noise_dir);
    % the multisine cache is generator scratch, not dataset content (D-208): move it out
    c = fullfile(out, '_cache');
    if isfolder(c), movefile(c, fullfile(cn, 'outputs', 'g5b_ms_cache')); end
    fprintf('G5b done: %d records in %s\n', numel(dir(fullfile(out, '*.mat'))), out);
end
