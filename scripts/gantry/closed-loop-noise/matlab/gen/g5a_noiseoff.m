function g5a_noiseoff()
% G5 (a) (CN-009): build the encoder-noise model, regenerate T3_standstill_Y000 and TP1_telica_y000
% with v = 0 through it, and compare every field of the production records bitwise.
    here = fileparts(mfilename('fullpath'));
    cn   = fileparts(fileparts(here));
    repo = fileparts(fileparts(fileparts(cn)));
    prod = fullfile(repo, 'data', 'gantry', 'matlab', 'trajectory', ...
                    'augmentation_ma50_z03_b140-230_a6all_telica_coulomb');
    out  = fullfile(cn, 'outputs', 'g5a_records');
    make_noise_model();
    cn_gen_multisine('off', {'T3'}, out, '');
    cn_gen_telica('off', {'TP1'}, out, '');
    allok = true;
    for id = {'T3_standstill_Y000', 'TP1_telica_y000'}
        P = load(fullfile(prod, [id{1} '.mat']));
        N = load(fullfile(out, [id{1} '.mat']));
        fn = fieldnames(P);
        bad = {};
        for k = 1:numel(fn)
            if ~isfield(N, fn{k}) || ~isequal(P.(fn{k}), N.(fn{k})) || ~strcmp(class(P.(fn{k})), class(N.(fn{k})))
                bad{end+1} = fn{k}; %#ok<AGROW>
                if isfield(N, fn{k}) && isnumeric(P.(fn{k})) && isequal(size(P.(fn{k})), size(N.(fn{k})))
                    fprintf('  %s.%s: max |diff| %.3e\n', id{1}, fn{k}, max(abs(double(P.(fn{k})(:)) - double(N.(fn{k})(:)))));
                end
            end
        end
        fprintf('G5a %s: %d production fields, %d differ %s -> %s\n', id{1}, numel(fn), numel(bad), ...
                strjoin(bad, ','), string(isempty(bad)));
        allok = allok && isempty(bad);
    end
    fprintf('G5a PASS %s\n', string(allok));
end
