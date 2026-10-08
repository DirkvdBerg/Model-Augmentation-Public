function check_reproduction(file, repro_dir)
% CHECK_REPRODUCTION  Does the CURRENT generator reproduce a saved record? (audit finding A4)
%   check_reproduction('TF-1_r1', D) compares the saved record (both versions) with the one
%   regenerated into folder D by generate_thesis_data({file}, struct('out_dir', D)).
%
% Stated before the run: PASS when every field of both versions is bitwise equal (isequaln),
% recursively, except meta.generator (run date, MATLAB version, code hashes: provenance of the
% run itself, expected to differ). Every differing field is printed; the code files whose hash
% differs from the one stored in the saved record are listed (information, not pass/fail).
    here = fileparts(fileparts(mfilename('fullpath')));                % scripts/gantry/thesis-data-verification
    addpath(fullfile(fileparts(fileparts(fileparts(here))), 'Thesis-writeup', 'Code', 'Data'));
    cfg = tdg_config();
    man = tdg_manifest(cfg);
    e = man(strcmp({man.file}, file));
    assert(numel(e) == 1, 'record %s not in the manifest', file);
    sd = cfg.split_dir.(e.set);
    allok = true;
    vers = {'noisy', cfg.out_noisy, fullfile(repro_dir, cfg.name_noisy); ...
            'noise-free', cfg.out_free, fullfile(repro_dir, cfg.name_free)};
    for v = 1:2
        A = load(fullfile(vers{v,2}, sd, [file '.mat']));
        B = load(fullfile(vers{v,3}, sd, [file '.mat']));
        diffs = compare(A, B, '');
        diffs = diffs(~startsWith(diffs, 'meta.generator'));
        if isempty(diffs)
            fprintf('C16 %s %s: PASS, every field bitwise equal outside meta.generator\n', file, vers{v,1});
        else
            allok = false;
            fprintf('C16 %s %s: FAIL, %d field(s) differ: %s\n', file, vers{v,1}, numel(diffs), strjoin(diffs, ', '));
        end
        ha = A.meta.generator.code_sha256;  hb = B.meta.generator.code_sha256;
        fa = fieldnames(ha);  fb = fieldnames(hb);
        changed = {};
        for k = 1:numel(fb)
            if ~isfield(ha, fb{k}), changed{end+1} = [fb{k} ' (new)']; %#ok<AGROW>
            elseif ~strcmp(ha.(fb{k}), hb.(fb{k})), changed{end+1} = fb{k}; end %#ok<AGROW>
        end
        gone = cellfun(@(s) [s ' (removed)'], reshape(setdiff(fa, fb), 1, []), 'UniformOutput', false);
        fprintf('    code changed since the saved record: %s\n', strjoin([changed, gone], ', '));
        fprintf('    model sha256 saved %s, now %s\n', A.meta.generator.model_sha256(1:16), B.meta.generator.model_sha256(1:16));
    end
    if allok, fprintf('C16 reproduction %s: PASS\n', file); else, error('C16 reproduction %s: FAIL', file); end   % a FAIL stops the job
end

function d = compare(a, b, p)
% Recursive field comparison; returns the paths of fields that differ.
    d = {};
    if isstruct(a) && isstruct(b) && isscalar(a) && isscalar(b)
        fa = fieldnames(a);  fb = fieldnames(b);
        x = setxor(fa, fb);
        for k = 1:numel(x), d{end+1} = [p pre(p) x{k} ' (only in one)']; end %#ok<AGROW>
        f = intersect(fa, fb);
        for k = 1:numel(f)
            d = [d, compare(a.(f{k}), b.(f{k}), [p pre(p) f{k}])]; %#ok<AGROW>
        end
    elseif ~isequaln(a, b)
        d = {p};
    end
end

function s = pre(p)
    if isempty(p), s = ''; else, s = '.'; end
end
