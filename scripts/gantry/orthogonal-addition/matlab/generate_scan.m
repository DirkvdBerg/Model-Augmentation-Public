function generate_scan(names, additions)
% GENERATE_SCAN  Several training-record-only datasets in ONE MATLAB session (OA-019 scans).
%   generate_scan({'augmentation_oa_s185', ...}, {'design/x185.mat', ...})
% Each call is generate_oa_data(NAME, ADDITION, SELECT = the 14 training tags); the folder gets
% the generator's _n14 suffix, so a scan set can never be mistaken for a full dataset.
tags = arrayfun(@(k) sprintf('T%d', k), 1:14, 'UniformOutput', false);
here = fileparts(mfilename('fullpath'));
addpath(here);
for k = 1:numel(names)
    fprintf('\n##### scan %d/%d: %s <- %s\n', k, numel(names), names{k}, additions{k});
    generate_oa_data('NAME', names{k}, 'ADDITION', additions{k}, 'SELECT', tags);
end
end
