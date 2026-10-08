% Generate the 12-record cubic-absorber test set with the thesis generator's verified replica (D-231, about 1 GB RAM).
% The copied ODE in this folder shadows the thesis one; out_dir keeps everything out of the thesis data set.
here = fileparts(mfilename('fullpath'));
gen  = fullfile(here, '..', '..', '..', '..', 'Thesis-writeup', 'Code', 'Data');
addpath(gen);                                  % the generator (it also adds itself to the top of the path)
cd(here);                                      % the CURRENT folder outranks the path: this folder's ODE copy wins
w = which('gantrySystemExtendedTanh'); fprintf('ODE in use: %s\n', w);
assert(contains(w, 'cubic-absorber'), 'cubic ODE is not first on the path');
out = fullfile(here, '..', '..', '..', '..', 'Thesis-writeup', 'Data', 'Cubic-absorber-test');
sel = {'E1_r1'};
generate_thesis_data(sel, struct('versions', {{'noisy'}}, 'simulator', 'replica', 'out_dir', out));
fprintf('GEN_CUBIC DONE\n');
