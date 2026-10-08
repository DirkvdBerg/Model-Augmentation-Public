% RUN_FULL_NOISY  D-238: generate every T_AF record in the noisy version only (low-pass encoder noise),
%   replica simulator, into Thesis-writeup/Data/Coulomb-tanh-and-MSD-lowpass-noise. The noise-free set is
%   read (multisine scale), not written. Existing files are skipped, so a stopped run can be restarted.
here = fileparts(mfilename('fullpath'));
cd(fullfile(here, '..', '..', 'Thesis-writeup', 'Code', 'Data'));
tdg_init_paths;
generate_thesis_data([], struct('versions', {{'noisy'}}, 'simulator', 'replica'));
