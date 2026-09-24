function g5c_cc0()
% G5 (c) attempt 2 (CN-009a): T1, T3, T5, V1 with cc = 0, noise on and off, double, into outputs/.
    here = fileparts(mfilename('fullpath'));
    cn   = fileparts(fileparts(here));
    sel  = {'T1_','T3_','T5_','V1_'};
    cn_gen_multisine('off64', sel, fullfile(cn, 'outputs', 'g5c_cc0_off'), '', true);
    cn_gen_multisine('noise', sel, fullfile(cn, 'outputs', 'g5c_cc0_noise'), fullfile(cn, 'matlab', 'noise_ss'), true);
end
