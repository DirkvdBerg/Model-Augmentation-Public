function g5c_ref()
% G5 (c)/(d) reference (CN-009): noise OFF, double precision (q_true), for the 8 standstill-class
% records and the 7 Telica-profile records, into outputs/g5c_records (not a dataset).
    here = fileparts(mfilename('fullpath'));
    cn   = fileparts(fileparts(here));
    out  = fullfile(cn, 'outputs', 'g5c_records');
    cn_gen_multisine('off64', {'T1_','T2_','T3_','T4_','T5_','V1_','E1_','E2_'}, out, '');
    cn_gen_telica('off64', {}, out, '');
end
