function export_record_reference()
% EXPORT_RECORD_REFERENCE  Machine-precision reference for the record-level gate.
%
%   The record-based check in verify_controller.py cannot reach machine precision, and the
%   reason is not the controller. gtd_run_simulation.m:33 computes
%
%       u_fb = lsim(plant.Cfb, r_sim - q_with)
%
%   with r_sim and q_with in DOUBLE, and gtd_save_record.m:25-31 then stores u_fb, y and
%   r_sim as SINGLE. Python therefore drives the controller with a quantised error signal
%   while the stored u_fb came from an unquantised one. The difference is a constant bias in
%   e of order one float32 step, which the controller's pole at z = 1 integrates into a ramp:
%   on V1_standstill_Yp10 the Y residual is 99.99 % a straight line of +0.199 N/s.
%
%   This script removes that mismatch. It re-runs MATLAB's own lsim on exactly the signal
%   Python can reconstruct, double(r_sim) - double(y) formed from the STORED single values,
%   and saves the result in double. Both sides then see identical input bits, so the residual
%   is arithmetic only.
%
%   No Simulink, no plant simulation, no record is modified. Reads the records, writes one
%   new file next to this script.

    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(fileparts(here)));
    addpath(fullfile(root, 'Matlab-scripts', 'Augmentation', 'data'));

    traj = fullfile(root, 'data', 'gantry', 'matlab', 'trajectory', 'augmentation');
    names = {'V1_standstill_Yp10', 'T10_aprbs_60'};
    Y_ops = [0.10, 0.00];                          % gtd_build_records.m

    cfg = gtd_config('augmentation', true, 0.10);

    % One file per record, saved -v7 so scipy.io.loadmat can read it without h5py.
    for i = 1:numel(names)
        rec = load(fullfile(traj, [names{i} '.mat']), 'r_sim', 'y');

        % Exactly the signal Python forms: the stored single values widened to double.
        e = double(rec.r_sim) - double(rec.y);

        plant = gtd_build_plant(Y_ops(i), cfg);
        Cfb   = plant.Cfb;
        u_ref = lsim(Cfb, e);

        Ctf = tf(Cfb);
        num = cell(3,1); den = cell(3,1);
        for j = 1:3
            [nj, dj] = tfdata(Ctf(j,j), 'v');
            num{j} = nj(:).';  den{j} = dj(:).';
        end

        S = struct();
        S.name = names{i};   S.Y_op = Y_ops(i);
        S.u_ref = u_ref;
        S.A = Cfb.A;  S.B = Cfb.B;  S.C = Cfb.C;  S.D = Cfb.D;
        S.num = num;  S.den = den;

        out = fullfile(here, sprintf('record_reference_%s.mat', names{i}));
        save(out, '-struct', 'S', '-v7');
        fprintf('%-22s Y_op %.2f  N %6d  e rms [%.3e %.3e %.3e] m  u_ref rms [%.4e %.4e %.4e] N\n', ...
            names{i}, Y_ops(i), size(e,1), std(e), std(u_ref));
        fprintf('   wrote %s\n', out);
    end
end
