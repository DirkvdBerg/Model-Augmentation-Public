% verify_exports.m
% ----------------
% Runs export_lpv_matrices.m and export_lpv_sim.m, then verifies their
% outputs with data-driven tolerances.
%
% Must be run from the repo root:
%   cd('<path>/Baseline-LPV-Augmentation')
%   run('Matlab-scripts/verify_exports.m')
%
% Tolerances for simulation checks are derived from an observed baseline run:
%   X1/X2 max deviation: 0.35 um observed  -> tolerance 10 um  (28x margin)
%   Y tracking error:    4.38 um observed  -> tolerance 50 um  (11x margin)
%   See docs/fp-model-structure.md for physical limits source (ETEL datasheet).

% ------------------------------------------------------------------
% Resolve paths and cd to repo root
% ------------------------------------------------------------------
% verify_exports.m lives in Matlab-scripts/. Repo root is one level up.
% The export scripts rely on pwd == repo root for addpath and file saving,
% so we cd there explicitly regardless of where the user called us from.
script_dir = fileparts(mfilename('fullpath'));   % .../Matlab-scripts
repo_root  = fileparts(script_dir);             % .../Baseline-LPV-Augmentation
orig_dir   = pwd;
cd(repo_root)

if ~exist(fullfile(repo_root, 'kamtin-fp-model'), 'dir')
    cd(orig_dir)
    error('Could not find kamtin-fp-model/ under: %s', repo_root)
end

n_pass = 0;
n_fail = 0;

% ------------------------------------------------------------------
% SECTION 1: export_lpv_matrices.m
% ------------------------------------------------------------------
fprintf('\n%s\n', repmat('=', 1, 60))
fprintf('  Running export_lpv_matrices.m ...\n')
fprintf('%s\n', repmat('=', 1, 60))
run(fullfile(script_dir, 'export_lpv_matrices.m'))
fprintf('\n--- Checking lpv_matrices outputs ---\n\n')

% 1.1 Variable sizes
[n_pass, n_fail] = chk(n_pass, n_fail, 'A_all size = (6,6,50)', ...
    isequal(size(A_all), [6 6 50]), ...
    sprintf('got %s', mat2str(size(A_all))));

[n_pass, n_fail] = chk(n_pass, n_fail, 'B_all size = (6,3,50)', ...
    isequal(size(B_all), [6 3 50]), ...
    sprintf('got %s', mat2str(size(B_all))));

[n_pass, n_fail] = chk(n_pass, n_fail, 'C_all size = (3,6,50)', ...
    isequal(size(C_all), [3 6 50]), ...
    sprintf('got %s', mat2str(size(C_all))));

[n_pass, n_fail] = chk(n_pass, n_fail, 'D_all size = (3,3,50)', ...
    isequal(size(D_all), [3 3 50]), ...
    sprintf('got %s', mat2str(size(D_all))));

% 1.2 Y_values within physical range
[n_pass, n_fail] = chk(n_pass, n_fail, 'Y_values within [-0.35, 0.35] m', ...
    all(Y_values >= -0.35) && all(Y_values <= 0.35), ...
    sprintf('range [%.3f, %.3f] m', min(Y_values), max(Y_values)));

% 1.3 D = 0
d_max = max(abs(D_all(:)));
[n_pass, n_fail] = chk(n_pass, n_fail, 'D_all == 0  (max abs < 1e-12)', ...
    d_max < 1e-12, ...
    sprintf('max abs D = %.2e', d_max));

% 1.4 M(Y) positive definite: det > 0 at all Y
[n_pass, n_fail] = chk(n_pass, n_fail, 'det(M(Y)) > 0 at all Y', ...
    all(det_M > 0), ...
    sprintf('min det = %.4f', min(det_M)));

% 1.5 C constant across Y
C_std = max(std(reshape(C_all, 18, 50), 0, 2));
[n_pass, n_fail] = chk(n_pass, n_fail, 'C_all constant across Y  (max std < 1e-10)', ...
    C_std < 1e-10, ...
    sprintf('max std = %.2e', C_std));

% 1.6 All eigenvalues of A(Y) inside unit circle
max_eig_mag = 0;
for k = 1:size(A_all, 3)
    max_eig_mag = max(max_eig_mag, max(abs(eig(A_all(:,:,k)))));
end
[n_pass, n_fail] = chk(n_pass, n_fail, 'All eig(A(Y)) inside unit circle', ...
    max_eig_mag <= 1.0, ...
    sprintf('max |eig| = %.6f', max_eig_mag));

% ------------------------------------------------------------------
% SECTION 2: export_lpv_sim.m
% ------------------------------------------------------------------
fprintf('\n%s\n', repmat('=', 1, 60))
fprintf('  Running export_lpv_sim.m  (Simulink -- takes ~1-2 min) ...\n')
fprintf('%s\n', repmat('=', 1, 60))
run(fullfile(script_dir, 'export_lpv_sim.m'))
fprintf('\n--- Checking lpv_sim_varying_y outputs ---\n\n')

N = size(q1, 1);

% 2.1 Consistent sizes
[n_pass, n_fail] = chk(n_pass, n_fail, 'q1 size = (N,3)', ...
    size(q1,2) == 3 && N > 0, ...
    sprintf('got %dx%d', size(q1,1), size(q1,2)));

[n_pass, n_fail] = chk(n_pass, n_fail, 'u_q1 size = (N,3)', ...
    isequal(size(u_q1), [N 3]), ...
    sprintf('got %dx%d', size(u_q1,1), size(u_q1,2)));

[n_pass, n_fail] = chk(n_pass, n_fail, 'Y_trajectory size = (N,1)', ...
    isequal(size(Y_trajectory), [N 1]), ...
    sprintf('got %dx%d', size(Y_trajectory,1), size(Y_trajectory,2)));

[n_pass, n_fail] = chk(n_pass, n_fail, 'r_sim size = (N,3)', ...
    isequal(size(r_sim), [N 3]), ...
    sprintf('got %dx%d', size(r_sim,1), size(r_sim,2)));

[n_pass, n_fail] = chk(n_pass, n_fail, 'q_simscape size = (N,3)', ...
    isequal(size(q_simscape), [N 3]), ...
    sprintf('got %dx%d', size(q_simscape,1), size(q_simscape,2)));

% 2.2 Y within physical machine range
[n_pass, n_fail] = chk(n_pass, n_fail, 'Y_trajectory within physical range [-0.4, 0.4] m', ...
    all(Y_trajectory >= -0.4) && all(Y_trajectory <= 0.4), ...
    sprintf('range [%.3f, %.3f] m', min(Y_trajectory), max(Y_trajectory)));

% 2.3 Initial Y near 0.3 m
[n_pass, n_fail] = chk(n_pass, n_fail, 'Initial Y = 0.3 m  (tolerance +-2 mm)', ...
    abs(Y_trajectory(1) - 0.3) < 0.002, ...
    sprintf('got %.4f m', Y_trajectory(1)));

% 2.4 Final Y near 0.1 m
[n_pass, n_fail] = chk(n_pass, n_fail, 'Final Y = 0.1 m  (tolerance +-2 mm)', ...
    abs(Y_trajectory(end) - 0.1) < 0.002, ...
    sprintf('got %.4f m', Y_trajectory(end)));

% 2.5 Feedback was active: F_Y RMS > 1 N
FY_rms = sqrt(mean(u_q1(:,3).^2));
[n_pass, n_fail] = chk(n_pass, n_fail, 'F_Y RMS > 1 N  (feedback active)', ...
    FY_rms > 1.0, ...
    sprintf('F_Y RMS = %.3f N', FY_rms));

% 2.6 X1, X2 stay near zero: max abs < 10 um
X1_max = max(abs(q1(:,1)));
X2_max = max(abs(q1(:,2)));
[n_pass, n_fail] = chk(n_pass, n_fail, 'X1 max deviation < 10 um  (X ref = 0)', ...
    X1_max < 10e-6, ...
    sprintf('X1 max = %.2f um', X1_max*1e6));
[n_pass, n_fail] = chk(n_pass, n_fail, 'X2 max deviation < 10 um  (X ref = 0)', ...
    X2_max < 10e-6, ...
    sprintf('X2 max = %.2f um', X2_max*1e6));

% 2.7 Y tracking error in final hold period < 50 um
n_last = round(0.5 * fs);
Y_err_final = q1(end-n_last+1:end, 3) - r_sim(end-n_last+1:end, 3);
Y_err_max = max(abs(Y_err_final));
[n_pass, n_fail] = chk(n_pass, n_fail, 'Y tracking error (last 0.5 s) < 50 um', ...
    Y_err_max < 50e-6, ...
    sprintf('max = %.2f um', Y_err_max*1e6));

% 2.8 Forces within ETEL peak limits (datasheet)
FX1_max = max(abs(u_q1(:,1)));
FX2_max = max(abs(u_q1(:,2)));
FY_max  = max(abs(u_q1(:,3)));
[n_pass, n_fail] = chk(n_pass, n_fail, 'F_X1 within peak limit (< 2000 N)', ...
    FX1_max < 2000, sprintf('max = %.1f N', FX1_max));
[n_pass, n_fail] = chk(n_pass, n_fail, 'F_X2 within peak limit (< 2000 N)', ...
    FX2_max < 2000, sprintf('max = %.1f N', FX2_max));
[n_pass, n_fail] = chk(n_pass, n_fail, 'F_Y within peak limit (< 1420 N)', ...
    FY_max < 1420,  sprintf('max = %.1f N', FY_max));

% ------------------------------------------------------------------
% Summary
% ------------------------------------------------------------------
fprintf('\n%s\n', repmat('=', 1, 60))
fprintf('  RESULTS: %d passed,  %d failed\n', n_pass, n_fail)
if n_fail == 0
    fprintf('  ALL PASS\n')
else
    fprintf('  SOME FAILED -- review output above\n')
end
fprintf('%s\n\n', repmat('=', 1, 60))

% ------------------------------------------------------------------
% Local helper function
% ------------------------------------------------------------------
function [np, nf] = chk(np, nf, name, condition, detail)
    if condition
        fprintf('  [PASS]  %s\n', name);
        np = np + 1;
    else
        fprintf('  [FAIL]  %s\n          -> %s\n', name, detail);
        nf = nf + 1;
    end
end
