% export_zpk_bhl  Exact zeros, poles, gains of the BHL rows of the Telica controller (G3).
%   matlab -nojvm -sd telica-real/controller -batch export_zpk_bhl
% Reads  kamtin-data/dFeedbackControllersTelica.mat (6x6 diagonal zpk, Ts 5e-5; rows LX1 LX2 LY
%        RX1 RX2 RY, supervisor-confirmed). Read only.
% Writes telica-real/controller/telica_zpk_bhl.mat: z{1..3}, p{1..3}, k(1..3), Ts, rows, and the
%        off-diagonal check (every off-diagonal entry of the 6x6 must be identically zero).
here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
S = load(fullfile(repo, 'kamtin-data', 'dFeedbackControllersTelica.mat'));
fn = fieldnames(S);
fprintf('[export_zpk_bhl] variables in file: %s\n', strjoin(fn', ', '));
K = S.(fn{1});
fprintf('[export_zpk_bhl] class %s, size %s, Ts %g\n', class(K), mat2str(size(K)), K.Ts);
[Z, Pp, Kg] = zpkdata(K);
rows = {'LX1', 'LX2', 'LY'};
z = cell(1, 3); p = cell(1, 3); k = zeros(1, 3);
for i = 1:3
    z{i} = Z{i, i}; p{i} = Pp{i, i}; k(i) = Kg(i, i);
    fprintf('%s: %d zeros, %d poles, k = %.10e, max|p| = %.12f, poles at |z|=1: %d\n', ...
        rows{i}, numel(z{i}), numel(p{i}), k(i), max(abs(p{i})), sum(abs(abs(p{i}) - 1) < 1e-12));
end
off = 0;
for i = 1:size(K, 1)
    for j = 1:size(K, 2)
        if i ~= j && Kg(i, j) ~= 0
            off = off + 1;
        end
    end
end
fprintf('[export_zpk_bhl] nonzero off-diagonal entries: %d\n', off);
Ts = K.Ts;
save(fullfile(here, 'telica_zpk_bhl.mat'), 'z', 'p', 'k', 'Ts', 'rows', 'off', '-v7');
fprintf('[export_zpk_bhl] saved telica_zpk_bhl.mat\n');
