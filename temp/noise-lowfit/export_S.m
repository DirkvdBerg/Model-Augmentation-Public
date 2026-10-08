% Export the simulation's output sensitivity S = (I + G K1)^-1 at Y = 0 (truth at rest, tanh slope), discrete ts.
repo = 'C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation';
addpath(fullfile(repo, 'Thesis-writeup', 'Code', 'Data'));
cfg = tdg_config();
plant0 = gtd_build_plant(0, cfg, 1);
Gd = c2d(tdg_truth_lin(0, cfg, cfg.friction_gain), cfg.ts, 'zoh');
S = feedback(eye(3), Gd * plant0.Cfb);
[A, B, C, D] = ssdata(S);
f = logspace(0, log10(9999), 3000)';
H = freqresp(S, 2*pi*f);
S2 = zeros(numel(f), 3, 3);
for r = 1:3, for c = 1:3, S2(:, r, c) = abs(squeeze(H(r, c, :))).^2; end, end
ts = cfg.ts;
save(fullfile(pwd, 'S_Y0.mat'), 'A', 'B', 'C', 'D', 'f', 'S2', 'ts');
fprintf('done, order %d\n', size(A, 1));
