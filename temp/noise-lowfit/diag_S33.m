repo = fullfile(pwd, '..', '..'); addpath(fullfile(repo, 'Thesis-writeup', 'Code', 'Data'));
cfg = tdg_config();
Gd = c2d(tdg_truth_lin(0, cfg, cfg.friction_gain), cfg.ts, 'zoh');
plant0 = gtd_build_plant(0, cfg, 1); S = feedback(eye(3), Gd * plant0.Cfb);
fq = 80:10:300; H = freqresp(S, 2*pi*fq);
fprintf('f [Hz]: '); fprintf('%5d', fq); fprintf('\n|S33| : '); fprintf('%5.2f', abs(squeeze(H(3,3,:)))); fprintf('\n');
fprintf('absorber: f_a = sqrt(ka/ma)/2pi = %.0f Hz\n', sqrt(cfg.ka / cfg.ma) / 2 / pi);
