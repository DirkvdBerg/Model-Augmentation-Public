repo = fullfile(pwd, '..', '..');
addpath(fullfile(repo, 'Thesis-writeup', 'Code', 'Data'));
cfg = tdg_config(); fs = cfg.fs;
T = load(cfg.noise_target); f = T.f(2:end);
Gd = c2d(tdg_truth_lin(0, cfg, cfg.friction_gain), cfg.ts, 'zoh');
plant0 = gtd_build_plant(0, cfg, 1); S = feedback(eye(3), Gd * plant0.Cfb);
fq = [9.77 15 19.5 25 29.3 40 50 60 80 100 150 200 300 500 700 1000 1500 2000 3000 5000 9000];
Sq = freqresp(S, 2*pi*fq);
fprintf('f [Hz] | |S11| |S22| |S33| | measured e X1 X2 Y | required v = e/|S| X1 X2 Y\n');
for i = 1:numel(fq)
    [~, k] = min(abs(f - fq(i)));
    e = sqrt([real(T.P(k+1,1,1)) real(T.P(k+1,2,2)) real(T.P(k+1,3,3))]);
    s = abs([Sq(1,1,i) Sq(2,2,i) Sq(3,3,i)]);
    fprintf('%6.1f | %.2f %.2f %.2f | %.1e %.1e %.1e | %.1e %.1e %.1e\n', fq(i), s, e, e./s);
end
