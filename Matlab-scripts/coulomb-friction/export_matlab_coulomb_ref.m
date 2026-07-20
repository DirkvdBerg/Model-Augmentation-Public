% export_matlab_coulomb_ref.m
% ---------------------------
% Independent reference for the Python LPV-LFR Coulomb implementation:
% integrate the supervisor's DIRECT EOM (gantrySystem, self-scheduled M(Y),
% numerical M\, no Coriolis -- the same LPV model as the Python baseline) with
% Coulomb friction added, on a prescribed open-loop input. Export to .mat so
% Python's simulate_coulomb (Cramer N(Y)/d(Y) through G) can be checked against
% it: same physics, different realization and language.
%
% Run from repo root (from MATLAB):
%   run('Matlab-scripts/coulomb-friction/export_matlab_coulomb_ref.m')

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
addpath(genpath(fullfile(repo, 'kamtin-fp-model', '03 Simulink gantry')));
addpath(here);

% Kamtin params (main.m) -- identical to the Python physics.py baseline
mb=22.8; mh=10.1; m1=10.2; m2=10.7; Jb=1.0; Jh=0.05;
cg1=14.5; cg2=20.3; cy=10; cb1=9; cb2=9; kb1=1987.5; kb2=1987.5;
Lb=0.725; d=0.1;
% THEORY: garcia2013 Coulomb magnitudes [N]
cc1=16.8; cc2=18.35; ccy=11.6; cc=[cc1;cc2;ccy];

P = [1, 1, 0; Lb/2, -Lb/2, 0; 0, 0, 1];

fs = 20000; ts = 1/fs; T = 0.3; nt = round(T*fs); t = (0:nt-1)'*ts;

% Prescribed open-loop STAGE force, per-axis sinusoids at different freqs so
% each axis velocity reverses sign (activates Coulomb) and axes stay decoupled.
u_stage = [60*sin(2*pi*7*t), 50*sin(2*pi*11*t), 40*sin(2*pi*9*t)];

% Base = gantrySystem (no Coriolis, LPV M(Y)) -- matches the Python LPV baseline.
baseFn = @(uu, xx) gantrySystem(uu, xx, m1,m2,mb,mh,Lb,Jb,Jh,d, ...
                                cg1,cg2,cb1,cb2,cy,kb1,kb2);

x = zeros(6,1);            % rest, Y=0
q = zeros(nt,3);           % stage positions [X1,X2,Y]
for k = 1:nt
    q(k,:) = (P.' * x(1:3)).';                 % logical pos -> stage
    u_log  = P * u_stage(k,:).';                % stage force -> logical
    f1 = gantrySystemCoulomb(baseFn, u_log, x,             Lb, cc);
    f2 = gantrySystemCoulomb(baseFn, u_log, x + ts/2*f1,   Lb, cc);
    f3 = gantrySystemCoulomb(baseFn, u_log, x + ts/2*f2,   Lb, cc);
    f4 = gantrySystemCoulomb(baseFn, u_log, x + ts*f3,     Lb, cc);
    x  = x + ts/6*(f1 + 2*f2 + 2*f3 + f4);
end

out = fullfile(here, 'matlab_coulomb_ref.mat');
save(out, 't', 'ts', 'fs', 'u_stage', 'q', 'cc', 'P', 'Lb');
fprintf('Saved %s  (nt=%d, |q|max=%.3e m)\n', out, nt, max(abs(q(:))));
