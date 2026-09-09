% CAS benchmark, MATLAB side. Companion to cas_benchmark.py.
%
% Identical task list, but M, C, K are transcribed from the ground truth
% kamtin-fp-model/03 Simulink gantry/main.m lines 51-63 rather than from the
% Python implementation, so this run also re-checks provenance.
%
% Timings exclude engine startup on both sides.
%
% Run: matlab -batch "cd('scripts/gantry/orthogonality/code'); cas_benchmark"

clear; clc;
syms kb1 kb2 cg1 cg2 cy cb1 cb2 mh m1 m2 mb Jb Jh d Lb Y real

M = [          m1 + m2 + mb + mh,                          (m1 - m2) * Lb / 2 - mh * Y,        0;
     (m1 - m2) * Lb / 2 - mh * Y, Jb + Jh + (m1 + m2) * Lb^2 / 4 + mh * d^2 + mh * Y^2,  -mh * d;
                               0,                                              -mh * d,       mh];

C = [           cg1 + cg2,               (cg1 - cg2) * Lb / 2,  0;
     (cg1 - cg2) * Lb / 2, cb1 + cb2 + (cg1 + cg2) * Lb^2 / 4,  0;
                        0,                                  0, cy];

K = [0, 0, 0; 0, kb1 + kb2, 0; 0, 0, 0];
M1 = [0, -mh, 0; -mh, 0, 0; 0, 0, 0];
M2 = [0, 0, 0; 0, mh, 0; 0, 0, 0];

fprintf('MATLAB %s, Symbolic Toolbox\n', version('-release'));
fprintf('\ntimings (engine startup excluded):\n');

t = tic; detM = expand(det(M));            t1 = toc(t);
fprintf('  %-34s %8.3f s\n', 'T1 det(M(Y)) expanded', t1);

t = tic; adjM = expand(adjoint(M));        t2 = toc(t);
fprintf('  %-34s %8.3f s\n', 'T2 adj(M(Y)) expanded', t2);

t = tic; Minv = simplify(inv(M));          t3 = toc(t);
fprintf('  %-34s %8.3f s\n', 'T3 simplify(M(Y)^-1)', t3);

t = tic;
Mi = adjM / detM;
P1 = simplify(Mi * K); P2 = simplify(Mi * C);
P3 = simplify(Mi * M1); P4 = simplify(Mi * M2);
t4 = toc(t);
fprintf('  %-34s %8.3f s\n', 'T4 simplify(M^-1 X), X in K,C,M1,M2', t4);

t = tic;
cf = coeffs(collect(detM, Y), Y, 'All');
degY = numel(cf) - 1;
t5 = toc(t);
fprintf('  %-34s %8.3f s\n', 'T5 degree of det(M(Y)) in Y', t5);

total = t1 + t2 + t3 + t4 + t5;
fprintf('  %-34s %8.3f s\n', 'TOTAL', total);

% Side result: does the LFR rational rewrite close at second order in Y?
adjdeg = zeros(1, 9); n = 0;
for i = 1:3
    for j = 1:3
        n = n + 1;
        c = coeffs(collect(expand(adjM(i,j)), Y), Y, 'All');
        adjdeg(n) = numel(c) - 1;
    end
end
fprintf('\ndet(M(Y)) degree in Y      : %d\n', degY);
fprintf('adj(M(Y)) entry degrees in Y: %s\n', mat2str(unique(adjdeg)));
