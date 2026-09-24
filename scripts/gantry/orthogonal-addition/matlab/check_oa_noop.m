function check_oa_noop()
% CHECK_OA_NOOP  Class A gate for gantrySystemOA (G3), outside Simulink.
%   A1  addition off (all oa_* zero, 4 empty absorber slots) reproduces gantrySystem.m
%       BIT-IDENTICALLY on the 6 physical states, and the 8 slot states have zero derivative.
%   A2  each stateless term, switched on alone, changes the derivative.
%   A3  the absorber EOM satisfies the coupled Lagrangian form it was derived from:
%         M qdd + SUM m_i b_i (b_i' qdd + dd_i'') + C qd + K q = u   and
%         m_i (b_i' qdd + dd_i'') + m_i w_i^2 d_i = 0
%       to round-off, at random states with all four slots active.
%   A4  the stateless terms enter as f_a = -(M_a qdd + C_a qd + K_a q): residual of
%         (M + M_a) qdd + (C + C_a) qd + (K + K_a) q = u  to round-off.
% Run: matlab -batch "addpath('scripts/gantry/orthogonal-addition/matlab'); check_oa_noop"

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(fileparts(here))));
addpath(fullfile(repo, 'kamtin-fp-model', '03 Simulink gantry', 'functions'));
addpath(here);

p = struct('m1',10.2,'m2',10.7,'mb',22.8,'mh',10.1,'Lb',0.725,'Jb',1.0,'Jh',0.05,'d',0.1, ...
           'cg1',14.5,'cg2',20.3,'cb1',9,'cb2',9,'cy',10,'kb1',1987.5,'kb2',1987.5);
args = {p.m1,p.m2,p.mb,p.mh,p.Lb,p.Jb,p.Jh,p.d,p.cg1,p.cg2,p.cb1,p.cb2,p.cy,p.kb1,p.kb2};
NS = 4;
Z3 = zeros(3); off = {Z3, Z3, Z3, Z3, Z3, Z3, Z3, zeros(1,NS), ones(1,NS), zeros(3,NS), zeros(3,NS)};   % OA-018: + oa_Ma2

rng(1);
nS = 200; worst = 0; worst_slot = 0;
for s = 1:nS
    x6 = [0.2*randn; 1e-3*randn; 0.3*(2*rand-1); 0.5*randn; 0.05*randn; 0.5*randn];
    u  = 100*randn(3,1);
    x  = [x6; zeros(2*NS,1)];
    ref = gantrySystem(u, x6, args{:});
    oa  = gantrySystemOA(u, x, args{:}, off{:});
    worst = max(worst, max(abs(oa(1:6) - ref)));
    worst_slot = max(worst_slot, max(abs(oa(7:end))));
end
fprintf('A1  addition off: max |OA - gantrySystem| over %d random states = %.3e  (slot states %.3e)\n', ...
        nS, worst, worst_slot);
A1 = (worst == 0) && (worst_slot == 0);

names = {'Ma0','Ma1','Ma2','Ca0','Ka0','Ka1','Ka2'};
A2 = true;
x6 = [0.1; 2e-4; 0.2; 0.3; 0.02; -0.4]; u = [50; -3; 20]; x = [x6; zeros(2*NS,1)];
base = gantrySystemOA(u, x, args{:}, off{:});
for k = 1:numel(names)
    on = off; E = zeros(3); E(1,3) = 1; E(3,1) = 1; E(1,1) = 1;
    on{k} = E;
    dd = max(abs(gantrySystemOA(u, x, args{:}, on{:}) - base));
    fprintf('A2  %s on alone: max change %.3e\n', names{k}, dd);
    A2 = A2 && dd > 0;
end

% A3: all four slots active, random geometry.
worst3 = 0; worst3b = 0;
for s = 1:50
    m = 0.2 + rand(1,NS); w = 2*pi*(50 + 300*rand(1,NS));
    b0 = randn(3,NS); b1 = randn(3,NS);
    x6 = [0.2*randn; 1e-3*randn; 0.3*(2*rand-1); 0.5*randn; 0.05*randn; 0.5*randn];
    da = 1e-5*randn(NS,1); dda = 1e-2*randn(NS,1);
    x = [x6; da; dda]; u = 100*randn(3,1);
    on = off; on{8} = m; on{9} = w; on{10} = b0; on{11} = b1;
    f = gantrySystemOA(u, x, args{:}, on{:});
    qdd = f(4:6); ddd = f(6+NS+(1:NS));
    Y = x6(3);
    M = [p.m1+p.m2+p.mb+p.mh, (p.m1-p.m2)*p.Lb/2 - p.mh*Y, 0;
         (p.m1-p.m2)*p.Lb/2 - p.mh*Y, p.Jb+p.Jh+(p.m1+p.m2)*p.Lb^2/4 + p.mh*p.d^2 + p.mh*Y^2, -p.mh*p.d;
         0, -p.mh*p.d, p.mh];
    C = [p.cg1+p.cg2, (p.cg1-p.cg2)*p.Lb/2, 0; (p.cg1-p.cg2)*p.Lb/2, p.cb1+p.cb2+(p.cg1+p.cg2)*p.Lb^2/4, 0; 0, 0, p.cy];
    K = [0 0 0; 0 p.kb1+p.kb2 0; 0 0 0];
    % d/dt dT/dqd with T = qd' M_r qd / 2 + SUM m_i (b_i' qd + dd_i)^2 / 2 and
    % M_r = M - SUM m_i b_i b_i' gives M_r qdd + SUM m_i b_i (b_i' qdd + dd_i'') = M qdd + SUM m_i b_i dd_i''.
    r = M*qdd + C*x6(4:6) + K*x6(1:3) - u;
    for i = 1:NS
        bi = b0(:,i) + Y*b1(:,i);
        r = r + m(i)*bi*ddd(i);
        ra = m(i)*(bi.'*qdd + ddd(i)) + m(i)*w(i)^2*da(i);
        worst3b = max(worst3b, abs(ra) / max(abs(m(i)*w(i)^2*da(i)), 1e-12));
    end
    worst3 = max(worst3, norm(r) / norm(u));
end
fprintf('A3  coupled Lagrangian residual: carrier %.3e (rel. to |u|), absorbers %.3e (rel.)\n', ...
        worst3, worst3b);
A3 = worst3 < 1e-10 && worst3b < 1e-8;

% A4: stateless terms as -(M_a qdd + C_a qd + K_a q).
worst4 = 0;
for s = 1:50
    x6 = [0.2*randn; 1e-3*randn; 0.3*(2*rand-1); 0.5*randn; 0.05*randn; 0.5*randn];
    u = 100*randn(3,1); Y = x6(3);
    S = @(a) (a + a.')/2;
    Ma0 = S(randn(3)); Ma1 = S(randn(3)); Ma2 = S(randn(3)); Ca0 = randn(3); Ka0 = S(1e3*randn(3));
    Ka1 = S(1e3*randn(3)); Ka2 = S(1e3*randn(3));
    on = off; on(1:7) = {Ma0, Ma1, Ma2, Ca0, Ka0, Ka1, Ka2};
    f = gantrySystemOA(u, [x6; zeros(2*NS,1)], args{:}, on{:});
    qdd = f(4:6);
    M = [p.m1+p.m2+p.mb+p.mh, (p.m1-p.m2)*p.Lb/2 - p.mh*Y, 0;
         (p.m1-p.m2)*p.Lb/2 - p.mh*Y, p.Jb+p.Jh+(p.m1+p.m2)*p.Lb^2/4 + p.mh*p.d^2 + p.mh*Y^2, -p.mh*p.d;
         0, -p.mh*p.d, p.mh];
    C = [p.cg1+p.cg2, (p.cg1-p.cg2)*p.Lb/2, 0; (p.cg1-p.cg2)*p.Lb/2, p.cb1+p.cb2+(p.cg1+p.cg2)*p.Lb^2/4, 0; 0, 0, p.cy];
    K = [0 0 0; 0 p.kb1+p.kb2 0; 0 0 0];
    r = (M + Ma0 + Y*Ma1 + Y^2*Ma2)*qdd + (C + Ca0)*x6(4:6) + (K + Ka0 + Y*Ka1 + Y^2*Ka2)*x6(1:3) - u;
    worst4 = max(worst4, norm(r)/norm(u));
end
fprintf('A4  stateless residual (M+M_a) qdd + (C+C_a) qd + (K+K_a) q - u: %.3e (rel.)\n', worst4);
A4 = worst4 < 1e-10;

fprintf('\nCHECK_OA_NOOP  A1 %s  A2 %s  A3 %s  A4 %s\n', pf(A1), pf(A2), pf(A3), pf(A4));
if ~(A1 && A2 && A3 && A4), error('check_oa_noop:fail', 'a Class A check failed'); end
end

function s = pf(b)
if b, s = 'PASS'; else, s = 'FAIL'; end
end
