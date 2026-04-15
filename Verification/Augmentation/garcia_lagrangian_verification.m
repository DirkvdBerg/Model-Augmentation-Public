%% garcia_lagrangian_verify.m
% Derives M, C, K from Garcia equations (3)(4)(5) via symbolic toolbox.
% Verifies against supervisor's matrices at Y = 0.3.

clear; clc;

%% Symbolic variables
syms mb mh m1 m2 Jb Jh Lb d real
syms cg1 cg2 cy cb1 cb2 kb1 kb2 real
syms X Theta Y real
syms dX dTheta dY real

%% Supervisor's numerical parameters (exact copy)
mb_n  = 22.8;
mh_n  = 10.1;
m1_n  = 10.2;
m2_n  = 10.7;
Jb_n  = 1.0;
Jh_n  = 0.05;
cg1_n = 14.5;
cg2_n = 20.3;
cy_n  = 10;
cb1_n = 9;
cb2_n = 9;
kb1_n = 1987.5;
kb2_n = 1987.5;
Lb_n  = 0.725;
d_n   = 0.1;
Y_n   = 0.3;

%% Supervisor's matrices at Y = 0.3 (exact copy from MATLAB file)
M_supervisor = [ ...
    m1_n + m2_n + mb_n + mh_n, ...
    (m1_n - m2_n) * Lb_n / 2 - mh_n * Y_n, ...
    0; ...
    (m1_n - m2_n) * Lb_n / 2 - mh_n * Y_n, ...
    Jb_n + Jh_n + (m1_n + m2_n) * Lb_n^2 / 4 + mh_n * d_n^2 + mh_n * Y_n^2, ...
    -mh_n * d_n; ...
    0, -mh_n * d_n, mh_n];

C_supervisor = [ ...
    cg1_n + cg2_n, (cg1_n - cg2_n) * Lb_n / 2, 0; ...
    (cg1_n - cg2_n) * Lb_n / 2, cb1_n + cb2_n + (cg1_n + cg2_n) * Lb_n^2 / 4, 0; ...
    0, 0, cy_n];

K_supervisor = [0, 0, 0; 0, kb1_n + kb2_n, 0; 0, 0, 0];

%% Garcia equation (3) — kinetic energy per body
% Small angle: cos(Theta)=1, sin(Theta)=0 per Garcia Section 2.3
% mh sits at position Y along crossarm, offset d perpendicular
% absolute x-velocity of mh: dX - Y*dTheta  (negative: rightward Y gives leftward moment)
% absolute y-velocity of mh: dY - d*dTheta
T_mb = 0.5*mb*dX^2 + 0.5*Jb*dTheta^2;
T_m1 = 0.5*m1*(dX + Lb/2*dTheta)^2;
T_m2 = 0.5*m2*(dX - Lb/2*dTheta)^2;
T_mh = 0.5*mh*((dX - Y*dTheta)^2 + (dY - d*dTheta)^2) + 0.5*Jh*dTheta^2;

T = expand(T_mb + T_m1 + T_m2 + T_mh);

%% Garcia equation (4) — potential energy
V = 0.5*(kb1 + kb2)*Theta^2;

%% Garcia equation (5) — Rayleigh dissipation
% Small angle: cos(Theta)=1
D = 0.5*(cg1 + cg2)*dX^2 ...
  + (cg1 - cg2)*(Lb/2)*dX*dTheta ...
  + 0.5*(cb1 + cb2 + (cg1 + cg2)*Lb^2/4)*dTheta^2 ...
  + 0.5*cy*dY^2;

%% Extract M, C, K via hessian
dq = [dX; dTheta; dY];
q  = [X;  Theta;  Y];

M_sym = simplify(hessian(T, dq));
K_sym = simplify(hessian(V, q));
C_sym = simplify(hessian(D, dq));

fprintf('=== Symbolic M ===\n'); disp(M_sym)
fprintf('=== Symbolic K ===\n'); disp(K_sym)
fprintf('=== Symbolic C ===\n'); disp(C_sym)

%% Substitute numerical values at Y = 0.3
params = {mb,    mh,    m1,    m2,    Jb,    Jh,    Lb,    d, ...
          cg1,   cg2,   cy,    cb1,   cb2,   kb1,   kb2,   Y};
values = {mb_n,  mh_n,  m1_n,  m2_n,  Jb_n,  Jh_n,  Lb_n,  d_n, ...
          cg1_n, cg2_n, cy_n,  cb1_n, cb2_n, kb1_n, kb2_n, Y_n};

M_num = double(subs(M_sym, params, values));
K_num = double(subs(K_sym, params, values));
C_num = double(subs(C_sym, params, values));

fprintf('=== Numerical M at Y=0.3 ===\n'); disp(M_num)
fprintf('=== Numerical K at Y=0.3 ===\n'); disp(K_num)
fprintf('=== Numerical C at Y=0.3 ===\n'); disp(C_num)

fprintf('=== Supervisor M at Y=0.3 ===\n'); disp(M_supervisor)
fprintf('=== Supervisor K at Y=0.3 ===\n'); disp(K_supervisor)
fprintf('=== Supervisor C at Y=0.3 ===\n'); disp(C_supervisor)

%% Residuals — all entries should be zero
fprintf('=== Residual M (should be all zero) ===\n'); disp(M_num - M_supervisor)
fprintf('=== Residual K (should be all zero) ===\n'); disp(K_num - K_supervisor)
fprintf('=== Residual C (should be all zero) ===\n'); disp(C_num - C_supervisor)