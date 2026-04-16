%% garcia_extended_lagrangian.m
% Step 1: Derives Garcia M, C, K from Lagrangian and verifies against
%         supervisor's matrices.
% Step 2: Extends with hidden mass ma at relative displacement delta_a
%         from mh, connected via spring ka and damper ca.

clear; clc;

%% -----------------------------------------------------------------------
%% SYMBOLIC VARIABLES
%% -----------------------------------------------------------------------

% Garcia coordinates and velocities
syms X Theta Y real
syms dX dTheta dY real

% Extended coordinate and velocity - hidden mass relative displacement
syms delta_a vdelta_a real    % delta_a = position, vdelta_a = velocity

% Garcia parameters
syms mb mh m1 m2 Jb Jh Lb d real
syms cg1 cg2 cy cb1 cb2 kb1 kb2 real

% New parameters for hidden mass
syms ma ka ca real

% Forces
syms F1 F2 Fy real

%% -----------------------------------------------------------------------
%% NUMERICAL VALUES
%% -----------------------------------------------------------------------

mb_n  = 22.8;    % Mass of the moving cross-arm (kg)
mh_n  = 10.1;    % Mass of the payload (Y-axis) (kg)
m1_n  = 10.2;    % Mass of actuator X1 (kg)
m2_n  = 10.7;    % Mass of actuator X2 (kg)
Jb_n  = 1.0;     % Rotary inertia of the cross-arm (kg.m^2)
Jh_n  = 0.05;    % Rotary inertia of the payload (Y-axis) (kg.m^2)
cg1_n = 14.5;    % Viscous friction of actuator X1 (N/(m/s))
cg2_n = 20.3;    % Viscous friction of actuator X2 (N/(m/s))
cy_n  = 10;      % Viscous friction of the payload (Y-axis) (N/(m/s))
cb1_n = 9;       % Viscous Friction of elastic joints 1 (Nm/(rad/s))
cb2_n = 9;       % Viscous Friction of elastic joints 2 (Nm/(rad/s))
kb1_n = 1987.5;  % Stiffness of elastic joint 1 (N.m/rad)
kb2_n = 1987.5;  % Stiffness of elastic joint 2 (N.m/rad)
Lb_n  = 0.725;   % Length of the moving cross-arm (m)
d_n   = 0.1;     % Distance between cross-arm and payload (m)
Y_n   = 0.3;     % Payload Y position for verification (m)

% Hidden mass parameters - chosen for visibility (tune later)
ma_n  = 1.0;     % Hidden mass (kg) - ~10% of mh
ka_n  = 500;     % Spring stiffness (N/m) - resonance ~3.6 Hz
ca_n  = 2;       % Damping (Ns/m)

%% -----------------------------------------------------------------------
%% STEP 1: GARCIA BASELINE ENERGIES (verified)
%% -----------------------------------------------------------------------

% Kinetic energy - Garcia equation (3), small angle applied
% cos(Theta)=1, sin(Theta)=0 per Garcia Section 2.3
T_mb = 0.5*mb*dX^2 + 0.5*Jb*dTheta^2;
T_m1 = 0.5*m1*(dX + Lb/2*dTheta)^2;
T_m2 = 0.5*m2*(dX - Lb/2*dTheta)^2;
T_mh = 0.5*mh*((dX - Y*dTheta)^2 + (dY - d*dTheta)^2) + 0.5*Jh*dTheta^2;

T_garcia = expand(T_mb + T_m1 + T_m2 + T_mh);

% Potential energy - Garcia equation (4)
V_garcia = 0.5*(kb1+kb2)*Theta^2;

% Rayleigh dissipation - Garcia equation (5), small angle applied
D_garcia = 0.5*(cg1+cg2)*dX^2 ...
         + (cg1-cg2)*(Lb/2)*dX*dTheta ...
         + 0.5*(cb1+cb2+(cg1+cg2)*Lb^2/4)*dTheta^2 ...
         + 0.5*cy*dY^2;

%% Verify Garcia baseline against supervisor's matrices
dq3 = [dX; dTheta; dY];
q3  = [X;  Theta;  Y];

M_garcia_sym = simplify(hessian(T_garcia, dq3));
K_garcia_sym = simplify(hessian(V_garcia, q3));
C_garcia_sym = simplify(hessian(D_garcia, dq3));

% Supervisor's matrices (exact copy)
M_supervisor = [ ...
    m1_n+m2_n+mb_n+mh_n, (m1_n-m2_n)*Lb_n/2-mh_n*Y_n, 0; ...
    (m1_n-m2_n)*Lb_n/2-mh_n*Y_n, Jb_n+Jh_n+(m1_n+m2_n)*Lb_n^2/4+mh_n*d_n^2+mh_n*Y_n^2, -mh_n*d_n; ...
    0, -mh_n*d_n, mh_n];
C_supervisor = [ ...
    cg1_n+cg2_n, (cg1_n-cg2_n)*Lb_n/2, 0; ...
    (cg1_n-cg2_n)*Lb_n/2, cb1_n+cb2_n+(cg1_n+cg2_n)*Lb_n^2/4, 0; ...
    0, 0, cy_n];
K_supervisor = [0,0,0; 0,kb1_n+kb2_n,0; 0,0,0];

params3 = {mb,mh,m1,m2,Jb,Jh,Lb,d,cg1,cg2,cy,cb1,cb2,kb1,kb2,Y};
values3 = {mb_n,mh_n,m1_n,m2_n,Jb_n,Jh_n,Lb_n,d_n,cg1_n,cg2_n,cy_n,cb1_n,cb2_n,kb1_n,kb2_n,Y_n};

M_garcia_num = double(subs(M_garcia_sym, params3, values3));
K_garcia_num = double(subs(K_garcia_sym, params3, values3));
C_garcia_num = double(subs(C_garcia_sym, params3, values3));

fprintf('=== STEP 1: Garcia baseline verification ===\n')
fprintf('Residual M (should be ~0): max abs = %.2e\n', max(abs(M_garcia_num - M_supervisor),[],'all'))
fprintf('Residual K (should be ~0): max abs = %.2e\n', max(abs(K_garcia_num - K_supervisor),[],'all'))
fprintf('Residual C (should be ~0): max abs = %.2e\n', max(abs(C_garcia_num - C_supervisor),[],'all'))

%% -----------------------------------------------------------------------
%% STEP 2: EXTENDED ENERGIES - add hidden mass ma
%% -----------------------------------------------------------------------

% Hidden mass ma sits at absolute position (Y + delta_a) along crossarm
% delta_a is the RELATIVE displacement from mh - same geometry as mh
% absolute x-velocity: dX - (Y+delta_a)*dTheta
% absolute y-velocity: dY + ddelta_a - d*dTheta
T_ma = 0.5*ma*((dX - (Y+delta_a)*dTheta)^2 + (dY + vdelta_a - d*dTheta)^2);

% Spring between mh and ma - only relative displacement delta_a
V_spring = 0.5*ka*delta_a^2;

% Damper between mh and ma - only relative velocity ddelta_a
D_damper = 0.5*ca*vdelta_a^2;

% Extended total energies
T = expand(T_garcia + T_ma);
V = V_garcia + V_spring;
D = D_garcia + D_damper;

%% -----------------------------------------------------------------------
%% STEP 3: EXTRACT EXTENDED MATRICES
%% -----------------------------------------------------------------------

dq4 = [dX; dTheta; dY; vdelta_a];
q4  = [X;  Theta;  Y;  delta_a];

M_ext = simplify(hessian(T, dq4));
K_ext = simplify(hessian(V, q4));
C_ext = simplify(hessian(D, dq4));

% Force vector - no direct force on hidden mass
f_ext = [F1+F2; (F1-F2)*Lb/2; Fy; 0];

fprintf('\n=== STEP 2: Extended symbolic matrices ===\n')
fprintf('--- M_ext ---\n'); disp(M_ext)
fprintf('--- K_ext ---\n'); disp(K_ext)
fprintf('--- C_ext ---\n'); disp(C_ext)

%% -----------------------------------------------------------------------
%% STEP 4: VERIFY - setting ma=0 must recover Garcia 3x3 exactly
%% -----------------------------------------------------------------------

M_check = simplify(subs(M_ext(1:3,1:3), ma, 0));
K_check = simplify(subs(K_ext(1:3,1:3), ma, 0));
C_check = simplify(subs(C_ext(1:3,1:3), ma, 0));

% Should match M_garcia_sym, K_garcia_sym, C_garcia_sym exactly
diff_M = simplify(M_check - M_garcia_sym);
diff_K = simplify(K_check - K_garcia_sym);
diff_C = simplify(C_check - C_garcia_sym);

fprintf('=== STEP 4: Recovery check (ma=0 must recover Garcia) ===\n')
fprintf('M diff (should be zero matrix): %s\n', mat2str(isequal(diff_M, sym(zeros(3)))))
fprintf('K diff (should be zero matrix): %s\n', mat2str(isequal(diff_K, sym(zeros(3)))))
fprintf('C diff (should be zero matrix): %s\n', mat2str(isequal(diff_C, sym(zeros(3)))))

%% -----------------------------------------------------------------------
%% STEP 5: EQUATIONS OF MOTION
%% -----------------------------------------------------------------------

ddq = simplify(M_ext \ (f_ext - C_ext*dq4 - K_ext*q4));

fprintf('\n=== STEP 5: Accelerations ddq = M_ext \\ (f - C*dq - K*q) ===\n')
fprintf('ddX     = '); disp(ddq(1))
fprintf('ddTheta = '); disp(ddq(2))
fprintf('ddY     = '); disp(ddq(3))
fprintf('ddelta_a= '); disp(ddq(4))