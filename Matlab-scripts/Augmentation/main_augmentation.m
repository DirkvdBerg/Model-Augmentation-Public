showfig = false; 
addpath(genpath(pwd))
set(0,"DefaultTextInterpreter", "latex")

freqsHz = logspace(-1, 3, 1000);

% Equations of Motion of Simplified Dual-Drive Gantry Stage (H-Type)
% M ddq + C dq + K q = f
% q = [X, Theta, Y]^T

% Given parameters
mb = 22.8;   % Mass of the moving cross-arm (kg)
mh = 10.1;   % Mass of the payload (Y-axis) (kg)
m1 = 10.2;   % Mass of actuator X1 (kg)
m2 = 10.7;   % Mass of actuator X2 (kg)

Jb = 1.0;    % Rotary inertia of the cross-arm (kg.m^2)
Jh = 0.05;   % Rotary inertia of the payload (Y-axis) (kg.m^2)

cg1 = 14.5;  % Viscous friction of actuator X1 (N/(m/s))
cg2 = 20.3;  % Viscous friction of actuator X2 (N/(m/s))
cy = 10;     % Viscous friction of the payload (Y-axis) (N/(m/s))

cb1 = 9;     % Viscous Friction of elastic joints 1 (Nm/(rad/s))
cb2 = 9;     % Viscous Friction of elastic joints 2 (Nm/(rad/s))

cc1 = 16.8;  % Coulomb friction of actuator X1 (N)
cc2 = 18.35; % Coulomb friction of actuator X2 (N)
ccy = 11.6;  % Coulomb friction of the payload (Y-axis) (N)

kb1 = 1987.5; % Stiffness of elastic joint 1 (N.m/rad)
kb2 = 1987.5; % Stiffness of elastic joint 2 (N.m/rad)

Lb = 0.725;   % Length of the moving cross-arm (m)
Lh = 0.25;    % Length of the payload (m)
d = 0.1;      % Distance between cross-arm and payload (m)

% Decoupled Coordinates
% X = (X1 + X2) / 2;
% Theta = asin((X1 - X2) / Lb);
% Y = Y;

% Time Derivatives of Decoupled Coordinates
% dX = (dX1 + dX2) / 2;
% dTheta = (dX1 - dX2) / (2 * sqrt(1 - 1/4 *  (X1 - X2)^2));
% dY = dY;

% Indicate Y location;
Y = 0.3;

% Mass Matrix
M = [          m1 + m2 + mb + mh,                          (m1 - m2) * Lb / 2 - mh * Y,        0;
     (m1 - m2) * Lb / 2 - mh * Y, Jb + Jh + (m1 + m2) * Lb^2 / 4 + mh * d^2 + mh * Y^2,  -mh * d;
                               0,                                              -mh * d,       mh];

% Viscous Damping Matrix
C = [           cg1 + cg2,               (cg1 - cg2) * Lb / 2,  0;
     (cg1 - cg2) * Lb / 2, cb1 + cb2 + (cg1 + cg2) * Lb^2 / 4,  0;
                        0,                                  0, cy];

% Stiffness Matrix
K = [0,         0, 0;
     0, kb1 + kb2, 0;
     0,         0, 0];

% % Vector of Forces
% f = [F1 + F2 - cc1 * sign(dX1) - cc2 * sign(dX2);
%      (F1 - F2 - cc1 * sign(dX1) + cc2 * sign(dX2)) * Lb / 2;
%      Fy - ccy * sign(dY)];

n = 3;
[Psi, lambda] = eigs(K,M,n,'smallestabs');
order = [1 3 2]; 
Psi = Psi(:, order); % Put mode shapes in order X, Theta, Y 
lambda = lambda(order, order); 

% Normalize with respect to the 2-norm of the modes. 
% In the paper they normalize with respect to the diagonal, but then the
% decoupling does not work properly. 
Psi = Psi./vecnorm(Psi, 2, 1); 

% Note: In the paper the influence of the rotation on the Y-axis
% is neglected. Here it is not done, but to do so we could set Psi(3, 2) to
% 0. 
naturalFreqsHz = sqrt(diag(lambda)) / (2 * pi);

%% State-space form: 
sys = getss(n,M,C,K);
[sys.OutputName, sys.InputName] = deal(["X1", "Theta", "Y"]); 

frdObjs.LogicalCoordinatesPlant = frd(sys, freqsHz, FrequencyUnit = "Hz");

% Input Output Transformation
% X1 -> X;
% X2 -> Theta;
% Y -> Y;

P = [1    1     0;
     Lb/2 -Lb/2 0;
     0    0     1]; 
T = pinv(P.');

StageCoordinatesSystem = P.'*sys*P; 
frdObjs.StageCoordinatesPlant = P.'*frdObjs.LogicalCoordinatesPlant*P;
[frdObjs.StageCoordinatesPlant.OutputName, frdObjs.StageCoordinatesPlant.InputName] = deal(["X1", "X2", "Y"]); 

Ty_modal = Psi \ T;
Tu_modal = T.' / Psi.';

% From stage coordinates to modal: 
frdObjs.ModalCoordinatesPlant = Ty_modal * frdObjs.StageCoordinatesPlant * Tu_modal;

% From logical coordinates to modal: 
% frdObjs.ModalCoordinatesPlant = Psi \ frdObjs.LogicalCoordinatesPlant / Psi.';
[frdObjs.ModalCoordinatesPlant.OutputName, frdObjs.ModalCoordinatesPlant.InputName] = deal(["X", "Theta", "Y"]); 

BodeOpts = bodeoptions();
BodeOpts.PhaseVisible = 'off';
BodeOpts.MagUnits = 'abs'; 
BodeOpts.MagScale = 'log'; 
BodeOpts.FreqUnits = 'Hz'; 
linespec = ["-", "-.", "--"]; 

plantNames = ["Stage coordinates", "Logical coordinates", "Modal coordinates"]; 
plantFieldNames = ["StageCoordinatesPlant", "LogicalCoordinatesPlant", "ModalCoordinatesPlant"]; 
%%
if showfig
figure(1); clf
for j = 1: 3
    mybode(frdObjs.(plantFieldNames(j)),BodeOpts,linespec(j),freqsHz);
end
legend(plantNames, 'fontsize', 12, 'location', 'northoutside');
title('Plant', 'fontsize', 14); 

%%
figure(2); clf
for j = 1: 3
    mybode(rga1(frdObjs.(plantFieldNames(j))),BodeOpts,linespec(j),freqsHz);
end
legend(plantNames, 'fontsize', 12, 'location', 'northoutside');
title('Relative gain array', 'fontsize', 14); 

%%
fig = figure(3); clf
tiledlayout(3, 1)
for j = 1: 3
    plantFieldName = plantFieldNames(j); 
    for i = 1: 3
        nexttile(i)
        RGA = rga1(frdObjs.(plantFieldName)); 
        bodeplot(RGA(i,i), BodeOpts, linespec(j));grid on;xlim([freqsHz(1) freqsHz(end)]); hold on
    end
end
nexttile(1); title(''); 
legend(plantNames, "fontsize", 12, 'location', 'northoutside');

nexttile(1); ylim([0.5 2])
nexttile(2); ylim([0.5 2])

sgtitle('Relative gain array diagonals', 'fontsize', 14); 
end

%% Setpoint
fs = 20e3; 
ts = 1/fs;
pmax = 400e-3; % [m]
vmax = 2; % [m/s]
amax = 20; % [m/s^2]
jerkTime = 25e-3; %[s]
jmax = amax/jerkTime; % [m/s^3]
smax = Inf; 
order = 3; 
if jerkTime == 0
    order = 2; 
end

[pvajs] = thirdOrderSetpointETEL(pmax, vmax, amax, jmax, smax, ts); 
nt = size(pvajs, 1); 
t = ts*(0:nt-1)'; 

ynames = ["Position $[m]$", "Speed $[m/s]$", "Acceleration $[m/s^2]$", "Jerk $[m/s^3]$"]; 
xname = "Time [s]"; 

if showfig
ax = gobjects(order+1, 1); 
figure(1); clf; 
for j = 1: order+1
    ax(j) = nexttile; 
    plot(t, pvajs(:, j)); 
    xlabel(xname)
    ylabel(ynames(j))
    grid minor
end
linkaxes(ax, 'x'); 
end

%% Feedback
fbw = 100; 
Cfb = tf(num2cell(zeros(3)), num2cell(ones(3))); 
for j = 1: 3
    Cfb(j,j) = ruleOfThumb(fbw, StageCoordinatesSystem(j, j), ts); 
end
if showfig
figure(4); clf
bodeplot(StageCoordinatesSystem*d2c(Cfb, 'tustin'))
xlim([freqsHz(1) freqsHz(end)]); 
end

%% Simulation
mdl = 'gantry_2025a';

f = 0*randn(nt, n); 
r = repmat(pvajs(:, 1), 1, 3); 
r(:, 3) = -r(:, 3) + Y; 

sim(mdl); 

G = c2d(StageCoordinatesSystem, ts, 'zoh'); 

Cfb.InputName = "e";  
Cfb.OutputName = "ufb";

G.InputName = "u";  
G.OutputName = "q";

S1 = sumblk("e = r - q",n);
S2 = sumblk("u = ufb + f",n);

T = connect(G,Cfb,S1,S2,{"r","f"},{"q","e"});

simout = lsim(T, [r,f]-[zeros(nt, 2), Y*ones(nt, 1), zeros(nt, 3)], t); 
q3 = simout(:, 1:3); 
q3(:, 3) = q3(:, 3) + Y; 

figure(4); 
nexttile(1)
plot(t, q)
title('Simscape result')
nexttile(2)
plot(t, q-q1)
title('residual eom without coriolis-centripetal vs simscape')
nexttile(3)
plot(t, q-q2)
title('residual eom with coriolis-centripetal vs simscape')
nexttile(4)
plot(t, q-q3)
title('residual lsim vs simscape')

legend('x1', 'x2', 'y', "fontsize", 12, 'location', 'northoutside')
