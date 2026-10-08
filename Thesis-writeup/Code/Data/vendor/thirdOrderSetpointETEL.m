function [pvajs, position, speed, acceleration, jerk, snap] = ...
    thirdOrderSetpointETEL(d, vmax, amax, jmax, smax, Ts)
%{
Setpoint computation for the third-order motion profile
Written by Jasper Gerritsen

Based on: 
FIR Filters for Online Trajectory Planning with Time- and Frequency-Domain Specifications
Luigi Biagiotti, Claudio Melchiorri

Note that is generator is NOT time-optimal. For time-optimal generation use
"thirdOrderSetpoint". This generator is to mimic the setpoints as generated
by the ETEL motion system. 

The main difference is that for triangular acceleration trajectories its
maximum jerk value is 2*jmax. 
---------------------------------------------------------------------
Inputs:
                Note: for the following the preferred unit can be selected as long as they are internally consistent
  d             - Motion distance
  vmax          - Maximum speed
  amax          - Maximum acceleration
  jmax          - Maximum jerk
  Ts            - Sampling time in [s]

Outputs: 
  pvajs         - m-by-5 array of position, speed, acceleration, jerk and snap

Optional outputs: 
  position      - m-by-1 array of position
  speed         - m-by-1 array of speed
  acceleration  - m-by-1 array of acceleration
  jerk          - m-by-1 array of jerk
  snap          - m-by-1 array of snap
%}

% Enforce snap limitation via the jerk: 
jmax = min([smax*Ts/2, jmax]); 

T(1) = ceil(d/vmax/Ts)*Ts;                  % Speed time Tv [s]
T(2) = ceil(vmax/amax/Ts)*Ts;               % Acceleration time Ta
if T(1) < T(2) && d~=0
    % Enforce T(1) == T(2), see page 10 of [1] for details. 
    % d/vmax = vmax/amax
    vmax = sqrt(d*amax); 
end

T(1) = ceil(d/vmax/Ts)*Ts;                  % Speed time Tv [s]
T(2) = ceil(vmax/amax/Ts)*Ts;               % Acceleration time Ta
T(3) = ceil(amax/jmax/Ts)*Ts;               % Jerk time Tj 

N = round(sum(T)/Ts); 

% The rectangular pulse function mi is made equivalent to the m function 
% mentioned in the paper above. 

% It is equivalent to the symbolic function:
% mi = @(Ti) 1/Ti * rectangularPulse(Ts, Ti, Ts*(0:N-1))
mi = @(Ti) 1/Ti * rectPulse(1, round(Ti/Ts), N); 

order = 3; 
if T(3) == 0
    order = 2; 
end

position = d*ones(N, 1);  
for i = 1: order
    rk = conv(position, mi(T(i)))*Ts; 
    position = rk; 
end
motionend = 2+find((position==max(abs(position))), 1, 'first'); 
position = position(1:motionend);
speed = diff([0;position])/Ts; 
acceleration = diff([0;speed])/Ts; 
jerk = diff([0;acceleration])/Ts; 
snap = diff([0;jerk])/Ts; 

pvajs = [position, speed, acceleration, jerk, snap];
pvajs = [zeros(1, 5); pvajs]; 

end

function u = rectPulse(x0, x1, N)
u = zeros(N, 1); 
u(x0:x1) = 1; 
end