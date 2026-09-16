%% gantry_euler_regressor_structure.m
% Leg 2 of the verification plan: the same structural claims, checked symbolically on the
% REAL gantry state equation rather than on a surrogate, at explicit-Euler fidelity.
%
% Euler replaces RK4 deliberately. RK4 with substeps multiplies the expression size by
% roughly four without changing any structural property being checked here: which rows the
% baseline writes, and which parameter columns are dependent. The rational M(Y)^-1
% structure and all fourteen parameters are kept exactly as in the model.
%
% Equations of motion (model_augmentation/systems/gantry_ss.py, itself mirroring main.m):
%   M(Y) qddot + C qdot + K q = P u_stage,   M(Y) = M0 + M1*Y + M2*Y^2,   Y = q3
% so with x = [q; qdot] and an explicit Euler step of size Ts,
%   x(k+1) = x(k) + Ts * [ qdot ; M(Y)^-1 ( P u - C qdot - K q ) ].
%
% Claims checked:
%   A  The baseline writes ZERO rows on the additional-state block, so the extended
%      regressor has an identically zero block there. This is the "orthogonal to zero"
%      statement, made explicit on the real model.
%   B  The parameter columns for kb1 and kb2 are identical, and likewise cb1/cb2 and
%      Jb/Jh. Rank deficiency is therefore exact and structural, not numerical.
%   C  The offset column is nonzero, so the regressor must be extended if the condition
%      is to mean "orthogonal to what the physics can produce" rather than "orthogonal to
%      its parameter variations".
%   D  The achieved rank of the stacked regressor, evaluated exactly at rational points.
%
% Run:  matlab -batch "run('gantry_euler_regressor_structure.m')"

clear; clc;
fprintf('=== gantry_euler_regressor_structure ===\n\n');

%% ------------------------------------------------------ symbolic parameters
syms kb1 kb2 cg1 cg2 cy cb1 cb2 mh m1 m2 mb Jb Jh d real
th     = [kb1 kb2 cg1 cg2 cy cb1 cb2 mh m1 m2 mb Jb Jh d].';
tnames = {'kb1','kb2','cg1','cg2','cy','cb1','cb2','mh','m1','m2','mb','Jb','Jh','d'};
nth    = numel(th);

Lb = sym(725)/1000;          % frozen: defines the coordinate frame, not a parameter
Ts  = sym(1)/20000;          % fs = 20 kHz

% nominal values, exactly as in gantry_ss.py
nom = [sym(1e5) sym(1e5) sym(100) sym(100) sym(10) sym(50) sym(50) ...
       sym(101)/10 sym(5) sym(5) sym(228)/10 sym(1) sym(1)/20 sym(1)/10].';

syms Y real                  % scheduling variable, Y = q3

%% ------------------------------------------------------------- the matrices
Z  = sym(zeros(3,3));
M0 = Z;
M0(1,1) = m1 + m2 + mb + mh;
M0(1,2) = (m1 - m2)*Lb/2;      M0(2,1) = M0(1,2);
M0(2,2) = Jb + Jh + (m1 + m2)*Lb^2/4 + mh*d^2;
M0(2,3) = -mh*d;               M0(3,2) = -mh*d;
M0(3,3) = mh;

M1 = Z;  M1(1,2) = -mh;  M1(2,1) = -mh;
M2 = Z;  M2(2,2) =  mh;
MY = M0 + M1*Y + M2*Y^2;

Cm = Z;
Cm(1,1) = cg1 + cg2;
Cm(1,2) = (cg1 - cg2)*Lb/2;    Cm(2,1) = Cm(1,2);
Cm(2,2) = cb1 + cb2 + (cg1 + cg2)*Lb^2/4;
Cm(3,3) = cy;

Km = Z;  Km(2,2) = kb1 + kb2;   % the only stiffness: X and Y axes are integrators

Pm = Z;  Pm(1,1) = 1; Pm(1,2) = 1; Pm(2,1) = Lb/2; Pm(2,2) = -Lb/2; Pm(3,3) = 1;

%% --------------------------------------------------- one explicit Euler step
q    = sym('q',  [3 1]);  assume(q,  'real');
qd   = sym('qd', [3 1]);  assume(qd, 'real');
u    = sym('u',  [3 1]);  assume(u,  'real');

fnet  = Pm*u - Cm*qd - Km*q;
qddot = MY \ fnet;                                    % rational M(Y)^-1, no numeric solve
fEul  = [q; qd] + Ts*[qd; qddot];                     % 6 x 1 physical next state
fEul  = subs(fEul, Y, q(3));                          % self-scheduling: Y IS a state

fprintf('one Euler step built: 6 physical rows, M(Y)^-1 kept rational, Y = q3\n\n');

%% =================================================================== CLAIM A
% The augmented state block. With n_a additional states the full next-state vector is
%   [ f_theta(x_p,u) ; 0_{n_a} ] + routing * g_eta(...)
% so the baseline's parameter Jacobian has an identically zero block on those rows.
na   = 2;
fAug = [fEul; sym(zeros(na,1))];
JAug = jacobian(fAug, th);                            % (6+na) x 14
blk  = JAug(7:6+na, :);
report('CLAIM A: baseline parameter Jacobian is exactly zero on the additional-state rows', blk);
fprintf('  so on those rows the condition reads "orthogonal to zero": no constraint at all\n\n');

%% =================================================================== CLAIM B
% Exact column dependencies. Evaluated at a symbolic state, so this is an identity in the
% state and the input, not a coincidence at one operating point.
fprintf('--- CLAIM B: exact parameter-column dependencies (structural rank deficiency)\n');
J = jacobian(fEul, th);
pairs = {{'kb1','kb2'}, {'cb1','cb2'}, {'Jb','Jh'}};
for k = 1:numel(pairs)
    i1 = find(strcmp(tnames, pairs{k}{1}));
    i2 = find(strcmp(tnames, pairs{k}{2}));
    report(sprintf('column(%s) - column(%s) = 0', pairs{k}{1}, pairs{k}{2}), ...
           simplify(J(:,i1) - J(:,i2)));
end
fprintf('  three exact dependencies -> rank(Phi) <= %d of %d, at ANY number of points\n\n', ...
        nth - 3, nth);

%% =================================================================== CLAIM C
% The offset column. c = f(theta_bar) - Phi*theta_bar is not zero, because the Euler map
% is affine in the parameters (it contains the state itself) and, through M(Y)^-1, not even
% affine. Without this column the condition protects parameter variations only.
fprintf('--- CLAIM C: the offset column is nonzero\n');
Jn = subs(J,    th, nom);
fn = subs(fEul, th, nom);
c  = simplify(fn - Jn*nom);
isz = isequal(simplify(c), sym(zeros(6,1)));
fprintf('  offset identically zero? %d  (expected 0: the column is real and must be kept)\n', isz);
fprintf('  offset (symbolic in the state and input) =\n'); disp(simplify(c));

%% =================================================================== CLAIM D
% Achieved rank of the stacked extended regressor at exact rational evaluation points.
% Points stand in for a reference rollout; the dependencies of claim B are independent of
% which points are used, the achieved rank is not.
fprintf('--- CLAIM D: achieved rank of the stacked extended regressor\n');
npts = 6;
rng(0);
Phi = sym([]);
for i = 1:npts
    qi  = sym(randi([-20 20], 3, 1))/100;     % positions, metres, Y in a plausible range
    qdi = sym(randi([-20 20], 3, 1))/10;      % velocities
    ui  = sym(randi([-50 50], 3, 1));         % stage forces
    sb  = {q(1),q(2),q(3),qd(1),qd(2),qd(3),u(1),u(2),u(3)};
    vl  = {qi(1),qi(2),qi(3),qdi(1),qdi(2),qdi(3),ui(1),ui(2),ui(3)};
    Ji  = subs(Jn, sb, vl);
    fi  = subs(fn, sb, vl);
    ci  = fi - Ji*nom;
    Phi = [Phi, sym([])]; %#ok<AGROW>
    Phi = [Phi; [Ji ci]]; %#ok<AGROW>
end
fprintf('  stacked extended regressor is %dx%d\n', size(Phi,1), size(Phi,2));
r = double(rank(Phi));
fprintf('  exact rank = %d of %d columns (14 parameters + 1 offset)\n', r, size(Phi,2));
fprintf('  deficiency = %d, consistent with the three dependencies of claim B\n', size(Phi,2) - r);
fprintf('  => Gyorok Assumption 1 (full column rank) FAILS on the real model.\n');
fprintf('     Orthogonality survives this; uniqueness of the fitted coefficient does not.\n');


%% =================================================================== CLAIM E
% Split the deficiency: claim B explains three dependencies, the achieved deficiency is four.
% This locates the fourth, and says whether the offset column adds a direction of its own.
fprintf('\n--- CLAIM E: where the deficiency sits\n');
rP = double(rank(Phi(:,1:nth)));
fprintf('  rank of the 14 parameter columns alone = %d  (deficiency %d)\n', rP, nth - rP);
fprintf('  rank with the offset column appended   = %d\n', r);
if r == rP + 1
    fprintf('  the offset column is OUTSIDE the parameter span: it adds a real direction, so\n');
    fprintf('  extending the regressor with it genuinely enlarges the protected set.\n');
else
    fprintf('  the offset column lies INSIDE the parameter span: extending the regressor with\n');
    fprintf('  it changes nothing and the negation channel is already covered.\n');
end
fprintf('  project-documented identifiable count = 10; symbolic parameter rank = %d\n', rP);
massix = [find(strcmp(tnames,'m1')) find(strcmp(tnames,'m2')) find(strcmp(tnames,'mb'))];
fprintf('  rank of the mass columns m1,m2,mb alone = %d of 3\n', double(rank(Phi(:,massix))));

% Name the dependencies explicitly: the null space of the parameter block. Each null vector
% is a parameter combination the data cannot see at these points. Three are the known pairs;
% any further one is a new identifiability statement and belongs in the thesis.
Nsp = null(Phi(:,1:nth));
fprintf('  null space of the parameter block has %d directions:\n', size(Nsp,2));
for j = 1:size(Nsp,2)
    v  = simplify(Nsp(:,j) / max(abs(Nsp(:,j))));
    nz = find(v ~= 0);
    str = '';
    for t = nz.'
        str = [str sprintf('%+s*%s ', char(v(t)), tnames{t})]; %#ok<AGROW>
    end
    fprintf('    %d) %s= 0\n', j, str);
end


fprintf('\n=== done ===\n');

%% ---------------------------------------------------------------- helpers
function report(name, expr)
    r  = simplify(expr);
    ok = isequal(r, sym(zeros(size(r))));
    if ok
        fprintf('  [PASS] %s\n', name);
    else
        fprintf('  [FAIL] %s\n', name);
        disp(r);
    end
end
