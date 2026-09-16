%% surrogate_obc_symbolic.m
% Leg 1 of the verification plan for the state-level orthogonal-by-construction
% augmentation: exact symbolic checks on a minimal surrogate that carries every
% structural feature of the gantry augmentation and nothing else.
%
% Surrogate (all features it must have, and why):
%   x_p = [p; v]      two physical states
%   x_a = a           ONE additional state, for which the baseline defines no equation
%   theta = [t1;t2;t3]  physical parameters. t1,t2 enter only as the sum (t1+t2),
%                       mimicking kb1+kb2 / cb1+cb2 / Jb+Jh -> rank deficiency.
%                       t3 is the scheduling-dependent mass, entering RATIONALLY,
%                       mimicking M(Y)^-1 -> baseline NOT linear in the parameters.
%   scheduling        the mass depends on p, a STATE, mimicking Y = x_p(3) self-scheduling
%   routing           the learned block writes the velocity row and the latent row only
%   output            y = p, discarding v, mimicking C keeping 3 of 6 physical rows
%   learned block     g_eta(x,u) = W*[p;v;a;u] + b, affine in its weights so that
%                     every derivative below is exact and small
%
% Numbers are used where the result does not depend on them (Ts, c0, nominal theta,
% the reference inputs) and symbols where it does (the network weights). All arithmetic
% is exact rational / symbolic, never floating point.
%
% Checks, in order:
%   1  Orthogonality of the construction holds at the actual (rank deficient) regressor.
%   2  Reduction: with the offset column dropped and a full-rank regressor, our
%      expressions ARE Gyorok Eqs. (8), (10), (11).
%   3  Zero slice: the fitted coefficient is exactly independent of every weight that
%      reads or writes the latent state, so the projection never charges the latent path.
%   4  Non-propagation: the one-step condition can hold EXACTLY while the accumulated
%      output contribution of the learned component is non-orthogonal to the baseline
%      parameter sensitivity over a horizon.
%
% Run:  matlab -batch "run('surrogate_obc_symbolic.m')"

clear; clc;
fprintf('=== surrogate_obc_symbolic ===\n\n');

%% ---------------------------------------------------------------- constants
Ts   = sym(1)/10;      % step
c0   = sym(1)/2;       % fixed viscous damping, NOT a parameter
thb  = [sym(1); sym(1); sym(2)];   % nominal theta_bar: mass = t3 + p = 2 + p
nth  = 3;

syms t1 t2 t3 real
theta = [t1; t2; t3];

% Baseline one-step physical map (explicit Euler), rows = [p; v].
% Depends on theta only, never on the latent state: the baseline cannot see a.
fbase = @(p, v, u, th) [ p + Ts*v ; ...
                         v + Ts*( u - (th(1)+th(2))*p - c0*v ) / (th(3) + p) ];

% Routing. n_w = 2 learned outputs: w1 -> physical velocity row, w2 -> latent row.
Rp = sym([0 0; 1 0]);     % 2 x 2, writes the velocity row only
Ra = sym([0 1]);          % 1 x 2, writes the latent row
Cy = sym([1 0]);          % output keeps p, discards v

% Learned block, affine in its parameters: g(x,u) = W*[p;v;a;u] + b
syms W11 W12 W13 W14 W21 W22 W23 W24 b1 b2 real
W = [W11 W12 W13 W14; W21 W22 W23 W24];
b = [b1; b2];
gfun = @(p, v, a, u) W*[p; v; a; u] + b;

%% ------------------------------------------------- the frozen reference set
% Z_ref is the rollout of the BASELINE ALONE at theta_bar: learned block gated off,
% latent row therefore never written. This is how D-185 specifies the point set, and
% it is the origin of the zero slice found in check 3.
Mref  = 3;
uref  = [sym(1); sym(-1); sym(1)/2];
xp    = [sym(0); sym(0)];
aref  = sym(zeros(Mref,1));    % latent state on the reference rollout: identically zero
Pts   = sym(zeros(Mref,3));    % rows: [p, v, u]
for i = 1:Mref
    Pts(i,:) = [xp(1), xp(2), uref(i)];
    xp = subs(fbase(xp(1), xp(2), uref(i), theta), theta, thb);
end
fprintf('Reference points [p v u] (baseline-only rollout, latent state = 0 at every point):\n');
disp(Pts);

%% --------------------------------------------- baseline regressor and offset
% Nonlinear parameter dependence: expand at theta_bar with the evaluation pair held
% fixed (the device of gyorok2025l4dc, Sect. 4).
%   Phi_p = d f / d theta at theta_bar        (2 x 3 per point)
%   c     = f(theta_bar) - Phi_p * theta_bar  (2 x 1 per point)  the offset column
Phi = sym([]); off = sym([]);
for i = 1:Mref
    p = Pts(i,1); v = Pts(i,2); u = Pts(i,3);
    fi  = fbase(p, v, u, theta);
    Ji  = subs(jacobian(fi, theta), theta, thb);
    ci  = subs(fi, theta, thb) - Ji*thb;
    Phi = [Phi; Ji];        %#ok<AGROW>
    off = [off; ci];        %#ok<AGROW>
end
Phit = simplify([Phi off]);          % extended regressor, 2*Mref x (nth+1)
fprintf('Extended regressor Phi_tilde (%dx%d):\n', size(Phit,1), size(Phit,2));
disp(Phit);

% The offset column is not zero: the baseline map is affine, not homogeneous, in theta.
fprintf('offset column identically zero? %d  (0 expected: the offset is a real column)\n\n', ...
        isequal(simplify(off), sym(zeros(size(off)))));

%% ------------------------------- the rank deficiency, proved not assumed
dupe = simplify(Phit(:,1) - Phit(:,2));
report('columns of t1 and t2 are identical (rank deficiency, exact)', dupe);
fprintf('  rank(Phi_tilde) = %d of %d columns\n\n', double(rank(Phit)), size(Phit,2));

%% ================================================================ CHECK 1
% Orthogonality at the actual rank-deficient regressor.
%
% The condition is the NORMAL EQUATIONS: Phi' * (G - Phi*eta_aux) = 0. Any solution of
% Phi'*Phi*eta_aux = Phi'*G gives orthogonality, so rank deficiency costs the UNIQUENESS
% of eta_aux and nothing else. Here the duplicate column is dropped to get a full-column-
% rank basis B with range(B) = range(Phi_tilde); the residual is then checked against the
% FULL Phi_tilde, which is what the condition actually asks for.
fprintf('--- CHECK 1: orthogonality of the construction at the true (deficient) rank\n');
Gref = sym(zeros(2*Mref,1));
for i = 1:Mref
    Gref(2*i-1:2*i) = Rp * gfun(Pts(i,1), Pts(i,2), aref(i), Pts(i,3));
end
B      = Phit(:,[1 3 4]);                    % duplicate t2 column removed
etaB   = (B.'*B) \ (B.'*Gref);               % exact symbolic solve
Gtil   = Gref - B*etaB;                      % projected learned write on Z_ref
report('Phi_tilde'' * G_tilde = 0 on the reference set', simplify(Phit.'*Gtil));
fprintf('  ||G_eta_ref|| (unprojected, for scale) = %s\n\n', char(simplify(Gref.'*Gref)));

%% ================================================================ CHECK 2
% Reduction to the published expressions. Switch off the extra structure: no latent
% state, no offset column, full-rank regressor. Our expressions must BE Gyorok's.
fprintf('--- CHECK 2: reduction to Gyorok Eqs. (8), (10), (11)\n');
A = sym('A', [4 2]); Gg = sym('Gg', [4 1]);   % generic full-rank regressor and write
aux   = (A.'*A) \ (A.'*Gg);                             % Eq. (8)
Gt_g  = (eye(4) - A*((A.'*A) \ A.')) * Gg;              % Eq. (10)
report('Eq. (10) equals G - A*Eq.(8), i.e. the two forms coincide', simplify(Gt_g - (Gg - A*aux)));
report('Eq. (11): A'' * G_tilde = 0', simplify(A.'*Gt_g));
fprintf('\n');

%% ================================================================ CHECK 3
% The zero slice. Because Z_ref comes from the baseline-only rollout, the latent state
% is zero at every reference point, so the fitted coefficient cannot depend on any weight
% that reads the latent state (W13) or writes it (W21..W24, b2).
fprintf('--- CHECK 3: the fitted coefficient is blind to the whole latent path\n');
blind = [W13, W21, W22, W23, W24, b2];
bn    = {'W13 (a -> physical)','W21','W22','W23','W24','b2 (-> latent row)'};
for j = 1:numel(blind)
    report(sprintf('d eta_aux / d %s = 0', bn{j}), simplify(diff(etaB, blind(j))));
end
% and the same derivative is NOT zero along a trajectory where the latent state is live
syms pk vk ak uk real
dwr = simplify(diff(Rp*gfun(pk, vk, ak, uk), W13));
fprintf('  d(learned physical write)/dW13 along a trajectory = [%s; %s]  (nonzero when a ~= 0)\n\n', ...
        char(dwr(1)), char(dwr(2)));

%% ================================================================ CHECK 4
% Non-propagation. Take the sparsest live latent path: p feeds the latent state, the
% latent state feeds the physical velocity row. Then the learned write is IDENTICALLY
% ZERO on the reference set, so the one-step condition holds exactly and trivially,
% while the contribution to the OUTPUT over a horizon is nonzero and generically
% non-orthogonal to the baseline parameter sensitivity.
fprintf('--- CHECK 4: one-step orthogonality does not survive accumulation\n');
syms w13 w21 real
Wc = sym(zeros(2,4)); Wc(1,3) = w13; Wc(2,1) = w21;   % only a->v and p->a
bc = sym(zeros(2,1));
gc = @(p, v, a, u) Wc*[p; v; a; u] + bc;

% the one-step condition, for this choice, on the reference set
Gc = sym(zeros(2*Mref,1));
for i = 1:Mref
    Gc(2*i-1:2*i) = Rp * gc(Pts(i,1), Pts(i,2), aref(i), Pts(i,3));
end
etaC = (B.'*B) \ (B.'*Gc);
report('the learned write is exactly zero on Z_ref, so Phi'' * G_tilde = 0 holds trivially', ...
       simplify(Phit.'*(Gc - B*etaC)));

% now roll out 3 steps and compare the output with and without the learned component
% The horizon must be long enough for the latent path to REACH the output:
% a(k+1) = w21*p(k), the latent state enters the velocity row one step later, and the
% output reads the position, which is one integration further still. A nonzero initial
% position starts the chain immediately.
T   = 6;
uu  = [sym(1); sym(-1); sym(1)/2; sym(0); sym(1)/4; sym(-1)/2];
x0  = [sym(1)/5; sym(-1)/10];
xa_ = x0; aa_ = sym(0);                    % augmented rollout
xb_ = x0;                                  % baseline-only rollout
Ya  = sym(zeros(T,1)); Yb = sym(zeros(T,1));
for k = 1:T
    Ya(k) = Cy*xa_;  Yb(k) = Cy*xb_;
    wk    = gc(xa_(1), xa_(2), aa_, uu(k));
    xa_n  = fbase(xa_(1), xa_(2), uu(k), theta) + Rp*wk;
    aa_   = Ra*wk;
    xa_   = xa_n;
    xb_   = fbase(xb_(1), xb_(2), uu(k), theta);
end
dout = simplify(subs(Ya - Yb, theta, thb));           % learned contribution to the output
S    = simplify(subs(jacobian(Yb, theta), theta, thb)); % baseline output parameter sensitivity
fprintf('  learned contribution to the stacked output d_out = \n'); disp(dout);
fprintf('  baseline output parameter sensitivity S (%dx%d) = \n', size(S,1), size(S,2)); disp(S);
ip = simplify(S.'*dout);
fprintf('  S'' * d_out = \n'); disp(ip);
isz = isequal(ip, sym(zeros(size(ip))));
fprintf('  identically zero? %d  -> %s\n', isz, ...
    ternary(isz, 'FAIL: expected a nonzero expression', ...
                 'PASS: nonzero, so the one-step condition is NOT sufficient at the output'));

fprintf('\n=== done ===\n');

%% ---------------------------------------------------------------- helpers
function report(name, expr)
    r  = simplify(expr);
    ok = isequal(r, sym(zeros(size(r))));
    if ok
        fprintf('  [PASS] %s\n', name);
    else
        fprintf('  [FAIL] %s\n         residual = %s\n', name, char(r(:).'));
    end
end

function s = ternary(c, a, b)
    if c, s = a; else, s = b; end
end
