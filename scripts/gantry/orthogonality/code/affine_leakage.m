function affine_leakage
% AFFINE_LEAKAGE  MATLAB companion to affine_leakage.py (Rule 2, engine cross-check).
%
% SCOPE NOTE. Rule 2 clause 1 requires a dual proof for structural claims ABOUT THE FP
% MODEL, where MATLAB reads kamtin-fp-model/ as an independent source. This derivation
% contains no FP-model transcription: it is abstract algebra on a generic affine
% augmentation. So this file is an ENGINE cross-check (a wrong solve branch, a bad
% simplify), not the two-independent-sources check. Claimed at that strength only.
%
% INCIDENT, 2026-09-04, and the reason for the null-space parameterisation below.
% The first version of this file used solve() for both constraints. For a 4-equation,
% 8-unknown system it reported "solved 8 of 8 entries", i.e. it returned the TRIVIAL
% branch L = M = bf = 0, and likewise K = 0 in Part C. That made the K = 0 control
% vanish for the wrong reason and put MATLAB in disagreement with sympy on two rows.
% Both engines now parameterise each constraint by an explicit null-space basis, which
% is unique up to a change of basis and has no branch to pick. Recorded here because
% Rule 2 clause 6 requires engine failures to be written down rather than argued away.
%
% Mirrors the four parts of the Python:
%   A  pointwise orthogonality, imposed as an identity on the xa = 0 slice, leaves the
%      state-to-physical map K entirely free, and the windowed projection stays nonzero.
%      Control at K = 0 separates propagation (inherited from Gyorok) from the state path.
%   B  the missing term in closed form, verified against the symbolic rollout.
%   C  the extension  Psi_T' A^a K = 0  annihilates it identically.
%   D  the rank / Cayley-Hamilton count that decides penalty vs by-construction.
%
% Run:
%   matlab.exe -batch "cd('scripts/gantry/orthogonality/code'); affine_leakage"

NX = 2; NA = 1; NU = 1; NTH = 1;    % minimal, hand-checkable
T_LIST = [2 3 4];

A  = sym('A',  [NX NX],  'real');
G  = sym('G',  [NA NA],  'real');
F  = sym('F',  [NA NX],  'real');
Hm = sym('Hm', [NA NU],  'real');
bg = sym('bg', [NA 1 ],  'real');
K  = sym('K',  [NX NA],  'real');
L  = sym('L',  [NX NX],  'real');
M  = sym('M',  [NX NU],  'real');
bf = sym('bf', [NX 1 ],  'real');
Ph = sym('Ph', [NX NTH], 'real');   % baseline parameter sensitivity

S = struct('A',A,'G',G,'F',F,'Hm',Hm,'bg',bg,'K',K,'L',L,'M',M,'bf',bf);
bar = repmat('=',1,78);

%% ---------------------------------------------------------------- PART A
fprintf('\n%s\nPART A  NEGATIVE: pointwise does not bound the rollout contribution\n%s\n', bar, bar);

cons = [reshape(Ph.'*L, [],1); reshape(Ph.'*M, [],1); reshape(Ph.'*bf, [],1)];
fprintf('Pointwise constraint as an IDENTITY in (x,u) on the whole slice xa = 0:\n');
fprintf('    Ph.T ( L x + M u + bf ) = 0   for ALL x, u\n');
for r = 1:numel(cons)
    fprintf('        %s = 0\n', char(cons(r)));
end
assert(~any(has(cons, K(:))), 'K appears in the pointwise constraint -- setup is wrong');
fprintf('    K appears in NONE of them: the state-to-physical map is unconstrained.\n');

Nb = null(Ph.');                     % see the INCIDENT note above: no solve() here
nk = size(Nb,2);
Lf  = sym('lf',  [nk NX], 'real');
Mf  = sym('mf',  [nk NU], 'real');
bff = sym('bff', [nk 1 ], 'real');
Ls  = Nb*Lf;   Ms = Nb*Mf;   bfs = Nb*bff;
pwK = [L(:); M(:); bf(:)];
pwV = [Ls(:); Ms(:); bfs(:)];
assert(~nz(expand(Ph.'*Ls)) && ~nz(expand(Ph.'*Ms)) && ~nz(expand(Ph.'*bfs)), ...
       'null-space parameterisation does not satisfy the pointwise constraint');
fprintf('    ker(Ph.T) has dimension %d in R^%d; L, M, bf are parameterised by it, so\n', nk, NX);
fprintf('    the pointwise constraint holds by construction, not by a solve branch.\n');
fprintf('    verified: Ph.T L = Ph.T M = Ph.T bf = 0 identically after substitution.\n');
fprintf('    all %d entries of K remain free.\n', numel(K));

for t = 1:numel(T_LIST)
    T = T_LIST(t);
    [xs, us] = traj(T, NX, NU);
    dx   = rollout(S, T, xs, us);
    Psi  = psiT(A, Ph, T);
    proj = expand(subs(Psi.'*dx, pwK, pwV));
    prop = expand(subs(proj, K(:), zeros(numel(K),1)));   % control: ANN cannot read xa
    st   = expand(proj - prop);
    fprintf('\n  T = %d:  Psi_T.T dx_T with the pointwise constraint enforced exactly\n', T);
    fprintf('      nonzero overall                     : %d\n', nz(proj));
    fprintf('      CONTROL K = 0 (static, Gyorok)      : %d   <- propagation, NOT our gap\n', nz(prop));
    fprintf('      STATE PATH terms carrying K         : %d   <- our gap\n', nz(st));
end

%% ---------------------------------------------------------------- PART B
fprintf('\n%s\nPART B  THE MISSING TERM in closed form\n%s\n', bar, bar);
fprintf('    dx_T^state = sum_{i=0}^{T-2} Lam_T(i) g_i ,  g_i = F x_i + Hm u_i + bg\n');
fprintf('    Lam_T(i)   = sum_{j=i+1}^{T-1} A^(T-1-j) K G^(j-1-i)\n\n');

for t = 1:numel(T_LIST)
    T = T_LIST(t);
    [xs, us] = traj(T, NX, NU);
    dx   = rollout(S, T, xs, us);
    Psi  = psiT(A, Ph, T);
    proj = expand(subs(Psi.'*dx, pwK, pwV));
    st   = expand(proj - subs(proj, K(:), zeros(numel(K),1)));
    closed = sym(zeros(NX,1));
    for i = 0:T-2
        closed = closed + lamT(A,K,G,T,i) * (F*xs{i+1} + Hm*us{i+1} + bg);
    end
    closed = subs(expand(Psi.'*closed), pwK, pwV);
    ok = ~nz(closed - st);
    fprintf('  T = %d:  closed form reproduces the symbolic rollout exactly : %d\n', T, ok);
    assert(ok, 'closed form mismatch at T=%d', T);
end

%% ---------------------------------------------------------------- PART C
fprintf('\n%s\nPART C  THE EXTENSION: orthogonality on the state-to-physical map K\n%s\n', bar, bar);
fprintf('    Psi_T.T A^a K = 0 ,  a = 0 ... T-2                              (EXT)\n');
fprintf('    <=> range(K) orthogonal to span{ Psi_T, A.T Psi_T, ... }\n');
fprintf('    At n_a = 0 there is no K, (EXT) is vacuous, method collapses to Gyorok.\n\n');

for t = 1:numel(T_LIST)
    T   = T_LIST(t);
    Psi = psiT(A, Ph, T);
    Sk  = sym([]);
    for a = 0:T-2
        Sk = [Sk; Psi.'*A^a]; %#ok<AGROW>
    end
    Nk = null(Sk);
    dk = size(Nk,2);
    if dk == 0
        extV = sym(zeros(numel(K),1));   % only K = 0 satisfies (EXT)
        triv = true;
    else
        Kf   = sym('kf', [dk NA], 'real');
        Kp   = Nk*Kf;
        extV = Kp(:);
        triv = ~nz(Kp);
    end

    [xs, us] = traj(T, NX, NU);
    dx   = rollout(S, T, xs, us);
    proj = expand(subs(Psi.'*dx, pwK, pwV));
    st   = expand(proj - subs(proj, K(:), zeros(numel(K),1)));
    killed = ~nz(subs(st, K(:), extV));

    fprintf('  T = %d:  %d scalar conditions on %d entries of K, dim ker(S_T) = %d\n', ...
            T, size(Sk,1)*NA, numel(K), dk);
    fprintf('          state-path term annihilated identically : %d\n', killed);
    fprintf('          forced K = 0 (condition vacuous?)       : %d\n', triv);
    assert(killed, 'EXT fails to annihilate the state path at T=%d', T);
end

%% ---------------------------------------------------------------- PART D
fprintf('\n%s\nPART D  BY-CONSTRUCTION OR PENALTY? the counting that decides it\n%s\n', bar, bar);
fprintf('(EXT): range(K) in ker(S_T), S_T the Krylov stack of Psi_T under A.T\n');
fprintf('Cayley-Hamilton: A.T has degree n_x, so the span saturates at a = n_x - 1.\n\n');
for T = 2:NX+3
    Psi = psiT(A, Ph, T);
    St  = sym([]);
    for a = 0:T-2
        St = [St; Psi.'*A^a]; %#ok<AGROW>
    end
    r = rank(St);
    fprintf('    T = %d:  stack %dx%d   rank = %d   dim ker = %d\n', ...
            T, size(St,1), size(St,2), r, NX - r);
end
fprintf('\n  Saturation is at rank = n_x, so dim ker(S_T) = 0 and (EXT) forces K = 0.\n');
fprintf('  A by-construction K = (I - Pi) Ktilde would delete the additional-state\n');
fprintf('  pathway entirely. The extension must be a SOFT PENALTY over realised data.\n');

fprintf('\nMATLAB run complete: every assertion passed.\n');
end

% ---------------------------------------------------------------- helpers
function b = nz(e)
b = ~isequal(simplify(expand(e)), sym(zeros(size(e))));
end

function [xs, us] = traj(T, NX, NU)
xs = cell(T,1); us = cell(T,1);
for k = 1:T
    xs{k} = sym(sprintf('x%d_', k-1), [NX 1], 'real');
    us{k} = sym(sprintf('u%d_', k-1), [NU 1], 'real');
end
end

function dx = rollout(S, T, xs, us)
% Cumulative ANN contribution to the physical state after T steps, from xa_0 = 0.
xa = sym(zeros(size(S.G,1),1));
f  = cell(T,1);
for k = 1:T
    f{k} = expand(S.K*xa + S.L*xs{k} + S.M*us{k} + S.bf);
    xa   = expand(S.G*xa + S.F*xs{k} + S.Hm*us{k} + S.bg);
end
dx = sym(zeros(size(S.A,1),1));
for j = 1:T
    dx = dx + S.A^(T-j) * f{j};
end
dx = expand(dx);
end

function P = psiT(A, Ph, T)
P = sym(zeros(size(Ph)));
for j = 0:T-1
    P = P + A^(T-1-j) * Ph;
end
P = expand(P);
end

function Lm = lamT(A, K, G, T, i)
Lm = sym(zeros(size(K)));
for j = i+1:T-1
    Lm = Lm + A^(T-1-j) * K * G^(j-1-i);
end
Lm = expand(Lm);
end
