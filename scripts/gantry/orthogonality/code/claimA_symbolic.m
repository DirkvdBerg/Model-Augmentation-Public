% Claim A, cross-checked against the MATLAB ground truth.
%
% Companion to claimA_symbolic.py. That script transcribes the matrices from
% model_augmentation/systems/gantry_ss.py, so on its own it proves only that
% the PYTHON model factors through the 10 identifiable combinations. If the
% Python transcription has drifted from the first-principles model, the proof
% would be about the wrong object.
%
% This script transcribes M, C, K from the ground truth instead:
%   kamtin-fp-model/03 Simulink gantry/main.m, lines 51-63
% (that folder is read-only; nothing here writes to it).
%
% Passing BOTH scripts means the factorization holds for the FP model and the
% Python implementation agrees with it on this property.
%
% Run: matlab -batch "run('scripts/gantry/orthogonality/code/claimA_symbolic.m')"
% Requires the Symbolic Math Toolbox.

clear; clc;
fprintf('Claim A cross-check against kamtin-fp-model main.m\n\n');

syms kb1 kb2 cg1 cg2 cy cb1 cb2 mh m1 m2 mb Jb Jh d Lb Y real

% --- transcribed verbatim from main.m lines 51-63 ------------------------
M = [          m1 + m2 + mb + mh,                          (m1 - m2) * Lb / 2 - mh * Y,        0;
     (m1 - m2) * Lb / 2 - mh * Y, Jb + Jh + (m1 + m2) * Lb^2 / 4 + mh * d^2 + mh * Y^2,  -mh * d;
                               0,                                              -mh * d,       mh];

C = [           cg1 + cg2,               (cg1 - cg2) * Lb / 2,  0;
     (cg1 - cg2) * Lb / 2, cb1 + cb2 + (cg1 + cg2) * Lb^2 / 4,  0;
                        0,                                  0, cy];

K = [0,         0, 0;
     0, kb1 + kb2, 0;
     0,         0, 0];

% Downstream ingredients of the rational inverse. Anything built from M, C, K
% is a function of these, so annihilating them settles the transition too.
detM = det(M);
adjM = adjoint(M);

mats  = {M, C, K, detM, adjM};
mnames = {'M(Y)', 'C', 'K', 'det(M)', 'adj(M)'};

pars  = [kb1 kb2 cg1 cg2 cy cb1 cb2 mh m1 m2 mb Jb Jh d];
pnames = {'kb1','kb2','cg1','cg2','cy','cb1','cb2','mh','m1','m2','mb','Jb','Jh','d'};

% --- the four predicted kernel directions, as coefficient vectors --------
V = sym(zeros(14, 4));
V(1,1) =  1;  V(2,1) = -1;                                   % v1: kb1 - kb2
V(6,2) =  1;  V(7,2) = -1;                                   % v2: cb1 - cb2
V(12,3) =  1; V(13,3) = -1;                                  % v3: Jb - Jh
V(9,4) =  1;  V(10,4) = 1; V(11,4) = -2; V(13,4) = -Lb^2/2;  % v4: mass-inertia
vnames = {'v1 kb1-kb2','v2 cb1-cb2','v3 Jb-Jh','v4 mass-inertia'};

% --- negative controls: must NOT be annihilated -------------------------
W = sym(zeros(14, 3));
W(1,1) = 1; W(2,1) = 1;                                      % kb1 + kb2
W(9,2) = 1;                                                  % m1 alone
W(9,3) = 1; W(10,3) = 1; W(11,3) = -2; W(13,3) = -Lb^2/3;    % v4, wrong coeff
wnames = {'kb1+kb2','m1 alone','v4 with Lb^2/3'};

    function D = dirderiv(Mx, v, pars)
        D = sym(zeros(size(Mx)));
        for i = 1:numel(pars)
            if v(i) ~= 0
                D = D + v(i) * diff(Mx, pars(i));
            end
        end
        D = simplify(D);
    end

fprintf('[A] PREDICTED DIRECTIONS: every entry must be identically zero\n');
allzero = true;
for k = 1:size(V,2)
    ok = true;
    for j = 1:numel(mats)
        D = dirderiv(mats{j}, V(:,k), pars);
        if any(D(:) ~= 0)
            ok = false;
            fprintf('    NONZERO in %s for %s\n', mnames{j}, vnames{k});
        end
    end
    allzero = allzero && ok;
    fprintf('  %-18s %s\n', vnames{k}, ternary(ok, 'ALL ZERO', 'FAILED'));
end

fprintf('\n[C] NEGATIVE CONTROLS: each must move at least one matrix\n');
negok = true;
for k = 1:size(W,2)
    moved = {};
    for j = 1:numel(mats)
        D = dirderiv(mats{j}, W(:,k), pars);
        if any(D(:) ~= 0)
            moved{end+1} = mnames{j}; %#ok<AGROW>
        end
    end
    ok = ~isempty(moved);
    negok = negok && ok;
    fprintf('  %-18s %s\n', wnames{k}, ...
        ternary(ok, ['moves ' strjoin(moved, ',')], 'FAILED: annihilated'));
end

% --- kernel dimension of d kappa ----------------------------------------
kappa = [kb1 + kb2; cg1; cg2; cy; cb1 + cb2; mh; ...
         m1 + m2 + mb; m1 - m2; Jb + Jh + (m1 + m2) * Lb^2 / 4; d];
Jk = jacobian(kappa, pars);
rk = double(rank(Jk));
fprintf('\n[D] rank(d kappa) = %d, dim ker = %d (predicted 4)\n', rk, 14 - rk);

verdict = allzero && negok && (14 - rk == 4);
fprintf('\nCLAIM A (MATLAB ground truth): %s\n', ternary(verdict, 'PROVED', 'NOT PROVED'));

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
