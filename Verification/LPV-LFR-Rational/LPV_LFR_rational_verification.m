%% verify_lfr_loop_symbolic.m
%
% Verification of the LPV-LFR loop matrix derivation.
%
% FIXES FROM PREVIOUS VERSION
%
%   Root cause of C2/C3 failures:
%     ga (gamma) was defined WITH mh*d^2 absorbed, but the document's
%     det and adj formulas use gamma WITHOUT mh*d^2.  Specifically:
%
%     N0(1,1) is the (1,1) cofactor of M(0):
%       det([ga+mhd^2, -mhd; -mhd, mh]) = mh*(ga+mhd^2) - mh^2*d^2 = mh*ga
%       => d^2 terms cancel; correct only when ga does NOT contain mhd^2.
%
%     N0(3,3) is the (3,3) cofactor of M(0):
%       det([al, be; be, ga+mhd^2]) = al*(ga+mhd^2) - be^2
%       => mhd^2 must appear explicitly here.
%
%     det(M(Y)): the mhd^2 terms cancel entirely in the cofactor expansion,
%       giving det = mh*(al*ga - be^2 + 2*be*mh*Y + mh*(al-mh)*Y^2)
%       where ga does NOT contain mhd^2.
%
%     Fix: ga = Jb+Jh+(m1+m2)*Lb^2/4  (no mhd^2 absorbed).
%          N0(3,3) written explicitly as al*(ga+mh*d^2) - be^2.
%
%   Root cause of C10 failures:
%     isAlways cannot prove rational expressions in all physical parameters
%     equal to zero, even when they are.  The expression Linv_block-Linv_rat
%     involves products of M0^{-1} denominators which the simplifier cannot
%     fully cancel.
%
%     Fix: C10 is split into C10a/C10b, checking L*Linv_rat=I6 and
%          Linv_rat*L=I6 directly. This avoids the rational subtraction.
%          C10 is logically implied by C3 + C9 passing; the numerical
%          fallback is the primary evidence.
%
% DESIGN PRINCIPLE
%   Every check compares hard-coded document expressions against an
%   independently computed ground truth.  No check derives both sides
%   from the same MATLAB expression.

clear; clc;

%% ========================================================================
%  0.  Symbols and assumptions
%% ========================================================================

syms Y m1 m2 mb mh Jb Jh Lb d real
syms f1 f2 f3 real
assume([m1 m2 mb mh Jb Jh Lb d] > 0)

I3 = sym(eye(3));  Z3 = sym(zeros(3));  I6 = sym(eye(6));
fnet = [f1; f2; f3];

%% ========================================================================
%  1.  GROUND TRUTH
%      M(Y) built directly from physical parameters — no M0/M1/M2 used.
%% ========================================================================

MY_gt = ...
  [ m1+m2+mb+mh,              (m1-m2)*Lb/2 - mh*Y,                        0;
    (m1-m2)*Lb/2 - mh*Y,      Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y^2,       -mh*d;
    0,                        -mh*d,                                        mh  ];

detMY_gt  = collect(expand(det(MY_gt)), Y);
adjMY_gt  = expand(adjoint(MY_gt));
MYinv_gt  = adjMY_gt / detMY_gt;

%% ========================================================================
%  2.  DOCUMENT EXPRESSIONS  (hard-coded verbatim from the document)
%% ========================================================================

% --- Inertia decomposition ---
M0 = [ m1+m2+mb+mh,         (m1-m2)*Lb/2,                        0;
       (m1-m2)*Lb/2,         Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2,       -mh*d;
       0,                   -mh*d,                                  mh ];

M1 = [  0,  -mh,  0;
       -mh,   0,  0;
        0,    0,  0 ];

M2 = [ 0,   0,  0;
       0,  mh,  0;
       0,   0,  0 ];

% --- Shorthand constants ---
% ga does NOT include mh*d^2 — this is the original document convention.
% The (2,2) entry of M(Y) is (ga + mh*d^2 + mh*Y^2); the mh*d^2 part
% appears explicitly in N0(3,3) and cancels in N0(1,1) and det(M(Y)).

al = m1+m2+mb+mh;
be = (m1-m2)*Lb/2;
ga = Jb+Jh+(m1+m2)*Lb^2/4;          % WITHOUT mh*d^2

% --- det(M(Y)) as stated in document ---
detMY_doc = mh*( al*ga - be^2  +  2*be*mh*Y  +  mh*(al-mh)*Y^2 );

% --- adj(M(Y)) = N0 + Y*N1 + Y^2*N2 as stated in document ---
N0_doc = ...
  [ mh*ga,               -be*mh,           -be*d*mh;
   -be*mh,                al*mh,             al*d*mh;
   -be*d*mh,              al*d*mh,           al*(ga+mh*d^2) - be^2 ];

N1_doc = ...
  [ 0,          mh^2,       d*mh^2;
    mh^2,       0,           0;
    d*mh^2,     0,           2*be*mh ];

N2_doc = ...
  [ mh^2,   0,   0;
    0,       0,   0;
    0,       0,   al*mh - mh^2 ];

adjMY_doc = expand(N0_doc  +  Y*N1_doc  +  Y^2*N2_doc);

% --- LFR objects ---
Dzw   = [ -(M0\M1),  -(M0\M2);
            I3,        Z3     ];
Delta = Y*I6;
L     = expand(I6 - Dzw*Delta);

L_explicit = [ I3 + Y*(M0\M1),    Y*(M0\M2);
              -Y*I3,               I3        ];
L_explicit  = expand(L_explicit);

S_doc = I3  +  Y*(M0\M1)  +  Y^2*(M0\M2);

% --- Block inverse (document structure; M(Y)^{-1} from ground truth) ---
Linv_block = simplify( ...
  [ MYinv_gt*M0,              -Y*(MYinv_gt*M2);
    Y*(MYinv_gt*M0),   I3 - Y^2*(MYinv_gt*M2) ] );

% --- Rational inverse (document adj/det formulas) ---
% Uses hard-coded adj_doc and det_doc — tests Sections 7 and 6 jointly.
% Genuine cross-check: if C2 and C3 pass, this form equals Linv_block.
MYinv_doc  = adjMY_doc / detMY_doc;
Linv_rat   = simplify( ...
  [ MYinv_doc*M0,              -Y*(MYinv_doc*M2);
    Y*(MYinv_doc*M0),   I3 - Y^2*(MYinv_doc*M2) ] );

% --- Loop solution (document Final Result) ---
z_doc    = [ MYinv_gt*fnet;
             Y*(MYinv_gt*fnet) ];
w_doc    = [ Y*(MYinv_gt*fnet);
             Y^2*(MYinv_gt*fnet) ];
rhs_loop = [ (M0\I3)*fnet;
              zeros(3,1)  ];

%% ========================================================================
%  3.  SYMBOLIC CHECKS
%% ========================================================================

fprintf('=== Symbolic checks ===\n\n');

chk('C1',  expand(M0 + Y*M1 + Y^2*M2  -  MY_gt), ...
    'M0 + Y*M1 + Y^2*M2 = M(Y)  [decomposition]');

chk('C2',  expand(detMY_doc - detMY_gt), ...
    'det(M(Y)) explicit formula vs ground truth');

chk('C3',  expand(adjMY_doc - adjMY_gt), ...
    'adj(M(Y)) = N0+Y*N1+Y^2*N2  every entry vs ground truth');

% C3b: degree of adj(M(Y)) is at most 2
N3_gt = matCoeff(adjMY_gt, Y, 3);
N4_gt = matCoeff(adjMY_gt, Y, 4);
chk('C3b', simplify([N3_gt; N4_gt]), ...
    'adj(M(Y)) has degree <= 2  (Y^3 and Y^4 coefficients zero)');

chk('C4',  expand(MY_gt*adjMY_gt  -  detMY_gt*I3), ...
    'M(Y)*adj(M(Y)) = det(M(Y))*I3');
chk('C5',  expand(adjMY_gt*MY_gt  -  detMY_gt*I3), ...
    'adj(M(Y))*M(Y) = det(M(Y))*I3');

chk('C6',  expand(L - L_explicit), ...
    'I - Dzw*Delta matches document explicit block form');

chk('C7',  expand(S_doc  -  (M0\MY_gt)), ...
    'S(Y) = I + Y*M0inv*M1 + Y^2*M0inv*M2 = M0^{-1}*M(Y)');

fprintf('  [C8: 6x6 symbolic det — may be slow]\n');
detL_computed = collect(expand(det(L)), Y);
chk('C8',  expand(detL_computed  -  det(M0\I3)*detMY_gt), ...
    'det(L(Y)) = det(M0^{-1})*det(M(Y))');

chk('C9a', simplify(L*Linv_block  -  I6), ...
    'L(Y) * L(Y)^{-1} = I6  (document block formula)');
chk('C9b', simplify(Linv_block*L  -  I6), ...
    'L(Y)^{-1} * L(Y) = I6  (document block formula)');

% C10a-b: rational form from document adj/det also inverts L.
% Avoids the rational subtraction Linv_block - Linv_rat that stalls isAlways.
% C10 is logically implied by C3 (adj correct) + C9 (block formula correct).
% isAlways warnings here are likely false negatives; see numerical fallback.
fprintf('\n  [C10: isAlways may warn on rational parameter expressions]\n');
chk('C10a', simplify(L*Linv_rat   -  I6), ...
    'L(Y) * Linv_rat (doc adj/det) = I6');
chk('C10b', simplify(Linv_rat*L   -  I6), ...
    'Linv_rat (doc adj/det) * L(Y) = I6');

chk('C11a', simplify(L*z_doc  -  rhs_loop), ...
    'L(Y)*z = [M0^{-1}*fnet; 0]  (loop solution z)');
chk('C11b', simplify(Delta*z_doc  -  w_doc), ...
    'w = Delta(Y)*z  (loop solution w)');

chk('C12',  simplify(MY_gt*(MYinv_gt*fnet)  -  fnet), ...
    'M(Y)*z_upper = fnet  (physical loop reduction)');

ratio = simplify(detL_computed / detMY_gt);
chk('C13',  simplify(ratio  -  det(M0\I3)), ...
    'det(L(Y))/det(M(Y)) = det(M0^{-1}),  constant in Y');

%% ========================================================================
%  4.  NUMERICAL FALLBACK
%      Evaluates residuals at physically admissible parameter values.
%      Three trials: different mass ratios, sign of beta, positive and
%      negative Y.  Primary safety net for C10 and any isAlways stalls.
%% ========================================================================

fprintf('\n=== Numerical fallback (3 parameter trials) ===\n\n');

all_syms = [m1, m2, mb, mh, Jb, Jh, Lb, d,  Y,  f1, f2, f3];
%            m1   m2   mb   mh   Jb   Jh   Lb    d     Y    f1  f2  f3
trials = [ 1.2, 0.8, 0.5, 0.3, 0.4, 0.2, 0.6, 0.10,  0.7,  1,  0,  0;
           2.1, 1.5, 0.9, 0.6, 1.1, 0.7, 1.2, 0.30, -0.4,  0,  1,  0;
           0.5, 0.5, 0.2, 0.1, 0.3, 0.1, 0.4, 0.05,  1.5,  0,  0,  1 ];

num_cases = { ...
  expand(M0 + Y*M1 + Y^2*M2  -  MY_gt),           'Num-C1    decomposition';
  expand(detMY_doc  -  detMY_gt),                   'Num-C2    det formula';
  expand(adjMY_doc  -  adjMY_gt),                   'Num-C3    adj formula';
  simplify(L*Linv_rat   -  I6),                     'Num-C10a  L*Linv_rat = I6';
  simplify(Linv_rat*L   -  I6),                     'Num-C10b  Linv_rat*L = I6';
  simplify(L*z_doc      -  rhs_loop),               'Num-C11a  loop z';
  simplify(Delta*z_doc  -  w_doc),                  'Num-C11b  loop w';
  simplify(MY_gt*(MYinv_gt*fnet)  -  fnet),        'Num-C12   M(Y)*z_upper=fnet' };

for k = 1:size(num_cases, 1)
    expr  = num_cases{k, 1};
    label = num_cases{k, 2};
    max_err = 0;
    for t = 1:size(trials, 1)
        val = double(subs(expr, all_syms, num2cell(trials(t, :))));
        max_err = max(max_err, max(abs(val(:))));
    end
    if max_err < 1e-10
        fprintf('  PASS  %-36s  max residual = %.2e\n', label, max_err);
    else
        fprintf('  FAIL  %-36s  max residual = %.2e\n', label, max_err);
    end
end

%% ========================================================================
%  LOCAL HELPER FUNCTIONS
%% ========================================================================

function chk(id, E, label)
% Symbolic zero-check.  PASS is reliable.  FAIL on rational expressions
% may be a false negative due to isAlways limitations; numerical fallback
% provides the independent safety net in those cases.
    E_flat = E(:);
    n      = numel(E_flat);
    n_fail = 0;
    for k = 1:n
        s = simplify(expand(E_flat(k)));
        if ~isAlways(s == sym(0))
            n_fail = n_fail + 1;
        end
    end
    if n_fail == 0
        fprintf('  PASS  %-6s  %s\n', id, label);
    else
        fprintf('  FAIL  %-6s  %s  (%d / %d entries)\n', id, label, n_fail, n);
    end
end

function C = matCoeff(M, var, pow)
% Extract coefficient of var^pow from each entry of symbolic matrix M.
    [n, m] = size(M);
    C = sym(zeros(n, m));
    for i = 1:n
        for j = 1:m
            C(i,j) = simplify( ...
                subs(diff(M(i,j), var, pow), var, 0) / factorial(pow));
        end
    end
end

















% %% verify_lfr_loop_symbolic.m
% clear; clc;
% 
% % --- symbols and assumptions ---
% syms Y m1 m2 mb mh Jb Jh Lb d real
% assumeAlso([m1 m2 mb mh Jb Jh Lb d] > 0)
% 
% I3 = sym(eye(3)); Z3 = sym(zeros(3)); I6 = sym(eye(6));
% 
% % --- inertia decomposition: M(Y) = M0 + Y*M1 + Y^2*M2 ---
% M0 = [m1 + m2 + mb + mh,          (m1 - m2)*Lb/2,                               0;
%       (m1 - m2)*Lb/2,             Jb + Jh + (m1 + m2)*Lb^2/4 + mh*d^2,        -mh*d;
%       0,                          -mh*d,                                         mh];
% 
% M1 = [0,  -mh,  0;
%      -mh,  0,   0;
%       0,   0,   0];
% 
% M2 = [0, 0, 0;
%       0, mh, 0;
%       0, 0, 0];
% 
% MY = expand(M0 + Y*M1 + Y^2*M2);
% 
% % --- LPV-LFR loop objects: L(Y) = I - Dzw*Delta(Y) ---
% A  = M0\M1;                % = inv(M0)*M1
% B  = M0\M2;                % = inv(M0)*M2
% Dzw   = [-A, -B;
%           I3, Z3];
% Delta = Y*I6;
% L     = simplify(I6 - Dzw*Delta);
% 
% % --- Schur complement and candidate inverse ---
% S = simplify(I3 + Y*A + Y^2*B);   % Schur complement of lower-right I3 block
% Linv_cand = simplify([MY\M0,            -Y*(MY\M2);
%                       Y*(MY\M0),  I3 - Y^2*(MY\M2)]);
% 
% % --- Rational / polynomial form of M(Y)^{-1} ---
% AdjMY     = simplify(adjoint(MY));
% DetMY     = collect(expand(det(MY)), Y);
% MYinv_rat = simplify(AdjMY / DetMY);
% 
% Linv_rat = simplify([MYinv_rat*M0,          -Y*MYinv_rat*M2;
%                      Y*MYinv_rat*M0,   I3 - Y^2*MYinv_rat*M2]);
% 
% % --- Polynomial coefficient extraction ---
% N0 = matCoeff(AdjMY, Y, 0);
% N1 = matCoeff(AdjMY, Y, 1);
% N2 = matCoeff(AdjMY, Y, 2);
% 
% d0 = polyCoeff(DetMY, Y, 0);
% d1 = polyCoeff(DetMY, Y, 1);
% d2 = polyCoeff(DetMY, Y, 2);
% 
% % --- Verification checks ---
% checks.schur_identity        = allZero(simplify(S - (M0\MY)));
% checks.det_identity          = isAlways(simplify(det(L) - DetMY/det(M0)) == 0);
% checks.inverse_left          = allZero(simplify(L*Linv_cand - I6));
% checks.inverse_right         = allZero(simplify(Linv_cand*L - I6));
% checks.adj_identity_left     = allZero(simplify(MY*AdjMY - DetMY*I3));
% checks.adj_identity_right    = allZero(simplify(AdjMY*MY - DetMY*I3));
% checks.rational_inverse      = allZero(simplify(Linv_cand - Linv_rat));
% checks.adj_expansion         = allZero(simplify(AdjMY - (N0 + Y*N1 + Y^2*N2)));
% checks.det_expansion         = isAlways(simplify(DetMY - (d0 + d1*Y + d2*Y^2)) == 0);
% 
% disp(checks)
% 
% disp('Det(M(Y)) ='); disp(DetMY)
% disp('d0 ='); disp(d0)
% disp('d1 ='); disp(d1)
% disp('d2 ='); disp(d2)
% 
% disp('N0 ='); disp(N0)
% disp('N1 ='); disp(N1)
% disp('N2 ='); disp(N2)
% 
% %% --- local helper functions ---
% function ok = allZero(E)
%     E  = E(:);
%     ok = all(arrayfun(@(k) isAlways(simplify(E(k)) == 0), 1:numel(E)));
% end
% 
% function C = matCoeff(M, var, pow)
%     [n,m] = size(M);
%     C = sym(zeros(n,m));
%     for i = 1:n
%         for j = 1:m
%             C(i,j) = simplify(subs(diff(M(i,j), var, pow), var, 0) / factorial(pow));
%         end
%     end
% end
% 
% function c = polyCoeff(p, var, pow)
%     c = simplify(subs(diff(p, var, pow), var, 0) / factorial(pow));
% end