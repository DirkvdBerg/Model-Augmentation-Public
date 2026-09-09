function trajectory_ownership_geometry()
% Independent MATLAB transcription of the revised-ownership reference identities.
%
% Twin of trajectory_ownership_geometry.py. It transcribes the model equations from the WRITTEN
% specification, not from the Python file, and reads nothing the Python run produces. Agreement
% between two engines reading two transcriptions is the check (algebra-tooling.md rule 1);
% either alone is weaker than it looks.
%
% SCOPE. This covers the identities that the revised ownership introduces and that the
% 2026-09-07 report could not contain, because that report's example shares one encoder scalar
% between the physical and the latent initialisation:
%
%   M1  the closed-loop sensitivity recurrence equals direct differentiation, for lambda, for
%       eta INCLUDING the latent initialisation, and for the physical-initialisation nuisance
%   M2  p_off and p_clamped are exactly independent of every augmentation parameter, latent
%       initialisation included, and the clamped latent state is null at every step
%   M3  E_w = Gamma_w J_w, the D-184 chain rule, per window
%   M4  D_ca p = (D_a0 p)(D_ca a_0), the latent-initialisation factorisation
%   M5  S n = 0 for the structural null direction of the rank-deficient coordinate set
%   M6  a nonzero MIXED derivative D_lambda D_eta V, which is the obstruction the selectively
%       routed update needs in order not to be the gradient of any scalar objective
%
% Every symbol is declared with sym('name','real'): `syms` cannot inject into the static
% workspace of a function that contains nested functions (algebra-tooling.md Sect. 4b).
%
% Run:
%   "C:\Program Files\MATLAB\R2025a\bin\matlab.exe" -batch ...
%       "addpath('scripts/gantry/orthogonality/code'); trajectory_ownership_geometry"

t_start = tic;
outdir = fullfile(fileparts(mfilename('fullpath')), 'trajectory_ownership_geometry');
if ~exist(outdir, 'dir'); mkdir(outdir); end

checks = struct('name', {}, 'level', {}, 'ok', {}, 'statement', {}, 'got', {});
undecided = {};

% ---------------------------------------------------------------- symbols and constants
kap1 = sym('kap1','real'); kap2 = sym('kap2','real');
nu_  = sym('nu','real');   sig  = sym('sig','real');
Kk = sym('Kk','real'); Ff = sym('Ff','real'); Gg = sym('Gg','real');
Pp = sym('Pp','real'); ca = sym('ca','real');
xp1 = sym('xp1','real'); xp2 = sym('xp2','real'); xp3 = sym('xp3','real');

LAM = [kap1 kap2 nu_ sig];
ETA = [Kk Ff Gg Pp ca];
XIP = [xp1 xp2 xp3];
ALLS = [LAM ETA XIP];

Ac = sym(4)/5; Bc = sym(1)/4; Cc = sym(3)/10; Dc = sym(1)/5;
Bx = [sym(1); sym(1)/3];
A12 = sym(1)/2; A21 = sym(-1)/5; A22 = sym(3)/4;

NWIN = 3; TSC = 3; NX = 2;
UD = [3 -5 2; -4 1 6; 2 7 -3] / 10;
YD = [1 4 -2; 5 -3 2; -1 2 3] / 10;
UD = sym(UD); YD = sym(YD);
% (h1 h2 h3 h4 ha ha2) per window
H = sym([2 -1 3 5 2 -3; -5 6 1 -4 -4 2; 2 5 -4 3 6 4]) / 10;

WIT = [2 3 1 2 3 2 1 2 3 1 2 3;
       1 2 3 1 2 1 3 1 2 2 1 1;
       3 1 2 3 1 3 2 3 1 3 2 2];

% free per-window initial states, for Gamma_w and the latent factorisation
Zs = sym(zeros(NWIN, 3));
for w = 1:NWIN
    Zs(w,1) = sym(sprintf('z1_%d', w), 'real');
    Zs(w,2) = sym(sprintf('z2_%d', w), 'real');
    Zs(w,3) = sym(sprintf('b_%d',  w), 'real');
end

% ---------------------------------------------------------------------------- checks
p_full  = stackAll('full',  false);
p_off   = stackAll('off',   false);
p_clamp = stackAll('clamped', false);
p_free  = stackAll('full',  true);
N = numel(p_full);

% M1 --------------------------------------------------------------------------------
groups = {LAM, ETA, XIP};
gnames = {'lambda','eta','xi_p'};
for gi = 1:3
    R = recurrenceJac(groups{gi}, 'full', false);
    D = jacobian(p_full, groups{gi});
    addZero(sprintf('M1.recurrence_equals_direct[%s]', gnames{gi}), R - D, ...
        'the forward sensitivity recurrence along chi=[x;a;xc] equals direct differentiation');
end
Roff = recurrenceJac(LAM, 'off', false);
addZero('M1.recurrence_equals_direct[lambda,off]', Roff - jacobian(p_off, LAM), ...
    'the same recurrence reproduces the off-mode physical sensitivity');
Rbad = recurrenceJac(LAM, 'full', true);
addNonzero('M1.negctrl_controller_path_detected', Rbad - jacobian(p_full, LAM), ...
    'severing the controller-state coupling changes the physical sensitivity, so an omitted controller path is detectable');

% M2 --------------------------------------------------------------------------------
for k = 1:numel(ETA)
    addZero(sprintf('M2.off_independent_of[%s]', char(ETA(k))), diff(p_off, ETA(k)), ...
        'p_off is exactly independent of this augmentation parameter');
end
addZero('M2.clamped_independent_of_latent_init', diff(p_clamp, ca), ...
    'p_clamped is exactly independent of the latent initialisation');
addZero('M2.clamped_latent_state_null', clampedLatentStates(), ...
    'the clamped latent state is null at initialisation and after every transition');
addZero('M2.full_equals_off_at_zero_gains', subs(p_full - p_off, [Kk Pp], [0 0]), ...
    'zero routed ANN gains make the full rollout equal the off rollout');
addZero('M2.clamped_equals_off_at_zero_direct', subs(p_clamp - p_off, Pp, 0), ...
    'with the direct route off, clamped equals off');
addZero('M2.full_equals_clamped_at_zero_readout', subs(p_full - p_clamp, Kk, 0), ...
    'with the latent readout zero, full equals clamped');

% M3 --------------------------------------------------------------------------------
subsInit = sym([]); subsFrom = sym([]);
for w = 1:NWIN
    ep = ePhys(w); ea = eLat(w);
    subsFrom = [subsFrom, Zs(w,1), Zs(w,2), Zs(w,3)];      %#ok<AGROW>
    subsInit = [subsInit, ep(1), ep(2), ea];               %#ok<AGROW>
end
Echain = sym([]);
Gams = cell(NWIN,1); Jws = cell(NWIN,1);
for w = 1:NWIN
    rows = (w-1)*TSC + (1:TSC);
    pw = p_free(rows);
    Gw = subs(jacobian(pw, [Zs(w,1) Zs(w,2)]), subsFrom, subsInit);
    Jw = jacobian(ePhys(w), XIP);
    Gams{w} = expand(Gw); Jws{w} = expand(Jw);
    Echain = [Echain; Gw*Jw];                              %#ok<AGROW>
end
addZero('M3.E_equals_Gamma_J', Echain - jacobian(p_full, XIP), ...
    'E_w = Gamma_w J_w for every window, hence E = blockdiag(Gamma_w) stack(J_w)');
addZero('M3.xip_only_through_initial_state', jacobian(p_free, XIP), ...
    'with the initial states free, the stack has no residual dependence on xi_p');

% M4 --------------------------------------------------------------------------------
dchain = sym(zeros(N,1));
for i = 1:N
    w = floor((i-1)/TSC) + 1;
    dchain(i) = subs(diff(p_free(i), Zs(w,3)), subsFrom, subsInit) * diff(eLat(w), ca);
end
addZero('M4.latent_init_chain_rule', dchain - diff(p_full, ca), ...
    'D_ca p = (D_a0 p)(D_ca a_0)');
addNonzero('M4.latent_init_reaches_output', diff(p_full, ca), ...
    'the trainable latent initialisation reaches the scored output');

% M5 --------------------------------------------------------------------------------
S = jacobian(p_full, LAM);
addZero('M5.structural_null_direction', S * [1; -1; 0; 0], ...
    'S n = 0 for n = e_kap1 - e_kap2: the predictor factors through kap1+kap2');

% M6 -------------------------------------------------------------------------------
% Frozen reference geometry at an integer witness point. The RATIONAL projector P_S is used
% rather than an orthonormal Q, because normalisation introduces surds that block exact
% rational arithmetic downstream (algebra-tooling.md Sect. 6).
w0 = WIT(1,:);
Snum = subs(S, ALLS, w0);
Enum = subs(jacobian(p_full, XIP), ALLS, w0);
PE = Enum * pinv(Enum);
Sbar = simplify((eye(N) - PE) * Snum);
PS = Sbar * pinv(Sbar);
d = p_full - p_off;
V = (d.' * PS * d) / N;                    % beta factored out; PLAIN-MSE 1/N kept explicitly
addNonzero('M6.penalty_gradient_reaches_latent_init', diff(V, ca), ...
    'the penalty gradient reaches the latent-initialisation parameter');
addNonzero('M6.penalty_has_nonzero_physical_derivative', diff(V, kap1), ...
    'D_lambda V is not identically zero, so routing only D_eta V is not the gradient of J_base + V');
addNonzero('M6.mixed_derivative_witness', diff(diff(V, kap1), ca), ...
    'D_lambda D_eta V is not identically zero, so the selectively routed field has an asymmetric Jacobian and is the gradient of NO scalar objective');

% ------------------------------------------------------------------------- report
nok = sum([checks.ok]); nfail = numel(checks) - nok;
fprintf('\n%d passed, %d failed, %d undecided, %.1f s\n', nok, nfail, numel(undecided), toc(t_start));
if nfail > 0
    fprintf('FAILED: %s\n', strjoin({checks(~[checks.ok]).name}, ', '));
end
out = struct('engine', 'matlab', 'version', version, ...
    'category', 'structural (RULES.md Rule 1): model only, no data, no simulation truth', ...
    'n_pass', nok, 'n_fail', nfail, 'undecided', {undecided}, ...
    'seconds', toc(t_start), 'checks', {checks});
fid = fopen(fullfile(outdir, 'trajectory_ownership_geometry_matlab.json'), 'w');
fwrite(fid, jsonencode(out, 'PrettyPrint', true));
fclose(fid);
fprintf('wrote %s\n', fullfile(outdir, 'trajectory_ownership_geometry_matlab.json'));

% =========================================================================== nested
    function ep = ePhys(w)
        h = H(w,:);
        ep = [xp1*h(1) + xp2*h(2); xp3*h(1) + xp2*h(3) + xp1*xp3*h(4)];
    end

    function ea = eLat(w)
        h = H(w,:);
        ea = ca*h(5) + ca^2*h(6);
    end

    function [x1n, x2n, an, xcn, y] = stepMap(x1, x2, a, xc, ud, yd, mode, dropCtrl) %#ok<INUSD>
        kap = kap1 + kap2;
        y  = sig*(x1 + 2*x2);
        e  = yd - y;
        u  = ud + Cc*xc + Dc*e;
        gp = double(~strcmp(mode,'off'));
        ga = double(~(strcmp(mode,'off') || strcmp(mode,'clamped')));
        wp = gp*(Kk*a + Pp*u);
        wa = ga*(Ff*x1 + Gg*a);
        x1n = expand(kap*x1 + A12*x2 + nu_*x1^2 + Bx(1)*u + wp);
        x2n = expand(A21*x1 + A22*x2 + Bx(2)*u);
        an  = expand(wa);
        xcn = expand(Ac*xc + Bc*e);
    end

    function [ys, traj] = rollWindow(w, x0, a0, mode)
        x1 = expand(x0(1)); x2 = expand(x0(2));
        if strcmp(mode,'full'); a = expand(a0); else; a = sym(0); end
        xc = sym(0);
        ys = sym(zeros(TSC,1)); traj = sym(zeros(TSC,4));
        for k = 1:TSC
            traj(k,:) = [x1 x2 a xc];
            [x1n, x2n, an, xcn, y] = stepMap(x1, x2, a, xc, UD(w,k), YD(w,k), mode, false);
            ys(k) = y;
            if k < TSC
                x1 = x1n; x2 = x2n; a = an; xc = xcn;
            end
        end
    end

    function p = stackAll(mode, freeInit)
        p = sym([]);
        for w = 1:NWIN
            if freeInit
                x0 = [Zs(w,1); Zs(w,2)]; a0 = Zs(w,3);
            else
                x0 = ePhys(w); a0 = eLat(w);
            end
            p = [p; rollWindow(w, x0, a0, mode)];           %#ok<AGROW>
        end
    end

    function v = clampedLatentStates()
        v = sym([]);
        for w = 1:NWIN
            [~, tr] = rollWindow(w, ePhys(w), eLat(w), 'clamped');
            v = [v; tr(:,3)];                               %#ok<AGROW>
        end
    end

    function R = recurrenceJac(group, mode, dropCtrl)
        ng = numel(group);
        R = sym([]);
        c = [sym('c1','real') sym('c2','real') sym('c3','real') sym('c4','real')];
        for w = 1:NWIN
            x0 = ePhys(w); a0 = eLat(w);
            if strcmp(mode,'full'); aInit = a0; else; aInit = sym(0); end
            X = [jacobian(x0(1), group); jacobian(x0(2), group); ...
                 jacobian(aInit, group); sym(zeros(1, ng))];
            x1 = expand(x0(1)); x2 = expand(x0(2)); a = expand(aInit); xc = sym(0);
            for k = 1:TSC
                [f1, f2, fa, fc, y] = stepMap(c(1), c(2), c(3), c(4), UD(w,k), YD(w,k), mode, false);
                F = [f1; f2; fa; fc];
                Fchi = jacobian(F, c);
                if dropCtrl; Fchi(:,4) = sym(0); end
                Fg = jacobian(F, group);
                Cm = jacobian(y, c);
                Dg = jacobian(y, group);
                vals = [x1 x2 a xc];
                R = [R; expand(subs(Cm, c, vals)*X + subs(Dg, c, vals))];  %#ok<AGROW>
                if k < TSC
                    X = expand(subs(Fchi, c, vals)*X + subs(Fg, c, vals));
                    [x1n, x2n, an, xcn, ~] = stepMap(x1, x2, a, xc, UD(w,k), YD(w,k), mode, false);
                    x1 = x1n; x2 = x2n; a = an; xc = xcn;
                end
            end
        end
    end

    function addZero(name, expr, statement)
        e = expand(expr);
        if all(logical(e(:) == 0))
            ok = true; got = 'zero (syntactic)';
        else
            r = simplify(e);
            try
                z = isAlways(r == 0, 'Unknown', 'error');
                ok = all(z(:)); got = 'zero (decided)';
                if ~ok; got = 'nonzero'; end
            catch
                ok = false; got = 'UNDECIDED';
                undecided{end+1} = name; %#ok<AGROW>
            end
        end
        checks(end+1) = struct('name', name, 'level', 'L1', 'ok', ok, ...
            'statement', statement, 'got', got);
        fprintf('  [%s] L1 %s\n', ternary(ok,'PASS','FAIL'), name);
    end

    function addNonzero(name, expr, statement)
        ok = false; got = 'every witness gave zero';
        for i = 1:size(WIT,1)
            v = simplify(subs(expr, ALLS, WIT(i,:)));
            if any(v(:) ~= 0)
                ok = true; got = sprintf('witness %d nonzero', i); break
            end
        end
        checks(end+1) = struct('name', name, 'level', 'L1', 'ok', ok, ...
            'statement', statement, 'got', got);
        fprintf('  [%s] L1 %s\n', ternary(ok,'PASS','FAIL'), name);
    end
end

function s = ternary(c, a, b)
if c; s = a; else; s = b; end
end
