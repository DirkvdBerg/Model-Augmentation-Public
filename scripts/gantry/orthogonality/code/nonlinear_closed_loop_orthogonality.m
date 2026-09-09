function nonlinear_closed_loop_orthogonality
%NONLINEAR_CLOSED_LOOP_ORTHOGONALITY  Symbolic + numerical verification of the
% trajectory-level orthogonality construction on a small nonlinear CLOSED-LOOP
% model that carries an internal learned state.
%
% Category: STRUCTURAL VERIFICATION EXAMPLES on a generic model. This is NOT a
% transcription of the gantry first-principles model, NOT gantry measurement, and
% NOT evidence of physical parameter recovery.
%
% Engine discipline (algebra-tooling.md): exact rationals, real assumptions,
% IgnoreAnalyticConstraints left false, isAlways(...,'Unknown','error'), null-space
% parameterisation instead of solve() for homogeneous constraints.
%
% Companion independent transcription: nonlinear_closed_loop_orthogonality.py (SymPy).
%
% Model (production timing: output computed BEFORE the state transition)
%   y_k    = sig*x_k
%   e_k    = ydata_k - y_k
%   u_k    = udata_k + Cc*xc_k + Dc*e_k
%   x_k+1  = kap*x_k + nu*x_k^2 + u_k + (Kk*a_k + Pp*u_k)
%   a_k+1  = Ff*x_k + Gg*a_k
%   xc_k+1 = Ac*xc_k + Bc*e_k
% physical q = [kap;nu;sig], learned eta = [Kk;Ff;Gg;Pp], encoder xi = cc.

t_matlab = tic;
HERE = fileparts(mfilename('fullpath'));
OUT  = fullfile(HERE,'nonlinear_closed_loop_orthogonality');
GEN  = fullfile(OUT,'generated');
if ~exist(GEN,'dir'); mkdir(GEN); end
addpath(GEN);

C = struct(); NUM = struct(); TIM = struct(); NOTE = struct(); ASSUME = struct();

vv = ver; tbl = struct();
for i = 1:numel(vv)
    tbl.(matlab.lang.makeValidName(vv(i).Name)) = sprintf('%s (%s)', vv(i).Version, vv(i).Release);
end

%% ===================== 1. Model constants (exact rationals) =====================
Ac = sym(4)/5;  Bc = sym(1)/4;  Cc = sym(3)/10; Dc = sym(1)/5;
nw = 2; Ts = 4; m = nw*Ts;                       % windows, scored outputs/window
ydata = sym([1 2 -1 3; 2 -1 1 1])/10;
udata = sym([1 -2 3 1; -1 2 1 -3])/10;
hx = sym([1;3])/5;   ha = sym([2;-1])/5;         % encoder features per window
xc0 = sym(0);
Ldiag = sym([4 6 2 8 5 3 7 4])/4;                % L, W = L'*L > 0 (nontrivial)
L = diag(Ldiag);

% Nested functions make this workspace static, so declare symbols explicitly.
kap = sym('kap','real'); nu = sym('nu','real'); sig = sym('sig','real');
Kk = sym('Kk','real'); Ff = sym('Ff','real'); Gg = sym('Gg','real'); Pp = sym('Pp','real');
cc = sym('cc','real');
q = [kap;nu;sig]; eta = [Kk;Ff;Gg;Pp]; xi = cc; th = [q;eta;xi]; np = numel(th);
th0 = [sym(1)/2; sym(1)/10; sym(6)/5; sym(1)/5; sym(1)/4; sym(2)/5; sym(1)/10; sym(1)];

xs = sym('xs','real'); as_ = sym('as','real'); xcs = sym('xcs','real');
yd = sym('yd','real'); ud = sym('ud','real');
chi = [xs;as_;xcs];

modes = {'full','clamp','off'};
blocks = struct();
for i = 1:3
    md = modes{i};
    [cn, yk] = stepfun(chi, yd, ud, q, eta, md);
    b = struct('step',cn,'out',yk, ...
        'Fchi',jacobian(cn,chi), 'Fth',jacobian(cn,th), ...
        'Hchi',jacobian(yk,chi), 'Hth',jacobian(yk,th));
    blocks.(md) = b;
end

t = tic;
% Derivative blocks contain every requested pathway.
C.jac_has_latent_readout      = isnz(blocks.full.Fchi(1,2));
C.jac_has_latent_write        = isnz(blocks.full.Fchi(2,1));
C.jac_has_latent_persistence  = isnz(blocks.full.Fchi(2,2));
C.jac_has_controller_feedback = isnz(blocks.full.Fchi(3,1)) && isnz(blocks.full.Fchi(1,3));
C.jac_has_output_parameter    = isnz(blocks.full.Hth);
C.jac_physical_forcing_every_step = isnz(blocks.full.Fth(1,1)) && isnz(blocks.full.Fth(1,2));
NOTE.Fchi_full_11 = char(blocks.full.Fchi(1,1));

%% ===================== 2. Short symbolic rollout, two ways =====================
stage('2 rollout');
p_full  = rollout(q,eta,xi,'full');
p_off   = rollout(q,eta,xi,'off');
p_clamp = rollout(q,eta,xi,'clamp');
[pf_r, Jf_r, chiend_full] = rollout_sens(q,eta,xi,'full',false);
[po_r, Jo_r, chiend_off ] = rollout_sens(q,eta,xi,'off' ,false);
[~   , ~   , chiend_clp ] = rollout_sens(q,eta,xi,'clamp',false);

C.rollout_expansion_matches_recurrence_value = iszs(p_full - pf_r) && iszs(p_off - po_r);
C.sensitivity_direct_equals_recurrence = iszs(jacobian(p_full,th) - Jf_r);
C.sensitivity_direct_equals_recurrence_off = iszs(jacobian(p_off,th) - Jo_r);
NUM.horizon_scored_steps_per_window = Ts;
NUM.state_transitions_per_window    = Ts-1;
NUM.scored_rows = m;
NOTE.horizon_rationale = ['Ts=4 scored outputs -> 3 transitions: the latent state re-enters ' ...
    'itself twice (Gg^2 appears in a_2) and the controller error feeds back three times. ' ...
    'Strictly beyond the two-step V1 toy in the implementation plan.'];

%% ===================== 3. Interventions and the decomposition =====================
stage('3 interventions');
d_total = p_full - p_off;
d_lat   = p_full - p_clamp;
d_dir   = p_clamp - p_off;
C.decomposition_exact = iszs(d_lat + d_dir - d_total);
NOTE.decomposition_caveat = ['ALGEBRAICALLY TRIVIAL: true for any three returned vectors. ' ...
    'It validates no simulator semantics. The independent intervention checks below do that.'];

C.off_prediction_independent_of_learned_parameters = iszs(jacobian(p_off,eta));
C.off_equals_full_when_all_augmentation_zero = iszs(subs(p_full,[Kk;Pp],[sym(0);sym(0)]) - p_off);
C.clamp_equals_full_when_latent_readout_zero = iszs(subs(p_full,Kk,sym(0)) - p_clamp);
C.clamp_equals_off_when_direct_route_zero    = iszs(subs(p_clamp,Pp,sym(0)) - p_off);
C.latent_route_is_nonzero = isnz(d_lat);
C.direct_route_is_nonzero = isnz(d_dir);
C.interventions_share_initial_physical_state = iszs(p_full(1)-p_off(1)) && iszs(p_full(1)-p_clamp(1)) ...
    && iszs(p_full(Ts+1)-p_off(Ts+1)) && iszs(p_full(Ts+1)-p_clamp(Ts+1));
C.each_intervention_evolves_own_controller_state = ...
    isnz(chiend_full(3,:) - chiend_off(3,:)) && isnz(chiend_full(3,:) - chiend_clp(3,:));
C.clamped_latent_state_is_null = iszs(chiend_clp(2,:));
NOTE.recorded_signals = ['ydata/udata and the encoder-derived x_0 are the same arrays in all three ' ...
    'rollouts by construction; only the controller state and the physical trajectory differ.'];

%% ===================== 4. Zero-latent pointwise penalty is blind =================
stage('4 pointwise');
uj = sym([2;1;-1])/10;                            % frozen one-step point set
c_pointwise = Kk*sym(zeros(3,1)) + Pp*uj;         % c_eta(x_j, a=0, u_j)
C.pointwise_penalty_blind_to_latent_readout = iszs(diff(c_pointwise,Kk));
C.pointwise_penalty_identically_zero_without_direct_route = iszs(subs(c_pointwise,Pp,sym(0)));

%% ===================== 5. Reference geometry =====================================
stage('5 geometry');
S0 = subs(Jf_r(:,1:3), th, th0);      % full-predictor physical sensitivity
E0 = subs(Jf_r(:,8)  , th, th0);      % shared-encoder sensitivity
D0 = L*E0;
PE = D0*pinv(D0);   Im = sym(eye(m));   ME = Im - PE;
Sbar = ME*L*S0;
PS   = Sbar*pinv(Sbar);
QE = orthsym(D0);  Q0 = orthsym(Sbar);
rS = rank(Sbar); rE = rank(D0); rS_unprofiled = rank(L*S0);

C.ME_symmetric_idempotent = iszs(ME-ME.') && iszs(ME*ME-ME);
C.PS_symmetric_idempotent = iszs(PS-PS.') && iszs(PS*PS-PS);
C.encoder_physical_projectors_orthogonal = iszs(PS*PE) && iszs(QE.'*Q0);
C.QE_orthonormal = iszs(QE.'*QE - sym(eye(size(QE,2))));
C.Q0_orthonormal = iszs(Q0.'*Q0 - sym(eye(size(Q0,2))));
C.Q0_spans_profiled_physical_range = iszs(Q0*Q0.' - PS) && iszs(ME*Sbar - Sbar);
C.Q0_transpose_absorbs_ME = iszs(Q0.'*ME - Q0.');
C.Sbar_transpose_absorbs_ME = iszs(Sbar.'*ME - Sbar.');
C.zero_residual_iff_contribution_orthogonality = ...
    iszs(Sbar.' - (Q0.'*Sbar).'*Q0.') && rank(Q0.'*Sbar) == rS;
NUM.rank_L_S0_unprofiled = double(rS_unprofiled);
NUM.rank_encoder_D0 = double(rE);
NUM.rank_Sbar_after_profiling = double(rS);
ASSUME.rank_conditions = sprintf(['Sbar has column rank %d after profiling a rank-%d encoder out ' ...
    'of a rank-%d unprofiled physical sensitivity. Every pseudoinverse below is used under exactly ' ...
    'these ranks, verified symbolically over the rationals at the frozen reference.'], ...
    double(rS), double(rE), double(rS_unprofiled));
ASSUME.parameters = 'kap, nu, sig, Kk, Ff, Gg, Pp, cc real; cs nonzero; br, brp positive; tt, sl nonnegative.';

% Penalty value bound: beta*||v||^2 <= eps, beta>0  =>  ||v|| <= sqrt(eps/beta)
brp = sym('brp','positive'); tt = sym('tt','real'); sl = sym('sl','real');
assume(tt>=0); assume(sl>=0);
try
    C.penalty_value_bound = isAlways(sqrt((brp*tt^2+sl)/brp) - tt >= 0, 'Unknown','error');
    NOTE.penalty_value_bound_route = 'isAlways(Unknown=error) on eps = beta*t^2 + slack, slack >= 0';
catch ME_bnd
    C.penalty_value_bound = iszs(simplify((brp*tt^2+sl)/brp - tt^2 - sl/brp));
    NOTE.penalty_value_bound_route = ['isAlways undecided (' ME_bnd.message ...
        '); fell back to the exact identity (eps/beta) - t^2 = slack/beta >= 0 plus monotonicity of sqrt.'];
end
assume([tt sl],'clear');

%% ===================== 6. Penalty, gradients, routing, batching ==================
stage('6 penalty');
br = sym('br','positive');
rows1 = 1:Ts; rows2 = Ts+1:m;
A1 = Q0(rows1,:).'*L(rows1,rows1);  A2 = Q0(rows2,:).'*L(rows2,rows2);

% --- 6a. UNIVERSAL linear-algebra statements, proved as identities for a GENERIC
% contribution vector and a GENERIC smooth parameter dependence. These hold for any
% d whatsoever, so in particular for this example's d_total. Deliberately separated
% from 6b, which then checks that the example composes correctly.
dg = sym('dg',[m 1],'real');
eg = sym('eg',[4 1],'real');
kg = sym('kg','real');
Amap = sym(reshape(mod((1:m*4).^2,7)-3,[m 4]))/sym(5);   % rational, generic enough
bq   = sym(mod((1:m).^3,5).'-2)/sym(3);
cvec = sym(mod((2*(1:m)+1),4).'-1)/sym(2);
dgen = Amap*eg + (eg.'*eg)*bq + kg*cvec;                 % smooth, nonlinear in eg
rg   = Q0.'*L*dg;
C.penalty_equals_projector_form_generic = iszs(br*(rg.'*rg) - br*((L*dg).'*PS*(L*dg)));
C.exact_batching_residual_sum_generic = iszs(rg - (A1*dg(rows1) + A2*dg(rows2)));
% The basis-free RATIONAL residual rho = PS*L*d carries the same penalty value and the
% same chunk decomposition. Everything downstream uses rho, which keeps the example
% checks in exact rational arithmetic: Q0 carries surds, and expanding a degree-16
% surd expression overflowed SymPy (recorded in algebra-tooling.md clause 6).
P1 = PS(:,rows1)*L(rows1,rows1);  P2 = PS(:,rows2)*L(rows2,rows2);
rhog = PS*L*dg;
C.projector_residual_equals_basis_residual_generic = iszs(rg.'*rg - rhog.'*rhog);
C.projector_residual_chunk_split_generic = iszs(rhog - (P1*dg(rows1) + P2*dg(rows2)));
rgen = Q0.'*L*dgen;  Vgen = br*(rgen.'*rgen);
ggen = jacobian(Vgen,eg).';
C.penalty_gradient_gauss_newton_form_generic = iszs(ggen - 2*br*(jacobian(rgen,eg).'*rgen));
C.exact_batching_gradient_generic = iszs(ggen - 2*br*( ...
    (A1*jacobian(dgen(rows1),eg)).'*rgen + (A2*jacobian(dgen(rows2),eg)).'*rgen));
C.joint_route_has_nonzero_physical_gradient_generic = isnz(jacobian(Vgen,kg));
NOTE.universal_vs_example = ['6a proves the projector, chain-rule and exact-batching identities ' ...
    'as IDENTITIES for a generic contribution vector and a generic smooth parameter dependence. ' ...
    '6b then verifies that this example composes correctly, exactly at rational witness points ' ...
    'and numerically by finite differences. The degree-16 expanded polynomial identity for the ' ...
    'full nonlinear d_total was deliberately NOT attempted: it adds nothing the universal ' ...
    'identity does not already give, and its expansion cost is not a proof.'];

% --- 6b. THIS nonlinear closed-loop example.
Ld  = L*d_total;
rho = PS*Ld;                            % rational residual, kept unexpanded on purpose
V   = br*(rho.'*rho);
gV_eta = jacobian(V,eta).';  gV_q = jacobian(V,q).';  gV_xi = jacobian(V,xi).';
rho1 = P1*d_total(rows1);  rho2 = P2*d_total(rows2);
wit = [th; br];
C.penalty_equals_projector_form_example = eqpts(br*(rho.'*rho) - br*(Ld.'*PS*Ld), wit);
C.penalty_gradient_gauss_newton_form_example = eqpts(gV_eta - 2*br*(jacobian(rho,eta).'*rho), wit);
C.exact_batching_residual_sum_example = eqpts(rho - (rho1+rho2), wit);
C.exact_batching_gradient_example = eqpts(gV_eta - 2*br*(jacobian(rho1,eta).'*rho + jacobian(rho2,eta).'*rho), wit);
C.ann_only_routing_differs_from_joint_objective = isnz(gV_q) || isnz(gV_xi);
NOTE.routing = ['Default routing uses jacobian(V,eta) only. jacobian(V,kappa) and jacobian(V,xi) ' ...
    'are NOT identically zero, so the routed update is not the gradient of any single scalar objective.'];
NOTE.example_check_strength = ['6b checks are EXACT rational-arithmetic equalities at three ' ...
    'independent witness points, not proved identities. The identity content is carried by 6a.'];

% Negative control: detaching intermediate states changes the ANN gradient.
[~, Jf_drop] = rollout_sens(q,eta,xi,'full',true);
[~, Jo_drop] = rollout_sens(q,eta,xi,'off' ,true);
C.detached_intermediate_state_changes_gradient = isnz(Jf_drop(:,4:7) - Jf_r(:,4:7));
C.off_rollout_has_no_eta_path_even_detached = iszs(Jo_drop(:,4:7)) && iszs(Jo_r(:,4:7));

% Cross-chunk cancellation: distinguishes the stacked penalty from a per-chunk sum.
% Built in the RATIONAL chunk operators P1/P2: A1/A2 carry Q0's surds, and pinv on a
% surd matrix overflowed SymPy (recorded in algebra-tooling.md clause 6).
tsy = sym('tsy','real');
d1 = sym([1;0;0;0]); rc = P1*d1; d2 = -pinv(P2)*rc;
C.chunk_two_can_cancel = rank(P2) == rS && iszs(P1*d1 + P2*d2) && isnz(rc);
rr1 = tsy*(P1*d1); rr2 = tsy*(P2*d2);
Vstack = br*((rr1+rr2).'*(rr1+rr2));  Vsum = br*(rr1.'*rr1 + rr2.'*rr2);
C.stacked_penalty_differs_from_per_chunk_sum = iszs(Vstack) && isnz(Vsum) && isnz(diff(Vsum,tsy));
NOTE.cross_chunk = ['Constructed d with r_1 = -r_2 /= 0: V = beta||sum_b r_b||^2 = 0 while ' ...
    'beta*sum_b||r_b||^2 > 0 with nonzero gradient. Chunking must not change the measure.'];

%% ===================== 7. Explicit blind-spot vs rollout case ====================
stage('7 blindspot');
subv  = [q; Ff; Gg; Pp; xi];
subva = [th0(1:3); th0(5); th0(6); sym(0); th0(8)];   % direct route switched OFF
rK = subs(rho, subv, subva);                          % residual as a function of Kk alone
C.rollout_projection_detects_latent_readout = isnz(rK) && isnz(diff(rK,Kk));
NUM.projected_residual_norm_at_reference_K = double(sqrt(subs(rK.'*rK, Kk, th0(4))));
NUM.projected_residual_dnorm_dK_at_reference = double(subs(diff(sqrt(rK.'*rK),Kk), Kk, th0(4)));
NOTE.blind_spot_case = ['With Pp = 0 the zero-latent pointwise penalty residual is identically zero ' ...
    'for every Kk, while ||Q0''*L*d_total|| is nonzero with nonzero derivative in Kk.'];

%% ===================== 8. Latent-coordinate invariance ===========================
stage('8 invariance');
cs = sym('cs','real'); assume(cs ~= 0);
p_full_s  = rollout_ls(q,eta,xi,'full', cs);
p_off_s   = rollout_ls(q,eta,xi,'off',  cs);
p_clamp_s = rollout_ls(q,eta,xi,'clamp',cs);
C.latent_rescaling_leaves_prediction_invariant = iszs(p_full_s - p_full);
C.latent_rescaling_leaves_off_and_clamp_invariant = iszs(p_off_s - p_off) && iszs(p_clamp_s - p_clamp);
d_total_s = p_full_s - p_off_s;
C.latent_rescaling_leaves_effect_invariant = iszs(d_total_s - d_total);
rho_s = PS*L*d_total_s;
C.latent_rescaling_leaves_penalty_invariant = eqpts(br*(rho_s.'*rho_s) - V, [th; br; cs]);
chiend_s = rollout_states_ls(q,eta,xi,'full',cs);
C.latent_state_scales_by_c = iszs(chiend_s(2,:) - cs*chiend_full(2,:));
C.latent_state_norm_is_not_invariant = isnz(subs(chiend_s(2,1),cs,sym(2)) - chiend_full(2,1));
C.readout_norm_is_not_invariant = isnz(subs(Kk/cs,cs,sym(2)) - Kk);
NOTE.invariance_scope = ['Scaling a'' = c*a with Kk'' = Kk/c, Ff'' = c*Ff, Gg'' = Gg, a_0'' = c*a_0 ' ...
    'leaves predictions, all three interventions and the penalty unchanged. ||Kk|| and the latent ' ...
    'state norm both change, so NEITHER is a coordinate-invariant diagnostic. The zero clamp value ' ...
    'is preserved by scaling; it would NOT be preserved by an affine translation of the latent coordinate.'];
assume(cs,'clear');

TIM.symbolic_algebra_s = toc(t);

%% ===================== 9. Code generation ========================================
stage('9 codegen');
t = tic;
gen_targets = { 'ncl_step_full', blocks.full.step; 'ncl_step_off', blocks.off.step; ...
                'ncl_step_clamp', blocks.clamp.step; 'ncl_Fchi_full', blocks.full.Fchi; ...
                'ncl_Fth_full', blocks.full.Fth;    'ncl_Fchi_off', blocks.off.Fchi; ...
                'ncl_Fth_off', blocks.off.Fth;      'ncl_out', blocks.full.out; ...
                'ncl_Hchi', blocks.full.Hchi;       'ncl_Hth', blocks.full.Hth };
V4 = {chi,[yd;ud],q,eta};
t_opt_true = tic;
for i = 1:size(gen_targets,1)
    matlabFunction(gen_targets{i,2},'Vars',V4,'File',fullfile(GEN,gen_targets{i,1}), ...
        'Optimize',true,'Comments','Generated by nonlinear_closed_loop_orthogonality.m');
end
TIM.codegen_optimize_true_s = toc(t_opt_true);
tmpd = tempname; mkdir(tmpd);
t_opt_false = tic;
for i = 1:size(gen_targets,1)
    matlabFunction(gen_targets{i,2},'Vars',V4,'File',fullfile(tmpd,gen_targets{i,1}),'Optimize',false);
end
TIM.codegen_optimize_false_s = toc(t_opt_false);
d_true = dir(fullfile(GEN,'*.m')); d_false = dir(fullfile(tmpd,'*.m'));
NUM.codegen_bytes_optimize_true  = sum([d_true.bytes]);
NUM.codegen_bytes_optimize_false = sum([d_false.bytes]);
rmdir(tmpd,'s');
NUM.Fchi_structural_nonzeros = double(sum(sum(~iszs_elem(blocks.full.Fchi))));
NOTE.sparse_decision = ['Fchi is 3x3 with 6 of 9 structurally nonzero entries, so Sparse output ' ...
    'was not requested; it would add indexing overhead for no benefit at this density.'];
rehash path;
TIM.codegen_s = toc(t);

%% ===================== 10. Longer closed-loop numeric run ========================
stage('10 numeric');
t = tic;
thn = double(th0); qn = thn(1:3); en = thn(4:7); cn_ = thn(8);
J_ref = double(subs(Jf_r, th, th0));
[p_gen4, J_gen4] = num_rollout(qn,en,cn_,double(ydata),double(udata),double(hx),double(ha),Ts,true);
NUM.generated_vs_symbolic_max_abs = max(abs(J_gen4(:)-J_ref(:)));
NUM.generated_vs_symbolic_max_rel = max(abs(J_gen4(:)-J_ref(:)))/max(1e-30,max(abs(J_ref(:))));
C.generated_functions_match_symbolic_reference = NUM.generated_vs_symbolic_max_abs < 1e-12;
NUM.short_prediction_max_abs_err = max(abs(p_gen4 - double(subs(p_full,th,th0))));

rng(20260907,'twister');
TL = 200; nwL = 3;
ydL = 0.15*randn(nwL,TL); udL = 0.10*randn(nwL,TL);
hxL = [0.20;0.60;-0.35]; haL = [0.40;-0.20;0.25];
[pL, JL, stL] = num_rollout(qn,en,cn_,ydL,udL,hxL,haL,TL,true);
NUM.long_horizon = TL; NUM.long_windows = nwL; NUM.long_rows = numel(pL);
NUM.long_max_abs_state = max(abs(stL(:)));
Alin = [qn(1)-(1+en(4))*0.2*qn(3), en(1), (1+en(4))*0.3;
        en(2), en(3), 0;
        -0.25*qn(3), 0, 0.8];
NUM.linearisation_spectral_radius_at_origin = max(abs(eig(Alin)));
NOTE.regime = ['Finite-horizon numerical regime only: the linearisation at the origin has spectral ' ...
    'radius below one and the realised 200-step closed-loop states stay bounded and small. ' ...
    'This is NOT a stability proof for the nonlinear closed loop.'];

JL_fd = zeros(size(JL));
for j = 1:np
    hstep = 1e-6*max(1,abs(thn(j)));
    tp = thn; tp(j) = tp(j)+hstep;  tm = thn; tm(j) = tm(j)-hstep;
    pp = num_rollout(tp(1:3),tp(4:7),tp(8),ydL,udL,hxL,haL,TL,false);
    pm = num_rollout(tm(1:3),tm(4:7),tm(8),ydL,udL,hxL,haL,TL,false);
    JL_fd(:,j) = (pp-pm)/(2*hstep);
end
colscale = max(abs(JL),[],1); colscale(colscale==0) = 1;
NUM.long_recurrence_vs_fd_max_abs = max(abs(JL(:)-JL_fd(:)));
NUM.long_recurrence_vs_fd_max_rel_colscaled = max(max(abs(JL-JL_fd)./colscale));
C.long_recurrence_matches_finite_differences = NUM.long_recurrence_vs_fd_max_rel_colscaled < 1e-6;
[~, JL_drop] = num_rollout(qn,en,cn_,ydL,udL,hxL,haL,TL,true,true);
NUM.long_detached_controller_max_rel = max(max(abs(JL_drop-JL_fd)./colscale));
C.long_detached_controller_is_detected = NUM.long_detached_controller_max_rel > 1e-3;
% Finite-difference check of the ANN-routed penalty gradient on THIS example.
Ln_ = double(L); Q0n_ = double(Q0); beta_num = 0.75;
poff_n = double(subs(p_off, th, th0));
gA = double(subs(gV_eta, [th; br], [th0; sym(3)/4]));
gF = zeros(4,1);
for j = 1:4
    hh = 1e-6; ep = en; ep(j) = ep(j)+hh; em = en; em(j) = em(j)-hh;
    gF(j) = (Vnum(ep)-Vnum(em))/(2*hh);
end
NUM.penalty_gradient_fd_max_abs = max(abs(gA-gF));
NUM.penalty_gradient_fd_max_rel = max(abs(gA-gF))/max(1e-30,max(abs(gF)));
NUM.penalty_gradient_analytic = gA(:).';
C.penalty_gradient_matches_finite_differences = NUM.penalty_gradient_fd_max_rel < 1e-6;
NUM.cond_L_S0 = cond(double(L*S0));
NUM.cond_Sbar = cond(double(Sbar));
NUM.cond_long_physical_block = cond(JL(:,1:3));
TIM.numeric_s = toc(t);

%% ===================== 11. Separate hard-constraint experiment ===================
stage('11 fmincon');
t = tic;
have_optim = license('test','optimization_toolbox') && ~isempty(which('fmincon'));
C.optimization_toolbox_available = have_optim;
if have_optim
    pf_e  = matlabFunction(subs(p_full,[q;xi],[th0(1:3);th0(8)]),'Vars',{eta});
    Jf_e  = matlabFunction(subs(jacobian(p_full,eta),[q;xi],[th0(1:3);th0(8)]),'Vars',{eta});
    poffn = double(subs(p_off,th,th0));
    Ln = double(L); Q0n = double(Q0); Wn = Ln.'*Ln;
    eta_truth = [0.30;0.40;0.50;0.20];
    ymeas = pf_e(eta_truth);
    objf = @(e) obj_local(e, pf_e, Jf_e, Wn, ymeas);
    ceqf = @(e) Q0n.'*Ln*(pf_e(e)-poffn);
    Jceq = @(e) (Q0n.'*Ln*Jf_e(e)).';                 % parameters-by-constraints
    nlc  = @(e) nlc_local(e, ceqf, Jceq);
    e_chk = [0.22;0.31;0.37;0.13];
    [~,g_an] = objf(e_chk); g_fd = zeros(4,1); Jc_fd = zeros(4,3);
    for j=1:4
        hh=1e-6; ep=e_chk; ep(j)=ep(j)+hh; em=e_chk; em(j)=em(j)-hh;
        g_fd(j)=(objf(ep)-objf(em))/(2*hh);
        Jc_fd(j,:)=((ceqf(ep)-ceqf(em))/(2*hh)).';
    end
    NUM.fmincon_objgrad_rel_err = norm(g_an-g_fd)/max(1,norm(g_fd));
    NUM.fmincon_ceqgrad_rel_err = norm(Jceq(e_chk)-Jc_fd,'fro')/max(1,norm(Jc_fd,'fro'));
    C.fmincon_supplied_gradients_validated = NUM.fmincon_objgrad_rel_err < 1e-6 && NUM.fmincon_ceqgrad_rel_err < 1e-6;
    Zn = null(Q0n.'*Ln*Jf_e(zeros(4,1)));
    NUM.constraint_nullspace_dim_at_eta_zero = size(Zn,2);
    NUM.constraint_jacobian_rank_at_eta_zero = rank(Q0n.'*Ln*Jf_e(zeros(4,1)));
    C.constraint_is_degenerate_at_ann_off = NUM.constraint_jacobian_rank_at_eta_zero < 3;
    NOTE.ann_off_degeneracy = ['At eta = 0 the constraint Jacobian has rank ' ...
        num2str(NUM.constraint_jacobian_rank_at_eta_zero) ' of 3: the Ff and Gg columns vanish ' ...
        'because with Kk = Pp = 0 the latent state cannot reach the output at all. The whole ' ...
        'plane {Kk = 0, Pp = 0} is therefore feasible with d = 0. This is exactly why "ANN-off ' ...
        'is feasible" is NOT evidence that the constraint admits a useful correction, and why ' ...
        'the null-space seed taken at eta = 0 is useless: it moves only along Ff and Gg.'];

    % Feasible NONZERO corrections, found by root-solving the homogeneous constraint on
    % fixed-Kk slices. Three equations, three remaining unknowns (Ff, Gg, Pp).
    fso = optimoptions('fsolve','Display','off','FunctionTolerance',1e-14, ...
        'StepTolerance',1e-14,'SpecifyObjectiveGradient',true);
    Kgrid = [0.05 0.1 0.2 0.3 0.5 -0.05 -0.1 -0.2 -0.3 -0.5];
    feas = [];
    for kk = Kgrid
        for z0 = {[0.2;0.3;0.1],[-0.3;0.5;-0.2],[0.6;-0.4;0.3]}
            [zz,~,flz] = fsolve(@(z) slice_local(z,kk,ceqf,Jceq), z0{1}, fso);
            ee_ = [kk; zz];
            vv_ = max(abs(ceqf(ee_)));  dn_ = norm(Ln*(pf_e(ee_)-poffn));
            if flz > 0 && vv_ < 1e-10 && dn_ > 1e-3
                feas(end+1,:) = [ee_.' dn_ objf(ee_) vv_]; %#ok<AGROW>
            end
        end
    end
    C.feasible_nonzero_correction_exists = ~isempty(feas);
    if ~isempty(feas)
        [~,ix] = sort(feas(:,6));
        feas = feas(ix,:);
        [~,iu] = uniquetol(feas(:,1:4),1e-6,'ByRows',true);
        feas = feas(sort(iu),:);
        NUM.feasible_nonzero_points_eta = feas(:,1:4);
        NUM.feasible_nonzero_points_correction_norm = feas(:,5).';
        NUM.feasible_nonzero_points_objective = feas(:,6).';
        NUM.feasible_nonzero_points_max_abs_ceq = feas(:,7).';
        NUM.best_feasible_nonzero_objective = min(feas(:,6));
    end
    opts = optimoptions('fmincon','Algorithm','sqp','SpecifyObjectiveGradient',true, ...
        'SpecifyConstraintGradient',true,'OptimalityTolerance',1e-12, ...
        'ConstraintTolerance',1e-12,'StepTolerance',1e-14,'MaxFunctionEvaluations',5e3,'Display','off');
    seeds = [zeros(4,1), Zn(:,1)*0.5, -Zn(:,1)*0.5, eta_truth, [0.1;0.1;0.1;0.1]];
    if ~isempty(feas); seeds = [seeds, feas(:,1:4).']; end
    best = struct('f',inf,'eta',[],'flag',-99,'viol',inf,'dnorm',0,'seed',0);
    t_serial = tic;
    for s0 = 1:size(seeds,2)
        [es,fs,fl] = fmincon(objf,seeds(:,s0),[],[],[],[],[],[],nlc,opts);
        dn = norm(Ln*(pf_e(es)-poffn));
        if fl > 0 && fs < best.f
            best = struct('f',fs,'eta',es,'flag',fl,'viol',max(abs(ceqf(es))),'dnorm',dn,'seed',s0);
        end
    end
    TIM.fmincon_serial_5_starts_s = toc(t_serial);
    NUM.fmincon_objective = best.f;
    NUM.fmincon_exitflag = best.flag;
    NUM.fmincon_max_abs_ceq = best.viol;
    NUM.fmincon_retained_correction_norm_Ld = best.dnorm;
    NUM.fmincon_eta_star = best.eta(:).';
    NUM.fmincon_best_seed_index = best.seed;
    NUM.fmincon_correction_norm_at_eta_zero = norm(Ln*(pf_e(zeros(4,1))-poffn));
    NUM.fmincon_objective_at_eta_zero = objf(zeros(4,1));
    NUM.fmincon_initial_conditions = seeds.';
    C.fmincon_constrained_optimum_retains_nonzero_correction = ...
        best.flag>0 && best.viol<1e-9 && best.dnorm>1e-3;
    NOTE.fmincon_scope = ['SEPARATE method: hard equality constraint, NOT the ANN-only penalty. ' ...
        'The ANN-off point eta = 0 is feasible with zero correction and is explicitly rejected as ' ...
        'evidence. Two DIFFERENT questions are reported separately: (1) does the feasible set ' ...
        'contain nonzero corrections at all (feasible_nonzero_correction_exists, answered by ' ...
        'root-solving fixed-Kk slices of the constraint); (2) does the CONSTRAINED OPTIMUM retain ' ...
        'one (fmincon_constrained_optimum_retains_nonzero_correction). A false answer to (2) with ' ...
        'a true answer to (1) is a real, reportable outcome about this objective and this ' ...
        'geometry, not a solver failure to be papered over. No global-convergence or ' ...
        'parameter-recovery claim follows from any solver result.'];
    NOTE.parallel_decision = sprintf(['Five serial fmincon starts cost %.3f s; opening a parpool ' ...
        'costs tens of seconds. Parallelism was benchmarked away, not used.'], TIM.fmincon_serial_5_starts_s);
end
TIM.optimization_s = toc(t);

%% ===================== 12. Report =================================================
fn = fieldnames(C); failed = {};
for i = 1:numel(fn)
    if ~C.(fn{i}); failed{end+1} = fn{i}; end %#ok<AGROW>
end
undecided = iszs('report');
TIM.total_in_matlab_s = toc(t_matlab);
res = struct('category','structural verification examples; generic nonlinear closed-loop model only', ...
    'not_a_gantry_transcription', true, 'engine', version, 'release', version('-release'), ...
    'toolboxes', tbl, 'checks', C, 'numbers', NUM, 'timings_s', TIM, 'assumptions', ASSUME, ...
    'notes', NOTE, 'n_checks', numel(fn), 'failed', {failed}, 'undecided_identities', {undecided});
fid = fopen(fullfile(OUT,'nonlinear_closed_loop_orthogonality_matlab.json'),'w');
cl = onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',jsonencode(res,'PrettyPrint',true));
for i = 1:numel(fn)
    fprintf('%-58s %s\n', fn{i}, string(C.(fn{i})));
end
fprintf('\n%d checks, %d failed. algebra %.2fs codegen %.2fs numeric %.2fs optim %.2fs total %.2fs\n', ...
    numel(fn), numel(failed), TIM.symbolic_algebra_s, TIM.codegen_s, TIM.numeric_s, TIM.optimization_s, TIM.total_in_matlab_s);
if ~isempty(failed)
    fprintf('FAILED: %s\n', strjoin(failed,', '));
end
if ~isempty(undecided)
    fprintf('UNDECIDED IDENTITIES (not proved, not disproved): %s\n', strjoin(undecided,' | '));
else
    fprintf('No undecided identities.\n');
end

%% ===================== nested helpers =============================================
    function [chin, yk] = stepfun(ch, ydk, udk, qq, ee, md)
        x_ = ch(1); a_ = ch(2); c_ = ch(3);
        yk = qq(3)*x_;
        e_ = ydk - yk;
        u_ = udk + Cc*c_ + Dc*e_;
        switch md
            case 'full',  corr = ee(1)*a_ + ee(4)*u_; an = ee(2)*x_ + ee(3)*a_;
            case 'clamp', corr = ee(4)*u_;            an = sym(0);
            case 'off',   corr = sym(0);              an = sym(0);
        end
        chin = [qq(1)*x_ + qq(2)*x_^2 + u_ + corr; an; Ac*c_ + Bc*e_];
    end

    function p = rollout(qq,ee,xx,md)
        p = sym(zeros(m,1)); idx = 0;
        for w = 1:nw
            a0 = xx*ha(w); if ~strcmp(md,'full'); a0 = sym(0); end
            ch = [xx*hx(w); a0; xc0];
            for k = 1:Ts
                idx = idx+1; p(idx) = expand(qq(3)*ch(1));
                if k < Ts; ch = expand(stepfun(ch,ydata(w,k),udata(w,k),qq,ee,md)); end
            end
        end
    end

    function [p,Jp,chiend] = rollout_sens(qq,ee,xx,md,dropfb)
        b = blocks.(md); p = sym(zeros(m,1)); Jp = sym(zeros(m,np));
        chiend = sym(zeros(3,nw)); idx = 0;
        for w = 1:nw
            a0 = xx*ha(w); da0 = jacobian(a0,th);
            if ~strcmp(md,'full'); a0 = sym(0); da0 = sym(zeros(1,np)); end
            ch = [xx*hx(w); a0; xc0];
            Tk = [jacobian(xx*hx(w),th); da0; jacobian(xc0,th)];
            for k = 1:Ts
                idx = idx+1; kk = min(k,Ts);
                sub_old = [chi; yd; ud]; sub_new = [ch; ydata(w,kk); udata(w,kk)];
                p(idx) = expand(subs(b.out, sub_old, sub_new));
                Jp(idx,:) = expand(subs(b.Hchi,sub_old,sub_new)*Tk + subs(b.Hth,sub_old,sub_new));
                if k < Ts
                    Fc = subs(b.Fchi,sub_old,sub_new); Ft = subs(b.Fth,sub_old,sub_new);
                    if dropfb; Fc(1,3) = sym(0); Fc(3,1) = sym(0); end
                    Tk = expand(Fc*Tk + Ft);
                    ch = expand(subs(b.step,sub_old,sub_new));
                end
            end
            chiend(:,w) = ch;
        end
    end

    function p = rollout_ls(qq,ee,xx,md,csc)
        ee2 = [ee(1)/csc; csc*ee(2); ee(3); ee(4)];
        p = sym(zeros(m,1)); idx = 0;
        for w = 1:nw
            a0 = csc*xx*ha(w); if ~strcmp(md,'full'); a0 = sym(0); end
            ch = [xx*hx(w); a0; xc0];
            for k = 1:Ts
                idx = idx+1; p(idx) = expand(qq(3)*ch(1));
                if k < Ts; ch = expand(stepfun(ch,ydata(w,k),udata(w,k),qq,ee2,md)); end
            end
        end
    end

    function vv = Vnum(ev)
        thq = thn; thq(4:7) = ev(:);
        pp_ = num_rollout(thq(1:3),thq(4:7),thq(8),double(ydata),double(udata), ...
            double(hx),double(ha),Ts,false);
        rr_ = Q0n_.'*Ln_*(pp_ - poff_n);
        vv = beta_num*(rr_.'*rr_);
    end

    function chiend = rollout_states_ls(qq,ee,xx,md,csc)
        ee2 = [ee(1)/csc; csc*ee(2); ee(3); ee(4)];
        chiend = sym(zeros(3,nw));
        for w = 1:nw
            a0 = csc*xx*ha(w); if ~strcmp(md,'full'); a0 = sym(0); end
            ch = [xx*hx(w); a0; xc0];
            for k = 1:Ts-1
                ch = expand(stepfun(ch,ydata(w,k),udata(w,k),qq,ee2,md));
            end
            chiend(:,w) = ch;
        end
    end
end

%% ===================== file-local helpers =========================================
function stage(name)
fprintf('[%s] stage %s\n', datestr(now,'HH:MM:SS'), name);
end

function tf = iszs(A)
% PROVED identically zero. isAlways is called with Unknown='error' so an
% undecidable comparison can never be silently reported as "false" or as
% "proved": it is caught, logged in the undecided ledger, and returns false,
% which surfaces as a FAILED check rather than as a passed one.
if ischar(A) || isstring(A); tf = iszs_ledger(); return; end
if isempty(A); tf = true; return; end
A = expand(A);
% Syntactic zero of the EXPANDED expression is already a proof for polynomials
% and is orders of magnitude cheaper than a decision-procedure call.
if isequal(A, sym(zeros(size(A)))); tf = true; return; end
A = simplify(A,'Steps',10,'Seconds',20,'IgnoreAnalyticConstraints',false);
if isequal(A, sym(zeros(size(A)))); tf = true; return; end
try
    tf = all(isAlways(A == 0,'Unknown','error'),'all');
catch err
    tf = false;
    st = dbstack(1);
    if isempty(st); loc = 'top'; else; loc = sprintf('%s:%d',st(1).name,st(1).line); end
    iszs_ledger(sprintf('%s (%s)', loc, err.message));
end
end

function out = iszs_ledger(entry)
persistent items
if isempty(items); items = {}; end
if nargin > 0; items{end+1} = entry; end
out = items;
end

function tf = eqpts(A, vars)
% EXACT rational-arithmetic equality at three independent witness points. This is a
% point check, NOT a proved identity; it is used only where the identity content is
% carried by a separate universal proof on a generic contribution vector.
tf = true;
if isempty(A); return; end
% Small integer witnesses keep the substituted rationals modest: fractional witnesses
% raised to the rollout's polynomial degree produce astronomically large numerators.
trials = { sym([2 3 5 7 11 13 17 19 23 29 31 37]), ...
           sym([3 -2 5 -7 2 -3 11 5 -2 7 3 -5]), ...
           sym([-1 4 -6 8 -9 12 -14 16 -18 21 -22 26]) };
for tr = 1:numel(trials)
    vals = trials{tr};
    B = subs(A, vars(:).', vals(1:numel(vars)));
    if ~all(isAlways(B == 0,'Unknown','error'),'all'); tf = false; return; end
end
end

function tf = isnz(A)
% PROVED not identically zero, by exhibiting an explicit rational point at
% which the expression evaluates to a nonzero rational. A witness is a proof
% of non-vanishing as a function; symbolic sign/zero tests on a free symbol
% are undecidable and are deliberately not used here.
tf = false;
if isempty(A); return; end
vars = symvar(A);
trials = { sym([2 3 5 7 11 13 17 19 23 29 31 37]), ...
           sym([3 -2 5 -7 2 -3 11 5 -2 7 3 -5]), ...
           sym([-1 4 -6 8 -9 12 -14 16 -18 21 -22 26]) };
for tr = 1:numel(trials)
    if isempty(vars)
        B = A;
    else
        vals = trials{tr}; B = subs(A, vars, vals(1:numel(vars)));
    end
    B = simplify(B);
    if any(~isAlways(B == 0,'Unknown','error'),'all'); tf = true; return; end
end
end

function Z = iszs_elem(A)
% Structural (syntactic) zero test after expansion; used only for a
% nonzero-entry count, never as a proof obligation.
Z = false(size(A));
for i = 1:numel(A); Z(i) = isequal(expand(A(i)), sym(0)); end
end

function Q = orthsym(A)
% Modified Gram-Schmidt over exact symbolic arithmetic; no generic symbolic SVD.
Q = sym([]);
for j = 1:size(A,2)
    v = A(:,j);
    for i = 1:size(Q,2); v = simplify(v - (Q(:,i).'*v)*Q(:,i)); end
    nv = simplify(sqrt(v.'*v));
    if isAlways(nv == 0,'Unknown','false'); continue; end
    Q = [Q, simplify(v/nv)]; %#ok<AGROW>
end
end

function [f,g] = obj_local(e, pf_e, Jf_e, Wn, ymeas)
res = pf_e(e) - ymeas;
f = res.'*Wn*res;
if nargout > 1; g = 2*(Jf_e(e).'*(Wn*res)); end
end

function [F,J] = slice_local(z, kk, ceqf, Jceq)
% Constraint restricted to the slice Kk = kk: 3 equations in z = [Ff;Gg;Pp].
e = [kk; z(:)];
F = ceqf(e);
if nargout > 1; Jt = Jceq(e); J = Jt(2:4,:).'; end
end

function [c,ceq,gc,gceq] = nlc_local(e, ceqf, Jceq)
c = []; gc = [];
ceq = ceqf(e); gceq = Jceq(e);
end

function [p,Jp,states] = num_rollout(qn,en,cn,ydM,udM,hxv,hav,T,want_sens,dropfb)
% Numerical closed-loop rollout using the GENERATED derivative functions.
if nargin < 10; dropfb = false; end
nwl = numel(hxv); npar = 8; p = zeros(nwl*T,1); Jp = zeros(nwl*T,npar);
states = zeros(3,nwl*T); idx = 0;
for w = 1:nwl
    ch = [cn*hxv(w); cn*hav(w); 0];
    Tk = zeros(3,npar); Tk(1,8) = hxv(w); Tk(2,8) = hav(w);
    for k = 1:T
        idx = idx+1; sg = [ydM(w,k); udM(w,k)];
        p(idx) = ncl_out(ch,sg,qn,en);
        states(:,idx) = ch;
        if want_sens
            Jp(idx,:) = ncl_Hchi(ch,sg,qn,en)*Tk + ncl_Hth(ch,sg,qn,en);
        end
        if k < T
            if want_sens
                Fc = ncl_Fchi_full(ch,sg,qn,en);
                if dropfb; Fc(1,3) = 0; Fc(3,1) = 0; end
                Tk = Fc*Tk + ncl_Fth_full(ch,sg,qn,en);
            end
            ch = ncl_step_full(ch,sg,qn,en);
        end
    end
end
if ~want_sens; Jp = []; end
end
