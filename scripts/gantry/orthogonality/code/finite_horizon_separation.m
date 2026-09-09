function finite_horizon_separation
% Independent CAS cross-check of abstract identities in gaps.md Section 9.
% Category: structural verification examples. Not a gantry-model transcription.
% General-dimensional proofs are written in gaps.md, not inferred from examples.
syms a b real
I = sym(eye(5)); L = diag(sym([2 3 4 5 6]));
E = L\I(:,5);
S = L\[I(:,1)+I(:,5), I(:,2)];
R = L\[a*I(:,1)+I(:,3)+I(:,5), b*I(:,2)+I(:,4)];
PE = (L*E)*pinv(L*E); M = I-PE;
Sb = M*L*S; Rb = M*L*R;
PR = Rb*inv(Rb.'*Rb)*Rb.';
PN = PE+PR; N = L*[E R];
H = simplify(Sb.'*(I-PR)*Sb);
C = struct;
C.weighted_profiled_curvature = z(H-diag([1/(1+a^2),1/(1+b^2)]));
C.sequential_equals_joint_projection = z(PN-N*inv(N.'*N)*N.');
C.projector_symmetric_idempotent = z(PN-PN.') && z(PN*PN-PN);
C.nonorthogonal_can_be_identifiable = isAlways(det(subs(H,[a b],[1 2]))>0) && ~z(Sb.'*subs(Rb,[a b],[1 2]));
C.orthogonal_limit_preserves_curvature = z(subs(H,[a b],[0 0])-Sb.'*Sb);
RR = subs(Rb,[a b],[1 2]); RR = [RR 2*RR];
C.redundant_neural_columns_allowed = z(RR*pinv(RR)-subs(PR,[a b],[1 2]));
D = sym('d',[5 1],'real'); res = sym('e',[5 1],'real');
before = inv(Sb.'*Sb)*Sb.'*M*L*res;
after = inv(Sb.'*Sb)*Sb.'*M*L*(res-D);
C.fixed_contribution_displacement = z(after-before+pinv(Sb)*M*L*D);
C.displacement_normal_equation = z(Sb.'*Sb*(after-before)+Sb.'*M*L*D);
eye3 = sym(eye(3)); e1=eye3(:,1); e2=eye3(:,2); enc=e1+e2;
me=eye3-enc*pinv(enc);
C.encoder_creates_cancellation = rank([e1 e2])==2 && z(me*(e1+e2)) && rank([enc e2 e1])==2;
syms t real
d=t*e1;
C.zero_value_not_zero_tangent = z(subs(d,t,0)) && isAlways(e1.'*diff(d,t)==1);
C.physical_rank_is_required = rank([e1 e1])<2 && z([e1 e1]*sym([1;-1]));
beta_reg = sym('beta_reg','positive');
sp=[sym(1);sym(0)]; rp=[sym(1);sqrt(beta_reg)];
hb=simplify(sp.'*(sym(eye(2))-rp*pinv(rp))*sp);
C.soft_penalty_restores_local_curvature = isAlways(hb==beta_reg/(1+beta_reg));
C.blind_penalty_cannot_restore_curvature = z(sp.'*(sym(eye(2))-sp*pinv(sp))*sp);
overlap=subs(Sb.'*PR*Sb,[a b],[1 2]); rho2=max(eig(overlap));
C.principal_angle_bound_attained = isAlways(rho2==sym(4)/5) && isAlways(min(eig(subs(H,[a b],[1 2])))==1-rho2);

% Polynomial dynamic model written independently in MATLAB.
syms x za u th k g c real
q=[x;za]; p=[th;k;g;c];
F=[th*x+x^2/10+k*za+u;g*za+x/5+u/7];
J=jacobian(F,q); B=jacobian(F,p);
state=[c;2*c]; T=jacobian(state,p);
ys=sym(zeros(3,1)); deriv=sym(zeros(3,4));
inputs=sym([1 -1 3])./[5 10 10];
for i=1:3
    values=[state;inputs(i)];
    T=subs(J,[q;u],values)*T+subs(B,[q;u],values);
    state=subs(F,[q;u],values);
    ys(i)=state(1); deriv(i,:)=T(1,:);
end
C.symbolic_rollout_chain_rule = z(jacobian(ys,p)-deriv);
C.state_scheduler_derivative_included = isAlways(J(1,1)==th+x/5);
C.latent_K_path_present = isAlways(simplify(deriv(1,2)-2*c)==0);
C.encoder_path_present = isAlways(simplify(deriv(1,4)-(th+2*k+c/5))==0);
fields=fieldnames(C);
for i=1:numel(fields)
    assert(C.(fields{i}), 'Failed: %s', fields{i});
end
result=struct('category','structural verification; generic examples only',...
    'engine',version,'checks',C,'curvature',{{char(H(1,1)),char(H(2,2))}});
out=fullfile(fileparts(mfilename('fullpath')),'finite_horizon_separation_matlab.json');
fid=fopen(out,'w'); assert(fid>=0); cleanup=onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',jsonencode(result,PrettyPrint=true));
fprintf('All %d symbolic checks passed.\n',numel(fields));
end

function yes=z(A)
yes=all(isAlways(simplify(A)==0),'all');
end
