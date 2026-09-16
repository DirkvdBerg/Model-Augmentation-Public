%% Exact IO / delay-state equivalence, with two additional states.
% Companion derivation: ../documentation/io-additional-state-realisation.md
% No production model imports or training. Requires Symbolic Math Toolbox.
clear;
fprintf('IO / delay-state additional-state verification\n');
fprintf('MATLAB %s; Symbolic Math Toolbox %s\n', version, ver('symbolic').Version);
syms h1 h2 v a1 a2 u t1 t2 t3 w1 w2 w3 r1 r2 c1 c2 c3 e real
h = [h1;h2;v]; a = [a1;a2]; th = [t1;t2;t3]; coef = [c1;c2;c3];
weights = [w1;w2;w3;r1;r2];
phi = @(h,u) [h(1),h(2),u];
gy = @(h,a,u) w1*a(1)+w2*a(2)+w3*h(1)*u;
ga = @(h,a,u) [r1*a(1)+a(2)+w1*h(1); r2*a(2)+u*h(3)];
T = sym([0 0 0;1 0 0;0 0 0]); E = sym([1;0;0]); B = sym([0;0;1]);
Eaug = [E;sym(zeros(2,1))];
f = phi(h,u)*th;
d = gy(h,a,u)-phi(h,u)*coef;
io_y = f+d;
io_next = [io_y;h1;u;ga(h,a,u)];
baseline = [T*h+B*u+E*f;sym(zeros(2,1))];
learned = [E*d;ga(h,a,u)];
ss_next = baseline+learned;
checkzero(io_next-ss_next,'1: arbitrary one-step IO / SS transition');
checkzero(io_y-Eaug.'*ss_next,'2: output timing is newest NEXT history slot');
checkzero(subs(io_next-ss_next,coef,sym(zeros(3,1))), ...
    '3: unprojected additional-state model');
checkzero((T*h+B*u).'*E*d,'4: shift rows orthogonal to learned write');

%% Reference-set projection, independent of parameters and rollout states.
% Columns: h1,h2,previous input,a1,a2,current input.
Z = sym([1 0 0 1 0 0;0 1 1 0 1 0;0 0 -1 1 1 1; ...
         1 1 2 -1 2 1;2 -1 1 2 -1 -1]);
N = size(Z,1); Phi = sym(zeros(N,3)); Gy = sym(zeros(N,1));
W = sym(zeros(5*N,1)); Fb = sym(zeros(5*N,1));
for i=1:N
    hi=Z(i,1:3).'; ai=Z(i,4:5).'; ui=Z(i,6);
    rows=(i-1)*5+(1:5);
    Phi(i,:)=phi(hi,ui); Gy(i)=gy(hi,ai,ui);
    W(rows)=[E*Gy(i);ga(hi,ai,ui)];
    Fb(rows)=[T*hi+B*ui+E*Phi(i,:)*th;sym(zeros(2,1))];
end
Q=kron(sym(eye(N)),Eaug); Psi=Q*Phi;
aux=pinv(Phi)*Gy; corrected=W-Psi*aux;
checkzero(pinv(Psi)*W-aux,'5: IO and SS auxiliary coefficients identical');
checkzero(Phi.'*(Gy-Phi*aux),'6: IO stacked orthogonality');
checkzero(Psi.'*corrected,'7: SS stacked orthogonality');
checkzero(Fb.'*corrected,'8: full baseline transition orthogonality');
checkzero(Phi.'*jacobian(Gy-Phi*aux,weights),'9: differentiated projection');
% Deliberately duplicate a column to test rank deficiency, not just full rank.
Phir=[Phi,Phi(:,1)]; Psir=Q*Phir;
assert(rank(Phir)==3 && size(Phir,2)==4);
checkzero(pinv(Psir)*W-pinv(Phir)*Gy,'10: rank-deficient coefficient equivalence');
checkzero(Phir.'*(Gy-Phir*pinv(Phir)*Gy),'11: rank-deficient orthogonality');

%% Four-step free simulation: independent history and state implementations.
% Keep inputs and initial conditions symbolic; rational parameters limit growth.
syms u0 u1 u2 u3 real
U=[u0;u1;u2;u3];
old=[th;weights;coef];
new=[sym([1; -1; 2])/3; sym([1;2;1;-1;1])/5; sym([1;-1;2])/7];
H=io_y; A=ga(h,a,u);
H=subs(H,old,new); A=subs(A,old,new);
S=subs(ss_next,old,new);
history=h; latent=a; state=[h;a]; args=[h;a;u];
for k=1:4
    yi=subs(H,args,[history;latent;U(k)]);
    anext=subs(A,args,[history;latent;U(k)]);
    ys=subs(H,args,[state;U(k)]);
    snext=subs(S,args,[state;U(k)]);
    history=[yi;history(1);U(k)]; latent=anext; state=snext;
    checkzero([yi-ys;history-state(1:3);latent-state(4:5)], ...
        sprintf('12.%d: free simulation output/history/latent equality',k));
end

%% Measured-history prediction and noise use the same feedback in both forms.
syms measured real
measured_io=[measured;h1;u;ga(h,a,u)];
measured_ss=ss_next+Eaug*(measured-io_y);
checkzero(measured_io-measured_ss,'13: measured-history predictor equivalence');
stochastic_io=[io_y+e;h1;u;ga(h,a,u)];
checkzero(stochastic_io-(ss_next+Eaug*e),'14: stochastic history injection');
% Prove a counterexample, rather than treating an undecidable identity as false.
witness=subs(measured_io(1)-ss_next(1),measured,io_y+1);
assert(isAlways(simplify(witness)==1));
fprintf('PASS 15: measured and predicted feedback are not interchangeable\n');
fprintf('ALL CHECKS PASSED. Physical gantry equivalence and statistical theorems not tested.\n');

function checkzero(expr,label)
    residual=simplify(expr);
    assert(all(isAlways(residual(:)==0)), 'Failed: %s',label);
    fprintf('PASS %s\n',label);
end
