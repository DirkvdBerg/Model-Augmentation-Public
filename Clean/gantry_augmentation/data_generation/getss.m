function [sys, A, B, Css, D] = getss(n,M,C,K)
A = [zeros(n)   eye(n);
         -M\K    -M\C];
B = [zeros(n); eye(3)/M];
Css = [eye(3), zeros(n)];
D = zeros(3);

sys = ss(A,B,Css,D);
end
