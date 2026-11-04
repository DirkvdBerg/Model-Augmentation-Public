function [q, da, u_plant] = tdg_replica(r, f, v, plant, cfg)
% TDG_REPLICA  The recorded loop of model/gantry_tdg_2025a.slx, stepped in plain MATLAB.
%   [q, da, u_plant] = TDG_REPLICA(r, f, v, plant, cfg) returns what the Simulink run logs: the
%   true stage position q (N x 3), the absorber state delta_a (N x 1) and the plant force
%   (N x 3). r, f (feedforward) and v (encoder noise) are N x 3 in stage coordinates. The loop,
%   block by block as in the model, with the same arithmetic, so both give the same records bitwise:
%     q = P' x(1:3)                        (integrator -> selector -> gain P')
%     e = r - (q + v)                      (encoder sum, then error sum)
%     u_fb = C xc + D e; xc <- A xc + B e  (K, discrete at ts)
%     u_plant = f + u_fb
%     x <- RK4 at ts with P u_plant held   (gantrySystemExtendedTanh, mh = mh_rigid; solver ode4)
    P = cfg.P;  PT = P.';  h = cfg.ts;
    [Ac, Bc, Cc, Dc] = ssdata(plant.Cfb);
    base = {cfg.m1, cfg.m2, cfg.mb, cfg.mh_rigid, cfg.Lb, cfg.Jb, cfg.Jh, cfg.d, ...
            cfg.cg1, cfg.cg2, cfg.cb1, cfg.cb2, cfg.cy, cfg.kb1, cfg.kb2, ...
            cfg.ma, cfg.ka, cfg.ca, cfg.L0, cfg.cc1, cfg.cc2, cfg.ccy};
    g = cfg.friction_gain;
    N = size(r, 1);
    rT = r.';  fT = f.';  vT = v.';
    x = [0; 0; plant.Y_op; 0; 0; 0; 0; 0];
    xc = zeros(size(Ac, 1), 1);
    q = zeros(3, N);  da = zeros(N, 1);  up = zeros(3, N);
    for k = 1:N
        qk = PT * x(1:3);
        e  = rT(:, k) - (qk + vT(:, k));
        uf = Cc * xc + Dc * e;
        xc = Ac * xc + Bc * e;
        us = fT(:, k) + uf;
        u  = P * us;
        s1 = gantrySystemExtendedTanh(u, x, base{:}, g);
        s2 = gantrySystemExtendedTanh(u, x + (h / 2) * s1, base{:}, g);
        s3 = gantrySystemExtendedTanh(u, x + (h / 2) * s2, base{:}, g);
        s4 = gantrySystemExtendedTanh(u, x + h * s3, base{:}, g);
        q(:, k) = qk;  da(k) = x(4);  up(:, k) = us;
        x = x + (h / 6) * (s1 + 2 * s2 + 2 * s3 + s4);
    end
    q = q.';  u_plant = up.';
end
