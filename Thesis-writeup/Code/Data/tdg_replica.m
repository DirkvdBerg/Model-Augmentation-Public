function [q, da, u_plant] = tdg_replica(r, f, v, plant, cfg)
% TDG_REPLICA  The recorded loop of model/gantry_tdg_2025a.slx, stepped outside Simulink (D-231).
%   [q, da, u_plant] = TDG_REPLICA(r, f, v, plant, cfg) returns what gtd_run_simulation reads
%   from the Simulink run: the TRUE stage position q (N x 3, the To Workspace1 log q_aug), the
%   absorber state delta_a (N x 1, delta_a_ode) and the plant force (N x 3, the PlantForceLog of
%   the Sum2 output). r, f (feedforward) and v (encoder noise) are N x 3 in stage coordinates.
%
%   The loop, block by block as in the model (tdg_make_model.m):
%     Reference1, Feedforward1, EncoderNoise: sources at ts, held over the step
%     q = P' x(1:3)                        (Integrator -> Selector1 -> Gain4)
%     e = r - (q + v)                      (EncoderSum, then Sum3)
%     u_fb = C xc + D e; xc <- A xc + B e  (LTI System1 = K1, discrete at ts)
%     u_plant = f + u_fb                   (Sum2)
%     x <- RK4 at ts with P u_plant held   (Gain3 -> Extended ODE -> Integrator, solver ode4)
%   with the Extended ODE = gantrySystemExtendedTanh at cfg.friction_gain and mh = mh_rigid (the
%   swap gtd_run_simulation makes for the run). The arithmetic is that of
%   scripts/gantry/coulomb-tanh-gain/cl_sim.m, which reproduced the Simulink generator bitwise
%   (D-223 gate, D-224 C17, D-225 C19); D-231 checks this file the same way on saved records.
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
