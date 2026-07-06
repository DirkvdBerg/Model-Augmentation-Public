function plant = gtd_build_plant(Y_op, cfg)
% GTD_BUILD_PLANT  Discrete plant and frozen-Y controller at operating point Y_op.
%   plant = GTD_BUILD_PLANT(Y_op, cfg) returns a struct with:
%     G       discrete plant in logical coords (r/f -> q), c2d ZOH
%     Cfb     discrete diagonal feedback controller (ruleOfThumb at cfg.fbw)
%     sys_cl  closed loop f -> q       = feedback(G, Cfb)
%     T_cl    complementary sens r -> q = feedback(G*Cfb, I)
%     Y_op    the operating point used
%
%   The controller is designed on the rigid nominal (cfg.mh, full payload mass)
%   and stays frozen for the whole record (D-039). Y-dependence enters only
%   through the mass matrix M_op. Mirrors generate_oscillatory_multisine_data.m
%   lines 250-256, 283, 293.

    m1=cfg.m1; m2=cfg.m2; mb=cfg.mb; mh=cfg.mh; Jb=cfg.Jb; Jh=cfg.Jh;
    Lb=cfg.Lb; d=cfg.d; P=cfg.P; ts=cfg.ts;

    M_op = [m1+m2+mb+mh,           (m1-m2)*Lb/2 - mh*Y_op,                       0;
            (m1-m2)*Lb/2 - mh*Y_op, Jb+Jh + (m1+m2)*Lb^2/4 + mh*d^2 + mh*Y_op^2, -mh*d;
            0,                      -mh*d,                                        mh];

    sys = P.' * getss(cfg.n, M_op, cfg.C_damp, cfg.K) * P;

    Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
    for j = 1:3
        Cfb(j,j) = ruleOfThumb(cfg.fbw, sys(j,j), ts);
    end
    Cfb = ss(Cfb);

    G      = c2d(sys, ts, 'zoh');
    sys_cl = feedback(G, Cfb);
    T_cl   = feedback(G*Cfb, eye(3));

    plant = struct('G',G, 'Cfb',Cfb, 'sys',sys, 'sys_cl',sys_cl, 'T_cl',T_cl, 'Y_op',Y_op);
end
