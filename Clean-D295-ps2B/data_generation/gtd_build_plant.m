function plant = gtd_build_plant(Y_op, cfg, K_gain)
% GTD_BUILD_PLANT  Rigid nominal plant at Y_op and the fixed controller K_gain x K1.
%   plant = GTD_BUILD_PLANT(Y_op, cfg, K_gain) returns a struct with:
%     Cfb     discrete diagonal feedback controller K_gain x K1 (K_gain = 1 except E5)
%     sys_cl  closed loop f -> q       = feedback(G, Cfb)
%     T_cl    complementary sensitivity r -> q = feedback(G*Cfb, I)
%     Y_op    the operating point of the pre-check plant, where the record starts
%   with G the discrete (ZOH) rigid nominal plant at Y_op in stage coordinates.
%   K1 is ruleOfThumb at cfg.fbw on each diagonal entry of the rigid nominal plant at
%   Y = cfg.K1_design_Y, so every record has the same controller; Y_op only sets the plant of
%   the linear limit pre-check and of the multisine sizing.
    sys_K = rigid_plant(cfg.K1_design_Y, cfg);
    Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
    for j = 1:3
        Cfb(j,j) = ruleOfThumb(cfg.fbw, sys_K(j,j), cfg.ts);
    end
    Cfb = ss(Cfb);
    if K_gain ~= 1
        Cfb = K_gain * Cfb;
    end

    G      = c2d(rigid_plant(Y_op, cfg), cfg.ts, 'zoh');
    sys_cl = feedback(G, Cfb);
    T_cl   = feedback(G*Cfb, eye(3));

    plant = struct('Cfb',Cfb, 'sys_cl',sys_cl, 'T_cl',T_cl, 'Y_op',Y_op);
end

function sys = rigid_plant(Y, cfg)
% Rigid nominal plant (full payload mass at Y, no absorber, no friction) in stage coordinates.
    m1=cfg.m1; m2=cfg.m2; mb=cfg.mb; mh=cfg.mh; Jb=cfg.Jb; Jh=cfg.Jh;
    Lb=cfg.Lb; d=cfg.d; P=cfg.P;
    M_op = [m1+m2+mb+mh,        (m1-m2)*Lb/2 - mh*Y,                       0;
            (m1-m2)*Lb/2 - mh*Y, Jb+Jh + (m1+m2)*Lb^2/4 + mh*d^2 + mh*Y^2, -mh*d;
            0,                   -mh*d,                                     mh];
    sys = P.' * getss(cfg.n, M_op, cfg.C_damp, cfg.K) * P;
end
