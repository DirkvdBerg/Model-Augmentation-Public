function loop = build_loop(Y_op, cfg)
% BUILD_LOOP  The controller K1 and the linear closed loop at Y_op that the multisine sizing and the limit pre-check use.
%   As in Jasper's main.m: rigid plant from getss, stage coordinates P' G P, ruleOfThumb per stage axis.
    % one controller for every record, designed at Y = K1_design_Y
    sys_K = rigid_plant(cfg.K1_design_Y, cfg);
    Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
    for j = 1:3
        Cfb(j,j) = ruleOfThumb(cfg.fbw, sys_K(j,j), cfg.ts);
    end
    Cfb = ss(Cfb);

    % linear closed loop at the record's own Y_op, discrete with zero-order hold
    G = c2d(rigid_plant(Y_op, cfg), cfg.ts, 'zoh');
    loop = struct('Cfb', Cfb, 'sys_cl', feedback(G, Cfb), 'T_cl', feedback(G*Cfb, eye(3)), 'Y_op', Y_op);
end

function sys = rigid_plant(Y, cfg)
% Jasper's rigid plant at Y in stage coordinates: the full payload mass, no absorber, no friction
    m1=cfg.m1; m2=cfg.m2; mb=cfg.mb; mh=cfg.mh; Jb=cfg.Jb; Jh=cfg.Jh; Lb=cfg.Lb; d=cfg.d;
    M = [m1+m2+mb+mh,        (m1-m2)*Lb/2 - mh*Y,                       0;
         (m1-m2)*Lb/2 - mh*Y, Jb+Jh + (m1+m2)*Lb^2/4 + mh*d^2 + mh*Y^2, -mh*d;
         0,                   -mh*d,                                     mh];
    sys = cfg.P.' * getss(cfg.n, M, cfg.C_damp, cfg.K) * cfg.P;
end
