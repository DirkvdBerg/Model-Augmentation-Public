function plant = gtd_build_plant(Y_op, cfg, K_gain)
% GTD_BUILD_PLANT  Discrete plant at Y_op and the FIXED controller K1 (or K2-g+).
%   plant = GTD_BUILD_PLANT(Y_op, cfg, K_gain) returns a struct with:
%     G       discrete plant in stage coords (f -> q), c2d ZOH, rigid baseline at Y_op
%     Cfb     discrete diagonal feedback controller: K_gain x K1
%     sys_cl  closed loop f -> q       = feedback(G, Cfb)
%     T_cl    complementary sens r -> q = feedback(G*Cfb, I)
%     Y_op    the operating point of the pre-check plant (and the record start)
%
%   THESIS-DATA (item 1, DATA-DESIGN 5.11 and section 10 item 1): the controller is
%   no longer designed at the record's Y_op. K1 is the generator's rule-of-thumb
%   diagonal controller (ruleOfThumb at cfg.fbw on each diagonal entry) designed
%   ONCE on the rigid nominal plant at Y = cfg.K1_design_Y = 0 and used unchanged in
%   every record; K_gain = cfg.K2_gain (1.2) gives K2-g+ for E5 only. Y_op keeps one
%   role: the plant of the linear limit pre-check (gtd_enforce_limits) and the
%   multisine crest-factor / yaw sizing, which is where the record starts.
%   Source: scripts/gantry/thesis-data-verification/vendor/original/gtd_build_plant.m (design at Y_op, no K_gain).

    if nargin < 3 || isempty(K_gain), K_gain = 1; end
    sys = rigid_plant(Y_op, cfg);

    % THESIS-DATA: controller from the design point, not from Y_op
    sys_K = rigid_plant(cfg.K1_design_Y, cfg);
    Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
    for j = 1:3
        Cfb(j,j) = ruleOfThumb(cfg.fbw, sys_K(j,j), cfg.ts);
    end
    Cfb = ss(Cfb);
    if K_gain ~= 1
        Cfb = K_gain * Cfb;                  % THESIS-DATA: K2-g+ = all K1 gains x 1.2
    end

    G      = c2d(sys, cfg.ts, 'zoh');
    sys_cl = feedback(G, Cfb);
    T_cl   = feedback(G*Cfb, eye(3));

    plant = struct('G',G, 'Cfb',Cfb, 'sys',sys, 'sys_cl',sys_cl, 'T_cl',T_cl, 'Y_op',Y_op, ...
                   'K_design_Y',cfg.K1_design_Y, 'K_gain',K_gain);
end

function sys = rigid_plant(Y, cfg)
% Rigid nominal plant (full payload mass at Y, no absorber, no friction) in stage
% coordinates, exactly the construction of the original gtd_build_plant.
    m1=cfg.m1; m2=cfg.m2; mb=cfg.mb; mh=cfg.mh; Jb=cfg.Jb; Jh=cfg.Jh;
    Lb=cfg.Lb; d=cfg.d; P=cfg.P;
    M_op = [m1+m2+mb+mh,        (m1-m2)*Lb/2 - mh*Y,                       0;
            (m1-m2)*Lb/2 - mh*Y, Jb+Jh + (m1+m2)*Lb^2/4 + mh*d^2 + mh*Y^2, -mh*d;
            0,                   -mh*d,                                     mh];
    sys = P.' * getss(cfg.n, M_op, cfg.C_damp, cfg.K) * P;
end
