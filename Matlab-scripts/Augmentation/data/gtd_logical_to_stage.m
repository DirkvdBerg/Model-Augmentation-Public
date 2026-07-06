function F_stage = gtd_logical_to_stage(F_logical, cfg)
% GTD_LOGICAL_TO_STAGE  Generalized (logical) forces -> stage rail forces.
%   F_stage = GTD_LOGICAL_TO_STAGE(F_logical, cfg) maps the logical force
%   channels [f_sym, f_anti, f_Y] to stage forces [F_X1, F_X2, F_Y].
%
%   The plant is built as sys = P' * G_phys * P with positions q_stage = P'*q_l.
%   By virtual-work invariance (F_stage . dq_stage = F_logical . dq_logical) the
%   force transform is the dual of the position transform:
%
%       F_stage = P^{-1} * F_logical
%              = [ 1/2,  1/Lb, 0 ] [ f_sym  ]
%                [ 1/2, -1/Lb, 0 ] [ f_anti ]
%                [ 0,    0,    1 ] [ f_Y    ]
%
%   NOTE: f_anti is a yaw TORQUE [N*m] (logical coord 2 is the tilt angle
%   theta ~ (X1-X2)/Lb), so it is divided by Lb to become a rail force. The
%   naive "F_X1 = f_sym + f_anti" map is both mis-normalized and dimensionally
%   wrong. Verified by gtd_check_transform.
%
%   F_logical may be (N x 3) with rows = time samples, or a 3-element vector.

    if isvector(F_logical) && numel(F_logical) == 3
        F_stage = cfg.P \ F_logical(:);
        if isrow(F_logical), F_stage = F_stage.'; end
    else
        assert(size(F_logical,2) == 3, 'F_logical must be (N x 3) or 3-vector');
        F_stage = (cfg.P \ F_logical.').';   % solve P*X = F_logical' -> X = P^{-1} F_logical'
    end
end
