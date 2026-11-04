function in = tdg_build_inputs(e, cfg)
% TDG_BUILD_INPUTS  Everything a record needs before simulation, from its manifest row.
%   in = TDG_BUILD_INPUTS(e, cfg) returns plant (pre-check plant at e.Y_op with the record's
%   controller), r and t (reference), pk (realised reference peaks), ms (multisine, seed
%   e.seed_ms) and chk (linear limit pre-check, with the multisine scale). The same row gives
%   the same r and f bitwise, so both versions of a record share them.
    rec = e;
    in.plant = gtd_build_plant(e.Y_op, cfg, e.K_gain);
    rec.seed = e.seed_sp;                                     % setpoint draws (aprbs only)
    [in.r, in.t] = gtd_make_reference(rec, cfg);
    in.pk = gtd_validate_ref(in.r, in.t, e.file, cfg.lim, e.p, cfg.acc_margin);
    rec.seed = e.seed_ms;                                     % multisine phases
    in.ms = gtd_make_multisine(rec, in.plant, cfg);
    [~, in.chk] = gtd_enforce_limits(in.plant, in.r, in.ms.f_stage, cfg);
end
