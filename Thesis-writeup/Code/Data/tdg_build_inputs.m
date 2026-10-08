function in = tdg_build_inputs(e, cfg, verbose)
% TDG_BUILD_INPUTS  Everything a record needs before simulation, from its manifest row.
%   in = TDG_BUILD_INPUTS(e, cfg) returns plant (pre-check plant at e.Y_op with the
%   record's controller), r, t (reference), pk (realised reference peaks), ms
%   (multisine, seed e.seed_ms), f_safe and chk (linear pre-check scale).
%   Deterministic: the same row gives the same r and f bitwise, so the noisy and the
%   noise-free versions of a record share them exactly.
    if nargin < 3, verbose = true; end
    rec = e;  rec.id = e.file;
    in.plant = gtd_build_plant(e.Y_op, cfg, e.K_gain);
    rec.seed = e.seed_sp;                                     % setpoint draws (aprbs only)
    [in.r, in.t] = gtd_make_reference_telica(rec, cfg);       % dispatcher for every class
    assert(size(in.r, 1) == cfg.N_record, '%s: %d samples', e.file, size(in.r, 1));
    if verbose
        in.pk = gtd_validate_ref(in.r, in.t, e.file, cfg.lim, e.p, cfg.acc_margin);
    else
        evalc('in.pk = gtd_validate_ref(in.r, in.t, e.file, cfg.lim, e.p, cfg.acc_margin);');
    end
    rec.seed = e.seed_ms;                                     % multisine phases
    in.ms = gtd_make_multisine(rec, in.plant, cfg);
    [in.f_safe, in.chk] = gtd_enforce_limits(in.plant, in.r, in.ms.f_stage, cfg);
end
