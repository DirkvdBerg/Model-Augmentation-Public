function pc = tdg_post_check(out, cfg)
% TDG_POST_CHECK  Binding limit check on the SIMULATED TRUTH (DATA-DESIGN section 10
% item 2). The copied generator checks limits only on the linear rigid-baseline
% pre-check (gtd_enforce_limits); this checks the Simulink T_AF response itself:
%   position |X1|, |X2| <= lim.pos_X, |Y| <= lim.pos_Y, yaw |X1 - X2| <= lim.diff,
%   velocity |dq/dt| <= lim.vel (first difference at 20 kHz, as validate_response),
%   plant force F = u_total (no force noise since D-225): peak <= lim.force_peak, rms over the record <= lim.force_rms.
% pc.pass is true iff all hold; the realised values and margins are returned for meta.
    lim = cfg.lim;
    q = out.q_with;
    v = diff(q) * cfg.fs;
    F = out.u_total;                  % D-225: no force noise; the plant receives u_total
    pc.pos_peak   = max(abs(q));
    pc.yaw_peak   = max(abs(q(:,1) - q(:,2)));
    pc.vel_peak   = max(abs(v));
    pc.force_peak = max(abs(F));
    pc.force_rms  = rms(F);
    pc.checks = struct( ...
        'pos',   all(pc.pos_peak <= [lim.pos_X, lim.pos_X, lim.pos_Y]), ...
        'yaw',   pc.yaw_peak <= lim.diff, ...
        'vel',   all(pc.vel_peak <= lim.vel), ...
        'fpeak', all(pc.force_peak <= lim.force_peak), ...
        'frms',  all(pc.force_rms <= lim.force_rms));
    c = struct2cell(pc.checks);
    pc.pass = all([c{:}]);
    nm = fieldnames(pc.checks);
    pc.failed = strjoin(nm(~[c{:}]), ',');
    pc.force_peak_frac = pc.force_peak ./ lim.force_peak;       % share of the limit used
    pc.force_rms_frac  = pc.force_rms  ./ lim.force_rms;
end
