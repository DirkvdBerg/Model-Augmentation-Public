function pc = tdg_post_check(out, cfg)
% TDG_POST_CHECK  Limit check on the simulated truth of one record version.
%   pc = TDG_POST_CHECK(out, cfg) checks the true stage position |X1|, |X2| <= lim.pos_X,
%   |Y| <= lim.pos_Y, yaw |X1 - X2| <= lim.diff, velocity |dq/dt| <= lim.vel (first difference at
%   20 kHz) and the plant force u_total: peak <= lim.force_peak, rms over the record <= lim.force_rms.
%   pc.pass is true iff all hold; pc.failed names the failed checks; the realised values and the
%   shares of the force limits are stored in the record.
    lim = cfg.lim;
    q = out.q_with;
    v = diff(q) * cfg.fs;
    F = out.u_total;                  % the plant force
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
