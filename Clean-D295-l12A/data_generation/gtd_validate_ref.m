function pk = gtd_validate_ref(r, t, id, lim, p, acc_margin)
% GTD_VALIDATE_REF  Assert that a reference is within the hardware limits; return its peaks.
%   pk = GTD_VALIDATE_REF(r, t, id, lim, p, acc_margin) checks positions, the 6 mm yaw budget,
%   velocity and acceleration of r and returns the realised peaks per stage axis [X1 X2 Y]
%   (pk.v, pk.a, pk.j) and of generalised X_sym and Y (pk.vl, pk.al, pk.jl). Acceleration exceptions
%   by name, from the record parameters p:
%     p.acc_exception  true: the acceleration assert is skipped and the value only reported
%                      (E6, the measured ILC move at 1.31 x max)
%     p.acc_margin     true: the acceleration limit is lim.acc x (1 + acc_margin) (E3 at 1.0 x max)
    ts  = t(2) - t(1);
    vel = diff(r)  / ts;
    acc = diff(vel) / ts;
    jrk = diff(acc) / ts;

    assert(max(abs(r(:,1)))          <= lim.pos_X, '%s: X1 position limit', id);
    assert(max(abs(r(:,2)))          <= lim.pos_X, '%s: X2 position limit', id);
    assert(max(r(:,3))               <=  lim.pos_Y, '%s: Y+ position limit', id);
    assert(min(r(:,3))               >= -lim.pos_Y, '%s: Y- position limit', id);
    assert(max(abs(r(:,1)-r(:,2)))   <= lim.diff,  '%s: yaw |X1-X2| limit', id);
    assert(max(abs(vel(:,1:2)),[],'all') <= lim.vel,   '%s: X velocity limit', id);
    assert(max(abs(vel(:,3)))        <= lim.vel,    '%s: Y velocity limit', id);
    fa = 1;
    if isfield(p, 'acc_margin') && p.acc_margin, fa = 1 + acc_margin; end
    if ~(isfield(p, 'acc_exception') && p.acc_exception)
        assert(max(abs(acc(:,1:2)),[],'all') <= fa*lim.acc_X, '%s: X acceleration limit', id);
        assert(max(abs(acc(:,3)))        <= fa*lim.acc_Y,  '%s: Y acceleration limit', id);
    else
        fprintf('  %s: acceleration assert skipped by name (realised X %.2f, Y %.2f m/s^2)\n', id, ...
                max(abs(acc(:,1:2)),[],'all'), max(abs(acc(:,3))));
    end

    lg = [(r(:,1) + r(:,2))/2, r(:,3)];
    vl = diff(lg)/ts;  al = diff(vl)/ts;  jl = diff(al)/ts;
    pk = struct('v', max(abs(vel)), 'a', max(abs(acc)), 'j', max(abs(jrk)), ...
                'vl', max(abs(vl)), 'al', max(abs(al)), 'jl', max(abs(jl)), ...
                'Xsym_range', [min(lg(:,1)), max(lg(:,1))], 'Y_range', [min(r(:,3)), max(r(:,3))], ...
                'Xanti_max', max(abs(r(:,1) - r(:,2)))/2);
    fprintf('  r OK %-12s: X1=[%+.0f %+.0f] X2=[%+.0f %+.0f] Y=[%+.0f %+.0f] mm | v %.3f/%.3f m/s a %.2f/%.2f m/s^2 j %.0f/%.0f m/s^3 (X_sym/Y)\n', id, ...
            min(r(:,1))*1e3, max(r(:,1))*1e3, min(r(:,2))*1e3, max(r(:,2))*1e3, ...
            min(r(:,3))*1e3, max(r(:,3))*1e3, pk.vl(1), pk.vl(2), pk.al(1), pk.al(2), pk.jl(1), pk.jl(2));
end
