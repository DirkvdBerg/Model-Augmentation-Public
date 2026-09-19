function gtd_validate_ref(r, t, id, lim)
% GTD_VALIDATE_REF  Assert a reference is within the enforced hardware limits.
%   GTD_VALIDATE_REF(r, t, id, lim) checks positions, the 6 mm yaw budget,
%   velocity and acceleration (acceleration on r only). Same checks as the
%   validate_ref local function in generate_oscillatory_multisine_data.m,
%   reading the anchored lim struct from cfg.lim.

    ts  = t(2) - t(1);
    vel = diff(r)  / ts;
    acc = diff(vel) / ts;

    assert(max(abs(r(:,1)))          <= lim.pos_X, '%s: X1 position limit', id);
    assert(max(abs(r(:,2)))          <= lim.pos_X, '%s: X2 position limit', id);
    assert(max(r(:,3))               <=  lim.pos_Y, '%s: Y+ position limit', id);
    assert(min(r(:,3))               >= -lim.pos_Y, '%s: Y- position limit', id);
    assert(max(abs(r(:,1)-r(:,2)))   <= lim.diff,  '%s: yaw |X1-X2| limit', id);
    assert(max(abs(vel(:,1:2)),[],'all') <= lim.vel,   '%s: X velocity limit', id);
    assert(max(abs(vel(:,3)))        <= lim.vel,    '%s: Y velocity limit', id);
    assert(max(abs(acc(:,1:2)),[],'all') <= lim.acc_X, '%s: X acceleration limit', id);
    assert(max(abs(acc(:,3)))        <= lim.acc_Y,  '%s: Y acceleration limit', id);

    fprintf('  r OK %-20s: X1=[%+.0f %+.0f] X2=[%+.0f %+.0f] Y=[%+.0f %+.0f] mm\n', id, ...
            min(r(:,1))*1e3, max(r(:,1))*1e3, min(r(:,2))*1e3, max(r(:,2))*1e3, ...
            min(r(:,3))*1e3, max(r(:,3))*1e3);
end
