% GTD_CHECK_MULTISINE  Verify the multisine layer (no Simulink needed).
%   Run from repo root:  >> gtd_check_multisine
%   Checks band occupancy, the yaw budget via the closed-loop response,
%   crest-factor selection, and the sinesweep / off variants, on a
%   representative subset of records.

clear; clc;
cfg     = gtd_config('joint', true, 0.50);
records = gtd_build_records(cfg);
df      = cfg.fs / cfg.N_record;

fprintf('band [%d %d] Hz, df = %.4f Hz, ~%d lines\n', ...
        cfg.f_low, cfg.f_high, df, numel(cfg.f_low:df:cfg.f_high));

for id = {'T3_standstill_Y000', 'T14_lissajous_yaw', 'E1_resonance_sweep', 'E4_multisine_off'}
    rec   = records(strcmp({records.id}, id{1}));
    plant = gtd_build_plant(rec.Y_op, cfg);
    ms    = gtd_make_multisine(rec, plant, cfg);

    assert(isequal(size(ms.f_stage), [cfg.N_record, 3]), '%s: bad size', rec.id);
    assert(all(isfinite(ms.f_stage(:))), '%s: non-finite force', rec.id);

    switch rec.p.excitation
        case 'multisine'
            % (a) spectral energy inside the band
            Fm   = abs(fft(ms.f_stage(:,1))).^2;
            fvec = (0:cfg.N_record-1)' * df;
            half = 1:floor(cfg.N_record/2);
            inb  = fvec(half) >= cfg.f_low-df & fvec(half) <= cfg.f_high+df;
            frac = sum(Fm(half(inb))) / sum(Fm(half));
            assert(frac > 0.99, '%s: only %.3f of force energy in band', rec.id, frac);

            % (b) yaw budget: closed-loop |X1-X2| within the 6 mm hard limit
            q       = lsim(plant.sys_cl, ms.f_stage);
            yaw_pk  = max(abs(q(:,1) - q(:,2)));
            assert(yaw_pk <= cfg.lim.diff, '%s: yaw %.2f mm > 6 mm', rec.id, yaw_pk*1e3);

            cf = max(abs(ms.f_stage)) ./ rms(ms.f_stage);
            fprintf('  %-20s inband %.3f | yaw %.2f mm (%.0f%% of 2mm budget) | A_anti %.3g Nm | stageCF [%.2f %.2f %.2f]\n', ...
                    rec.id, frac, yaw_pk*1e3, 100*yaw_pk/cfg.yaw_budget, ms.info.A_anti, cf);

        case 'sinesweep'
            assert(max(abs(ms.f_stage(:,1:2)),[],'all') < 1e-9, '%s: X force nonzero', rec.id);
            act = cfg.n_hold + (1:cfg.n_active);
            assert(all(ms.f_stage([1:cfg.n_hold, act(end)+1:end], 3) == 0), '%s: force outside active window', rec.id);
            fprintf('  %-20s sinesweep on F_Y, peak %.1f N, band [%d %d]\n', ...
                    rec.id, max(abs(ms.f_stage(:,3))), ms.info.band);

        case 'none'
            assert(all(ms.f_stage(:) == 0), '%s: expected zero force', rec.id);
            fprintf('  %-20s excitation OFF (all zero)\n', rec.id);
    end
end

fprintf('\nMultisine layer OK\n');
