function gtd_plot_record(out, record, cfg, show)
% GTD_PLOT_RECORD  Per-trajectory figure: positions and decomposed forces.
%   GTD_PLOT_RECORD(out, record, cfg, show) saves one PNG per record to
%   cfg.fig_dir. With show=true the figure is left open on screen (used when
%   generating a selected subset); with show=false (default) it is rendered
%   off-screen and closed, so a full batch does not open 22 windows.
%   Layout: 4 (or 5) rows x 3 channels [X1/FX1, X2/FX2, Y/FY]
%     row 1  stage positions: reference r_sim vs response y
%     row 2  total force      (u_fb + f_sim)
%     row 3  feedback force    (trajectory tracking)
%     row 4  multisine force   (injected excitation)
%     row 5  hidden MSD delta_a (full width) -- only when the MSD is present;
%            for E1 the resonance shows as a bulge where the sweep crosses ~150 Hz
%   Forces are split by TYPE with independent y-scales per row, so the small
%   injected multisine is visible instead of being crushed under the feedback.
%   No conclusions asserted in titles (lessons rule): the signals are shown; the
%   reader judges.

    if nargin < 4, show = false; end

    t   = double(out.t_sim);
    y   = double(out.q_with);
    r   = double(out.r_sim);
    uft = double(out.u_total);
    ufb = double(out.u_fb);
    ums = double(out.f_ms);

    pos_lbl = {'X_1', 'X_2', 'Y'};
    frc_lbl = {'F_{X1}', 'F_{X2}', 'F_{Y}'};
    col_ref = [0.0 0.45 0.74]; col_resp = [0.85 0.33 0.10];
    col_fb  = [0.47 0.67 0.19]; col_ms   = [0.49 0.18 0.56];

    has_da = isfield(out, 'da_with') && ~isempty(out.da_with);
    nrows  = 4 + has_da;

    vis = 'off'; if show, vis = 'on'; end
    fig = figure('Name', record.id, 'Position', [60 20 1500 190*nrows], ...
                 'Color', 'w', 'Visible', vis);
    tl = tiledlayout(nrows, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    % Row 1: positions, reference vs response
    for ch = 1:3
        nexttile(ch); hold on
        plot(t, r(:,ch), '--', 'Color', col_ref,  'LineWidth', 1.0, 'DisplayName', 'reference');
        plot(t, y(:,ch),        'Color', col_resp, 'LineWidth', 0.8, 'DisplayName', 'response');
        ylabel([pos_lbl{ch} ' [m]']); grid on
        if ch == 1, legend('Location','best','FontSize',8); end
        set(gca, 'XTickLabel', []);
        title(pos_lbl{ch});
    end

    % Rows 2-4: forces split by type (total / feedback / multisine), each row
    % autoscaled independently so the multisine is visible on its own scale.
    force_rows = {uft, 'total', col_resp; ufb, 'feedback', col_fb; ums, 'multisine', col_ms};
    for rw = 1:3
        U = force_rows{rw,1}; name = force_rows{rw,2}; col = force_rows{rw,3};
        for ch = 1:3
            nexttile(3*rw + ch); hold on
            plot(t, U(:,ch), 'Color', col, 'LineWidth', 0.7);
            ylabel(sprintf('%s [N]', frc_lbl{ch})); grid on
            if rw < 3, set(gca, 'XTickLabel', []); else, xlabel('Time [s]'); end
            if ch == 1, title(sprintf('%s force', name)); end
            % delivered stage-force RMS + peak, top-right (see D-085 discussion)
            text(0.98, 0.95, sprintf('RMS %.0f / pk %.0f N', rms(U(:,ch)), max(abs(U(:,ch)))), ...
                 'Units','normalized', 'HorizontalAlignment','right', 'VerticalAlignment','top', ...
                 'FontSize', 7, 'BackgroundColor', [1 1 1 0.6]);
        end
    end

    % Row 5: hidden MSD displacement delta_a (full width), in micrometres
    if has_da
        da = 1e6 * double(out.da_with);
        nexttile([1 3]); hold on
        plot(t, da, 'Color', [0.30 0.30 0.30], 'LineWidth', 0.7);
        ylabel('\delta_a [\mum]'); xlabel('Time [s]'); grid on
        title('hidden MSD displacement \delta_a');
        text(0.98, 0.95, sprintf('RMS %.2f / pk %.2f \\mum', rms(da), max(abs(da))), ...
             'Units','normalized', 'HorizontalAlignment','right', 'VerticalAlignment','top', ...
             'FontSize', 7, 'BackgroundColor', [1 1 1 0.6]);
    end

    title(tl, sprintf('%s  |  %s  |  %s  |  seed %d', ...
          record.id, record.split, cfg.track, record.seed), ...
          'Interpreter', 'none', 'FontWeight', 'bold');

    if ~exist(cfg.fig_dir, 'dir'), mkdir(cfg.fig_dir); end
    exportgraphics(fig, fullfile(cfg.fig_dir, [record.id '.png']), 'Resolution', 150);
    if ~show, close(fig); end   % keep the window open only for a selected subset
end
