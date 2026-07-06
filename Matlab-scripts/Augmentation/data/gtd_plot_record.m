function gtd_plot_record(out, record, cfg, show)
% GTD_PLOT_RECORD  Per-trajectory figure: positions and decomposed forces.
%   GTD_PLOT_RECORD(out, record, cfg, show) saves one PNG per record to
%   cfg.fig_dir. With show=true the figure is left open on screen (used when
%   generating a selected subset); with show=false (default) it is rendered
%   off-screen and closed, so a full batch does not open 22 windows.
%   Two rows of three channels:
%     row 1  stage positions [X1, X2, Y]: reference r_sim vs response y
%     row 2  stage forces   [FX1, FX2, FY]: total = feedback u_fb + multisine f_sim
%   No conclusions asserted in titles (lessons rule): the signals and the limit
%   line are shown; the reader judges.

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

    vis = 'off'; if show, vis = 'on'; end
    fig = figure('Name', record.id, 'Position', [60 60 1500 720], ...
                 'Color', 'w', 'Visible', vis);
    tl = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

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

    % Row 2: forces, total decomposed into feedback + multisine
    for ch = 1:3
        nexttile(3 + ch); hold on
        plot(t, uft(:,ch), 'Color', col_resp, 'LineWidth', 0.9, 'DisplayName', 'total');
        plot(t, ufb(:,ch), 'Color', col_fb,   'LineWidth', 0.7, 'DisplayName', 'feedback');
        plot(t, ums(:,ch), 'Color', col_ms,   'LineWidth', 0.7, 'DisplayName', 'multisine');
        ylabel([frc_lbl{ch} ' [N]']); xlabel('Time [s]'); grid on
        if ch == 1, legend('Location','best','FontSize',8); end
        title(frc_lbl{ch});
    end

    title(tl, sprintf('%s  |  %s  |  %s  |  seed %d', ...
          record.id, record.split, cfg.track, record.seed), ...
          'Interpreter', 'none', 'FontWeight', 'bold');

    if ~exist(cfg.fig_dir, 'dir'), mkdir(cfg.fig_dir); end
    exportgraphics(fig, fullfile(cfg.fig_dir, [record.id '.png']), 'Resolution', 150);
    if ~show, close(fig); end   % keep the window open only for a selected subset
end
