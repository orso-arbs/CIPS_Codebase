import numpy as np
import matplotlib.pyplot as plt
import warnings
import glob
import os
import pandas as pd
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D as _Line2D
import Format_1 as F_1

# ── LaTeX / font settings ─────────────────────────────────────────────────────
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb,amsfonts}'
plt.rcParams['mathtext.fontset'] = 'cm'
LATEX_FONT_SIZE = 16
plt.rcParams['font.size'] = LATEX_FONT_SIZE


@F_1.ParameterLog(max_size=1024 * 10)
def plot10_histogram_ridgeline(
    # Input
    input_dir,
    SRec_df=None,

    # ──────────────────────────────────────────────────────────────────────────
    # KEY PARAMETER: number of time steps (ridges) to display
    n_time_points=12,
    # ──────────────────────────────────────────────────────────────────────────

    # Distribution
    dist_column='d_cell_SRec_distribution_nonDim',
    x_label=r'$d_c \;/\; \delta_T$',

    # Time axis labelling
    time_column=None,
    time_normalization_col=None,

    # ── Histogram ─────────────────────────────────────────────────────────────
    bin_count=40,

    # ── Stagger: each successive (lower/later) ridge is shifted right ──────────
    x_stagger=True,             # Enable/disable horizontal staggering
    x_stagger_fraction=0.5,     # Total stagger as fraction of data span (0.5 → half the x-range)

    # ── Aesthetics ────────────────────────────────────────────────────────────
    ridge_color='#3cb371',          # even ridges (i=0,2,4,...) — green
    edge_color='#1a7a40',
    ridge_color_alt='#b8d44a',      # odd ridges (i=1,3,5,...) — greenish-yellow
    edge_color_alt='#7a9020',
    fill_alpha=0.72,
    edge_linewidth=1.0,
    overlap=0.65,

    # Mean line
    show_mean_lines=True,
    mean_line_color='#1a1a1a',
    mean_line_linewidth=2.5,
    mean_line_linestyle='--',

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir_manual='',
    output_dir_comment='',
    save_svg=True,
    save_png=True,
    show_plots=False,
    output_filename='ridgeline_diameter_distribution',

    # ── Figure ────────────────────────────────────────────────────────────────
    figure_width=11,
    figure_height=20,

    # ── Unified font size ─────────────────────────────────────────────────────
    fontsize=28,

    # ── Y-axis label position (axes fraction: 0=bottom, 1=top; x shifts left/right) ──
    ylabel_x=0.0,   # horizontal position in axes fraction
    ylabel_y=1.0,   # vertical position in axes fraction (1.0 = top)

    # ── Legend ───────────────────────────────────────────────────────────────
    show_legend=True,  # True = show legend, False = hide it
    legend_x=0.5,      # 0=left edge, 0.5=centre, 1=right edge
    legend_y=0.98,     # 0=bottom, 1=top

    # ── X-axis range ─────────────────────────────────────────────────────────
    x_trim_percentile=0.0,
    x_axis_extension=1.03,
    x_axis_limit=None,

    # ── Image selection ───────────────────────────────────────────────────────
    image_numbers=[],
    omit_image_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 106],

    Plot_log_level=1,
):
    """
    Publication-quality ridgeline (joy) plot of cell diameter distributions
    over time.

    Key design choices
    ------------------
    * **Time downward**: earliest step at top, latest at bottom.
    * **No white mask**: overflow bars (bars taller than the stride) flow
      uninterrupted into the space of the ridge above.  The dark thin baseline
      line is drawn at high zorder to mark the zero level without a white gap.
    * **Mean line full height**: the dashed mean-diameter line always spans the
      entire allocated stride, so it is readable regardless of bar height.
    * **Stagger**: each successive (lower/later) ridge is shifted slightly to
      the right by *x_stagger_fraction* × bin_width, giving a 3-D perspective
      that clarifies which bars belong to which ridge when they overlap.
    * **Frequency scale bar**: a reference bar on the right shows what a given
      frequency percentage (default 10 %) looks like as a bar height.

    Parameters
    ----------
    input_dir : str
    SRec_df : pd.DataFrame or None
    n_time_points : int
        Number of equally-spaced ridges. Includes the very first and very last
        available time steps.
    dist_column : str
    x_label : str
    time_column : str or None
    time_normalization_col : str or None
    bin_count : int
        Histogram bin count. Same global bins for all ridges.
    x_stagger : bool
        Enable horizontal stagger.
    x_stagger_fraction : float
        Shift per ridge step as a fraction of one bin width. Default 0.25.
    ridge_color, edge_color : str
    fill_alpha : float
    overlap : float in [0, 1]
    show_mean_lines : bool
    mean_line_color, mean_line_linewidth, mean_line_linestyle
    show_scale_bar : bool
    scale_bar_pct : float
        Reference frequency percentage shown by the scale bar.
    fontsize : int
        Controls ALL text uniformly.
    x_trim_percentile : float
    image_numbers : list
    omit_image_list : list

    Returns
    -------
    output_dir : str
    """
    if Plot_log_level < 2:
        warnings.filterwarnings('ignore', category=mpl.MatplotlibDeprecationWarning)

    ######################################################################  I/O
    output_dir = F_1.F_out_dir(
        input_dir, __file__,
        output_dir_comment=output_dir_comment,
        output_dir_manual=output_dir_manual,
    )
    png_dir = svg_dir = None
    if save_png:
        png_dir = os.path.join(output_dir, 'png_plots')
        os.makedirs(png_dir, exist_ok=True)
        print(f'PNG: {png_dir}') if Plot_log_level >= 1 else None
    if save_svg:
        svg_dir = os.path.join(output_dir, 'svg_plots')
        os.makedirs(svg_dir, exist_ok=True)
        print(f'SVG: {svg_dir}') if Plot_log_level >= 1 else None

    ######################################################################  Load
    if SRec_df is None:
        pkl_path = os.path.join(input_dir, 'SRec_DataFrame.pkl')
        if not os.path.exists(pkl_path):
            pkl_files = glob.glob(os.path.join(input_dir, '*.pkl'))
            if not pkl_files:
                raise FileNotFoundError(f'No .pkl in {input_dir}')
            pkl_path = pkl_files[0]
        print(f'Loading: {pkl_path}') if Plot_log_level >= 1 else None
        SRec_df = pd.read_pickle(pkl_path)

    #####################################################################  Clean
    dist_clean_col = f'{dist_column}_ridgeline_clean'
    SRec_df[dist_clean_col] = SRec_df[dist_column].apply(
        lambda x: x[~np.isnan(x)] if x is not None and len(x) > 0
        else np.array([])
    )
    SRec_df = SRec_df[SRec_df[dist_clean_col].apply(len) > 0].reset_index(drop=True)
    if len(SRec_df) == 0:
        print('Error: no valid data.') if Plot_log_level >= 0 else None
        return output_dir

    ####################################################################  Filter
    if image_numbers and 'image_number' in SRec_df.columns:
        SRec_df = SRec_df[SRec_df['image_number'].isin(image_numbers)].reset_index(drop=True)
    if omit_image_list and 'image_number' in SRec_df.columns:
        SRec_df = SRec_df[~SRec_df['image_number'].isin(omit_image_list)].reset_index(drop=True)
    if len(SRec_df) == 0:
        print('Error: no images after filtering.') if Plot_log_level >= 0 else None
        return output_dir

    ##########################################################  Auto time column
    if time_column is None:
        time_column = 'Time_VisIt' if 'Time_VisIt' in SRec_df.columns else 'image_number'
        print(f'Time column: "{time_column}"') if Plot_log_level >= 1 else None

    ###############################################################  Sort by time
    if time_column in SRec_df.columns:
        SRec_df = SRec_df.sort_values(time_column).reset_index(drop=True)
        print(f'Sorted by "{time_column}". Range: '
              f'{float(SRec_df[time_column].iloc[0]):.4g} – '
              f'{float(SRec_df[time_column].iloc[-1]):.4g}') if Plot_log_level >= 1 else None

    ##############################################################  Select rows
    # Always include the FIRST and LAST available time instance;
    # the remaining n-2 are equally spaced in between.
    N_avail = len(SRec_df)
    n = min(n_time_points, N_avail)
    if n < n_time_points and Plot_log_level >= 0:
        print(f'Warning: {N_avail} images → using {n} ridges.')
    indices = np.round(np.linspace(0, N_avail - 1, n)).astype(int)
    df = SRec_df.iloc[indices].reset_index(drop=True)
    print(f'Ridgeline: {n} ridges from {N_avail} images '
          f'(index {indices[0]}…{indices[-1]}).') if Plot_log_level >= 1 else None

    #############################################################  Build labels
    # Only the numeric value — the axis label "t" is shown once via set_ylabel.
    def _label(row, i):
        if time_column not in row.index:
            return rf'${i+1}$'
        tv = float(row[time_column])
        if time_normalization_col and time_normalization_col in row.index:
            nv = float(row[time_normalization_col])
            tv = tv / nv if nv != 0 else tv
            return rf'${tv:.2f}$'
        if time_column == 'image_number':
            return rf'${int(tv)}$'
        if time_column == 'Time_VisIt':
            return rf'${tv:.2f}$'
        return rf'${tv:.3f}$'

    labels = [_label(df.iloc[i], i) for i in range(n)]

    ###############################################################  X range
    all_data = np.concatenate(df[dist_clean_col].values)
    xp_lo = np.percentile(all_data, x_trim_percentile)
    xp_hi = np.percentile(all_data, 100.0 - x_trim_percentile)
    span = xp_hi - xp_lo
    if x_axis_limit is not None:
        if isinstance(x_axis_limit, (tuple, list)) and len(x_axis_limit) == 2:
            x_lo, x_hi = float(x_axis_limit[0]), float(x_axis_limit[1])
        else:
            x_lo = xp_lo - 0.02 * span
            x_hi = float(x_axis_limit)
    else:
        x_lo = xp_lo - 0.02 * span
        x_hi = xp_hi + 0.02 * span * x_axis_extension

    #########################################################  Histogram bins
    bins_global = np.linspace(x_lo, x_hi, bin_count + 1)
    # Step-wise x: [e0, e1, e1, e2, ..., e_{N-1}, eN]  (length 2*bin_count)
    x_step_base = np.repeat(bins_global, 2)[1:-1]

    # Stagger: ridge i (i=0=top=early, i=n-1=bottom=late) is shifted right by
    # i * stagger_step.  x_stagger_fraction is the TOTAL stagger as a fraction
    # of the data span (e.g. 0.5 → total shift = half the x-range).
    stagger_step = (x_stagger_fraction * span / max(n - 1, 1)) if x_stagger else 0.0
    total_stagger = (n - 1) * stagger_step   # = x_stagger_fraction * span

    ############################################################  Histograms
    hists_pct = []
    means = []
    counts = []
    for i in range(n):
        data = df.iloc[i][dist_clean_col]
        data_in = data[(data >= x_lo) & (data <= x_hi)]
        if len(data_in) == 0:
            data_in = data
        total = len(data_in)
        hist_vals, _ = np.histogram(data_in, bins=bins_global)
        hist_pct = hist_vals * 100.0 / total if total > 0 else np.zeros(bin_count)
        hists_pct.append(hist_pct)
        means.append(float(np.mean(data_in)))
        counts.append(int(len(data)))

    g_max = max((np.max(h) for h in hists_pct if np.max(h) > 0), default=1.0)

    ################################################################  Layout
    ridge_h = 1.0
    stride  = ridge_h * (1.0 - overlap)

    # TIME DOWNWARD: i=0 (earliest) → highest baseline (top)
    def _bl(i):  return float((n - 1 - i) * stride)
    def _z(i):   return (i + 1) * 5   # later (more recent) ridges drawn in front

    ################################################################  Figure
    fig, ax = plt.subplots(figsize=(figure_width, figure_height), facecolor='none')
    ax.set_facecolor('none')

    ################################################################  Draw ridges
    x_plot_lo = x_lo
    x_plot_hi = x_hi + total_stagger   # rightmost edge including all stagger

    for i in range(n):
        bl  = _bl(i)
        z   = _z(i)
        xs  = i * stagger_step          # horizontal shift for this ridge

        # Alternate fill/edge colours every ridge for visual separation
        rc = ridge_color     if i % 2 == 0 else ridge_color_alt
        ec = edge_color      if i % 2 == 0 else edge_color_alt

        h_norm = hists_pct[i] / g_max * ridge_h
        y_step = np.repeat(h_norm, 2)

        x_step_i = x_step_base + xs

        # ── Coloured fill ─────────────────────────────────────────────────────
        ax.fill_between(x_step_i, bl, bl + y_step,
                        color=rc, alpha=fill_alpha,
                        zorder=z, linewidth=0)

        # ── Step outline ──────────────────────────────────────────────────────
        x_out = np.r_[x_lo + xs, x_step_i, x_hi + xs]
        y_out = np.r_[bl,         bl + y_step,       bl]
        ax.plot(x_out, y_out, color=ec, lw=edge_linewidth,
                zorder=z + 1, solid_capstyle='butt')

        # ── Baseline line — thin, drawn OVER everything (high z) so it marks
        #    the ridge's zero level clearly, without a white gap in overflow bars.
        #    It is a single thin dark line (no white underlayer).
        bl_arr   = np.full(2, bl)
        x_b_ends = np.array([x_plot_lo, x_plot_hi])
        ax.plot(x_b_ends, bl_arr,
                color=edge_color, lw=0.7, alpha=0.55,
                zorder=z + 3, solid_capstyle='butt')

        # ── Mean line — spans the FULL stride height (always readable) ────────
        if show_mean_lines and x_lo <= means[i] <= x_hi:
            mean_x = means[i] + xs          # shift with the ridge
            mean_top = bl + stride          # full allotted height
            ax.plot([mean_x, mean_x], [bl, mean_top],
                    color=mean_line_color,
                    lw=mean_line_linewidth,
                    ls=mean_line_linestyle,
                    zorder=z + 4,
                    solid_capstyle='round')

    ########################################################  Frequency scale bar
    # (removed — frequency indicators not shown)

    #######################################################  Y-axis time labels
    ax.set_yticks([_bl(i) for i in range(n)])
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.tick_params(axis='y', length=0, pad=10)
    ax.spines['left'].set_visible(False)
    # Time-axis label — position tunable via ylabel_x / ylabel_y
    ax.set_ylabel(r'$t\,/\,\tau$', fontsize=fontsize, rotation=0,
                  position=(ylabel_x, ylabel_y), ha='center', va='bottom', labelpad=36)

    ##############################################################  X-axis
    y_top = _bl(0) + ridge_h * 1.15
    y_bot = _bl(n - 1) - stride * 0.30
    # x: left margin + right margin for stagger
    right_extra = total_stagger + 0.02 * span
    ax.set_xlim(float(x_lo - 0.01 * span), float(x_hi + right_extra))
    ax.set_ylim(y_bot, y_top)
    ax.set_xlabel(x_label, fontsize=fontsize, labelpad=12)

    # Slanted (parallelepiped) grid lines:
    # Ridge i=0 (top/earliest) has 0 stagger; ridge i=n-1 (bottom/latest) has total_stagger.
    # At data value x_d: top ridge draws at x_d, bottom ridge at x_d + total_stagger.
    # Slanted gridline: (x_d + total_stagger, y_bot) → (x_d, y_top)
    # X-axis ticks sit at x_d + total_stagger, labelled with x_d.
    # Ticks at every even integer within the data range
    first_even = int(np.ceil(float(x_lo) / 2.0)) * 2
    last_even  = int(np.floor(float(x_hi) / 2.0)) * 2
    tick_data_vals = np.arange(first_even, last_even + 1, 2, dtype=float)
    ax.xaxis.grid(False)
    for xd in tick_data_vals:
        ax.plot([float(xd + total_stagger), float(xd)],
                [float(y_bot),              float(y_top)],
                ls='--', color='#aaaaaa', alpha=0.30, lw=0.8, zorder=0)
    # Ticks at bottom-ridge positions, labels show unshifted data values
    ax.set_xticks(tick_data_vals + total_stagger)
    ax.set_xticklabels([f'{int(v):d}' for v in tick_data_vals], fontsize=fontsize)

    ax.set_axisbelow(False)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#555555')
    ax.spines['bottom'].set_linewidth(0.8)

    ###################################################################  Legend
    # Custom handler: draws two side-by-side coloured rectangles in one key
    from matplotlib.legend_handler import HandlerPatch
    import matplotlib.patches as _mpatches

    class _HandlerSplit(HandlerPatch):
        def __init__(self, c1, c2, ec1, ec2, a, **kw):
            super().__init__(**kw)
            self._c1, self._c2, self._ec1, self._ec2, self._a = c1, c2, ec1, ec2, a
        def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
            mid = w / 2
            r1 = _mpatches.Rectangle((float(-xd),       float(-yd)), mid, h,
                                     facecolor=self._c1, edgecolor=self._ec1,
                                     alpha=self._a, transform=trans)
            r2 = _mpatches.Rectangle((float(-xd + mid), float(-yd)), mid, h,
                                     facecolor=self._c2, edgecolor=self._ec2,
                                     alpha=self._a, transform=trans)
            return [r1, r2]

    dc_patch = Patch(facecolor=ridge_color, edgecolor=edge_color,
                     alpha=fill_alpha, label=r'$d_c$')
    legend_handles: list = [dc_patch]
    handler_map: dict = {
        dc_patch: _HandlerSplit(ridge_color, ridge_color_alt,
                                edge_color, edge_color_alt, fill_alpha)
    }
    if show_mean_lines:
        legend_handles.append(
            _Line2D([0], [0], color=mean_line_color, lw=mean_line_linewidth,
                    ls=mean_line_linestyle, label=r'$\overline{d}_c$')
        )
    if show_legend:
        leg = ax.legend(handles=legend_handles, handler_map=handler_map,
                        fontsize=fontsize,
                        loc='upper center', bbox_to_anchor=(legend_x, legend_y),
                        framealpha=1.0, handlelength=1.8, handletextpad=0.6,
                        ncol=1)
        leg.set_zorder(9999)  # above all histogram bars regardless of n

    ###################################################################  Layout
    plt.tight_layout()
    fig.subplots_adjust(left=0.22)

    ######################################################################  Save
    if save_png and png_dir is not None:
        path = os.path.join(png_dir, f'{output_filename}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', transparent=True)
        print(f'Saved PNG: {path}') if Plot_log_level >= 1 else None
    if save_svg and svg_dir is not None:
        path = os.path.join(svg_dir, f'{output_filename}.svg')
        plt.savefig(path, format='svg', bbox_inches='tight', transparent=True)
        print(f'Saved SVG: {path}') if Plot_log_level >= 1 else None
    if show_plots:
        plt.show()
    plt.close(fig)

    print('Ridgeline complete.') if Plot_log_level >= 1 else None
    return output_dir


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Running Ridgeline Diameter Distribution Plotter...')
    plot10_histogram_ridgeline(
        input_dir=(
            r'C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis'
            r'\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554'
            r'\20250625_1626096\20250626_1700136\20250628_2007187'
        ),
        output_dir_comment='3D Diameter Ridgeline Distribution',

        # ── Key parameter ────────────────────────────────────────────────────
        n_time_points=40, #30 was good for overview
        # ─────────────────────────────────────────────────────────────────────

        dist_column='d_cell_SRec_distribution_nonDim',
        x_label=r'$d_c \;/\; \delta_T$',
        bin_count=40,

        ridge_color='#3cb371',      # green — even ridges
        edge_color='#1a7a40',
        ridge_color_alt='#b8d44a',  # greenish-yellow — odd ridges
        edge_color_alt='#7a9020',
        fill_alpha=0.72,
        overlap=0.65,

        x_stagger=False,
        x_stagger_fraction=0.2,

        show_mean_lines=True,
        mean_line_color='#1a1a1a',
        mean_line_linewidth=2.5,
        mean_line_linestyle='--',

        fontsize=28,

        save_png=True,
        save_svg=True,
        show_plots=True,
        output_filename='Spost15_Diameter_3D_ridgeline',

        omit_image_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                         10, 11, 12, 13, 14, 15, 106],

        # t/τ label position (axes fraction: x=0 is left edge, y=1 is top)
        ylabel_x=0.1,    # shift left (<0) or right (>0)
        ylabel_y=0.95,   # 1.0 = top of axis, 0.5 = middle, 0.0 = bottom

        # legend
        show_legend=False,  # True = show, False = hide
        legend_x=0.5,      # horizontal anchor: 0=left, 0.5=centre, 1=right
        legend_y=0.98,     # vertical anchor:   0=bottom, 1=top

        Plot_log_level=1,
    )
