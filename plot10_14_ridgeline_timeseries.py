import os
import warnings
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def _load_A11_rdot():
    """Load A11 R_mean_dot manual extraction data."""
    path = (
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction"
        r"\A11_SF_R_mean_dot_as_first_time_derivative_of_the_average_radius_of_the_"
        r"wrinkled_flame_front_vs_time_manual_extraction.txt"
    )
    return pd.read_csv(path)


@F_1.ParameterLog(max_size=1024 * 10)
def plot10_14_ridgeline_timeseries(
    input_dir,

    # ── Distribution column (ridgeline) ───────────────────────────────────────
    dist_column='d_cell_SRec_distribution_CSTx6_nonDim',

    # ── Key parameter: number of ridges ───────────────────────────────────────
    n_time_points=15,

    # ── Image selection ───────────────────────────────────────────────────────
    image_numbers=[],
    omit_image_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 106],

    # ── Histogram ─────────────────────────────────────────────────────────────
    bin_count=40,
    x_trim_percentile=0.0,
    x_axis_limit=None,

    # ── Ridgeline aesthetics ──────────────────────────────────────────────────
    ridge_color='#3cb371',          # even ridges (i=0,2,4,...) — green
    edge_color='#1a7a40',
    ridge_color_alt='#b8d44a',      # odd ridges (i=1,3,5,...) — greenish-yellow
    edge_color_alt='#7a9020',
    fill_alpha=0.55,
    edge_linewidth=0.8,
    overlap=0.65,

    # ── Per-ridge mean marker ─────────────────────────────────────────────────
    show_mean_markers=False,
    mean_marker_color='#1a1a1a',
    mean_marker_linewidth=2.0,
    mean_marker_linestyle='--',

    # ── Font / figure ─────────────────────────────────────────────────────────
    fontsize=20,
    figsize=(18, 9),
    dpi=300,

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir_manual='',
    output_dir_comment='',
    save_svg=True,
    save_png=True,
    show_plots=False,
    output_filename='ridgeline_timeseries_combined',

    Plot_log_level=1,
):
    """
    Combined rotated ridgeline + time-series plot.

    Layout (single panel, shared x-axis = time t/τ)
    ------------------------------------------------
    Left y-axis         : R_dot  (blue)
    Right y-axis 1      : d_c/δ_T — shared by ridgeline histograms
                          AND the mean-diameter curve (green)
    Right y-axis 2      : N_cells (red, spine offset outward)

    Ridgeline orientation
    ---------------------
    Rotated vs. plot10: time on x-axis, d_c on y-axis.
    At each selected time t_i a horizontal histogram is drawn: bars extend
    to the right (toward later time) proportional to bin frequency.
    Earlier ridges have higher z-order (drawn in front).

    X-tick marks are placed at every histogram (ridge) location.
    """
    if Plot_log_level < 2:
        warnings.filterwarnings('ignore', category=mpl.MatplotlibDeprecationWarning)

    # ── Output dirs ───────────────────────────────────────────────────────────
    output_dir = F_1.F_out_dir(input_dir, __file__,
                               output_dir_comment=output_dir_comment,
                               output_dir_manual=output_dir_manual)
    png_dir = svg_dir = None
    if save_png:
        png_dir = os.path.join(output_dir, 'png_plots')
        os.makedirs(png_dir, exist_ok=True)
        print(f'PNG: {png_dir}') if Plot_log_level >= 1 else None
    if save_svg:
        svg_dir = os.path.join(output_dir, 'svg_plots')
        os.makedirs(svg_dir, exist_ok=True)
        print(f'SVG: {svg_dir}') if Plot_log_level >= 1 else None

    # ── Load Analysis df ──────────────────────────────────────────────────────
    pkl_files = glob.glob(os.path.join(input_dir, 'Analysis_A11_final_df.pkl'))
    if not pkl_files:
        raise FileNotFoundError(f'No Analysis_A11_final_df.pkl in {input_dir}')
    df = pd.read_pickle(pkl_files[0])
    if Plot_log_level >= 1:
        print(f'Loaded Analysis df from {pkl_files[0]}')

    # ── Load A11 R_mean_dot ───────────────────────────────────────────────────
    rdot_df = _load_A11_rdot()

    # ── Filter images ─────────────────────────────────────────────────────────
    if image_numbers and 'image_number' in df.columns:
        df = df[df['image_number'].isin(image_numbers)]
    df = df.sort_values('Time_VisIt').reset_index(drop=True)

    # Keep unfiltered copy for continuous lines (d_mean, N_cells span full time)
    df_lines = df.copy()

    if omit_image_list and 'image_number' in df.columns:
        df = df[~df['image_number'].isin(omit_image_list)]
    df = df.sort_values('Time_VisIt').reset_index(drop=True)
    if len(df) == 0:
        print('Error: no data after filtering.')
        return output_dir

    # ── Resolve distribution column ───────────────────────────────────────────
    # Priority: specified column → fallbacks in Analysis df → SRec_DataFrame.pkl
    dist_col_actual = dist_column
    _fallback_cols = [
        'd_cell_SRec_distribution_CSTx6_nonDim',
        'd_cell_SRec_distribution_nonDim',
        'd_cell_distribution_nonDim',
    ]

    if dist_col_actual not in df.columns:
        found = False
        for fc in _fallback_cols:
            if fc in df.columns:
                dist_col_actual = fc
                found = True
                print(f'Using fallback distribution column in Analysis df: {fc}') if Plot_log_level >= 1 else None
                break

        if not found:
            # Try SRec_DataFrame.pkl
            srec_path = os.path.join(input_dir, 'SRec_DataFrame.pkl')
            if not os.path.exists(srec_path):
                srec_candidates = [f for f in glob.glob(os.path.join(input_dir, '*.pkl'))
                                   if 'Analysis' not in os.path.basename(f)]
                if not srec_candidates:
                    raise FileNotFoundError(
                        f'No distribution column found in Analysis df and no SRec pkl in {input_dir}.\n'
                        f'Available Analysis df columns: {list(df.columns[:30])}'
                    )
                srec_path = srec_candidates[0]

            srec_df = pd.read_pickle(srec_path)
            if image_numbers and 'image_number' in srec_df.columns:
                srec_df = srec_df[srec_df['image_number'].isin(image_numbers)]
            if omit_image_list and 'image_number' in srec_df.columns:
                srec_df = srec_df[~srec_df['image_number'].isin(omit_image_list)]

            dist_col_actual = None
            for fc in _fallback_cols:
                if fc in srec_df.columns:
                    dist_col_actual = fc
                    break
            if dist_col_actual is None:
                raise ValueError(
                    f'No distribution column found in SRec df. '
                    f'Available: {list(srec_df.columns[:20])}'
                )

            srec_merge = srec_df[['image_number', dist_col_actual]].drop_duplicates('image_number')
            df = df.merge(srec_merge, on='image_number', how='inner')
            df = df.sort_values('Time_VisIt').reset_index(drop=True)
            if Plot_log_level >= 1:
                print(f'Merged distribution from SRec df, column: {dist_col_actual}')

    if Plot_log_level >= 1:
        print(f'Distribution column: {dist_col_actual}')

    # ── Clean distributions ───────────────────────────────────────────────────
    clean_col = f'{dist_col_actual}_clean'
    df[clean_col] = df[dist_col_actual].apply(
        lambda x: x[~np.isnan(x)] if x is not None and len(x) > 0 else np.array([])
    )
    df = df[df[clean_col].apply(len) > 0].reset_index(drop=True)
    if len(df) == 0:
        print('Error: no valid distribution data after cleaning.')
        return output_dir

    # ── Select n_time_points ridges (equally spaced, incl. first & last) ──────
    N_avail = len(df)
    n = min(n_time_points, N_avail)
    if n < n_time_points and Plot_log_level >= 0:
        print(f'Warning: only {N_avail} images available → using {n} ridges.')
    indices = np.round(np.linspace(0, N_avail - 1, n)).astype(int)
    df_ridges = df.iloc[indices].reset_index(drop=True)
    time_values = df_ridges['Time_VisIt'].values.astype(float)
    if Plot_log_level >= 1:
        print(f'Ridgeline: {n} ridges, t in [{time_values[0]:.4f}, {time_values[-1]:.4f}]')

    # ── Global histogram bins (in d space) ────────────────────────────────────
    all_data = np.concatenate(df_ridges[clean_col].values)
    if x_axis_limit is not None:
        if isinstance(x_axis_limit, (tuple, list)) and len(x_axis_limit) == 2:
            d_lo, d_hi = float(x_axis_limit[0]), float(x_axis_limit[1])
        else:
            d_lo = float(np.percentile(all_data, x_trim_percentile))
            d_hi = float(x_axis_limit)
    else:
        d_lo = float(np.percentile(all_data, x_trim_percentile))
        d_hi = float(np.percentile(all_data, 100.0 - x_trim_percentile))
    d_span = d_hi - d_lo
    d_lo -= 0.02 * d_span
    d_hi += 0.02 * d_span

    bins_global = np.linspace(d_lo, d_hi, bin_count + 1)

    # ── Compute per-ridge histograms ──────────────────────────────────────────
    hists_pct = []
    means = []
    for i in range(n):
        data = df_ridges.iloc[i][clean_col]
        data_in = data[(data >= d_lo) & (data <= d_hi)]
        if len(data_in) == 0:
            data_in = data
        total = len(data_in)
        hist_vals, _ = np.histogram(data_in, bins=bins_global)
        hist_pct = hist_vals * 100.0 / total if total > 0 else np.zeros(bin_count)
        hists_pct.append(hist_pct)
        means.append(float(np.mean(data_in)))

    g_max = max((np.max(h) for h in hists_pct if np.max(h) > 0), default=1.0)

    # ── Time stride / ridge width ─────────────────────────────────────────────
    # ridge_h_t: max bar extent in time units (a bar at 100% freq has this width)
    # stride_t = ridge_h_t * (1 - overlap) = dt_mean  →  ridge_h_t = dt_mean / (1-overlap)
    dt_mean = (time_values[-1] - time_values[0]) / max(n - 1, 1)
    ridge_h_t = dt_mean / max(1.0 - overlap, 1e-6)
    stride_t  = dt_mean  # spacing between consecutive ridge baselines

    # ── Figure and axes ───────────────────────────────────────────────────────
    plt.rcParams['font.size'] = fontsize
    fig, ax_main = plt.subplots(figsize=figsize, dpi=dpi)

    # ax_d   : right y-axis 1 — diameter d_c (ridgeline + d_mean curve)
    # ax_N   : right y-axis 2 — N_cells (offset outward)
    ax_d = ax_main.twinx()
    ax_N = ax_main.twinx()
    ax_N.spines['right'].set_position(('outward', 70))

    # ── Draw ridgeline on ax_d ────────────────────────────────────────────────
    for i in range(n):
        t_i = float(time_values[i])
        # Earlier ridges in front (higher z); mirrors plot10 convention
        z = (n - i) * 5 + 10

        # Alternate fill/edge colours every ridge for visual separation
        rc = ridge_color     if i % 2 == 0 else ridge_color_alt
        ec = edge_color      if i % 2 == 0 else edge_color_alt

        # Normalised bar widths in time units: shape (B,)
        h_norm_t = hists_pct[i] / g_max * ridge_h_t

        # ── Coloured fill: only non-zero bins to avoid zero-width artefacts ───
        for j in range(bin_count):
            if h_norm_t[j] > 0:
                ax_d.fill_betweenx(
                    [bins_global[j], bins_global[j + 1]],
                    t_i, t_i + h_norm_t[j],
                    color=rc, alpha=fill_alpha, zorder=z, linewidth=0,
                )

        # ── Step outline — NaN at zero-freq bins prevents horizontal baseline
        #    lines from appearing where bars return to t_i.
        # y_out = np.repeat(bins_global, 2)  shape (2*(B+1),)
        # x_out = [t_i, t_i+h0,t_i+h0, ..., t_i+h_{B-1},t_i+h_{B-1}, t_i]
        x_step = np.repeat(h_norm_t, 2)
        y_out = np.repeat(bins_global, 2).astype(float)
        x_out = np.r_[t_i, t_i + x_step, t_i].astype(float)
        for j in range(bin_count):
            if h_norm_t[j] < 1e-10:
                x_out[2 * j + 1] = np.nan
                x_out[2 * j + 2] = np.nan
        ax_d.plot(x_out, y_out, color=ec, lw=edge_linewidth,
                  zorder=z + 1, solid_capstyle='butt')

        # ── Per-ridge mean marker (short horizontal dashed line) ──────────────
        if show_mean_markers and d_lo <= means[i] <= d_hi:
            ax_d.plot([t_i, t_i + stride_t],
                      [means[i], means[i]],
                      color=mean_marker_color,
                      lw=mean_marker_linewidth,
                      ls=mean_marker_linestyle,
                      zorder=z + 4,
                      solid_capstyle='round')

    # ── Mean diameter curve (continuous) on ax_d — uses full time range ──────
    line_d_mean, = ax_d.plot(
        df_lines['Time_VisIt'], df_lines['d_cell_SRec_mean_CSTx6_nonDim'],
        color='green', lw=2.0, zorder=300,
        label=r'$\overline{d}_c/\delta_T$',
    )
    ax_d.set_ylabel(r'$d_c \;/\; \delta_T$', color='green', fontsize=fontsize)
    ax_d.tick_params(axis='y', labelcolor='green', direction='out', labelsize=fontsize)

    # ── N_cells on ax_N — uses full time range ────────────────────────────────
    line_N, = ax_N.plot(
        df_lines['Time_VisIt'], df_lines['N_cells_CSTx6'],
        color='red', lw=2.0, zorder=301,
        label=r'$N_{cells}$',
    )
    ax_N.set_ylabel(r'$N_{cells}$', color='red', fontsize=fontsize)
    ax_N.tick_params(axis='y', labelcolor='red', direction='out', labelsize=fontsize)

    # ── R_dot on ax_main (left y-axis) ────────────────────────────────────────
    line_rdot, = ax_main.plot(
        rdot_df['time'], rdot_df['R_mean_dot'],
        color='blue', lw=2.0, zorder=302,
        label=r'$\dot{R}_{SEF}/S_L$',
    )
    ax_main.set_ylabel(r'$\dot{R}_{SEF}/S_L$', color='blue', fontsize=fontsize)
    ax_main.tick_params(axis='y', labelcolor='blue', direction='out', labelsize=fontsize)

    # ── X-axis: ticks at every histogram location ─────────────────────────────
    ax_main.set_xticks(time_values)
    tick_labels = [rf'${tv:.2f}$' for tv in time_values]
    ax_main.set_xticklabels(tick_labels, fontsize=max(fontsize - 6, 9),
                             rotation=45, ha='right')
    ax_main.set_xlabel(r'$t\,/\,\tau$', fontsize=fontsize)

    # ── X limits ──────────────────────────────────────────────────────────────
    t_data_min = min(float(df_lines['Time_VisIt'].min()),
                     float(rdot_df['time'].min()))
    t_data_max = max(float(df_lines['Time_VisIt'].max()),
                     float(rdot_df['time'].max()))
    x_lo_plot = t_data_min - dt_mean * 0.3
    x_hi_plot = max(t_data_max, float(time_values[-1]) + ridge_h_t) + dt_mean * 0.2
    ax_main.set_xlim(x_lo_plot, x_hi_plot)

    # ── Y limits for ax_d (d axis) ────────────────────────────────────────────
    ax_d.set_ylim(d_lo - 0.05 * d_span, d_hi + 0.05 * d_span)

    # ── Subtle vertical grid at ridge positions ───────────────────────────────
    for t_i in time_values:
        ax_main.axvline(t_i, color='#cccccc', lw=0.5, ls=':', zorder=0)

    # ── Spines ────────────────────────────────────────────────────────────────
    ax_main.spines['top'].set_visible(False)
    ax_d.spines['top'].set_visible(False)

    # ── Legend ────────────────────────────────────────────────────────────────
    ridge_patch = Patch(facecolor=ridge_color, edgecolor=edge_color,
                        alpha=fill_alpha, label=r'$d_c$ distribution')
    mean_marker_handle = _Line2D([0], [0], color=mean_marker_color,
                                 lw=mean_marker_linewidth, ls=mean_marker_linestyle,
                                 label=r'ridge $\overline{d}_c$')
    legend_handles = [line_rdot, ridge_patch]
    if show_mean_markers:
        legend_handles.append(mean_marker_handle)
    legend_handles += [line_d_mean, line_N]
    ax_main.legend(handles=legend_handles, fontsize=max(fontsize - 2, 12),
                   loc='upper left', framealpha=0.85,
                   handlelength=1.8, handletextpad=0.6)

    # ── Layout ────────────────────────────────────────────────────────────────
    plt.tight_layout(rect=(0, 0, 0.88, 1))

    # ── Save ──────────────────────────────────────────────────────────────────
    if save_png and png_dir:
        path = os.path.join(png_dir, f'{output_filename}.png')
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f'Saved PNG: {path}') if Plot_log_level >= 1 else None
    if save_svg and svg_dir:
        path = os.path.join(svg_dir, f'{output_filename}.svg')
        plt.savefig(path, format='svg', bbox_inches='tight')
        print(f'Saved SVG: {path}') if Plot_log_level >= 1 else None
    if show_plots:
        plt.show()
    plt.close(fig)

    print('plot10_14_ridgeline_timeseries complete.') if Plot_log_level >= 1 else None
    return output_dir


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Running Ridgeline Timeseries Combined Plotter...')
    plot10_14_ridgeline_timeseries(
        input_dir=(
            r'C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis'
            r'\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554'
            r'\20250625_1626096\20250626_1700136\20250628_2007187'
        ),
        output_dir_comment='Ridgeline Timeseries Combined',

        n_time_points=30,
        dist_column='d_cell_SRec_distribution_CSTx6_nonDim',
        bin_count=40,
        overlap=0.65,

        ridge_color='#3cb371',
        edge_color='#1a7a40',
        fill_alpha=0.55,

        show_mean_markers=False,
        mean_marker_color='#1a1a1a',
        mean_marker_linewidth=2.0,
        mean_marker_linestyle='--',

        fontsize=20,
        figsize=(18, 9),

        save_png=True,
        save_svg=True,
        show_plots=True,
        output_filename='ridgeline_timeseries_combined',

        omit_image_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 106],
        Plot_log_level=1,
    )
