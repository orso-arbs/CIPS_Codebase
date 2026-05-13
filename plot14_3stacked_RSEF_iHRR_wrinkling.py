import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import Format_1 as F_1

# Enhanced LaTeX configuration
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb,amsfonts}'
plt.rcParams['mathtext.fontset'] = 'cm'

def load_A11_data():
    """
    Load A11 data files from fixed paths.

    Returns
    -------
    dict
        Dictionary containing all loaded A11 dataframes
    """
    A11_data = {}

    A11_data['K_mean'] = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction\A11_SF_K_mean_as_mean_stretch_rate_vs_time_manual_extraction.txt")
    A11_data['N_c'] = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction\A11_SF_N_c_as_number_of_cells_vs_time_manual_extraction.txt")
    A11_data['R_mean'] = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction\A11_SF_R_mean_as_average_radius_of_the_wrinkled_flame_fron_vs_time_manual_extraction.txt")
    A11_data['R_mean_dot'] = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction\A11_SF_R_mean_dot_as_first_time_derivative_of_the_average_radius_of_the_wrinkled_flame_front_vs_time_manual_extraction.txt")
    A11_data['s_a'] = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction\A11_SF_s_a_as_average_normal_component_of_the_absolute_propagation_velocity_vs_time_manual_extraction.txt")
    A11_data['s_d'] = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction\A11_SF_s_d_as_average_density_weighted_displacement_speed_vs_time_manual_extraction.txt")
    A11_data['A'] = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction\A11_SF_A_as_flame_surface_area_of_the_wrinkled_spherical_front_vs_time_manual_extraction.txt")
    A11_data['a_t'] = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction\A11_SF_a_t_as_average_total_aerodynamic_strain_vs_time_manual_extraction.txt")
    A11_data['iHRR'] = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction\A11_SF_iHRR_as_integral_heat_release_rate_vs_time_manual_extraction.txt")
    A11_data['K_geom'] = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction\A11_SF_K_geom_as_geometric_stretch_rate_vs_time_manual_extraction.txt")

    return A11_data


def plotter_14_3stacked_RSEF_iHRR_wrinkling(
    input_dir,
    output_dir_manual="",
    output_dir_comment="",
    image_list=[],
    omit_image_list=[106],
    figsize=(9.8, 9.8),
    dpi=300,
    show_plot=0,
    Plot_log_level=1,
    legend=True,
    legend_alpha=1.0,
    grid_alpha=0.5
):
    """
    Creates a 3x1 stacked-panel plot sharing the x-axis (time t/tau).

    Top:    R_SEF / delta_T  (A11 R_mean)
    Middle: iHRR / (delta_T * rho_u * c_p * S_L * (T_b - T_u))  (A11 iHRR, already nondim)
    Bottom: wrinkling factor Xi = A_SEF / (4 pi R_SEF^2)  with twin axis R_dot / S_L
    """
    output_dir = F_1.F_out_dir(input_dir=input_dir, script_path=__file__,
                               output_dir_comment=output_dir_comment,
                               output_dir_manual=output_dir_manual)

    if Plot_log_level >= 1:
        print(f"plotter_14_3stacked_RSEF_iHRR_wrinkling: Output directory: {output_dir}")

    png_dir = os.path.join(output_dir, "png")
    svg_dir = os.path.join(output_dir, "svg")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    pandas_wildcard_str = os.path.join(input_dir, "Analysis_A11_final_df.pkl")
    pkl_files = glob.glob(pandas_wildcard_str)

    if not pkl_files:
        print(f"No Analysis_A11_final_df.pkl file found in {input_dir}")
        return output_dir

    df_path = pkl_files[0]
    df = pd.read_pickle(df_path)

    if Plot_log_level >= 1:
        print(f"Loaded DataFrame from {df_path}")

    try:
        A11_data = load_A11_data()
        if Plot_log_level >= 1:
            print("Loaded A11 data successfully")
    except Exception as e:
        print(f"Failed to load A11 data: {e}")
        return output_dir

    if image_list:
        df = df[df['image_number'].isin(image_list)]
    if omit_image_list:
        df = df[~df['image_number'].isin(omit_image_list)]

    from scipy.interpolate import interp1d

    # Common time axis for the bottom-panel wrinkling factor (computed on df times)
    x_values = df['Time_VisIt']

    a11_A_y = A11_data['A']['A'].values
    a11_A_x = A11_data['A']['time'].values
    f_A = interp1d(a11_A_x, a11_A_y, bounds_error=False, fill_value="extrapolate")  # type: ignore
    A_interp = f_A(x_values)

    a11_R_mean_y = A11_data['R_mean']['R_mean'].values
    a11_R_mean_x = A11_data['R_mean']['time'].values
    f_R_mean = interp1d(a11_R_mean_x, a11_R_mean_y, bounds_error=False, fill_value="extrapolate")  # type: ignore
    R_mean_interp = f_R_mean(x_values)

    R_mean_interp_safe = np.where(R_mean_interp == 0, np.nan, R_mean_interp)
    wrinkling_factor = A_interp / (4 * np.pi * R_mean_interp_safe ** 2)

    label_fs = 16
    tick_fs = 14
    legend_fs = 14
    line_lw = 1.8

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Small spacer row above ax2 leaves room for the iHRR ×10^5 offset text
    # to sit outside the plot window without intruding on the top panel.
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 0.07, 1, 1], hspace=0, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    ax3 = fig.add_subplot(gs[3], sharex=ax1)
    plt.rcParams['font.size'] = label_fs

    panel_label_fs = 18

    # --- Top: R_SEF / delta_T ---
    ax1.plot(A11_data['R_mean']['time'], A11_data['R_mean']['R_mean'],
             color='black', linewidth=line_lw)
    ax1.set_ylabel(r'$R_{SEF}/\delta_T$', color='black', fontsize=label_fs)
    ax1.tick_params(axis='y', labelcolor='black', direction='out', labelsize=tick_fs)
    ax1.text(0.015, 0.95, r'(a)', transform=ax1.transAxes,
             fontsize=panel_label_fs, ha='left', va='top')

    # --- Middle: iHRR nondimensionalised ---
    ax2.plot(A11_data['iHRR']['time'], A11_data['iHRR']['iHRR'],
             color='black', linewidth=line_lw)
    ax2.set_ylabel(r'$\dfrac{iHRR}{\delta_T \rho_u c_p S_L (T_b - T_u)}$',
                   color='black', fontsize=label_fs)
    ax2.tick_params(axis='y', labelcolor='black', direction='out', labelsize=tick_fs)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    # Hide default offset (renders above the panel — collides with ax1 due to hspace=0).
    ax2.yaxis.get_offset_text().set_visible(False)
    # Reserve whitespace above the curve so the factor sits cleanly inside the panel.
    ax2.set_ylim(0, 1.05e6)
    ax2.set_yticks([0, 2e5, 4e5, 6e5, 8e5])
    ax2.text(0.985, 0.96, r'$\times 10^{5}$', transform=ax2.transAxes,
             fontsize=tick_fs, ha='right', va='top')
    ax2.text(0.015, 0.95, r'(b)', transform=ax2.transAxes,
             fontsize=panel_label_fs, ha='left', va='top')

    # --- Bottom: wrinkling factor (left) + R_dot/S_L (right twin) ---
    ax3_twin = ax3.twinx()

    line_b1, = ax3.plot(x_values, wrinkling_factor, color='purple', linestyle='--',
                        linewidth=line_lw, label=r'$A_{SEF}/(4\pi R_{SEF}^2)$')
    ax3.set_ylabel(r'$A_{SEF}/(4\pi R_{SEF}^2)$', color='purple', fontsize=label_fs)
    ax3.tick_params(axis='y', labelcolor='purple', direction='out', labelsize=tick_fs)

    line_b2, = ax3_twin.plot(A11_data['R_mean_dot']['time'],
                             A11_data['R_mean_dot']['R_mean_dot'],
                             color='blue', linewidth=line_lw,
                             label=r'$\dot{R}_{SEF}/S_L$')
    ax3_twin.set_ylabel(r'$\dot{R}_{SEF}/S_L$', color='blue', fontsize=label_fs)
    ax3_twin.tick_params(axis='y', labelcolor='blue', direction='out', labelsize=tick_fs)

    bottom_lines = [line_b1, line_b2]
    bottom_labels = [l.get_label() for l in bottom_lines]
    if legend:
        ax3.legend(bottom_lines, bottom_labels, loc='upper left',
                   bbox_to_anchor=(0.07, 1.0),
                   fontsize=legend_fs).get_frame().set_alpha(legend_alpha)
    ax3.text(0.015, 0.95, r'(c)', transform=ax3.transAxes,
             fontsize=panel_label_fs, ha='left', va='top')

    ax3.set_xlabel(r'$t/\tau$', fontsize=label_fs)
    ax1.tick_params(axis='x', direction='out', top=True, labeltop=True,
                    bottom=True, labelbottom=False, labelsize=tick_fs)
    ax2.tick_params(axis='x', direction='out', top=True, labeltop=False,
                    bottom=True, labelbottom=False, labelsize=tick_fs)
    ax3.tick_params(axis='x', direction='out', top=True, labeltop=False,
                    bottom=True, labelbottom=True, labelsize=tick_fs)

    ax1.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=grid_alpha)
    ax2.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=grid_alpha)
    ax3.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=grid_alpha)

    plt.tight_layout(rect=(0, 0, 0.95, 1))
    base_filename = "plot14_3stacked_RSEF_iHRR_wrinkling"
    png_path = os.path.join(png_dir, f"{base_filename}.png")
    svg_path = os.path.join(svg_dir, f"{base_filename}.svg")

    plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')

    if Plot_log_level >= 1:
        print(f"Saved figures to:\n  {png_path}\n  {svg_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return output_dir


if __name__ == "__main__":
    input_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187"

    plotter_14_3stacked_RSEF_iHRR_wrinkling(
        input_dir=input_dir,
        output_dir_comment="RSEF_iHRR_WrinklingFactor_supplementary",
        show_plot=0
    )
