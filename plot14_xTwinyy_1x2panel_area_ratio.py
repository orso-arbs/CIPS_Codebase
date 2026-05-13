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
    
    # Load A11 data with descriptive keys
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

def plotter_14_xTwinyy_1x2panel(
    input_dir,
    output_dir_manual="",
    output_dir_comment="",
    image_list=[],
    omit_image_list=[106],
    figsize=(14, 10),
    dpi=300,
    show_plot=0,
    Plot_log_level=1
):
    """
    Creates a 1x2 panel plot with a shared x-axis.
    """
    # Create output directory
    output_dir = F_1.F_out_dir(input_dir=input_dir, script_path=__file__, 
                             output_dir_comment=output_dir_comment, 
                             output_dir_manual=output_dir_manual)
    
    if Plot_log_level >= 1:
        print(f"plotter_14_xTwinyy_1x2panel: Output directory: {output_dir}")

    # Create directories for PNG and SVG outputs
    png_dir = os.path.join(output_dir, "png")
    svg_dir = os.path.join(output_dir, "svg")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    # Find the PKL file
    pandas_wildcard_str = os.path.join(input_dir, "Analysis_A11_final_df.pkl")
    pkl_files = glob.glob(pandas_wildcard_str)
    
    if not pkl_files:
        print(f"No Analysis_A11_final_df.pkl file found in {input_dir}")
        return output_dir
    
    # Load the DataFrame
    df_path = pkl_files[0]
    df = pd.read_pickle(df_path)
    
    if Plot_log_level >= 1:
        print(f"Loaded DataFrame from {df_path}")

    # Load A11 data
    try:
        A11_data = load_A11_data()
        if Plot_log_level >= 1:
            print("Loaded A11 data successfully")
    except Exception as e:
        print(f"Failed to load A11 data: {e}")
        return output_dir

    # Filter DataFrame
    if image_list:
        df = df[df['image_number'].isin(image_list)]
    if omit_image_list:
        df = df[~df['image_number'].isin(omit_image_list)]

    # Calculate total area from distribution
    if 'A_cell_SRec_distribution_CSTx6_nonDim2' in df.columns:
        df['A_cell_SRec_sum_CSTx6_nonDim2'] = df['A_cell_SRec_distribution_CSTx6_nonDim2'].apply(np.sum)
    else:
        print("Warning: 'A_cell_SRec_distribution_CSTx6_nonDim2' not found. Cannot calculate total area.")
        # Add an empty column to prevent errors later
        df['A_cell_SRec_sum_CSTx6_nonDim2'] = np.nan

    from scipy.interpolate import interp1d
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True, gridspec_kw={'hspace': 0})
    plt.rcParams['font.size'] = 20

    # Get target x-values from the main dataframe
    x_values = df['Time_VisIt']

    # --- Interpolate A11 data for calculations ---
    # Interpolate R_mean_dot for any calculations needing a common time axis
    a11_r_mean_dot_y = A11_data['R_mean_dot']['R_mean_dot'].values
    a11_r_mean_dot_x = A11_data['R_mean_dot']['time'].values
    f_r_mean_dot = interp1d(a11_r_mean_dot_x, a11_r_mean_dot_y, bounds_error=False, fill_value="extrapolate") # type: ignore
    r_mean_dot_interp = f_r_mean_dot(x_values)

    # Interpolate A for any calculations needing a common time axis
    a11_A_y = A11_data['A']['A'].values
    a11_A_x = A11_data['A']['time'].values
    f_A = interp1d(a11_A_x, a11_A_y, bounds_error=False, fill_value="extrapolate") # type: ignore
    A_interp = f_A(x_values)

    # Top plot
    ax1_twin1 = ax1.twinx()
    ax1_twin2 = ax1.twinx()

    # Move the second twin axis spine further out
    ax1_twin2.spines["right"].set_position(("outward", 60))
    
    # Left y-axis
    line1, = ax1.plot(A11_data['R_mean_dot']['time'], A11_data['R_mean_dot']['R_mean_dot'], color='blue', label=r'$\dot{\overline{R}}/S_L$')
    ax1.set_ylabel(r'$\dot{\overline{R}}/S_L$', color='blue', fontsize=20)
    ax1.tick_params(axis='y', labelcolor='blue', direction='out', labelsize=20)

    # Right y-axis 1 (diameter)
    line2, = ax1_twin1.plot(x_values, df['d_cell_SRec_mean_CSTx6_nonDim'], color='green', label=r'$\overline{d}_c/\delta_T$')
    ax1_twin1.set_ylabel(r'$\overline{d}_c/\delta_T$', color='green', fontsize=20)
    ax1_twin1.tick_params(axis='y', labelcolor='green', direction='out', labelsize=20)

    # Right y-axis 2 (cell count)
    line3, = ax1_twin2.plot(x_values, df['N_cells_CSTx6'], color='red', label=r'$N_{cells}$')
    ax1_twin2.set_ylabel(r'$N_{cells}$', color='red', fontsize=20)
    ax1_twin2.tick_params(axis='y', labelcolor='red', direction='out', labelsize=20)


    # Top plot legend
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=20)

    # Bottom plot
    ax2_twin = ax2.twinx()
    bottom_lines = []

    # Left y-axis (A_A11 and A_SRec, Total)
    line4, = ax2.plot(A11_data['A']['time'], A11_data['A']['A'], color='blue', label=r'$A_{DNS}/\delta_T^2$')
    bottom_lines.append(line4)

    if 'A_cell_SRec_sum_CSTx6_nonDim2' in df.columns:
        line5, = ax2.plot(x_values, df['A_cell_SRec_sum_CSTx6_nonDim2'], color='orange', label=r'$A_{Cells}/\delta_T^2$')
        bottom_lines.append(line5)

    ax2.set_ylabel(r'$A/\delta_T^2$', color='black', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='black', direction='out', labelsize=20)

    # Right y-axis (Difference)
    if 'A_cell_SRec_sum_CSTx6_nonDim2' in df.columns:
        srec_total_area = df['A_cell_SRec_sum_CSTx6_nonDim2'].values
        # Replace 0 with NaN to avoid division by zero errors in the plot
        srec_total_area_safe = np.where(srec_total_area == 0, np.nan, srec_total_area)
        difference = A_interp / srec_total_area_safe
        # Cap the difference at 5 to prevent early high values from skewing the plot
        difference_capped = np.clip(difference, a_min=None, a_max=4)
        line6, = ax2_twin.plot(x_values, difference_capped, color='black', linestyle='--', label=r'$A_{DNS} / A_{Cells}$')
        bottom_lines.append(line6)
        ax2_twin.set_ylabel(r'$A_{DNS} / A_{Cells}$', color='black', fontsize=20)
        ax2_twin.tick_params(axis='y', labelcolor='black', direction='out', labelsize=20)

    # Bottom plot legend
    bottom_labels = [l.get_label() for l in bottom_lines]
    ax2.legend(bottom_lines, bottom_labels, loc='upper left', fontsize=20)

    ax2.spines['top'].set_visible(False)

    # X-axis settings
    ax2.set_xlabel(r'$t/\tau$', fontsize=20)
    ax1.tick_params(axis='x', direction='out', top=True, labeltop=True, bottom=True, labelbottom=False, labelsize=20)
    ax2.tick_params(axis='x', direction='out', top=True, labeltop=False, bottom=True, labelbottom=True, labelsize=20)
    
    # Grid
    ax1.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    ax2.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # Layout and save
    plt.tight_layout(rect=(0, 0, 0.9, 1))
    base_filename = "plot14_xTwinyy_1x2panel"
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
    
    plotter_14_xTwinyy_1x2panel(
        input_dir=input_dir,
        output_dir_comment="Area_Ratio PROCI",
        show_plot=0
    )