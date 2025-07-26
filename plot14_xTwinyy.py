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

def plotter_14_xTwinyy(
    input_dir,
    # Main Y-axis parameters
    y1_columns=None,
    y1_colors=None,
    y1_label=None,
    y1_line_widths=None,
    y1_markers=None,
    y1_marker_sizes=None,
    y1_scale_factors=None,
    # First twin Y-axis parameters
    y2_columns=None,
    y2_colors=None,
    y2_label=None,
    y2_line_widths=None,
    y2_markers=None,
    y2_marker_sizes=None,
    y2_scale_factors=None,
    # Second twin Y-axis parameters
    y3_columns=None,
    y3_colors=None,
    y3_label=None,
    y3_line_widths=None,
    y3_markers=None,
    y3_marker_sizes=None,
    y3_scale_factors=None,
    # X-axis parameters
    x_column='R_SF_nonDim',
    # Output and display parameters
    output_dir_manual="",
    output_dir_comment="",
    image_list=[],
    omit_image_list=[106],
    connect_with_lines=True,
    show_grid=True,
    grid_style='--',
    grid_width=0.5,
    grid_color='gray',
    grid_alpha=0.5,
    x_label=None,
    x_label_fontsize=20,
    tick_label_fontsize=20,
    legend_fontsize=20,
    legend_title=None,
    legend_loc='best',
    show_legend=True,
    line_style=['-', '-', '-'],  # List of line styles for all axes: [main_axis, twin1, twin2]
    twin_axis_spacing=[60, 120],  # Spacing for the 2 twin axes
    figsize=(14, 6),
    dpi=300,
    show_plot=0,
    Plot_log_level=1,
    # A11 data parameters
    include_A11_data=False,
    A11_data=None,
    A11_x_column='time',
    A11_y_column='N_c',
    A11_y_scale_factor=1.0,
    A11_line_style='--',
    A11_line_width=1.5,
    A11_line_color='black',
    A11_marker_style='',
    A11_marker_size=6,
    A11_label='A11 Data',
    A11_axis_index=0  # Which axis to add A11 data to (0=main, 1=first twin, 2=second twin)
):
    """
    Creates an x-y plot with a main y-axis and exactly 2 twin y-axes, each with configurable parameters.
    
    Parameters
    ----------
    input_dir : str
        Directory containing the Analysis_A11_final_df.pkl file
    
    # Main Y-axis parameters
    y1_columns : list
        List of column names to plot on the main y-axis
    y1_colors : list
        List of colors for each y1 column
    y1_label : str
        Label for the main y-axis
    y1_line_widths : list
        List of line widths for each y1 column
    y1_markers : list
        List of marker styles for each y1 column
    y1_marker_sizes : list
        List of marker sizes for each y1 column
    y1_scale_factors : list
        List of scale factors for each y1 column
        
    # First twin Y-axis parameters
    y2_columns : list
        List of column names to plot on the first twin y-axis
    y2_colors : list
        List of colors for each y2 column
    y2_label : str
        Label for the first twin y-axis
    y2_line_widths : list
        List of line widths for each y2 column
    y2_markers : list
        List of marker styles for each y2 column
    y2_marker_sizes : list
        List of marker sizes for each y2 column
    y2_scale_factors : list
        List of scale factors for each y2 column
        
    # Second twin Y-axis parameters
    y3_columns : list
        List of column names to plot on the second twin y-axis
    y3_colors : list
        List of colors for each y3 column
    y3_label : str
        Label for the second twin y-axis
    y3_line_widths : list
        List of line widths for each y3 column
    y3_markers : list
        List of marker styles for each y3 column
    y3_marker_sizes : list
        List of marker sizes for each y3 column
    y3_scale_factors : list
        List of scale factors for each y3 column
    
    # Other parameters
    line_style : list
        Global list of line styles for all axes [main_axis, twin1, twin2].
    
    ...
    
    Returns
    -------
    str
        Path to the output directory
    """
    # Create output directory
    output_dir = F_1.F_out_dir(input_dir=input_dir, script_path=__file__, 
                             output_dir_comment=output_dir_comment, 
                             output_dir_manual=output_dir_manual)
    
    if Plot_log_level >= 1:
        print(f"plotter_14_xTwinyy: Output directory: {output_dir}")

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
    
    # Check if the required columns exist
    if x_column not in df.columns:
        print(f"Column '{x_column}' not found in DataFrame. Available columns: {df.columns.tolist()}")
        return output_dir
    
    # Default values if not provided
    if y1_columns is None:
        y1_columns = ['N_cells_CSTx6']
    if y2_columns is None:
        y2_columns = ['d_cell_SRec_mean_CSTx6_nonDim']
    if y3_columns is None:
        y3_columns = ['contour_length_SRec_total_CSTx6_nonDim']
    
    if y1_colors is None:
        y1_colors = ['red'] * len(y1_columns)
    if y2_colors is None:
        y2_colors = ['green'] * len(y2_columns)
    if y3_colors is None:
        y3_colors = ['blue'] * len(y3_columns)
    
    if y1_label is None:
        y1_label = '$N_{cells}$'
    if y2_label is None:
        y2_label = '$\\overline{d}_c/\\delta_T$'
    if y3_label is None:
        y3_label = '$L/\\delta_T$'
    
    # Ensure we have line styles for all axes
    if line_style is None:
        line_style = ['-', '-', '-']
    while len(line_style) < 3:
        line_style.append('-')
    
    if y1_line_widths is None:
        y1_line_widths = [1.5] * len(y1_columns)
    if y2_line_widths is None:
        y2_line_widths = [1.5] * len(y2_columns)
    if y3_line_widths is None:
        y3_line_widths = [1.5] * len(y3_columns)
    
    if y1_markers is None:
        y1_markers = [''] * len(y1_columns)
    if y2_markers is None:
        y2_markers = [''] * len(y2_columns)
    if y3_markers is None:
        y3_markers = [''] * len(y3_columns)
    
    if y1_marker_sizes is None:
        y1_marker_sizes = [6] * len(y1_columns)
    if y2_marker_sizes is None:
        y2_marker_sizes = [6] * len(y2_columns)
    if y3_marker_sizes is None:
        y3_marker_sizes = [6] * len(y3_columns)
    
    if y1_scale_factors is None:
        y1_scale_factors = [1.0] * len(y1_columns)
    if y2_scale_factors is None:
        y2_scale_factors = [1.0] * len(y2_columns)
    if y3_scale_factors is None:
        y3_scale_factors = [1.0] * len(y3_columns)
    
    # Validate that all y-columns exist in DataFrame
    all_y_columns = y1_columns + y2_columns + y3_columns
    for y_column in all_y_columns:
        if y_column not in df.columns:
            print(f"Column '{y_column}' not found in DataFrame. Available columns: {df.columns.tolist()}")
            return output_dir
    
    # Filter DataFrame based on image_list if provided
    if image_list:
        df = df[df['image_number'].isin(image_list)]
        if df.empty:
            print(f"No matching images found for the provided image_list: {image_list}")
            return output_dir
    
    # Exclude images in omit_image_list if provided
    if omit_image_list:
        df = df[~df['image_number'].isin(omit_image_list)]
        if df.empty:
            print(f"No images remaining after applying omit_image_list: {omit_image_list}")
            return output_dir
        if Plot_log_level >= 1:
            print(f"Excluded {len(omit_image_list)} images: {omit_image_list}")
    
    # Create figure
    plt.figure(figsize=figsize, dpi=dpi)
    
    # Set font size for all elements
    plt.rcParams['font.size'] = tick_label_fontsize
    
    # Create the main axis
    ax_main = plt.gca()
    
    # List to store all axes for legend and formatting
    all_axes = [ax_main]
    all_lines = []
    all_labels = []
    
    # Plot on the main axis (y1)
    for i, y_column in enumerate(y1_columns):
        color = y1_colors[i] if i < len(y1_colors) else 'red'
        line_width = y1_line_widths[i] if i < len(y1_line_widths) else 1.5
        marker = y1_markers[i] if i < len(y1_markers) else ''
        marker_size = y1_marker_sizes[i] if i < len(y1_marker_sizes) else 6
        scale_factor = y1_scale_factors[i] if i < len(y1_scale_factors) else 1.0
        
        line, = ax_main.plot(df[x_column], df[y_column] * scale_factor, 
                          marker=marker, markersize=marker_size,
                          linestyle=line_style[0], linewidth=line_width,
                          color=color, label=f"{y_column}")
        all_lines.append(line)
        all_labels.append(f"{y_column}")
    
    ax_main.set_ylabel(y1_label, color=y1_colors[0], fontsize=x_label_fontsize)
    ax_main.tick_params(axis='y', labelcolor=y1_colors[0], labelsize=tick_label_fontsize, direction='in')
    ax_main.spines['left'].set_color(y1_colors[0])
    
    # Create the first twin axis (y2)
    ax_twin1 = ax_main.twinx()
    all_axes.append(ax_twin1)
    
    # Move the axis to the right with the specified spacing
    spacing1 = twin_axis_spacing[0] if len(twin_axis_spacing) > 0 else 60
    ax_twin1.spines["right"].set_position(("outward", spacing1))
    ax_twin1.spines["right"].set_color(y2_colors[0])
    
    # Plot on the first twin axis (y2)
    for i, y_column in enumerate(y2_columns):
        color = y2_colors[i] if i < len(y2_colors) else y2_colors[0]
        line_width = y2_line_widths[i] if i < len(y2_line_widths) else 1.5
        marker = y2_markers[i] if i < len(y2_markers) else ''
        marker_size = y2_marker_sizes[i] if i < len(y2_marker_sizes) else 6
        scale_factor = y2_scale_factors[i] if i < len(y2_scale_factors) else 1.0
        
        line, = ax_twin1.plot(df[x_column], df[y_column] * scale_factor, 
                           marker=marker, markersize=marker_size,
                           linestyle=line_style[1], linewidth=line_width,
                           color=color, label=f"{y_column}")
        all_lines.append(line)
        all_labels.append(f"{y_column}")
    
    ax_twin1.set_ylabel(y2_label, color=y2_colors[0], fontsize=x_label_fontsize)
    ax_twin1.tick_params(axis='y', labelcolor=y2_colors[0], labelsize=tick_label_fontsize, direction='in')
    
    # Create the second twin axis (y3)
    ax_twin2 = ax_main.twinx()
    all_axes.append(ax_twin2)
    
    # Move the axis to the right with the specified spacing
    spacing2 = twin_axis_spacing[1] if len(twin_axis_spacing) > 1 else 120
    ax_twin2.spines["right"].set_position(("outward", spacing2))
    ax_twin2.spines["right"].set_color(y3_colors[0])
    
    # Plot on the second twin axis (y3)
    for i, y_column in enumerate(y3_columns):
        color = y3_colors[i] if i < len(y3_colors) else y3_colors[0]
        line_width = y3_line_widths[i] if i < len(y3_line_widths) else 1.5
        marker = y3_markers[i] if i < len(y3_markers) else ''
        marker_size = y3_marker_sizes[i] if i < len(y3_marker_sizes) else 6
        scale_factor = y3_scale_factors[i] if i < len(y3_scale_factors) else 1.0
        
        line, = ax_twin2.plot(df[x_column], df[y_column] * scale_factor, 
                           marker=marker, markersize=marker_size,
                           linestyle=line_style[2], linewidth=line_width,
                           color=color, label=f"{y_column}")
        all_lines.append(line)
        all_labels.append(f"{y_column}")
    
    ax_twin2.set_ylabel(y3_label, color=y3_colors[0], fontsize=x_label_fontsize)
    ax_twin2.tick_params(axis='y', labelcolor=y3_colors[0], labelsize=tick_label_fontsize, direction='in')
    
    # Add A11 data if requested
    if include_A11_data:
        if A11_data is None:
            # Load A11 data if not provided
            try:
                A11_data = load_A11_data()
                if Plot_log_level >= 1:
                    print("Loaded A11 data successfully")
            except Exception as e:
                print(f"Failed to load A11 data: {e}")
                include_A11_data = False
        
        if include_A11_data and A11_y_column in A11_data:
            a11_df = A11_data[A11_y_column]
            
            # Determine which axis to use for A11 data
            if A11_axis_index == 0:
                a11_ax = ax_main
            elif A11_axis_index == 1:
                a11_ax = ax_twin1
            elif A11_axis_index == 2:
                a11_ax = ax_twin2
            else:
                print(f"Invalid A11_axis_index {A11_axis_index}, using main axis")
                a11_ax = ax_main
            
            # Plot the A11 data on the selected axis
            if A11_x_column in a11_df.columns and A11_y_column in a11_df.columns:
                line, = a11_ax.plot(
                    a11_df[A11_x_column], 
                    a11_df[A11_y_column] * A11_y_scale_factor,
                    marker=A11_marker_style, markersize=A11_marker_size,
                    linestyle=A11_line_style, linewidth=A11_line_width,
                    color=A11_line_color, label=A11_label
                )
                all_lines.append(line)
                all_labels.append(A11_label)
                
                if Plot_log_level >= 1:
                    print(f"Added A11 data to plot: {A11_y_column} vs {A11_x_column}")
            else:
                print(f"A11 data missing required columns: {A11_x_column} or {A11_y_column}")
    
    # Set the x-axis label
    ax_main.set_xlabel(x_label if x_label else x_column, fontsize=x_label_fontsize)
    
    # Configure tick parameters for inward facing ticks
    for ax in all_axes:
        ax.tick_params(axis='both', direction='in', which='both', labelsize=tick_label_fontsize)
    
    # Add grid if requested
    if show_grid:
        ax_main.grid(True, linestyle=grid_style, linewidth=grid_width, 
                  alpha=grid_alpha, color=grid_color)
    
    # Add a combined legend if requested
    if show_legend:
        if legend_title:
            plt.figlegend(all_lines, all_labels, loc=legend_loc, fontsize=legend_fontsize, 
                         frameon=False, title=legend_title, title_fontsize=legend_fontsize+2)
        else:
            plt.figlegend(all_lines, all_labels, loc=legend_loc, fontsize=legend_fontsize, 
                         frameon=False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create filename from x and y columns
    y_columns_str = "_".join([y1_columns[0], y2_columns[0], y3_columns[0]]).replace(' ', '_')
    
    base_filename = f"{x_column.replace(' ', '_')}_vs_{y_columns_str}"
    png_path = os.path.join(png_dir, f"{base_filename}.png")
    svg_path = os.path.join(svg_dir, f"{base_filename}.svg")
    
    plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    
    if Plot_log_level >= 1:
        print(f"Saved figures to:\n  {png_path}\n  {svg_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return output_dir


# if __name__ == "__main__":
#     # Example usage with twin axes for different metrics
#     input_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187"
    
#     output_dir = plotter_14_xTwinyy(
#         input_dir=input_dir,
#         x_column="Time_VisIt",
#         # Main y-axis parameters
#         y1_columns=["nonDim_per_px"],
#         y1_colors=["black"],
#         y1_line_widths=[2],
#         y1_label=r'$D$',
#         # First twin y-axis parameters
#         y2_columns=["R_SF_nonDim"],
#         y2_colors=["blue"],
#         y2_line_widths=[2],
#         y2_label=r'$R_{SF}/\delta_T$',
#         # Second twin y-axis parameters
#         y3_columns=["R_SF_px"],
#         y3_colors=["red"],
#         y3_line_widths=[2],
#         y3_label=r'$R_{SF}[px]$',
#         # Other parameters
#         output_dir_comment="twin_dimentionalisation",
#         x_label=r"$\tau$",
#         legend_loc='upper left',
#         show_legend=False,
#         line_style=['-', '-', '--'],  # Control line styles globally: solid for main, dashed for twin1, dash-dot for twin2
#         twin_axis_spacing=[0, 60],
#         show_plot=0
#     )


if __name__ == "__main__":
    # Example usage with twin axes for different metrics
    input_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187"
    
    output_dir = plotter_14_xTwinyy(
        input_dir=input_dir,
        x_column="Time_VisIt",
        # Main y-axis parameters
        y1_columns=["nonDim_per_px"],
        y1_colors=["black"],
        y1_line_widths=[2],
        y1_label=r'$D$',
        # First twin y-axis parameters
        y2_columns=["R_SF_nonDim"],
        y2_colors=["blue"],
        y2_line_widths=[2],
        y2_label=r'$R_{SF}/\delta_T$',
        # Second twin y-axis parameters
        y3_columns=["R_SF_px"],
        y3_colors=["red"],
        y3_line_widths=[2],
        y3_label=r'$R_{SF}[px]$',
        # Other parameters
        output_dir_comment="cell stages",
        x_label=r"$\tau$",
        legend_loc='upper left',
        show_legend=False,
        line_style=['-', '-', '--'],  # Control line styles globally: solid for main, dashed for twin1, dash-dot for twin2
        twin_axis_spacing=[0, 60],
        show_plot=0
    )