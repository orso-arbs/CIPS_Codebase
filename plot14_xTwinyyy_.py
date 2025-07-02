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

def plotter_14_xTwinyyy(
    input_dir,
    y_columns_list=None,  # List of lists of y-columns, each sublist for one twin axis
    y_colors_list=None,   # List of lists of colors for each y-column
    y_labels_list=None,   # List of labels for each twin axis
    y_line_styles_list=None,  # List of lists of line styles for each y-column
    y_line_widths_list=None,  # List of lists of line widths for each y-column
    y_markers_list=None,  # List of lists of marker styles for each y-column
    y_marker_sizes_list=None,  # List of lists of marker sizes for each y-column
    y_scale_factors_list=None,  # List of lists of scale factors for each y-column
    x_column='R_SF_nonDim',  # x_column remains a single value
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
    show_legend=True,  # Parameter to control legend visibility
    axis_spacings=None,  # List of spacings for each twin axis (points)
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
    A11_axis_index=0  # Which twin axis to add A11 data to
):
    """
    Creates an x-y plot with multiple twin y-axes, each with configurable parameters.
    
    Parameters
    ----------
    input_dir : str
        Directory containing the Analysis_A11_final_df.pkl file
    y_columns_list : list of lists
        List of lists of column names to plot on each twin y-axis
    y_colors_list : list of lists
        List of lists of colors for each y-column
    y_labels_list : list
        List of labels for each twin y-axis
    y_line_styles_list : list of lists
        List of lists of line styles for each y-column
    y_line_widths_list : list of lists
        List of lists of line widths for each y-column
    y_markers_list : list of lists
        List of lists of marker styles for each y-column
    y_marker_sizes_list : list of lists
        List of lists of marker sizes for each y-column
    y_scale_factors_list : list of lists
        List of lists of scale factors for each y-column
    x_column : str, optional
        Column name to plot on x-axis, by default 'R_SF_nonDim'
    output_dir_manual : str, optional
        Manual output directory, by default ""
    output_dir_comment : str, optional
        Comment to append to the output directory name, by default ""
    image_list : list, optional
        List of image numbers to include in plot, if empty all images are used, by default []
    omit_image_list : list, optional
        List of image numbers to exclude from plot (applied after image_list filter), by default [106]
    connect_with_lines : bool, optional
        Whether to connect points with lines, by default True
    show_grid : bool, optional
        Whether to show grid lines, by default True
    grid_style : str, optional
        Style of grid lines, by default '--'
    grid_width : float, optional
        Width of grid lines, by default 0.5
    grid_color : str, optional
        Color of grid lines, by default 'gray'
    grid_alpha : float, optional
        Transparency of grid lines (0-1), by default 0.5
    x_label : str, optional
        Label for x-axis, by default None (will use x_column)
    x_label_fontsize : int, optional
        Font size for x-axis label, by default 20
    tick_label_fontsize : int, optional
        Font size for tick labels, by default 20
    legend_fontsize : int, optional
        Font size for legend, by default 20
    legend_title : str, optional
        Title for the legend, by default None
    legend_loc : str, optional
        Location of legend, by default 'best'
    show_legend : bool, optional
        Whether to display the legend, by default True
    axis_spacings : list, optional
        List of spacings for each twin axis in points, by default None (will use default spacing of [60, 120, 180, ...])
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (12, 8)
    dpi : int, optional
        DPI for the figure, by default 100
    show_plot : int, optional
        Whether to display the plot (1) or not (0), by default 0
    Plot_log_level : int, optional
        Logging level, by default 1
    include_A11_data : bool, optional
        Whether to include A11 simulation data in the plot, by default False
    A11_data : dict, optional
        Dictionary containing A11 simulation data, by default None (will load if include_A11_data=True)
    A11_x_column : str, optional
        Column name for A11 x-axis data, by default 'time'
    A11_y_column : str, optional
        Column name for A11 y-axis data, by default 'N_c'
    A11_y_scale_factor : float, optional
        Scale factor to multiply A11 y-axis values by, by default 1.0 (no scaling)
    A11_line_style : str, optional
        Line style for A11 data, by default '--'
    A11_line_width : float, optional
        Line width for A11 data, by default 1.5
    A11_line_color : str, optional
        Line color for A11 data, by default 'black'
    A11_marker_style : str, optional
        Marker style for A11 data, by default ''
    A11_marker_size : int, optional
        Marker size for A11 data, by default 6
    A11_label : str, optional
        Label for A11 data in legend, by default 'A11 Data'
    A11_axis_index : int, optional
        Index of the twin axis to add A11 data to, by default 0
    
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
        print(f"plotter_14_xTwinyyy: Output directory: {output_dir}")

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
    if y_columns_list is None:
        y_columns_list = [['N_cells_CSTx6'], ['d_cell_SRec_mean_CSTx6_nonDim'], ['contour_length_SRec_total_CSTx6_nonDim']]
    
    if y_colors_list is None:
        y_colors_list = [['red'], ['green'], ['blue']]
    
    if y_labels_list is None:
        y_labels_list = ['$N_{cells}$', '$\\overline{d}_c/\\delta_T$', '$L/\\delta_T$']
    
    if y_line_styles_list is None:
        y_line_styles_list = [['-'] for _ in range(len(y_columns_list))]
    
    if y_line_widths_list is None:
        y_line_widths_list = [[1.5] for _ in range(len(y_columns_list))]
    
    if y_markers_list is None:
        y_markers_list = [[''] for _ in range(len(y_columns_list))]
    
    if y_marker_sizes_list is None:
        y_marker_sizes_list = [[6] for _ in range(len(y_columns_list))]
    
    if y_scale_factors_list is None:
        y_scale_factors_list = [[1.0] for _ in range(len(y_columns_list))]
    
    # Validate that all y_columns exist in DataFrame
    for y_columns in y_columns_list:
        for y_column in y_columns:
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
    
    # Plot the first set of y_columns on the main axis
    for i, y_column in enumerate(y_columns_list[0]):
        color = y_colors_list[0][i] if i < len(y_colors_list[0]) else 'red'
        line_style = y_line_styles_list[0][i] if i < len(y_line_styles_list[0]) else '-'
        line_width = y_line_widths_list[0][i] if i < len(y_line_widths_list[0]) else 1.5
        marker = y_markers_list[0][i] if i < len(y_markers_list[0]) else ''
        marker_size = y_marker_sizes_list[0][i] if i < len(y_marker_sizes_list[0]) else 6
        scale_factor = y_scale_factors_list[0][i] if i < len(y_scale_factors_list[0]) else 1.0
        
        line, = ax_main.plot(df[x_column], df[y_column] * scale_factor, 
                          marker=marker, markersize=marker_size,
                          linestyle=line_style, linewidth=line_width,
                          color=color, label=f"{y_column}")
        all_lines.append(line)
        all_labels.append(f"{y_column}")
    
    ax_main.set_ylabel(y_labels_list[0], color=y_colors_list[0][0], fontsize=x_label_fontsize)
    ax_main.tick_params(axis='y', labelcolor=y_colors_list[0][0], labelsize=tick_label_fontsize, direction='in')
    ax_main.spines['left'].set_color(y_colors_list[0][0])
    
    # Default axis spacings if not provided (60, 120, 180, etc.)
    if axis_spacings is None:
        axis_spacings = [(i+1) * 60 for i in range(len(y_columns_list) - 1)]
    elif len(axis_spacings) < len(y_columns_list) - 1:
        # Extend axis_spacings if it's shorter than needed
        additional_spacings = [(len(axis_spacings) + i + 1) * 60 for i in range(len(y_columns_list) - 1 - len(axis_spacings))]
        axis_spacings = axis_spacings + additional_spacings
        if Plot_log_level >= 1:
            print(f"Extended axis_spacings to {axis_spacings}")
    
    # Create and plot on twin axes for each additional set of y_columns
    twin_axes = []
    
    for i in range(1, len(y_columns_list)):
        # Create a twin axis
        ax_twin = ax_main.twinx()
        twin_axes.append(ax_twin)
        all_axes.append(ax_twin)
        
        # Move the axis to the right with the specified spacing from axis_spacings
        spacing = axis_spacings[i-1] if i-1 < len(axis_spacings) else i * 60
        ax_twin.spines["right"].set_position(("outward", spacing))
        ax_twin.spines["right"].set_color(y_colors_list[i][0])
        
        # Plot each y_column in this group
        for j, y_column in enumerate(y_columns_list[i]):
            color = y_colors_list[i][j] if j < len(y_colors_list[i]) else y_colors_list[i][0]
            line_style = y_line_styles_list[i][j] if i < len(y_line_styles_list) and j < len(y_line_styles_list[i]) else '-'
            line_width = y_line_widths_list[i][j] if i < len(y_line_widths_list) and j < len(y_line_widths_list[i]) else 1.5
            marker = y_markers_list[i][j] if i < len(y_markers_list) and j < len(y_markers_list[i]) else ''
            marker_size = y_marker_sizes_list[i][j] if i < len(y_marker_sizes_list) and j < len(y_marker_sizes_list[i]) else 6
            scale_factor = y_scale_factors_list[i][j] if i < len(y_scale_factors_list) and j < len(y_scale_factors_list[i]) else 1.0
            
            line, = ax_twin.plot(df[x_column], df[y_column] * scale_factor, 
                              marker=marker, markersize=marker_size,
                              linestyle=line_style, linewidth=line_width,
                              color=color, label=f"{y_column}")
            all_lines.append(line)
            all_labels.append(f"{y_column}")
        
        # Set the y-axis label and color
        if i < len(y_labels_list):
            ax_twin.set_ylabel(y_labels_list[i], color=y_colors_list[i][0], fontsize=x_label_fontsize)
        else:
            ax_twin.set_ylabel(f"Group {i+1}", color=y_colors_list[i][0], fontsize=x_label_fontsize)
        
        # Configure the y-axis tick parameters
        ax_twin.tick_params(axis='y', labelcolor=y_colors_list[i][0], labelsize=tick_label_fontsize, direction='in')
    
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
            elif 0 < A11_axis_index <= len(twin_axes):
                a11_ax = twin_axes[A11_axis_index - 1]
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
    
    # Create filename from x and column groups
    y_columns_str = "_".join([f"Group{i+1}" for i in range(len(y_columns_list))])
    
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


if __name__ == "__main__":
    # Example usage with twin axes for different metrics
    input_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187"
    
    # Define column groups for each twin axis
    y_columns_list = [
        ['N_cells_CSTx6'],                      # First axis (red)
        ['d_cell_SRec_mean_CSTx6_nonDim'],      # Second axis (green)
        ['contour_length_SRec_total_CSTx6_nonDim'],  # Third axis (blue)
        ['Roundness_mean_SRec_CSTx6_nonDim'],   # Fourth axis (violet)
    ]
    
    # Define colors for each group
    y_colors_list = [
        ['red'],    # Colors for first axis
        ['green'],  # Colors for second axis
        ['blue'],   # Colors for third axis
        ['violet'], # Colors for fourth axis
    ]
    
    # Define labels for each axis
    y_labels_list = [
        '$N_{cells}$',                # Label for first axis
        '$\\overline{d}_c/\\delta_T$',  # Label for second axis
        '$L/\\delta_T$',               # Label for third axis
        '$\overline{Q}$'                           # Label for fourth axis
    ]
    
    # Define scale factors for each column
    y_scale_factors_list = [
        [1.0],    # No scaling for N_cells
        [1.0],    # No scaling for mean diameter
        [1.0],    # No scaling for contour length
        [1.0],    # No scaling for roundness
    ]
    
    # Custom spacings for each twin axis (in points)
    axis_spacings = [0, 60, 150]  # Spacings for the 2nd, 3rd, and 4th axes
    
    output_dir = plotter_14_xTwinyyy(
        input_dir=input_dir,
        x_column="R_SF_nonDim",
        y_columns_list=y_columns_list,
        y_colors_list=y_colors_list,
        y_labels_list=y_labels_list,
        y_scale_factors_list=y_scale_factors_list,
        output_dir_comment="twin_axes_metrics_vs_R_SF_nonDim",
        x_label=r'$R_{SF}/\delta_T$',
        legend_loc='upper left',
        show_legend=False,  # Set to False to hide the legend
        axis_spacings=axis_spacings,  # Custom spacings for each twin axis
        show_plot=0       # Set to 1 to display the plot
    )
