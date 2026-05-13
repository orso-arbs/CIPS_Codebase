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

def plotter_14_xTwinyyyy_A11(
    input_dir,
    y_columns_list=None,  # List of lists of y-columns, each sublist for one twin axis
    y_colors_list=None,   # List of lists of colors for each y-column
    y_labels_list=None,   # List of labels for each twin axis
    y_line_styles_list=None,  # List of lists of line styles for each y-column
    y_line_widths_list=None,  # List of lists of line widths for each y-column
    y_markers_list=None,  # List of lists of marker styles for each y-column
    y_marker_sizes_list=None,  # List of lists of marker sizes for each y-column
    y_scale_factors_list=None,  # List of lists of scale factors for each y-column
    x_column='Time_VisIt',  # x_column for the plots
    # LLS fit lines parameters
    fit_y_variables=None,      # List of y variables to fit (empty list = no fit lines)
    fit_x_start_values=None,   # List of x start values for each fit
    fit_x_stop_values=None,    # List of x stop values for each fit
    fit_colors=None,           # List of colors for each fit line
    fit_line_styles=None,      # List of line styles for each fit line
    fit_line_widths=None,      # List of line widths for each fit line
    output_dir_manual="",
    output_dir_comment="",
    image_list=[],             # List of image numbers to include
    omit_image_list=[106],     # List of image numbers to exclude
    connect_with_lines=True,
    show_grid=True,
    grid_style='--',
    grid_width=0.5,
    grid_color='gray',
    grid_alpha=0.5,
    x_label=None,
    x_label_fontsize=25,
    tick_label_fontsize=25,
    legend_fontsize=25,
    legend_title=None,
    legend_loc='best',
    show_legend=True,          # Parameter to control legend visibility
    axis_spacings=None,        # List of spacings for each twin axis (points)
    figsize=(16, 6),
    dpi=300,
    show_plot=0,
    Plot_log_level=1,
    # Enhanced A11 data parameters with multi-axis support
    include_A11_data=True,
    A11_data=None,
    A11_x_column='time',
    A11_y_columns_list=[['N_c']],    # List of lists of A11 y-columns, each sublist for one twin axis
    A11_y_scale_factors_list=[[1.0]], # List of lists of scale factors for each A11 y-column
    A11_line_styles_list=[['--']],   # List of lists of line styles for each A11 y-column
    A11_line_widths_list=[[1.5]],    # List of lists of line widths for each A11 y-column
    A11_line_colors_list=[['black']], # List of lists of colors for each A11 y-column
    A11_marker_styles_list=[['']],   # List of lists of marker styles for each A11 y-column
    A11_marker_sizes_list=[[6]],     # List of lists of marker sizes for each A11 y-column
    A11_labels_list=[['A11 Data']],  # List of lists of labels for each A11 y-column
    A11_axis_indices_list=[[0]],     # List of lists indicating which axis to use for each A11 curve
    y_legend_texts_list=None,  # New parameter: List of lists of custom legend texts for each y-column
    A11_use_same_x=False,     # Whether to use the same x-axis as main data (requires interpolation)
    # New vertical lines parameters
    vlines_list=None,         # List of x positions to draw vertical lines
    vlines_colors=None,       # List of colors for vertical lines
    vlines_styles=None,       # List of line styles for vertical lines
    vlines_widths=None,       # List of line widths for vertical lines
    # New text box parameters
    textboxes_list=None,      # List of [text, x, y] lists for each textbox
    textbox_fontsize=12,      # Font size for textboxes
    textbox_colors=None,      # List of colors for each textbox
):
    """
    Creates an x-y plot with multiple twin y-axes with configurable parameters.
    Enhanced version with support for A11 data on any axis.
    
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
        Column name to plot on x-axis, by default 'Time_VisIt'
    fit_y_variables : list, optional
        List of y variables to fit with LLS, by default None
    fit_x_start_values : list, optional
        List of x start values for each fit, by default None
    fit_x_stop_values : list, optional
        List of x stop values for each fit, by default None
    fit_colors : list, optional
        List of colors for each fit line, by default None
    fit_line_styles : list, optional
        List of line styles for each fit line, by default None
    fit_line_widths : list, optional
        List of line widths for each fit line, by default None
    output_dir_manual : str, optional
        Manual output directory, by default ""
    output_dir_comment : str, optional
        Comment to append to output directory name, by default ""
    image_list : list, optional
        List of image numbers to include in plot, by default []
    omit_image_list : list, optional
        List of image numbers to exclude from plot, by default [106]
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
        Transparency of grid lines, by default 0.5
    x_label : str, optional
        Label for x-axis, by default None
    x_label_fontsize : int, optional
        Font size for x-axis label, by default 20
    tick_label_fontsize : int, optional
        Font size for tick labels, by default 20
    legend_fontsize : int, optional
        Font size for legend, by default 20
    legend_title : str, optional
        Title for legend, by default None
    legend_loc : str, optional
        Location for legend, by default 'best'
    show_legend : bool, optional
        Whether to display the legend, by default True
    axis_spacings : list, optional
        List of spacings for each twin axis in points, by default None
    figsize : tuple, optional
        Figure size (width, height), by default (14, 6)
    dpi : int, optional
        DPI for output images, by default 300
    show_plot : int, optional
        Whether to display the plot, by default 0
    Plot_log_level : int, optional
        Logging verbosity, by default 1
    include_A11_data : bool, optional
        Whether to include A11 simulation data, by default True
    A11_data : dict, optional
        Dictionary of A11 data, by default None
    A11_x_column : str, optional
        Column name for A11 x-axis data, by default 'time'
    A11_y_columns_list : list of lists, optional
        List of lists of A11 column names to plot, by default [['N_c']]
    A11_y_scale_factors_list : list of lists, optional
        List of lists of scale factors for A11 data, by default [[1.0]]
    A11_line_styles_list : list of lists, optional
        List of lists of line styles for A11 data, by default [['--']]
    A11_line_widths_list : list of lists, optional
        List of lists of line widths for A11 data, by default [[1.5]]
    A11_line_colors_list : list of lists, optional
        List of lists of colors for A11 data, by default [['black']]
    A11_marker_styles_list : list of lists, optional
        List of lists of marker styles for A11 data, by default [['']]
    A11_marker_sizes_list : list of lists, optional
        List of lists of marker sizes for A11 data, by default [[6]]
    A11_labels_list : list of lists, optional
        List of lists of labels for A11 data, by default [['A11 Data']]
    A11_axis_indices_list : list of lists, optional
        List of lists indicating which axis to use for A11 data, by default [[0]]
        
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
        print(f"plotter_14_xTwinyyyy_A11: Output directory: {output_dir}")

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
    
    # Load A11 data if requested and not provided
    if include_A11_data and A11_data is None:
        try:
            A11_data = load_A11_data()
            if Plot_log_level >= 1:
                print("Loaded A11 data successfully")
        except Exception as e:
            print(f"Failed to load A11 data: {e}")
            include_A11_data = False
    
    # Default axis spacings if not provided (60, 120, 180, etc.)
    if axis_spacings is None:
        axis_spacings = [(i+1) * 60 for i in range(len(y_columns_list) - 1)]
    elif len(axis_spacings) < len(y_columns_list) - 1:
        # Extend axis_spacings if it's shorter than needed
        additional_spacings = [(len(axis_spacings) + i + 1) * 60 for i in range(len(y_columns_list) - 1 - len(axis_spacings))]
        axis_spacings = axis_spacings + additional_spacings
        if Plot_log_level >= 1:
            print(f"Extended axis_spacings to {axis_spacings}")
    
    # Create a figure for the plot
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
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
        
        # Use custom legend text if provided, otherwise use column name
        legend_text = y_legend_texts_list[0][i] if (y_legend_texts_list and len(y_legend_texts_list) > 0 and 
                                                  i < len(y_legend_texts_list[0])) else f"{y_column}"
        
        line, = ax_main.plot(df[x_column], df[y_column] * scale_factor, 
                            marker=marker, markersize=marker_size,
                            linestyle=line_style, linewidth=line_width,
                            color=color, label=legend_text)
        all_lines.append(line)
        all_labels.append(legend_text)
    
    ax_main.set_ylabel(y_labels_list[0], color=y_colors_list[0][0], fontsize=x_label_fontsize)
    ax_main.tick_params(axis='y', labelcolor=y_colors_list[0][0], labelsize=tick_label_fontsize, direction='in')
    ax_main.spines['left'].set_color(y_colors_list[0][0])
    
    # Create and plot on twin axes for each additional set of y_columns
    twin_axes = []
    
    for i in range(1, len(y_columns_list)):
        # Create a twin axis
        ax_twin = ax_main.twinx()
        twin_axes.append(ax_twin)
        all_axes.append(ax_twin)
        
        # Move the axis to the right with the specified spacing
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
            
            # Use custom legend text if provided, otherwise use column name
            legend_text = y_legend_texts_list[i][j] if (y_legend_texts_list and i < len(y_legend_texts_list) and 
                                                  j < len(y_legend_texts_list[i])) else f"{y_column}"
            
            line, = ax_twin.plot(df[x_column], df[y_column] * scale_factor, 
                                marker=marker, markersize=marker_size,
                                linestyle=line_style, linewidth=line_width,
                                color=color, label=legend_text)
            all_lines.append(line)
            all_labels.append(legend_text)
        
        # Set the y-axis label and color
        if i < len(y_labels_list):
            ax_twin.set_ylabel(y_labels_list[i], color=y_colors_list[i][0], fontsize=x_label_fontsize)
        else:
            ax_twin.set_ylabel(f"Group {i+1}", color=y_colors_list[i][0], fontsize=x_label_fontsize)
        
        # Configure the y-axis tick parameters
        ax_twin.tick_params(axis='y', labelcolor=y_colors_list[i][0], labelsize=tick_label_fontsize, direction='in')
    
    # Add A11 data if requested - Enhanced version with multi-axis support
    if include_A11_data and A11_data is not None:
        # Process each group of A11 data
        for group_idx, (y_columns, scale_factors, line_styles, line_widths, 
                        line_colors, marker_styles, marker_sizes, labels, axis_indices) in enumerate(
            zip(A11_y_columns_list, A11_y_scale_factors_list, A11_line_styles_list, 
                A11_line_widths_list, A11_line_colors_list, A11_marker_styles_list, 
                A11_marker_sizes_list, A11_labels_list, A11_axis_indices_list)):
            
            # Process each column in the group
            for col_idx, (y_column, scale_factor, line_style, line_width, 
                        line_color, marker_style, marker_size, label, axis_idx) in enumerate(
                zip(y_columns, scale_factors, line_styles, line_widths, 
                    line_colors, marker_styles, marker_sizes, labels, axis_indices)):
                
                # Check if the A11 data column exists
                if y_column in A11_data:
                    a11_df = A11_data[y_column]
                    
                    # Determine which axis to use for this A11 data
                    if axis_idx == 0:
                        a11_ax = ax_main
                    elif 0 < axis_idx <= len(twin_axes):
                        a11_ax = twin_axes[axis_idx - 1]
                    else:
                        if Plot_log_level >= 1:
                            print(f"Invalid A11 axis index {axis_idx} for column {y_column}, using main axis")
                        a11_ax = ax_main
                    
                    # Handle A11 data with different approaches based on whether we're using the same x or not
                    if A11_use_same_x:
                        # If using same x as main data, create x values matching the main data
                        if Plot_log_level >= 1:
                            print(f"Using main data x-axis values for A11 data ({x_column})")
                        
                        # Get x values from main dataframe
                        x_values = df[x_column].values
                        
                        # Handle interpolation for different length data
                        if len(a11_df) != len(x_values):
                            if Plot_log_level >= 1:
                                print(f"A11 data length ({len(a11_df)}) differs from main data ({len(x_values)}), interpolating...")
                            
                            # Import interp1d here to avoid dependency if not used
                            from scipy.interpolate import interp1d
                            
                            # Get original A11 x and y values - use index as source x
                            a11_x_original = np.arange(len(a11_df))
                            a11_y_original = a11_df[y_column].values * scale_factor
                            
                            # Create interpolation function
                            f = interp1d(a11_x_original, a11_y_original, bounds_error=False, fill_value="extrapolate")
                            
                            # Interpolate to match main data x range
                            x_interp = np.linspace(0, len(a11_df)-1, len(x_values))
                            y_interp = f(x_interp)
                            
                            # Plot with interpolated values
                            line, = a11_ax.plot(
                                x_values, 
                                y_interp,
                                marker=marker_style, markersize=marker_size,
                                linestyle=line_style, linewidth=line_width,
                                color=line_color, label=label
                            )
                            all_lines.append(line)
                            all_labels.append(label)
                        else:
                            # If lengths match, use main data x values directly
                            line, = a11_ax.plot(
                                x_values, 
                                a11_df[y_column].values * scale_factor,
                                marker=marker_style, markersize=marker_size,
                                linestyle=line_style, linewidth=line_width,
                                color=line_color, label=label
                            )
                            all_lines.append(line)
                            all_labels.append(label)
                    elif A11_x_column in a11_df.columns and y_column in a11_df.columns:
                        # Standard case - use A11's own x column
                        line, = a11_ax.plot(
                            a11_df[A11_x_column], 
                            a11_df[y_column] * scale_factor,
                            marker=marker_style, markersize=marker_size,
                            linestyle=line_style, linewidth=line_width,
                            color=line_color, label=label
                        )
                        all_lines.append(line)
                        all_labels.append(label)
                        if Plot_log_level >= 1:
                            print(f"Added A11 data to plot: {y_column} vs {A11_x_column}")
                    else:
                        if Plot_log_level >= 1:
                            print(f"Required columns {A11_x_column} or {y_column} not found in A11 data")
                else:
                    if Plot_log_level >= 1:
                        print(f"A11 data column {y_column} not found. Available columns: {list(A11_data.keys())}")

    # Add LLS fit lines if requested
    if fit_y_variables and len(fit_y_variables) > 0:
        
        # Initialize default values if not provided
        if fit_x_start_values is None:
            fit_x_start_values = [df[x_column].min()] * len(fit_y_variables)
        if fit_x_stop_values is None:
            fit_x_stop_values = [df[x_column].max()] * len(fit_y_variables)
        if fit_colors is None:
            fit_colors = ['black'] * len(fit_y_variables)
        if fit_line_styles is None:
            fit_line_styles = ['--'] * len(fit_y_variables)
        if fit_line_widths is None:
            fit_line_widths = [2.0] * len(fit_y_variables)
        
        # Ensure all lists are of same length
        n_fits = len(fit_y_variables)
        if len(fit_x_start_values) < n_fits:
            fit_x_start_values = fit_x_start_values + [df[x_column].min()] * (n_fits - len(fit_x_start_values))
        if len(fit_x_stop_values) < n_fits:
            fit_x_stop_values = fit_x_stop_values + [df[x_column].max()] * (n_fits - len(fit_x_stop_values))
        if len(fit_colors) < n_fits:
            fit_colors = fit_colors + ['black'] * (n_fits - len(fit_colors))
        if len(fit_line_styles) < n_fits:
            fit_line_styles = fit_line_styles + ['--'] * (n_fits - len(fit_line_styles))
        if len(fit_line_widths) < n_fits:
            fit_line_widths = fit_line_widths + [2.0] * (n_fits - len(fit_line_widths))
        
        # Process each fit
        for i, y_var in enumerate(fit_y_variables):
            if y_var in df.columns:
                # Filter data for fit range
                x_start = fit_x_start_values[i]
                x_stop = fit_x_stop_values[i]
                fit_df = df[(df[x_column] >= x_start) & (df[x_column] <= x_stop)]
                
                if len(fit_df) >= 2:  # Need at least 2 points for a line
                    try:
                        # Ensure we're working with numeric data
                        x_data = pd.to_numeric(fit_df[x_column], errors='coerce')
                        y_data = pd.to_numeric(fit_df[y_var], errors='coerce')
                        
                        # Remove NaN values that might have been introduced by coercion
                        valid_idx = ~(np.isnan(x_data) | np.isnan(y_data))
                        x_data = x_data[valid_idx].values
                        y_data = y_data[valid_idx].values
                        
                        # Check if we still have enough points after filtering
                        if len(x_data) >= 2:
                            # Perform linear fit
                            coeffs = np.polyfit(x_data, y_data, 1)
                            slope, intercept = coeffs
                            
                            # Generate points for the line
                            x_fit = np.array([x_start, x_stop])
                            y_fit = slope * x_fit + intercept
                            
                            # Determine which axis to plot on based on which axis contains this variable
                            target_ax = ax_main  # Default to main axis
                            
                            for j, y_cols in enumerate(y_columns_list):
                                if y_var in y_cols:
                                    if j == 0:
                                        target_ax = ax_main
                                    else:
                                        target_ax = twin_axes[j-1]
                                    break
                            
                            # Plot the fit line
                            fit_label = f"Fit {y_var}: {slope:.2f}x + {intercept:.2f}"
                            fit_line, = target_ax.plot(
                                x_fit, y_fit, 
                                color=fit_colors[i],
                                linestyle=fit_line_styles[i],
                                linewidth=fit_line_widths[i],
                                label=fit_label
                            )
                            
                            all_lines.append(fit_line)
                            all_labels.append(fit_label)
                        else:
                            if Plot_log_level >= 1:
                                print(f"Warning: Not enough valid numeric data points for fit of {y_var}")
                    except Exception as e:
                        if Plot_log_level >= 1:
                            print(f"Warning: Error fitting {y_var}: {e}")
                else:
                    if Plot_log_level >= 1:
                        print(f"Warning: Not enough points for fit of {y_var} between x={x_start} and x={x_stop}")
            else:
                if Plot_log_level >= 1:
                    print(f"Warning: Fit variable {y_var} not found in DataFrame")
    
    # Set the x-axis label
    ax_main.set_xlabel(x_label if x_label else x_column, fontsize=x_label_fontsize)
    
    # Configure tick parameters for inward facing ticks
    for ax in all_axes:
        ax.tick_params(axis='both', direction='in', which='both', labelsize=tick_label_fontsize)
    
    # Add grid if requested
    if show_grid:
        ax_main.grid(True, linestyle=grid_style, linewidth=grid_width, 
                  alpha=grid_alpha, color=grid_color)
    
    # Add vertical lines if specified
    if vlines_list:
        # Set default styles if not provided
        if vlines_colors is None:
            vlines_colors = ['black'] * len(vlines_list)
        if vlines_styles is None:
            vlines_styles = ['--'] * len(vlines_list)
        if vlines_widths is None:
            vlines_widths = [1.0] * len(vlines_list)
            
        # Extend lists if they're shorter than needed
        vlines_colors = vlines_colors * (len(vlines_list) // len(vlines_colors) + 1)
        vlines_styles = vlines_styles * (len(vlines_list) // len(vlines_styles) + 1)
        vlines_widths = vlines_widths * (len(vlines_list) // len(vlines_widths) + 1)
        
        # Get current y-limits to constrain vertical lines
        ymin, ymax = ax_main.get_ylim()
        
        # Draw each vertical line
        for i, x_pos in enumerate(vlines_list):
            # Use standard axvline with explicit y limits instead of using transform coordinates
            ax_main.axvline(x=x_pos, 
                         color=vlines_colors[i], 
                         linestyle=vlines_styles[i], 
                         linewidth=vlines_widths[i],
                         ymin=0, ymax=1.0,
                         zorder=1)  # Lower zorder to ensure it's behind data points
    
    # Add text boxes if specified
    if textboxes_list:
        # Set default text colors if not provided
        if textbox_colors is None:
            textbox_colors = ['black'] * len(textboxes_list)
        else:
            # Extend colors list if needed
            textbox_colors = textbox_colors * (len(textboxes_list) // len(textbox_colors) + 1)
            
        # Add each text box
        for i, textbox in enumerate(textboxes_list):
            if len(textbox) >= 3:  # Make sure we have text, x, and y values
                text, x, y = textbox[0], textbox[1], textbox[2]
                # Use bbox to create a text box with a light background
                ax_main.text(x, y, text, 
                         fontsize=textbox_fontsize, 
                         color=textbox_colors[i],
                         ha='center', va='center',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2),
                         transform=ax_main.transAxes if len(textbox) < 4 else None,  # Use axes coordinates by default
                         zorder=10)  # Higher zorder to ensure it's on top
    
    # Add a combined legend if requested - support for tuple coordinates
    if show_legend:
        if legend_title:
            if isinstance(legend_loc, tuple):
                plt.figlegend(all_lines, all_labels, loc='center', bbox_to_anchor=legend_loc, 
                            fontsize=legend_fontsize, frameon=False, title=legend_title, 
                            title_fontsize=legend_fontsize+2)
            else:
                plt.figlegend(all_lines, all_labels, loc=legend_loc, fontsize=legend_fontsize, 
                            frameon=False, title=legend_title, title_fontsize=legend_fontsize+2)
        else:
            if isinstance(legend_loc, tuple):
                plt.figlegend(all_lines, all_labels, loc='center', bbox_to_anchor=legend_loc, 
                            fontsize=legend_fontsize, frameon=False)
            else:
                plt.figlegend(all_lines, all_labels, loc=legend_loc, fontsize=legend_fontsize, 
                            frameon=False)
    
    # Don't use tight_layout() which can cause the layout issues
    # Instead, adjust the spacing manually
    plt.subplots_adjust(right=0.85, left=0.1, top=0.9, bottom=0.15)
    
    # Create filename from x and column groups
    y_columns_str = "_".join([f"Group{i+1}" for i in range(len(y_columns_list))])
    
    base_filename = f"{x_column.replace(' ', '_')}_vs_{y_columns_str}"
    png_path = os.path.join(png_dir, f"{base_filename}.png")
    svg_path = os.path.join(svg_dir, f"{base_filename}.svg")
    
    # Save with fixed size instead of using bbox_inches='tight'
    plt.savefig(png_path, dpi=dpi)
    plt.savefig(svg_path, format='svg')
    
    if Plot_log_level >= 1:
        print(f"Saved figures to:\n  {png_path}\n  {svg_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return output_dir

# Example usage
if __name__ == "__main__":
    # Example usage with twin axes for different metrics
    input_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187"
    
    # Define column groups for each twin axis (four axes)
    y_columns_list = [
        [],                                      # First axis (main) - empty as we'll only put A11 data here
        ['d_cell_SRec_mean_CSTx6_nonDim'],      # Second axis (green)
        ['N_cells_CSTx6'],                      # Third axis (red)
        ['contour_length_SRec_total_CSTx6_nonDim']  # Fourth axis (blue)
    ]
    
    # Define colors for each axis
    y_colors_list = [
        ['darkblue'],  # Color for first axis (A11 data)
        ['green'],     # Color for second axis
        ['red'],       # Color for third axis
        ['blue']       # Color for fourth axis
    ]
    
    # Define labels for each axis
    y_labels_list = [
        r'$\dot{R}_{SEF}/S_L$',       # First axis label for A11 data
        r'$\overline{d}_c/\delta_T$', # Second axis label
        r'$N_{\mathrm{cells}}$',      # Third axis label
        r'$L/\delta_T\times 10^4$'    # Fourth axis label
    ]
    
    # Define scale factors for each column
    y_scale_factors_list = [
        [1.0],  # No scaling for first axis (A11 data)
        [1.0],  # No scaling for second axis
        [1.0],  # No scaling for third axis
        [1e-4]  # Scale down contour length by 10^4
    ]
    
    # Custom spacings for each twin axis (in points)
    axis_spacings = [0, 70, 160]  # Spacings for the 2nd, 3rd, and 4th axes
    
    # A11 data parameters - plotting R_mean_dot on main axis
    A11_y_columns_list = [
        ['R_mean_dot'],  # Plot R_mean_dot on first axis
    ]
    
    A11_line_colors_list = [
        ['black'],  # Color for A11 data
    ]
    
    A11_labels_list = [
        ['$\dot{R}$'],  # Label for A11 data
    ]
    
    # Define custom legend texts for each column
    y_legend_texts_list = [
        ["$\dot{R}$"],  # Legend for the first axis (A11 data)
        ["$\overline{d}_c$"],  # Legend for second axis (green)
        ["$N_{\mathrm{cells}}$"],  # Legend for third axis (red)
        ["$L$"]  # Legend for fourth axis (blue)
    ]
    
    # Define vertical lines at specific x positions
    vlines = [30.0, 38.0, 60.0]  # Vertical lines at reasonable positions
    vlines_colors = ['gray', 'gray', 'gray']
    vlines_styles = ['--', '--', '--']
    vlines_widths = [2, 2, 2]
    
    # Define text boxes - use axis coordinates (0-1) instead of data coordinates
    # Each entry is [text, x-position, y-position] where x and y are between 0 and 1
    textboxes = [
        ["Geometric Expansion", 0.2, 1.1],  # Position at 20% from left, 90% from bottom
        ["Deepening", 0.5, 1.2],  # Position at 50% from left, 80% from bottom
        ["Flattening", 0.8, 1.3],  # Position at 80% from left, 70% from bottom
        ["Splitting + Deepening", 1, 1.3]  # Position at 80% from left, 70% from bottom
    ]
    
    # Load A11 data directly - don't modify it
    A11_data = load_A11_data()
    

    #### Old for thesis
    #
    # # Following plot14_xyyy.py's approach: use A11_use_same_x=True for interpolation
    # output_dir = plotter_14_xTwinyyyy_A11(
    #     input_dir=input_dir,
    #     x_column="R_SF_nonDim",
    #     y_columns_list=y_columns_list,
    #     y_colors_list=y_colors_list,
    #     y_labels_list=y_labels_list,
    #     y_scale_factors_list=y_scale_factors_list,
    #     y_legend_texts_list=y_legend_texts_list,
    #     output_dir_comment="A11_R_mean_dot_with_cell_metrics",
    #     x_label=r'$R_{SF}/\delta_T$',
    #     legend_loc=(0.2, 0.8),  # Position in normalized coordinates
    #     show_legend=True,
    #     axis_spacings=axis_spacings,
    #     include_A11_data=True,
    #     A11_data=A11_data,  # Use the original A11 data
    #     A11_x_column='R_mean',  # Use R_mean as x-axis for A11 data
    #     A11_use_same_x=True,  # Interpolate A11 data to match main x-axis values
    #     A11_y_columns_list=A11_y_columns_list,
    #     A11_line_colors_list=A11_line_colors_list,
    #     A11_labels_list=A11_labels_list,
    #     # Add vertical lines and text boxes
    #     vlines_list=vlines,
    #     vlines_colors=vlines_colors,
    #     vlines_styles=vlines_styles,
    #     vlines_widths=vlines_widths,
    #     textboxes_list=textboxes,
    #     textbox_fontsize=25,
    #     textbox_colors=['black', 'black', 'black', 'black'],
    #     show_plot=0
    # )


    # new for PROCI
    output_dir = plotter_14_xTwinyyyy_A11(
        input_dir=input_dir,
        x_column="Time_VisIt",
        y_columns_list=y_columns_list,
        y_colors_list=y_colors_list,
        y_labels_list=y_labels_list,
        y_scale_factors_list=y_scale_factors_list,
        y_legend_texts_list=y_legend_texts_list,
        output_dir_comment="A11_R_mean_dot_with_cell_metrics_vs_time_PROCI",
        x_label=r'$t/\tau$',
        legend_loc=(0.2, 0.8),  # Position in normalized coordinates
        show_legend=True,
        axis_spacings=axis_spacings,
        include_A11_data=True,
        A11_data=A11_data,  # Use the original A11 data
        A11_x_column='Time_VisIt',  # Use Time_VisIt as x-axis for A11 data
        A11_use_same_x=True,  # Interpolate A11 data to match main x-axis values
        A11_y_columns_list=A11_y_columns_list,
        A11_line_colors_list=A11_line_colors_list,
        A11_labels_list=A11_labels_list,
        # Add vertical lines and text boxes
        vlines_list=vlines,
        vlines_colors=vlines_colors,
        vlines_styles=vlines_styles,
        vlines_widths=vlines_widths,
        textboxes_list=textboxes,
        textbox_fontsize=25,
        textbox_colors=['black', 'black', 'black', 'black'],
        show_plot=0
    )

    
    print(f"Finished creating plot with A11 data in: {output_dir}")