import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import Format_1 as F_1

# LaTeX settings
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

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

def plotter_14_xyyy(
    input_dir,
    y_columns=['N_cells_CSTx6'],  # Changed to list of y-columns
    x_column='R_SF_nonDim',  # x_column remains a single value
    output_dir_manual="",
    output_dir_comment="",
    image_list=[],
    omit_image_list=[106],
    connect_with_lines=True,
    marker_styles=[''],  # List of marker styles
    marker_sizes=[6],    # List of marker sizes
    marker_colors=['blue'],  # List of marker colors
    line_styles=['-'],   # List of line styles
    line_widths=[1.5],   # List of line widths
    line_colors=['blue'],  # List of line colors
    show_grid=True,
    grid_style='--',
    grid_width=0.5,
    grid_color='gray',
    grid_alpha=0.5,
    x_label=None,
    y_label=None,
    legend_labels=None,  # List of legend labels
    legend_title=None,   # Optional title for the legend
    x_label_fontsize=20,
    y_label_fontsize=20,
    tick_label_fontsize=20,
    legend_fontsize=20,
    legend_loc='upper left',
    figsize=(10, 6),
    dpi=100,
    show_plot=0,
    Plot_log_level=1,
    # A11 data parameters
    include_A11_data=False,
    A11_data=None,
    A11_x_column='time',
    A11_use_same_x=False,  # Whether to use the same x-axis as main data
    A11_y_column='N_c',
    A11_y_scale_factor=1.0,  # New parameter: scale factor for A11 y-axis values
    A11_line_style='--',
    A11_line_width=1.5,
    A11_line_color='red',
    A11_marker_style='',
    A11_marker_size=6,
    A11_label='A11 Data'
):
    """
    Creates an x-y plot with multiple curves, each with configurable parameters.
    
    Parameters
    ----------
    input_dir : str
        Directory containing the Analysis_A11_final_df.pkl file
    y_columns : list
        List of column names to plot on y-axis
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
    marker_styles : list, optional
        List of styles of markers, by default ['']
    marker_sizes : list, optional
        List of sizes of markers, by default [6]
    line_styles : list, optional
        List of styles of lines, by default ['-']
    line_widths : list, optional
        List of widths of lines, by default [1.5]
    line_colors : list, optional
        List of colors of lines, by default ['blue']
    marker_colors : list, optional
        List of colors of markers, by default ['blue']
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
    y_label : str, optional
        Label for y-axis, by default None (will use a generic label)
    legend_labels : list, optional
        List of custom text for the legend labels. If None, y_columns will be used.
    legend_title : str, optional
        Title for the legend, by default None
    x_label_fontsize : int, optional
        Font size for x-axis label, by default 20
    y_label_fontsize : int, optional
        Font size for y-axis label, by default 20
    tick_label_fontsize : int, optional
        Font size for tick labels, by default 20
    legend_fontsize : int, optional
        Font size for legend, by default 20
    legend_loc : str, optional
        Location of legend, by default 'upper left'
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (10, 6)
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
    A11_use_same_x : bool, optional
        Whether to plot A11 data using the same x values as the main data, by default False
    A11_y_column : str, optional
        Column name for A11 y-axis data, by default 'N_c'
    A11_y_scale_factor : float, optional
        Scale factor to multiply A11 y-axis values by, by default 1.0 (no scaling)
    A11_line_style : str, optional
        Line style for A11 data, by default '--'
    A11_line_width : float, optional
        Line width for A11 data, by default 1.5
    A11_line_color : str, optional
        Line color for A11 data, by default 'red'
    A11_marker_style : str, optional
        Marker style for A11 data, by default ''
    A11_marker_size : int, optional
        Marker size for A11 data, by default 6
    A11_label : str, optional
        Label for A11 data in legend, by default 'A11 Data'
    
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
        print(f"plotter_14_xyyy: Output directory: {output_dir}")

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
    
    # Prepare styling parameters - extend lists if they're shorter than y_columns
    num_curves = len(y_columns)
    
    # Extend style lists if needed
    marker_styles = marker_styles * (num_curves // len(marker_styles) + 1)
    marker_sizes = marker_sizes * (num_curves // len(marker_sizes) + 1)
    marker_colors = marker_colors * (num_curves // len(marker_colors) + 1)
    line_styles = line_styles * (num_curves // len(line_styles) + 1)
    line_widths = line_widths * (num_curves // len(line_widths) + 1)
    line_colors = line_colors * (num_curves // len(line_colors) + 1)
    
    # Use y_columns as default legend labels if none provided
    if legend_labels is None:
        legend_labels = y_columns
    else:
        # Extend legend labels if needed
        legend_labels = legend_labels * (num_curves // len(legend_labels) + 1)
    
    # Plot each y-column
    for i, y_column in enumerate(y_columns):
        if i >= len(marker_styles):
            i_mod = i % len(marker_styles)
        else:
            i_mod = i
            
        # Get styling for this curve
        marker_style = marker_styles[i_mod] if i_mod < len(marker_styles) else ''
        marker_size = marker_sizes[i_mod] if i_mod < len(marker_sizes) else 6
        marker_color = marker_colors[i_mod] if i_mod < len(marker_colors) else 'blue'
        line_style = line_styles[i_mod] if i_mod < len(line_styles) else '-'
        line_width = line_widths[i_mod] if i_mod < len(line_widths) else 1.5
        line_color = line_colors[i_mod] if i_mod < len(line_colors) else 'blue'
        legend_label = legend_labels[i_mod] if i_mod < len(legend_labels) else y_column
        
        # Create plot
        if connect_with_lines:
            plt.plot(df[x_column], df[y_column], 
                    marker=marker_style, markersize=marker_size, 
                    linestyle=line_style, linewidth=line_width,
                    color=line_color, markerfacecolor=marker_color, 
                    markeredgecolor='black', label=legend_label)
        else:
            plt.scatter(df[x_column], df[y_column], 
                      s=marker_size**2, marker=marker_style,
                      color=marker_color, edgecolors='black', 
                      label=legend_label)
    
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
            
            # Decide which x column to use for A11 data
            a11_x_col = x_column if A11_use_same_x else A11_x_column
            
            # Apply scaling factor to A11 y values
            if A11_y_scale_factor != 1.0 and Plot_log_level >= 1:
                print(f"Applying scale factor of {A11_y_scale_factor} to A11 {A11_y_column} values")
            
            if A11_use_same_x:
                # If using same x as main data, create x values matching the main data
                if Plot_log_level >= 1:
                    print(f"Using main data x-axis values for A11 data ({x_column})")
                
                # Get x values from main dataframe
                x_values = df[x_column].values
                
                # If A11 data has a different length than main data, interpolate
                if len(a11_df) != len(x_values):
                    if Plot_log_level >= 1:
                        print(f"A11 data length ({len(a11_df)}) differs from main data ({len(x_values)}), interpolating...")
                    
                    # Get original A11 x and y values
                    a11_x_original = np.arange(len(a11_df))
                    a11_y_original = a11_df[A11_y_column].values * A11_y_scale_factor  # Apply scale factor
                    
                    # Create interpolation function
                    from scipy.interpolate import interp1d
                    f = interp1d(a11_x_original, a11_y_original, bounds_error=False, fill_value="extrapolate")
                    
                    # Interpolate to match main data x range
                    x_interp = np.linspace(0, len(a11_df)-1, len(x_values))
                    y_interp = f(x_interp)
                    
                    # Plot with interpolated values
                    plt.plot(x_values, y_interp,
                            marker=A11_marker_style, markersize=A11_marker_size,
                            linestyle=A11_line_style, linewidth=A11_line_width,
                            color=A11_line_color, label=A11_label)
                else:
                    # If lengths match, use main data x values directly
                    plt.plot(x_values, a11_df[A11_y_column].values * A11_y_scale_factor,  # Apply scale factor
                            marker=A11_marker_style, markersize=A11_marker_size,
                            linestyle=A11_line_style, linewidth=A11_line_width,
                            color=A11_line_color, label=A11_label)
            elif a11_x_col in a11_df.columns and A11_y_column in a11_df.columns:
                # Standard case - use A11's own x column
                plt.plot(a11_df[a11_x_col], a11_df[A11_y_column] * A11_y_scale_factor,  # Apply scale factor
                        marker=A11_marker_style, markersize=A11_marker_size,
                        linestyle=A11_line_style, linewidth=A11_line_width,
                        color=A11_line_color, label=A11_label)
                if Plot_log_level >= 1:
                    print(f"Added A11 data to plot: {A11_y_column} vs {a11_x_col}")
            else:
                print(f"A11 data missing required columns: {a11_x_col} or {A11_y_column}")
        else:
            print(f"A11 data for {A11_y_column} not found")
            
    # Set labels
    plt.xlabel(x_label if x_label else x_column, fontsize=x_label_fontsize)
    
    # For y-label, use the provided y_label or a generic label since we have multiple columns
    if y_label:
        plt.ylabel(y_label, fontsize=y_label_fontsize)
    else:
        # If all y-columns end with same unit, we can use that
        suffix_match = True
        common_suffix = ''
        for col in y_columns:
            if common_suffix and not col.endswith(common_suffix):
                suffix_match = False
                break
            elif not common_suffix and '_' in col:
                common_suffix = col.split('_')[-1]
        
        if suffix_match and common_suffix:
            plt.ylabel(f"Values ({common_suffix})", fontsize=y_label_fontsize)
        else:
            plt.ylabel("Values", fontsize=y_label_fontsize)
    
    # Set tick parameters for inward facing ticks
    plt.tick_params(axis='both', direction='in', which='both', labelsize=tick_label_fontsize)
    
    # Add grid if requested
    if show_grid:
        plt.grid(True, linestyle=grid_style, linewidth=grid_width, 
                alpha=grid_alpha, color=grid_color)
    
    # Add legend
    if legend_title:
        plt.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=False, title=legend_title, 
                  title_fontsize=legend_fontsize+2)
    else:
        plt.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create filename from x and first few y columns
    y_columns_str = "_".join([y_col.replace(' ', '_') for y_col in y_columns[:min(3, len(y_columns))]])
    if len(y_columns) > 3:
        y_columns_str += "_etc"
    
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
    

    # # print length of individual y_columns
    # print(f"Number of y_columns: {len(y_columns)}")
    # for i, y_col in enumerate(y_columns):
    #     print(f"  {i+1}. {y_col} (length: {len(df[y_col])})")

    # # print every tenth value of the y_collumns
    # for i, y_col in enumerate(y_columns):
    #     print(f"  {i+1}. {y_col} (every 10th value):")
    #     for j in range(0, len(df[y_col]), 10):
    #         print(f"    {j}: {df[y_col].iloc[j]}")


    # print(f"column contents of d_cell_SRec_distribution_nonDim: "
    #       f"{df['d_cell_SRec_distribution_nonDim'].tolist() if 'd_cell_SRec_distribution_nonDim' in df.columns else 'Column not found'}")

    return output_dir


if __name__ == "__main__":
    # Example usage with multiple curves for diameter metrics
    plotter_14_xyyy(
        input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187",
        x_column="R_SF_nonDim",
        y_columns=[
            "d_cell_mean_nonDim",
            "d_cell_mean_CST_nonDim",
            "d_cell_SRec_mean_nonDim",
            "d_cell_SRec_mean_CST_nonDim"  # Added the 3D contour length for comparison
        ],
        output_dir_comment="multiple_metrics_vs_R_SF_nonDim",
        line_colors=['green', 'green', 'green'],
        line_styles=[':', '-.', '--', '-'],
        marker_styles=['', '', '', ''],
        marker_sizes=[6, 6, 6],
        legend_labels=[
            r'2D',
            r'2D in Tile',
            r'3D',
            r'3D in Tile'
        ],
        legend_title="",
        legend_loc='upper left',
        x_label=r'$R_{SF}/\delta_T$',
        y_label=r'$\overline{d}_c/\delta_T$',
    )
    
    # Example usage with multiple curves for area metrics
    plotter_14_xyyy(
        input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187",
        x_column="R_SF_nonDim",
        y_columns=[
            "A_cell_mean_nonDim2",
            "A_cell_mean_CST_nonDim2",
            "A_cell_SRec_mean_nonDim2",
            "A_cell_SRec_mean_CST_nonDim2"
        ],
        output_dir_comment="area_metrics_vs_R_SF_nonDim",
        line_colors=['darkgreen', 'darkgreen', 'darkgreen'],
        line_styles=[':', '-.', '--', '-'],
        marker_styles=['', '', '', ''],
        marker_sizes=[6, 6, 6],
        legend_labels=[
            r'2D',
            r'2D in Tile',
            r'3D',
            r'3D in Tile'
        ],
        legend_title="",
        legend_loc='upper left',
        x_label=r'$R_{SF}/\delta_T$',
        y_label=r'$\overline{A}_c/\delta_T^2$',
    )
    
    # Example usage with multiple curves for cell count metrics (with A11 data)
    # Load A11 data first
    A11_data = load_A11_data()
        
    # Example with A11 data using same x-axis as main data
    plotter_14_xyyy(
        input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187",
        x_column="R_SF_nonDim",
        y_columns=[
            "N_cells",
            "N_cells_CST",
        ],
        output_dir_comment="cell_counts_vs_R_SF_with_A11_same_x",
        line_colors=['red', 'red'],
        line_styles=[':', '-'],
        marker_styles=['', ''],
        marker_sizes=[6, 7],
        legend_labels=[
            r'Cellpose ',
            r'Cellpose in Tile',
        ],
        legend_title="",
        legend_loc='upper left',
        x_label=r'$R_{SF}/\delta_T$',
        y_label=r'$N_{cells}$',

        include_A11_data=True,
        A11_data=A11_data,
        A11_y_column='N_c',
        A11_y_scale_factor=1/6.0,  # Scale the A11 data by a factor of 6
        A11_use_same_x=True,  # Use same x-axis as main data
        A11_line_style='--',
        A11_line_color='black',
        A11_label=r'Manual Count in Tile'
    )
    
    # Example usage with multiple curves for contour length metrics
    plotter_14_xyyy(
        input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187",
        x_column="R_SF_nonDim",
        y_columns=[
            "contour_length_total_nonDim",
            "contour_length_total_CST_nonDim",
            "contour_length_SRec_total_nonDim",
            "contour_length_SRec_total_CST_nonDim"
        ],
        output_dir_comment="contour_length_metrics_vs_R_SF_nonDim",
        line_colors=['blue', 'blue', 'blue'],
        line_styles=[':', '-.', '--', '-'],
        marker_styles=['', '', '', ''],
        marker_sizes=[6, 6, 6],
        legend_labels=[
            r'2D',
            r'2D in Tile',
            r'3D',
            r'3D in Tile'
        ],
        legend_title="",
        legend_loc='upper left',
        x_label=r'$R_{SF}/\delta_T$',
        y_label=r'$L$/$\delta_T$',
    )