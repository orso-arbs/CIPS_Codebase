import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import cv2
from PIL import Image
import Format_1 as F_1

# LaTeX settings
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

def plotter_14_xyyy_image_A11(
    input_dir,
    y_columns=['N_cells_CSTx6'],  # List of y-columns
    x_column='R_SF_nonDim',  # x_column for the plot
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
    legend_show=True,  # Whether to show the legend
    legend_fontsize=20,
    legend_loc='upper left',
    # Image display parameters
    image_width_ratio=0.3,     # Width ratio for image subplot
    plot_width_ratio=0.7,      # Width ratio for property plot subplot
    plot_spacing=0.2,          # Horizontal spacing between plots
    ScaleFactor=1.5,           # Scale factor for zooming in on the spherical flame
    show_segmentation=True,    # Whether to show segmentation overlay
    color_only_CST_cells=False, # Whether to color only cells included in the CST
    show_segmentation_on_both_halves=False, # Whether to show segmentation on both halves (True) or only right half (False)
    create_video=False,        # Whether to create a video from the images
    video_fps=5,               # Frames per second for the video
    figsize=(20, 8),
    dpi=100,
    show_plot=0,
    Plot_log_level=1,
    # A11 data parameters
    include_A11_data=False,
    A11_data=None,
    A11_x_column='time',
    A11_use_same_x=False,  # Whether to use the same x-axis as main data
    A11_y_column='N_c',
    A11_y_scale_factor=1.0,  # Scale factor for A11 y-axis values
    A11_line_style='--',
    A11_line_width=1.5,
    A11_line_color='red',
    A11_marker_style='',
    A11_marker_size=6,
    A11_label='A11 Data',
    # Linear fit parameters
    fit_enable=False,
    y_fit_list=[],
    x_fit_list=[],
    x_range_fit_list=[],
    fit_line_colors=['red'],
    fit_line_widths=[2.0],
    fit_line_styles=['--'],
    fit_labels=[],
    show_fit_equation=True,
    fit_equation_position='legend'  # or 'plot' to put equations on the plot
):
    """
    Creates an x-y plot with multiple curves alongside the current image.
    Linear fits can be added for specified y vs x data within given x ranges.
    
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
    image_width_ratio : float, optional
        Width ratio for image subplot, by default 0.3
    plot_width_ratio : float, optional
        Width ratio for property plot subplot, by default 0.7
    plot_spacing : float, optional
        Horizontal spacing between plots, by default 0.2
    ScaleFactor : float, optional
        Scale factor for zooming in on the spherical flame, by default 1.5
    show_segmentation : bool, optional
        Whether to show segmentation overlay, by default True
    color_only_CST_cells : bool, optional
        Whether to color only cells included in the CST, by default False
    show_segmentation_on_both_halves : bool, optional
        Whether to show segmentation on both halves of the image (True) or only on the right half (False), 
        by default False. When False, the left half shows the original image and right half shows segmentation overlay
    create_video : bool, optional
        Whether to create a video from the images, by default False
    video_fps : int, optional
        Frames per second for the video, by default 5
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (20, 8)
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
    fit_enable : bool, optional
        Whether to calculate and display linear fits, by default False
    y_fit_list : list, optional
        List of y column names to use for fitting, by default []
    x_fit_list : list, optional
        List of x column names to use for fitting (must match y_fit_list in length), by default []
    x_range_fit_list : list, optional
        List of [min_x, max_x] pairs defining the x range for each fit, by default []
    fit_line_colors : list, optional
        List of colors for fit lines, by default ['red']
    fit_line_widths : list, optional
        List of widths for fit lines, by default [2.0]
    fit_line_styles : list, optional
        List of styles for fit lines, by default ['--']
    fit_labels : list, optional
        List of custom labels for fits, by default [] (will generate automatic labels)
    show_fit_equation : bool, optional
        Whether to display fit equations, by default True
    fit_equation_position : str, optional
        Where to display fit equations ('legend' or 'plot'), by default 'legend'
    
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
        print(f"plotter_14_xyyy_image_A11: Output directory: {output_dir}")

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
    
    # Load A11 data if requested and not provided
    if include_A11_data and A11_data is None:
        try:
            A11_data = load_A11_data()
            if Plot_log_level >= 1:
                print("Loaded A11 data successfully")
        except Exception as e:
            print(f"Failed to load A11 data: {e}")
            include_A11_data = False
    
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

    # Create a figure for each time point
    all_image_paths = []
    for idx, row in df.iterrows():
        if Plot_log_level >= 1:
            print(f"Processing row {idx + 1}/{len(df)}: Image {row['image_number']}")
        
        # Create figure with GridSpec for image and plot
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = gridspec.GridSpec(1, 2, width_ratios=[image_width_ratio, plot_width_ratio])
        gs.update(wspace=plot_spacing)  # Add horizontal spacing
        
        # Left subplot for image
        ax_image = plt.subplot(gs[0])
        
        # Get image info
        image_num = row['image_number']
        image_file_path = row['image_file_path']
        current_time = row['Time_VisIt']
        current_x_value = row[x_column]  # Get current value of the selected x-variable
        
        # Display image if path exists
        try:
            original_img = cv2.imread(image_file_path)
            if original_img is None:
                print(f"Error: Could not read image file: {image_file_path}")
                ax_image.text(0.5, 0.5, "Image\nloading\nerror", 
                            ha='center', va='center', 
                            transform=ax_image.transAxes)
                ax_image.axis('off')
            else:
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                
                # Get mask from DataFrame if available
                mask = None
                if 'masks' in row and row['masks'] is not None:
                    mask = row['masks']
                    if not isinstance(mask, np.ndarray):
                        if Plot_log_level >= 1:
                            print(f"Warning: Mask for image {image_num} is not a NumPy array. Type: {type(mask)}")
                        mask = None
                
                # Get image dimensions
                img_height, img_width = original_img.shape[:2]
                
                # Create a combined image if segmentation is enabled
                if show_segmentation and mask is not None:
                    # Create a combined image
                    combined_img = original_img.copy()
                    right_half_width = img_width // 2
                    
                    # Process either just the right half or the whole image
                    if not show_segmentation_on_both_halves:
                        # Original behavior: only right half with segmentation
                        # Convert right half to grayscale
                        right_half = original_img[:, right_half_width:, :]
                        gray_right_half = cv2.cvtColor(right_half, cv2.COLOR_RGB2GRAY)
                        gray_right_half = cv2.cvtColor(gray_right_half, cv2.COLOR_GRAY2RGB)
                        combined_img[:, right_half_width:, :] = gray_right_half
                        
                        # Define regions where we'll apply overlays
                        display_region = np.zeros_like(mask, dtype=bool)
                        display_region[:, right_half_width:] = True
                    else:
                        # New behavior: whole image with segmentation
                        # Convert whole image to grayscale
                        gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
                        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
                        combined_img = gray_img.copy()
                        
                        # Define region for the whole image
                        display_region = np.ones_like(mask, dtype=bool)
                    
                    # Add colored segmentation overlay
                    color_mask = np.zeros_like(original_img)
                    
                    # Create a mask that will be used to track cells to be colored
                    cells_to_color = np.zeros_like(mask, dtype=bool)
                    
                    # Define a list of distinct colors
                    distinct_colors = [
                        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                        (0, 255, 255), (255, 0, 255), (255, 165, 0), (128, 0, 128),
                        (165, 42, 42), (255, 192, 203)
                    ]

                    # Get unique mask values greater than 0
                    unique_mask_values = np.unique(mask[mask > 0])
                    
                    # Check if we should filter cells by CST inclusion
                    cst_inclusion = None
                    if color_only_CST_cells and 'CST_inclusion' in row and row['CST_inclusion'] is not None:
                        cst_inclusion = row['CST_inclusion']
                        if Plot_log_level >= 2:
                            print(f"Filtering cells by CST inclusion, {sum(cst_inclusion)} included out of {len(cst_inclusion)}")

                    for i, mask_value in enumerate(unique_mask_values):
                        # Check if this cell should be colored
                        cell_idx = i  # Mask value - 1 would be the cell index if masks start at 1
                        
                        # Only process cells that meet the inclusion criteria
                        should_color = True
                        if color_only_CST_cells and cst_inclusion is not None:
                            # Skip cells not in CST if filtering is enabled
                            if cell_idx >= len(cst_inclusion) or not cst_inclusion[cell_idx]:
                                should_color = False
                        
                        if should_color:
                            color_index = int(mask_value) % 10
                            selected_color = distinct_colors[color_index]
                            color_mask[mask == mask_value] = selected_color
                            
                            # Mark this cell to be included in the coloring
                            cells_to_color[mask == mask_value] = True
                    
                    # Apply overlay only to cells that should be colored
                    # and only in the display region (right half or whole image)
                    cells_to_process = cells_to_color & display_region
                    
                    # Apply overlay with transparency
                    alpha = 0.5  # Transparency factor
                    combined_img[cells_to_process] = (
                        alpha * color_mask[cells_to_process] + 
                        (1 - alpha) * combined_img[cells_to_process]
                    ).astype(np.uint8)
                    
                    # Add a vertical separator line if only showing on right half
                    if not show_segmentation_on_both_halves:
                        # Calculate zoom region to focus on the spherical flame
                        D_SF_px = row.get('D_SF_px', min(img_height, img_width) / 2)
                        center_x, center_y = img_width // 2, img_height // 2
                        zoom_half_size = int(D_SF_px * ScaleFactor / 2)
                        
                        # Ensure zoom region is within image bounds
                        left = max(0, center_x - zoom_half_size)
                        right = min(img_width, center_x + zoom_half_size)
                        top = max(0, center_y - zoom_half_size)
                        bottom = min(img_height, center_y + zoom_half_size)
                        
                        # Extract the zoom region
                        zoom_img = combined_img[top:bottom, left:right, :]
                        
                        # Display the zoomed image
                        ax_image.imshow(zoom_img)
                        
                        # Add a vertical line at the center to separate original and segmented views
                        ax_image.axvline(x=(right-left)//2, color='yellow', linestyle='-', linewidth=1)
                    else:
                        # For both halves mode, just display the whole zoomed area
                        D_SF_px = row.get('D_SF_px', min(img_height, img_width) / 2)
                        center_x, center_y = img_width // 2, img_height // 2
                        zoom_half_size = int(D_SF_px * ScaleFactor / 2)
                        
                        # Ensure zoom region is within image bounds
                        left = max(0, center_x - zoom_half_size)
                        right = min(img_width, center_x + zoom_half_size)
                        top = max(0, center_y - zoom_half_size)
                        bottom = min(img_height, center_y + zoom_half_size)
                        
                        # Extract the zoom region
                        zoom_img = combined_img[top:bottom, left:right, :]
                        
                        # Display the zoomed image
                        ax_image.imshow(zoom_img)
                else:
                    # Display original image without segmentation
                    ax_image.imshow(original_img)
                
                # Add image information
                ax_image.set_title(f"$\\tau = {current_time:.2f}$", fontsize=x_label_fontsize)
                ax_image.axis('off')
        except Exception as e:
            print(f"Error processing image {image_file_path}: {e}")
            ax_image.text(0.5, 0.5, "Image\nprocessing\nerror", 
                        ha='center', va='center', 
                        transform=ax_image.transAxes)
            ax_image.axis('off')
        
        # Right subplot for plot
        ax_plot = plt.subplot(gs[1])

        # Plot each y-column on the main axis
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
                ax_plot.plot(df[x_column], df[y_column], 
                        marker=marker_style, markersize=marker_size, 
                        linestyle=line_style, linewidth=line_width,
                        color=line_color, markerfacecolor=marker_color, 
                        markeredgecolor='black', label=legend_label)
            else:
                ax_plot.scatter(df[x_column], df[y_column], 
                          s=marker_size**2, marker=marker_style,
                          color=marker_color, edgecolors='black', 
                          label=legend_label)

        # Add A11 data if requested
        if include_A11_data:
            if A11_data is not None and A11_y_column in A11_data:
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
                        ax_plot.plot(x_values, y_interp,
                                marker=A11_marker_style, markersize=A11_marker_size,
                                linestyle=A11_line_style, linewidth=A11_line_width,
                                color=A11_line_color, label=A11_label)
                    else:
                        # If lengths match, use main data x values directly
                        ax_plot.plot(x_values, a11_df[A11_y_column].values * A11_y_scale_factor,  # Apply scale factor
                                marker=A11_marker_style, markersize=A11_marker_size,
                                linestyle=A11_line_style, linewidth=A11_line_width,
                                color=A11_line_color, label=A11_label)
                elif a11_x_col in a11_df.columns and A11_y_column in a11_df.columns:
                    # Standard case - use A11's own x column
                    ax_plot.plot(a11_df[a11_x_col], a11_df[A11_y_column] * A11_y_scale_factor,  # Apply scale factor
                            marker=A11_marker_style, markersize=A11_marker_size,
                            linestyle=A11_line_style, linewidth=A11_line_width,
                            color=A11_line_color, label=A11_label)
                    if Plot_log_level >= 1:
                        print(f"Added A11 data to plot: {A11_y_column} vs {a11_x_col}")
                else:
                    print(f"A11 data missing required columns: {a11_x_col} or {A11_y_column}")
            else:
                print(f"A11 data for {A11_y_column} not found")
                
        # Add linear fits if requested
        if fit_enable and y_fit_list and x_fit_list and x_range_fit_list:
            # Check if the lists have compatible lengths
            if len(y_fit_list) != len(x_fit_list) or len(y_fit_list) != len(x_range_fit_list):
                print("Error: y_fit_list, x_fit_list, and x_range_fit_list must have the same length")
            else:
                # Extend fit style lists if needed
                num_fits = len(y_fit_list)
                fit_line_colors = fit_line_colors * (num_fits // len(fit_line_colors) + 1)
                fit_line_widths = fit_line_widths * (num_fits // len(fit_line_widths) + 1)
                fit_line_styles = fit_line_styles * (num_fits // len(fit_line_styles) + 1)
                
                # Use empty fit_labels as default if not provided
                if not fit_labels:
                    fit_labels = [""] * num_fits
                else:
                    fit_labels = fit_labels * (num_fits // len(fit_labels) + 1)
                
                # Perform linear fits for each y vs x pair
                for i, (y_fit, x_fit, x_range) in enumerate(zip(y_fit_list, x_fit_list, x_range_fit_list)):
                    # Check if columns exist in DataFrame
                    if y_fit not in df.columns:
                        print(f"Fit column '{y_fit}' not found in DataFrame. Skipping fit.")
                        continue
                    if x_fit not in df.columns:
                        print(f"Fit column '{x_fit}' not found in DataFrame. Skipping fit.")
                        continue
                    
                    # Filter data for fitting within the specified x range
                    x_min, x_max = x_range
                    fit_df = df[(df[x_fit] >= x_min) & (df[x_fit] <= x_max)]
                    
                    if len(fit_df) < 2:
                        print(f"Not enough points in range [{x_min}, {x_max}] for fitting. Skipping fit.")
                        continue
                    
                    # Get x and y data for fitting
                    x_data = fit_df[x_fit].values
                    y_data = fit_df[y_fit].values
                    
                    # Handle non-numeric data
                    try:
                        # Convert to numeric and handle NaN values
                        x_data = pd.to_numeric(x_data, errors='coerce')
                        y_data = pd.to_numeric(y_data, errors='coerce')
                        
                        # Create a mask for valid (non-NaN) values in both arrays
                        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
                        
                        # Filter data to keep only valid values
                        x_valid = x_data[valid_mask]
                        y_valid = y_data[valid_mask]
                        
                        if len(x_valid) < 2:
                            print(f"Not enough valid numeric points for fitting after filtering NaN values. Skipping fit.")
                            continue
                        
                        # Perform linear fit with valid data
                        slope, intercept = np.polyfit(x_valid, y_valid, 1)
                        
                        # Generate fit line data
                        x_fit_line = np.array([x_min, x_max])
                        y_fit_line = slope * x_fit_line + intercept
                        
                        # Get styling for this fit
                        fit_line_color = fit_line_colors[i % len(fit_line_colors)]
                        fit_line_width = fit_line_widths[i % len(fit_line_widths)]
                        fit_line_style = fit_line_styles[i % len(fit_line_styles)]
                        
                        # Create label for the fit
                        if show_fit_equation:
                            if slope >= 0:
                                equation = f"${y_fit} = {slope:.4f} {x_fit} + {intercept:.4f}$"
                            else:
                                equation = f"${y_fit} = {slope:.4f} {x_fit} {intercept:.4f}$"
                            
                            fit_label = f"{fit_labels[i]} {equation}" if fit_labels[i] else equation
                        else:
                            fit_label = f"{fit_labels[i]}" if fit_labels[i] else f"Fit: {y_fit} vs {x_fit}"
                        
                        # Plot the fit line
                        ax_plot.plot(x_fit_line, y_fit_line, 
                                 color=fit_line_color, 
                                 linewidth=fit_line_width,
                                 linestyle=fit_line_style,
                                 label=fit_label if fit_equation_position == 'legend' else "")
                        
                        # Add fit equation as text annotation if requested
                        if show_fit_equation and fit_equation_position == 'plot':
                            # Place equation at 10% from the left and 90% from the bottom of the plot
                            ax_plot.annotate(equation, 
                                        xy=(0.1, 0.9 - 0.05 * i), 
                                        xycoords='axes fraction',
                                        fontsize=legend_fontsize-2,
                                        color=fit_line_color)
                    
                    except Exception as e:
                        print(f"Error during fitting {y_fit} vs {x_fit}: {str(e)}")
        
        # Add vertical line at current x position
        ax_plot.axvline(x=current_x_value, color='black', linestyle='--', linewidth=2)
        
        # Set labels
        ax_plot.set_xlabel(x_label if x_label else x_column, fontsize=x_label_fontsize)
        
        # For y-label, use the provided y_label or a generic label since we have multiple columns
        if y_label:
            ax_plot.set_ylabel(y_label, fontsize=y_label_fontsize)
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
                ax_plot.set_ylabel(f"Values ({common_suffix})", fontsize=y_label_fontsize)
            else:
                ax_plot.set_ylabel("Values", fontsize=y_label_fontsize)
        
        # Set tick parameters for inward facing ticks
        ax_plot.tick_params(axis='both', direction='in', which='both', labelsize=tick_label_fontsize)
        
        # Add grid if requested
        if show_grid:
            ax_plot.grid(True, linestyle=grid_style, linewidth=grid_width, 
                    alpha=grid_alpha, color=grid_color)
        
        # Add legend
        if legend_show:
            if legend_title:
                ax_plot.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=False, title=legend_title, 
                        title_fontsize=legend_fontsize+2)
            else:
                ax_plot.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=False)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        base_filename = f"image_{int(image_num):04d}_{x_column}_{current_x_value:.2f}"
        png_path = os.path.join(png_dir, f"{base_filename}.png")
        svg_path = os.path.join(svg_dir, f"{base_filename}.svg")
        
        plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        all_image_paths.append(png_path)
        
        if Plot_log_level >= 1:
            print(f"Saved figures to:\n  {png_path}\n  {svg_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    # Create a video if requested
    if create_video and all_image_paths:
        try:
            # Sort image paths by filename
            all_image_paths.sort()
            
            # Read the first image to get dimensions
            first_img = cv2.imread(all_image_paths[0])
            height, width, layers = first_img.shape
            
            # Create video writer
            video_path = os.path.join(output_dir, "plot_sequence.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, video_fps, (width, height))
            
            # Add each image to the video
            for img_path in all_image_paths:
                img = cv2.imread(img_path)
                video.write(img)
            
            # Release the video writer
            video.release()
            
            if Plot_log_level >= 1:
                print(f"Created video: {video_path}")
                
        except Exception as e:
            print(f"Error creating video: {e}")

    return output_dir


# if __name__ == "__main__":
#     # Example with A11 data using same x-axis as main data
#     A11_data = load_A11_data()
        
#     # Example showing segmentation only on the right half of the image
#     plotter_14_xyyy_image_A11(
#         input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187",
#         x_column="Time_VisIt",
#         y_columns=[
#             "N_cells",
#             "N_cells_CST",
#         ],
#         output_dir_comment="CPaCPTcounts_vs_time",
#         line_colors=['red', 'red'],
#         line_styles=[':', '-'],
#         marker_styles=['', ''],
#         marker_sizes=[6, 7],
#         legend_show=False,
#         legend_labels=[
#             r'Cellpose ',
#             r'Cellpose in Tile',
#         ],
#         legend_title="",
#         legend_loc='upper left',
#         x_label=r'$\tau$',
#         y_label=r'$N_{cells}$',

#         include_A11_data=True,
#         A11_data=A11_data,
#         A11_y_column='N_c',
#         A11_y_scale_factor=1/6.0,  # Scale the A11 data by a factor of 6
#         A11_use_same_x=True,  # Use same x-axis as main data
#         A11_line_style='--',
#         A11_line_color='black',
#         A11_label=r'Manual Count in Tile',
        
#         # Image display parameters
#         show_segmentation=True,
#         color_only_CST_cells=True,
#         show_segmentation_on_both_halves=False,  # Show segmentation overlay only on right half
#         ScaleFactor=1.2,
#         create_video=True,
#         video_fps=5,
#         figsize=(18, 6),
#         dpi=300,
#         show_plot=0
#     )


if __name__ == "__main__":
    # Example with A11 data using same x-axis as main data
    A11_data = load_A11_data()
        
    # Example showing segmentation only on the right half of the image
    plotter_14_xyyy_image_A11(
        input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187",
        x_column="Time_VisIt",
        y_columns=[
            "N_cells_CSTx6",
        ],
        output_dir_comment="CPaManualCounts_vs_time",
        line_colors=['red'],
        line_styles=['-'],
        marker_styles=[''],
        marker_sizes=[7],
        legend_show=False,
        legend_labels=[
            r'Cellpose in Tile',
        ],
        legend_title="",
        legend_loc='upper left',
        x_label=r'$\tau$',
        y_label=r'$N_{cells}$',

        include_A11_data=True,
        A11_data=A11_data,
        A11_y_column='N_c',
        A11_y_scale_factor=1,  # Scale the A11 data by a factor of 6
        A11_use_same_x=True,  # Use same x-axis as main data
        A11_line_style='--',
        A11_line_color='black',
        A11_label=r'Manual Count in Tile',
        
        # Image display parameters
        show_segmentation=True,
        color_only_CST_cells=True,
        show_segmentation_on_both_halves=False,  # Show segmentation overlay only on right half
        ScaleFactor=1.2,
        create_video=True,
        video_fps=5,
        figsize=(18, 6),
        dpi=300,
        show_plot=0
    )
