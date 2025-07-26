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

def plotter_14_xTwinyyyy_image(
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
    figsize=(20, 8),
    dpi=300,
    # Image display parameters
    image_width_ratio=0.3,     # Width ratio for image subplot
    plot_width_ratio=0.7,      # Width ratio for property plot subplot
    plot_spacing=0.2,          # Horizontal spacing between plots
    ScaleFactor=1.5,           # Scale factor for zooming in on the spherical flame
    show_segmentation=True,    # Whether to show segmentation overlay
    color_only_CST_cells=False, # Whether to color only cells included in the CST
    show_segmentation_on_both_halves=False, # Whether to show segmentation on both halves of the image
    create_video=False,        # Whether to create a video from the images
    video_fps=5,               # Frames per second for the video
    # Other parameters
    show_plot=0,
    Plot_log_level=1,
    # A11 data parameters
    include_A11_data=True,
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
    Creates an x-y plot with multiple twin y-axes alongside the current image, each with configurable parameters.
    
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
        Figure size (width, height) in inches, by default (20, 8)
    dpi : int, optional
        DPI for the figure, by default 300
    image_width_ratio : float, optional
        Width ratio for image subplot, by default 0.3
    plot_width_ratio : float, optional
        Width ratio for property plot subplot, by default 0.7
    plot_spacing : float, optional
        Horizontal spacing between plots, by default 0.1
    ScaleFactor : float, optional
        Scale factor for zooming in on the spherical flame, by default 1.5
    show_segmentation : bool, optional
        Whether to show segmentation overlay, by default True
    color_only_CST_cells : bool, optional
        Whether to color only cells included in the CST, by default False
    show_segmentation_on_both_halves : bool, optional
        Whether to show segmentation overlay on both halves of the image, by default False (only right half)
    create_video : bool, optional
        Whether to create a video from the images, by default False
    video_fps : int, optional
        Frames per second for the video, by default 5
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
        print(f"plotter_14_xTwinyyyy_image: Output directory: {output_dir}")

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
                
                #ax_image.set_title(f"$\\tau = {current_time:.2f}$", fontsize=x_label_fontsize)
                ax_image.axis('off')
        except Exception as e:
            print(f"Error processing image {image_file_path}: {e}")
            ax_image.text(0.5, 0.5, "Image\nprocessing\nerror", 
                        ha='center', va='center', 
                        transform=ax_image.transAxes)
            ax_image.axis('off')
        
        # Right subplot for the plot
        ax_main = plt.subplot(gs[1])
        
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
                line_width = y_line_widths_list[i][j] if i < len(y_line_widths_list) and j < len(y_line_styles_list[i]) else 1.5
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
        
        # Add vertical line at current value for the selected x-axis variable
        current_x_value = row[x_column]  # Get current value of the selected x-variable
        time_line = ax_main.axvline(current_x_value, color='black', linestyle='--', linewidth=2, 
                                  label=f'{x_column} = {current_x_value:.2f}')
        all_lines.append(time_line)
        all_labels.append(f'{x_column} = {current_x_value:.2f}')
        
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
        
        # Save the figure
        base_filename = f"image_{int(image_num):04d}_time_{current_time:.2f}"
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

# out of CP data
# if __name__ == "__main__":
#     # Example usage with twin axes for different metrics
#     input_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187"
    
#     # Define column groups for each twin axis (three axes now)
#     y_columns_list = [
#         ['R_SF_nonDim'],  # Third axis
#         ['N_cells'],                      # First axis
#         ['d_cell_mean_nonDim'],      # Second axis
#     ]
    
#     # Define better colors using a professional color palette
#     y_colors_list = [
#         ['teal'],     # Green for third axis
#         ['red'],    # Blue for first axis
#         ['green'],    # Vermillion for second axis
#     ]
    
#     # Define labels for each axis with consistent formatting
#     y_labels_list = [
#         r'$R_{SF}/\delta_T$',                     # Label for third axis
#         r'$N_{\mathrm{cells}}$',                  # Label for first axis
#         r'$\overline{d}_c/\delta_T$',             # Label for second axis
#     ]
    
#     # Line styles for differentiation
#     y_line_styles_list = [
#         ['-'],    # Solid line for N_cells
#         ['-'],    # Solid line for mean diameter
#         ['-'],     # Solid line for contour length
#     ]
    
#     # Line widths for better visibility
#     y_line_widths_list = [
#         [2.0],    # Thicker line for N_cells
#         [2.0],    # Thicker line for mean diameter
#         [2.0],     # Thicker line for contour length
#     ]
    
#     # Add markers for data points
#     y_markers_list = [
#         [''],    # Circle markers for N_cells
#         [''],    # Square markers for mean diameter
#         [''],     # Triangle markers for contour length
#     ]
    
#     # Define scale factors for each column
#     y_scale_factors_list = [
#         [1.0],    # No scaling for N_cells
#         [1.0],    # No scaling for mean diameter
#         [1.0],     # No scaling for contour length
#     ]
    
#     # Better spacings for  eachtwin axis (in points)
#     axis_spacings = [0, 70]  # Increased spacings for clearer separation
    
#     output_dir = plotter_14_xTwinyyyy_image(
#         input_dir=input_dir,
#         x_column="Time_VisIt",
#         y_columns_list=y_columns_list,
#         y_colors_list=y_colors_list,
#         y_labels_list=y_labels_list,
#         y_line_styles_list=y_line_styles_list,
#         y_line_widths_list=y_line_widths_list,
#         y_markers_list=y_markers_list,
#         y_marker_sizes_list=[[6], [6], [6]],
#         y_scale_factors_list=y_scale_factors_list,
#         output_dir_comment="professional_style_plots",
#         x_label=r'$\tau$',
#         x_label_fontsize=20,
#         tick_label_fontsize=20,
#         legend_fontsize=20,
#         legend_loc='upper right',
#         legend_title=r'\textbf{Legend}',
#         show_legend=False,
#         axis_spacings=axis_spacings,
#         ScaleFactor=1.2,
#         show_segmentation=True,
#         create_video=True,
#         video_fps=10,
#         figsize=(18, 6),
#         dpi=300,
#         show_grid=True,
#         grid_style='--',
#         grid_width=0.7,
#         grid_color='#CCCCCC',
#         grid_alpha=0.5,
#         show_plot=0
#     )





# if __name__ == "__main__":
#     # Example usage with twin axes for different metrics
#     input_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187"
    
#     # Define column groups for each twin axis (three axes now)
#     y_columns_list = [
#         ['R_SF_nonDim'],  # Third axis
#         ['N_cells_CSTx6'],                      # First axis
#         ['d_cell_SRec_mean_CSTx6_nonDim'],      # Second axis
#     ]
    
#     # Define better colors using a professional color palette
#     y_colors_list = [
#         ['teal'],     # Green for third axis
#         ['red'],    # Blue for first axis
#         ['green'],    # Vermillion for second axis
#     ]
    
#     # Define labels for each axis with consistent formatting
#     y_labels_list = [
#         r'$\dot{R}_{SF}/S_L$', #r'$R_{SF}/\delta_T$',                     # Label for third axis
#         r'$N_{\mathrm{cells}}$',                  # Label for first axis
#         r'$\overline{d}_c/\delta_T$',             # Label for second axis
#     ]
    
#     # Line styles for differentiation
#     y_line_styles_list = [
#         ['-'],    # Solid line for N_cells
#         ['-'],    # Solid line for mean diameter
#         ['-'],     # Solid line for contour length
#     ]
    
#     # Line widths for better visibility
#     y_line_widths_list = [
#         [2.0],    # Thicker line for N_cells
#         [2.0],    # Thicker line for mean diameter
#         [2.0],     # Thicker line for contour length
#     ]
    
#     # Add markers for data points
#     y_markers_list = [
#         [''],    # Circle markers for N_cells
#         [''],    # Square markers for mean diameter
#         [''],     # Triangle markers for contour length
#     ]
    
#     # Define scale factors for each column
#     y_scale_factors_list = [
#         [1.0],    # No scaling for N_cells
#         [1.0],    # No scaling for mean diameter
#         [1.0],     # No scaling for contour length
#     ]
    
#     # Better spacings for each twin axis (in points)
#     axis_spacings = [0, 70]  # Increased spacings for clearer separation
    
#     # Define LLS fit parameters
#     fit_y_variables = []  # Variables to fit
#     fit_x_start_values = [16.0,
#                             30.0, 
#                             38.0, 
#                         ]      # Starting x value for each fit
#     fit_x_stop_values = [30.0, 
#                         38.0,
#                         57.0,
#                         ]       # Ending x value for each fit
#     fit_colors = ['darkgreen',
#                     'darkgreen',
#                     'darkgreen',
#                     ]      # Colors for each fit line
#     fit_line_styles = ['--',
#                        '--',
#                        '--',
#                         ]         # Line styles for each fit line
#     fit_line_widths = [2.5,
#                        2.5,
#                        2.5,
#                         ]          # Line widths for each fit line
    
#     output_dir = plotter_14_xTwinyyyy_image(
#         input_dir=input_dir,
#         image_list=[],
#         omit_image_list=[106],
#         x_column="Time_VisIt",
#         y_columns_list=y_columns_list,
#         y_colors_list=y_colors_list,
#         y_labels_list=y_labels_list,
#         y_line_styles_list=y_line_styles_list,
#         y_line_widths_list=y_line_widths_list,
#         y_markers_list=y_markers_list,
#         y_marker_sizes_list=[[6], [6], [6]],
#         y_scale_factors_list=y_scale_factors_list,
#         # LLS fit parameters
#         fit_y_variables=fit_y_variables,
#         fit_x_start_values=fit_x_start_values,
#         fit_x_stop_values=fit_x_stop_values,
#         fit_colors=fit_colors,
#         fit_line_styles=fit_line_styles,
#         fit_line_widths=fit_line_widths,
#         output_dir_comment="with_fits",
#         x_label=r'$\tau$',
#         x_label_fontsize=20,
#         tick_label_fontsize=20,
#         legend_fontsize=20,
#         legend_loc='upper right',
#         legend_title=r'\textbf{Legend}',
#         show_legend=False,  # Show legend to display fit lines
#         axis_spacings=axis_spacings,
#         ScaleFactor=1.2,
#         color_only_CST_cells=True,
#         show_segmentation=True,
#         show_segmentation_on_both_halves=True,
#         create_video=True,
#         video_fps=10,
#         figsize=(18, 6),
#         dpi=300,
#         show_grid=True,
#         grid_style='--',
#         grid_width=0.7,
#         grid_color='#CCCCCC',
#         grid_alpha=0.5,
#         show_plot=0
#     )


if __name__ == "__main__":
    # Example usage with twin axes for different metrics
    input_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187"
    
    # Define column groups for each twin axis (three axes now)
    y_columns_list = [
        ['R_SF_nonDim'],                        # Third axis
        ['N_cells_CSTx6'],                      # First axis
        ['d_cell_SRec_mean_CSTx6_nonDim'],      # Second axis
    ]
    
    # Define better colors using a professional color palette
    y_colors_list = [
        ['teal'],     # Green for third axis
        ['red'],    # Blue for first axis
        ['green'],    # Vermillion for second axis
    ]
    
    # Define labels for each axis with consistent formatting
    y_labels_list = [
        r'$\dot{R}_{SF}/S_L$', #r'$R_{SF}/\delta_T$',                     # Label for third axis
        r'$N_{\mathrm{cells}}$',                  # Label for first axis
        r'$\overline{d}_c/\delta_T$',             # Label for second axis
    ]
    
    # Line styles for differentiation
    y_line_styles_list = [
        ['-'],    # Solid line for N_cells
        ['-'],    # Solid line for mean diameter
        ['-'],     # Solid line for contour length
    ]
    
    # Line widths for better visibility
    y_line_widths_list = [
        [2.0],    # Thicker line for N_cells
        [2.0],    # Thicker line for mean diameter
        [2.0],     # Thicker line for contour length
    ]
    
    # Add markers for data points
    y_markers_list = [
        [''],    # Circle markers for N_cells
        [''],    # Square markers for mean diameter
        [''],     # Triangle markers for contour length
    ]
    
    # Define scale factors for each column
    y_scale_factors_list = [
        [1.0],    # No scaling for N_cells
        [1.0],    # No scaling for mean diameter
        [1.0],     # No scaling for contour length
    ]
    
    # Better spacings for each twin axis (in points)
    axis_spacings = [0, 70]  # Increased spacings for clearer separation
    
    # Define LLS fit parameters
    fit_y_variables = []  # Variables to fit
    fit_x_start_values = [16.0,
                            30.0, 
                            38.0, 
                        ]      # Starting x value for each fit
    fit_x_stop_values = [30.0, 
                        38.0,
                        57.0,
                        ]       # Ending x value for each fit
    fit_colors = ['darkgreen',
                    'darkgreen',
                    'darkgreen',
                    ]      # Colors for each fit line
    fit_line_styles = ['--',
                       '--',
                       '--',
                        ]         # Line styles for each fit line
    fit_line_widths = [2.5,
                       2.5,
                       2.5,
                        ]          # Line widths for each fit line
    
    output_dir = plotter_14_xTwinyyyy_image(
        input_dir=input_dir,
        image_list=[79],
        omit_image_list=[106],
        x_column="Time_VisIt",
        y_columns_list=y_columns_list,
        y_colors_list=y_colors_list,
        y_labels_list=y_labels_list,
        y_line_styles_list=y_line_styles_list,
        y_line_widths_list=y_line_widths_list,
        y_markers_list=y_markers_list,
        y_marker_sizes_list=[[6], [6], [6]],
        y_scale_factors_list=y_scale_factors_list,
        # LLS fit parameters
        fit_y_variables=fit_y_variables,
        fit_x_start_values=fit_x_start_values,
        fit_x_stop_values=fit_x_stop_values,
        fit_colors=fit_colors,
        fit_line_styles=fit_line_styles,
        fit_line_widths=fit_line_widths,
        output_dir_comment="with_fits",
        x_label=r't/$\tau$',
        x_label_fontsize=25,
        tick_label_fontsize=25,
        legend_fontsize=25,
        legend_loc='upper right',
        legend_title=r'\textbf{Legend}',
        show_legend=False,  # Show legend to display fit lines
        axis_spacings=axis_spacings,
        ScaleFactor=1.2,
        color_only_CST_cells=True,
        show_segmentation=True,
        show_segmentation_on_both_halves=True,
        create_video=True,
        video_fps=10,
        figsize=(18, 6),
        dpi=300,
        show_grid=True,
        grid_style='--',
        grid_width=0.7,
        grid_color='#CCCCCC',
        grid_alpha=0.5,
        show_plot=0
    )