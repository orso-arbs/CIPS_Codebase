import os
import sys
import pandas as pd
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import re
import glob
import cv2
import Format_1 as F_1

# LaTeX settings
plt.rcParams['text.usetex'] = True
LATEX_FONT_SIZE = 16  # Global font size for LaTeX text
plt.rcParams['font.size'] = LATEX_FONT_SIZE
plt.rcParams['font.family'] = 'serif'

def plotter_11_colortables_panel(
    DataFrames_dir,
    output_dir_manual="",
    output_dir_comment="",
    text_list=None,  # Optional list of text to show on each x,y plot
    show_images=True,  # Whether to show flame images above x,y plots
    image_num_to_show=50,  # Which image number to show from each DataFrame
    show_plot=0,
    Plot_log_level=1,
    colorbar_width=0.7,       # Width of colorbar relative to subplot width
    colorbar_height=0.7,      # Height of colorbar relative to subplot height
    left_colorbar_x_offset=0.1,  # X offset for left column colorbars
    right_colorbar_x_offset=0.1, # X offset for right column colorbars
    image_height_ratio=0.35,  # Height ratio of flame image to x,y plot
    image_x_offset=0.2,       # X position offset for flame images
    image_y_offset=0.05,      # Y position offset from top for flame images
    image_h_pos=0.5,          # Horizontal position within subplot (0-1)
    image_v_pos=0.8,          # Vertical position within subplot (0-1)
    ScaleFactor=1.5,          # Scale factor for zooming in on the spherical flame
    figsize=(20, 24),         # Figure size (width, height) in inches
    side_L_width=1,
    side_R_width=1.5,
    center_L_width=3,
    center_R_width=3,
    colortable_scale=0.8,     # Scale factor for colortable images (0-1)
    left_border_width=50,     # Width of right white border for left colortables (in pixels)
    right_border_width=50,    # Width of left white border for right colortables (in pixels)
    FontSizeFactor_Legends=1.0,  # Factor to adjust font size for legends
    FontSizeFactor_Axis=1.0,     # Factor to adjust font size for axes labels
    dpi=100,                  # DPI for the figure
    save_fig=True,             # Whether to save the figure
    image_positions=None,     # List of (x,y) positions for flame images in figure coordinates
    split_legends=False,        # Whether to split legends between left and right columns
):
    """
    Creates a 4x4 grid panel comparing results from 8 different CIPS pipe runs.
    Each row shows 2 sets of results with colortables and x,y plots arranged in columns.
    
    Parameters
    ----------
    DataFrames_dir : list
        List of directories containing the DataFrame .pkl files (8 directories)
    output_dir_manual : str, optional
        Manual output directory, by default ""
    output_dir_comment : str, optional
        Comment to append to the output directory name, by default ""
    text_list : list, optional
        List of text labels to show on each x,y plot (8 items), by default None
    show_images : bool, optional
        Whether to show flame images above x,y plots, by default True
    image_num_to_show : int, optional
        Which image number to show from each DataFrame, by default 50
    show_plot : int, optional
        Whether to display the plot (1) or not (0), by default 0
    Plot_log_level : int, optional
        Logging level, by default 1
    colorbar_width : float, optional
        Width of colorbar relative to subplot width, by default 0.8
    colorbar_height : float, optional
        Height of colorbar relative to subplot height, by default 0.7
    left_colorbar_x_offset : float, optional
        X offset for left column colorbars, by default 0.1
    right_colorbar_x_offset : float, optional
        X offset for right column colorbars, by default 0.1
    image_height_ratio : float, optional
        Height ratio of flame image to x,y plot, by default 0.35
    image_x_offset : float, optional
        X position offset for flame images, by default 0.2
    image_y_offset : float, optional
        Y position offset from top for flame images, by default 0.05
    image_h_pos : float, optional
        Horizontal position of images within subplot (0 = left, 0.5 = centered, 1 = right), by default 0.5
    image_v_pos : float, optional
        Vertical position of images within subplot (0 = bottom, 0.5 = centered, 1 = top), by default 0.8
    ScaleFactor : float, optional
        Scale factor for zooming in on the spherical flame, by default 1.5
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (20, 24)
    center_width : int, optional
        Width of center columns relative to side columns, by default 3
    side_width : int, optional
        Width of side columns with colorbars, by default 1
    colortable_scale : float, optional
        Scale factor for colortable images (0-1), by default 0.8
    left_border_width : int, optional
        Width of right white border for left colortables (in pixels), by default 50
    right_border_width : int, optional
        Width of left white border for right colortables (in pixels), by default 50
    FontSizeFactor_Legends : float, optional
        Factor to adjust font size for legends, by default 1.4
    FontSizeFactor_Axis : float, optional
        Factor to adjust font size for axes labels, by default 1.0
    dpi : int, optional
        DPI for the figure, by default 100
    save_fig : bool, optional
        Whether to save the figure, by default True
    image_positions : list of tuples, optional
        List of (x,y) positions for flame images in figure coordinates. 
        Should contain 8 tuples for the 8 plots, by default None
    split_legends : bool, optional
        If True, split legends between left and right columns
        - Left column shows diameter-related legends
        - Right column shows cell count and area coverage legends
        By default False
        
    Returns
    -------
    str
        Path to the output directory
    """
    # Validate input arguments
    if len(DataFrames_dir) < 8:
        print(f"Warning: Expected 8 DataFrame directories, but got {len(DataFrames_dir)}. Some plots may be empty.")
        # Pad with None values to ensure we have 8 elements
        DataFrames_dir = DataFrames_dir + [None] * (8 - len(DataFrames_dir))
    elif len(DataFrames_dir) > 8:
        print(f"Warning: More than 8 DataFrame directories provided. Only the first 8 will be used.")
        DataFrames_dir = DataFrames_dir[:8]
        
    if text_list and len(text_list) < 8:
        print(f"Warning: Text list has fewer than 8 items. Padding with empty strings.")
        text_list = text_list + [""] * (8 - len(text_list))
    elif text_list and len(text_list) > 8:
        print(f"Warning: Text list has more than 8 items. Only the first 8 will be used.")
        text_list = text_list[:8]
    elif not text_list:
        text_list = [""] * 8  # Empty text for all plots if not provided
    
    # Create output directory
    # Use the first valid DataFrame dir for output directory creation
    valid_input_dir = next((dir_path for dir_path in DataFrames_dir if dir_path), DataFrames_dir[0])
    output_dir = F_1.F_out_dir(input_dir=valid_input_dir, script_path=__file__, 
                             output_dir_comment=output_dir_comment, 
                             output_dir_manual=output_dir_manual)
    
    if Plot_log_level >= 1:
        print(f"plotter_11_colortables_panel: Output directory: {output_dir}")
    
    # Find and load all DataFrames
    dataframes = []
    colortable_dirs = []
    
    for df_dir in DataFrames_dir:
        if not df_dir:
            dataframes.append(None)
            colortable_dirs.append(None)
            continue
            
        # Find PKL files in the directory
        pandas_wildcard_str = os.path.join(df_dir, "*.pkl")
        pkl_files = glob.glob(pandas_wildcard_str)
        
        if not pkl_files:
            print(f"No PKL files found in {df_dir}")
            dataframes.append(None)
            colortable_dirs.append(None)
            continue
        
        # Prioritize files with "DataFrame" in their name
        df_path_candidates = [f for f in pkl_files if "DataFrame" in f]
        if df_path_candidates:
            df_path = df_path_candidates[0]
        else:
            df_path = pkl_files[0]
        
        # Load the DataFrame
        try:
            df = pd.read_pickle(df_path)
            dataframes.append(df)
            if Plot_log_level >= 1:
                print(f"Loaded DataFrame from {df_path}")
                
            # Find the colortable directory (two levels up from input_dir)
            parent_dir = os.path.dirname(df_dir)
            grandparent_dir = os.path.dirname(parent_dir)
            potential_colortable_dir = os.path.join(grandparent_dir, "colortables")
            
            if os.path.exists(potential_colortable_dir) and os.path.isdir(potential_colortable_dir):
                colortable_dirs.append(potential_colortable_dir)
                if Plot_log_level >= 2:
                    print(f"Found colortables directory at: {potential_colortable_dir}")
            else:
                if Plot_log_level >= 1:
                    print(f"Colortables directory not found at expected location: {potential_colortable_dir}")
                colortable_dirs.append(None)
                
        except Exception as e:
            print(f"Error loading DataFrame from {df_path}: {e}")
            dataframes.append(None)
            colortable_dirs.append(None)
    
    # Create the figure and grid
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Define width ratios for the 4 columns: left colortable, left xy plot, right xy plot, right colortable
    width_ratios = [side_L_width, center_L_width, center_R_width, side_R_width]
    
    # Create 4x4 grid with the specified width ratios
    gs = gridspec.GridSpec(4, 4, width_ratios=width_ratios, height_ratios=[1, 1, 1, 1])
    gs.update(hspace=0.0, wspace=0.0)  # No spacing between subplots
    
    # Create a second grid for the flame images with the same dimensions
    gs_images = gridspec.GridSpec(4, 4, width_ratios=width_ratios, height_ratios=[1, 1, 1, 1])
    gs_images.update(hspace=0.0, wspace=0.0)
    
    # Calculate max values across all DataFrames to use consistent y-axis scales
    max_diameter_mean = 0
    max_diameter_median = 0
    max_diameter_training = 0
    max_diameter_estimate = 0
    max_D_SF_px_scaled = 0
    max_N_cells = 0
    min_N_cells = float('inf')
    max_area_coverage = 0
    min_area_coverage = 1.0
    min_time = float('inf')
    max_time = 0
    
    for df in dataframes:
        if df is not None:
            max_diameter_mean = max(max_diameter_mean, df['diameter_mean_px'].max())
            if 'diameter_median_px' in df.columns:
                max_diameter_median = max(max_diameter_median, df['diameter_median_px'].max())
            max_diameter_training = max(max_diameter_training, df['diameter_training_px'].max())
            max_diameter_estimate = max(max_diameter_estimate, df['diameter_estimate_used_px'].max())
            S2 = 1e-1  # Scale factor as in the original plot
            max_D_SF_px_scaled = max(max_D_SF_px_scaled, df['D_SF_px'].max() * S2)
            max_N_cells = max(max_N_cells, df['N_cells'].max())
            min_N_cells = min(min_N_cells, df['N_cells'].min())
            if 'Ar_px2_CP_maskperSF' in df.columns:
                max_area_coverage = max(max_area_coverage, df['Ar_px2_CP_maskperSF'].max())
                min_area_coverage = min(min_area_coverage, df['Ar_px2_CP_maskperSF'].min())
            min_time = min(min_time, df['Time_VisIt'].min())
            max_time = max(max_time, df['Time_VisIt'].max())
    
    # Adjust limits for better visualization
    max_diameter = max(max_diameter_mean, max_diameter_median, max_diameter_training, 
                     max_diameter_estimate, max_D_SF_px_scaled) * 1.05
    max_N_cells = max_N_cells * 1.05
    
    # Load all colorbar images before plotting, resize them, and add white borders
    colorbar_images = []
    for i, colortable_dir in enumerate(colortable_dirs):
        if colortable_dir:
            labeled_path = os.path.join(colortable_dir, "PointWise_colorbar_labeled.png")
            unlabeled_path = os.path.join(colortable_dir, "PointWise_colorbar_unlabeled.png")
            
            img_path = labeled_path if os.path.exists(labeled_path) else unlabeled_path
            
            if os.path.exists(img_path):
                orig_img = Image.open(img_path)
                
                # Scale the image size according to colortable_scale
                if colortable_scale < 1.0:
                    # Create a new white background image first
                    new_width = int(orig_img.width * colortable_scale)
                    new_height = int(orig_img.height * colortable_scale)
                    
                    # Create a white background image
                    img = Image.new('RGB', (new_width, new_height), (255, 255, 255))
                    
                    # Resize the original image with high-quality resampling
                    resized_orig = orig_img.resize((new_width, new_height), Image.LANCZOS)
                    
                    # Paste the resized image onto the white background
                    img.paste(resized_orig, (0, 0))
                else:
                    img = orig_img
                
                # Determine if this is for left or right side (odd indices are right side)
                is_left_side = (i % 2 == 0)
                
                # Create a new image with white border on the appropriate side
                if is_left_side:
                    # For left colortables, add white border on the right
                    new_img = Image.new('RGB', (img.width + left_border_width, img.height), (255, 255, 255))
                    new_img.paste(img, (0, 0))
                    if Plot_log_level >= 2:
                        print(f"Added {left_border_width}px right border to left colortable {i}")
                else:
                    # For right colortables, add white border on the left
                    new_img = Image.new('RGB', (img.width + right_border_width, img.height), (255, 255, 255))
                    new_img.paste(img, (right_border_width, 0))
                    if Plot_log_level >= 2:
                        print(f"Added {right_border_width}px left border to right colortable {i}")
                
                colorbar_images.append(new_img)
            else:
                colorbar_images.append(None)
                if Plot_log_level >= 1:
                    print(f"No colorbar image found for directory {colortable_dir}")
        else:
            colorbar_images.append(None)
    
    # Create all plots
    xy_plot_axes = []  # Store xy plot axes for sharing references
    
    # Process all plots first (without images)
    for row in range(4):
        # Each row has 2 sets of colortable + xy plot
        for col_set in range(2):
            df_index = row * 2 + col_set  # Index into the dataframes list
            
            # Base column indices for this set
            if col_set == 0:
                colorbar_col = 0
                xy_plot_col = 1
            else:
                colorbar_col = 3
                xy_plot_col = 2
            
            # Get current DataFrame and colorbar image
            df = dataframes[df_index] if df_index < len(dataframes) else None
            colorbar_img = colorbar_images[df_index] if df_index < len(colorbar_images) else None
            
            # Create colortable subplot
            ax_colorbar = plt.subplot(gs[row, colorbar_col])
            
            # Display colorbar image if available
            if colorbar_img is not None:
                # Calculate the position to center the image in the subplot
                ax_colorbar.imshow(np.array(colorbar_img), aspect='auto')
                
                # Get min/max values for colorbar annotation
                min_val_colorbar = None
                max_val_colorbar = None
                
                if df is not None and 'Min_Psuedocolored_variable_SF_VisIt' in df.columns:
                    min_val_colorbar = df['Min_Psuedocolored_variable_SF_VisIt'].mean()
                if df is not None and 'Max_Psuedocolored_variable_SF_VisIt' in df.columns:
                    max_val_colorbar = df['Max_Psuedocolored_variable_SF_VisIt'].mean()
                
                # Add Min/Max annotations if values are available
                if min_val_colorbar is not None and max_val_colorbar is not None:
                    try:
                        max_text = f"max: {float(max_val_colorbar):.1f}"
                        min_text = f"min: {float(min_val_colorbar):.1f}"
                    except (ValueError, TypeError):
                        max_text = f"max: {str(max_val_colorbar)}"
                        min_text = f"min: {str(min_val_colorbar)}"
                    
                    # Position text within the colorbar image rather than outside
                    # For left colorbar (even index), text goes on the left side
                    # For right colorbar (odd index), text goes on the right side
                    
                    is_left_side = (df_index % 2 == 0)
                    h_align = 'left' if is_left_side else 'right'
                    x_pos = 0.15 if is_left_side else 0.85
                    
                    # Display min/max values inside the colorbar
                    ax_colorbar.text(x_pos, 0.95, max_text, 
                                   ha=h_align, va='top', transform=ax_colorbar.transAxes, 
                                   fontsize=LATEX_FONT_SIZE * 0.6, color='red', 
                                   bbox=dict(facecolor='black', alpha=0.0, pad=1))
                    ax_colorbar.text(x_pos, 0.05, min_text, 
                                   ha=h_align, va='bottom', transform=ax_colorbar.transAxes, 
                                   fontsize=LATEX_FONT_SIZE * 0.6, color='red',
                                   bbox=dict(facecolor='black', alpha=0.0, pad=1))
            else:
                ax_colorbar.text(0.5, 0.5, "No colorbar\navailable", 
                               ha='center', va='center', transform=ax_colorbar.transAxes)
                
            ax_colorbar.axis('off')
            
            # Create xy plot subplot
            ax_xy = plt.subplot(gs[row, xy_plot_col])
            xy_plot_axes.append(ax_xy)
            
            # If we have data, create x,y plot
            if df is not None:
                # Add grid
                ax_xy.grid(True, which='both', linestyle='--', alpha=0.5)
                
                # Plot data
                S2 = 1e-1  # Scale factor as in the original plot
                ax_xy.plot(df['Time_VisIt'], df['diameter_mean_px'], label='Cell Mean Diameter [px]', color='green')
                if 'diameter_median_px' in df.columns:
                    ax_xy.plot(df['Time_VisIt'], df['diameter_median_px'], label='Cell Median Diameter [px]', color='darkgreen')
                ax_xy.plot(df['Time_VisIt'], df['diameter_training_px'], label='Cellpose Training Diameter [px]', color='violet')
                ax_xy.plot(df['Time_VisIt'], df['diameter_estimate_used_px'], label='Cellpose Estimate Diameter [px]', color='purple')
                ax_xy.plot(df['Time_VisIt'], df['D_SF_px'] * S2, label=f'Flame Ball Diameter [px] * {S2:.3f}', color='orange')
                
                # Twin x-axis for cell count
                ax_xy_twin = ax_xy.twinx()
                ax_xy_twin.plot(df['Time_VisIt'], df['N_cells'], label='Number of cells', color='red')
                
                # Third y-axis for efficiency
                ax_xy_twin2 = None
                if 'Ar_px2_CP_maskperSF' in df.columns:
                    ax_xy_twin2 = ax_xy.twinx()
                    ax_xy_twin2.spines["right"].set_position(("outward", 60))
                    ax_xy_twin2.plot(df['Time_VisIt'], df['Ar_px2_CP_maskperSF'], 
                                   label=r'SF Area coverage by segmented cells [\%]', 
                                   color='gray')
                
                # Set consistent axes limits
                ax_xy.set_xlim(min_time, max_time)
                ax_xy.set_ylim(0, max_diameter)
                ax_xy_twin.set_ylim(min_N_cells * 0.95, max_N_cells)
                if ax_xy_twin2:
                    ax_xy_twin2.set_ylim(0, 1)
                
                # Only show certain tick labels based on row and column
                ax_xy.tick_params(axis='both', which='both', direction='in')
                ax_xy_twin.tick_params(axis='both', which='both', direction='in')
                if ax_xy_twin2:
                    ax_xy_twin2.tick_params(axis='both', which='both', direction='in')
                
                # X-axis labels and ticks
                if row == 3:  # Bottom row
                    ax_xy.set_xlabel('$\\tau$', fontsize=LATEX_FONT_SIZE)
                    ax_xy.tick_params(labelbottom=True)
                else:
                    ax_xy.tick_params(labelbottom=False)
                
                if row == 0:  # Top row
                    ax_xy.tick_params(labeltop=True)
                    ax_xy.xaxis.set_label_position('top')  # Set label position to top
                    ax_xy.set_xlabel('$\\tau$', fontsize=LATEX_FONT_SIZE)  # Add x-label to top
                else:
                    ax_xy.tick_params(labeltop=False)
                
                # Y-axis labels and ticks
                if col_set == 0:  # Left column of xy plots
                    if xy_plot_col == 1:  # First xy plot column
                        ax_xy.set_ylabel('Diameter [px]', fontsize=LATEX_FONT_SIZE *FontSizeFactor_Axis)
                        ax_xy.tick_params(labelleft=True)
                        ax_xy_twin.tick_params(labelright=False)
                        if ax_xy_twin2:
                            ax_xy_twin2.tick_params(labelright=False)
                else:  # Right column of xy plots
                    if xy_plot_col == 2:  # Second xy plot column
                        ax_xy.tick_params(labelleft=False)
                        ax_xy_twin.set_ylabel('Number of Cells', fontsize=LATEX_FONT_SIZE *FontSizeFactor_Axis)
                        ax_xy_twin.tick_params(labelright=True)
                        if ax_xy_twin2:
                            ax_xy_twin2.set_ylabel(r"SF Area coverage [\%]", fontsize=LATEX_FONT_SIZE *FontSizeFactor_Axis)
                            ax_xy_twin2.tick_params(labelright=True)
                
                # Add legend if it's the first row of plots
                if row == 0:
                    # Get all legend handles and labels
                    lines_xy, labels_xy = ax_xy.get_legend_handles_labels()
                    lines_twin, labels_twin = ax_xy_twin.get_legend_handles_labels()
                    
                    if ax_xy_twin2:
                        lines_twin2, labels_twin2 = ax_xy_twin2.get_legend_handles_labels()
                    else:
                        lines_twin2, labels_twin2 = [], []
                        
                    if split_legends:
                        # Split legends between left and right columns
                        if col_set == 0:  # Left column - show diameter legends only
                            ax_xy.legend(lines_xy, labels_xy, 
                                       loc='upper left', bbox_to_anchor=(0.0, 1.5),
                                       frameon=False, fontsize=LATEX_FONT_SIZE*FontSizeFactor_Legends)
                        else:  # Right column - show cell count and area coverage legends
                            combined_lines = lines_twin + lines_twin2
                            combined_labels = labels_twin + labels_twin2
                            ax_xy.legend(combined_lines, combined_labels, 
                                       loc='upper right', bbox_to_anchor=(1.0, 1.5),
                                       frameon=False, fontsize=LATEX_FONT_SIZE*FontSizeFactor_Legends)
                    else:
                        # Original behavior - show all legends in both columns
                        lines_combined = lines_xy + lines_twin + lines_twin2
                        labels_combined = labels_xy + labels_twin + labels_twin2
                        
                        # Place legend at the top
                        if col_set == 0:  # Left set
                            ax_xy.legend(lines_combined, labels_combined, 
                                       loc='upper left', bbox_to_anchor=(0.0, 1.45),
                                       frameon=False, fontsize=LATEX_FONT_SIZE*FontSizeFactor_Legends)
                        else:  # Right set
                            ax_xy.legend(lines_combined, labels_combined, 
                                       loc='upper right', bbox_to_anchor=(1.0, 1.45),
                                       frameon=False, fontsize=LATEX_FONT_SIZE*FontSizeFactor_Legends)
                
                # Add text label if provided
                if text_list and df_index < len(text_list) and text_list[df_index]:
                    ax_xy.text(0.5, 0.9, text_list[df_index], 
                             ha='center', va='center', transform=ax_xy.transAxes,
                             fontsize=LATEX_FONT_SIZE * 0.9, weight='bold')
    
    # Add all flame images in a second pass using the image grid
    if show_images:
        for row in range(4):
            for col_set in range(2):
                df_index = row * 2 + col_set
                df = dataframes[df_index] if df_index < len(dataframes) else None
                
                if df is not None:
                    try:
                        # Find the row with the specified image number
                        image_row = None
                        closest_image_num = float('inf')
                        closest_row = None
                        
                        for idx, row_data in df.iterrows():
                            if row_data['image_number'] == image_num_to_show:
                                image_row = row_data
                                break
                            # Keep track of closest image number as fallback
                            if abs(row_data['image_number'] - image_num_to_show) < abs(closest_image_num - image_num_to_show):
                                closest_image_num = row_data['image_number']
                                closest_row = row_data
                        
                        # If exact image not found, use closest one
                        if image_row is None and closest_row is not None:
                            image_row = closest_row
                            if Plot_log_level >= 1:
                                print(f"Image number {image_num_to_show} not found. Using closest image number {closest_image_num} instead.")
                        
                        if image_row is not None:
                            # Get image path and data
                            image_file_path = image_row['image_file_path']
                            mask_from_df = image_row['masks']
                            D_SF_px = image_row['D_SF_px']
                            
                            # Create image same as before
                            original_img = cv2.imread(image_file_path)
                            if original_img is not None:
                                # Process the image (same processing code as before)
                                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                                
                                # Get image dimensions - moved up to prevent errors
                                img_height, img_width = original_img.shape[:2]
                                
                                # Create a combined image: Left half original, right half with segmentation overlay
                                combined_img = original_img.copy()
                                
                                # Convert right half to grayscale
                                right_half_width = img_width // 2
                                right_half = original_img[:, right_half_width:, :]
                                gray_right_half = cv2.cvtColor(right_half, cv2.COLOR_RGB2GRAY)
                                gray_right_half = cv2.cvtColor(gray_right_half, cv2.COLOR_GRAY2RGB)
                                combined_img[:, right_half_width:, :] = gray_right_half
                                
                                # Add colored segmentation overlay to the right half if mask exists
                                if mask_from_df is not None:
                                    # Create colored mask for overlay
                                    color_mask = np.zeros_like(original_img)
                                    
                                    # Define distinct colors
                                    distinct_colors = [
                                        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                                        (0, 255, 255), (255, 0, 255), (255, 165, 0), (128, 0, 128),
                                        (165, 42, 42), (255, 192, 203)
                                    ]

                                    # Apply colors based on mask value
                                    unique_mask_values = np.unique(mask_from_df[mask_from_df > 0])
                                    for mask_value in unique_mask_values:
                                        color_index = int(mask_value) % 10
                                        selected_color = distinct_colors[color_index]
                                        color_mask[mask_from_df == mask_value] = selected_color
                                    
                                    # Apply overlay to right half with transparency
                                    alpha = 0.5
                                    mask_region = mask_from_df > 0
                                    mask_region_right = np.zeros_like(mask_region)
                                    mask_region_right[:, right_half_width:] = mask_region[:, right_half_width:]
                                    
                                    combined_img[mask_region_right] = (
                                        alpha * color_mask[mask_region_right] + 
                                        (1 - alpha) * combined_img[mask_region_right]
                                    ).astype(np.uint8)
                                
                                # Calculate zoom region to focus on the spherical flame
                                center_x, center_y = img_width // 2, img_height // 2
                                zoom_half_size = int(D_SF_px * ScaleFactor / 2)
                                
                                # Ensure zoom region is within image bounds
                                left = max(0, center_x - zoom_half_size)
                                right = min(img_width, center_x + zoom_half_size)
                                top = max(0, center_y - zoom_half_size)
                                bottom = min(img_height, center_y + zoom_half_size)
                                
                                # Extract the zoom region
                                zoom_img = combined_img[top:bottom, left:right, :]
                                
                                # Add a vertical line at the center to separate original and segmented views
                                zoom_img_with_line = zoom_img.copy()
                                center_x_zoomed = (right-left)//2
                                cv2.line(zoom_img_with_line, 
                                       (center_x_zoomed, 0), 
                                       (center_x_zoomed, bottom-top), 
                                       (255, 255, 0), 1)
                                
                                # Make all pure white pixels transparent
                                h, w, c = zoom_img_with_line.shape
                                rgba_img = np.zeros((h, w, 4), dtype=np.uint8)
                                rgba_img[:, :, 0:3] = zoom_img_with_line
                                rgba_img[:, :, 3] = 255  # Fully opaque
                                
                                # Find white pixels (255, 255, 255) and make them transparent
                                white_mask = (zoom_img_with_line[:, :, 0] == 255) & \
                                            (zoom_img_with_line[:, :, 1] == 255) & \
                                            (zoom_img_with_line[:, :, 2] == 255)
                                rgba_img[white_mask, 3] = 0  # Make white pixels transparent
                                
                                # Now use the image grid to position the image
                                # Column index based on col_set
                                # For col_set 0, use column 1 (left center)
                                # For col_set 1, use column 2 (right center)
                                img_col = 1 if col_set == 0 else 2
                                
                                # Create a subplot in the image grid
                                ax_image = plt.subplot(gs_images[row, img_col])
                                
                                # Calculate the size for the images as a fraction of subplot size
                                # Position the image in the upper part of the subplot
                                ax_pos = ax_image.get_position()
                                
                                # Fix image scaling: explicitly apply image_height_ratio to control size
                                aspect_ratio = w / h
                                
                                # Scale the image using image_height_ratio as a direct multiplier
                                # Reduce base_height to make images smaller by default
                                base_height = 1.0  # Reduced from 0.1 to make images smaller
                                h_size = base_height * image_height_ratio  # Apply scaling factor
                                w_size = h_size * aspect_ratio  # Maintain aspect ratio
                                
                                # Center horizontally at image_h_pos and vertically at image_v_pos
                                x_pos = image_h_pos - (w_size / 2)  # Center horizontally around image_h_pos
                                y_pos = image_v_pos - (h_size / 2)  # Center vertically around image_v_pos
                                
                                # Set position with proper scaling applied
                                ax_image.set_position([
                                    ax_pos.x0 + ax_pos.width * x_pos,  # X start
                                    ax_pos.y0 + ax_pos.height * y_pos,  # Y start
                                    ax_pos.width * w_size,              # Width
                                    ax_pos.height * h_size              # Height
                                ])
                                
                                # Display the image
                                ax_image.imshow(rgba_img)#, aspect='equal')
                                ax_image.axis('off')
                                ax_image.set_frame_on(False)  # Make sure there's no frame
                                
                                # Remove the border around the image
                                # Make the background of the image plot transparent
                                ax_image.patch.set_alpha(0.0)
                                
                    except Exception as e:
                        print(f"Error adding flame image to plot in row {row}, col_set {col_set}: {e}")
    
    # Adjust layout
    #plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        fig_path = os.path.join(output_dir, "colortable_panel_comparison.png")
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        if Plot_log_level >= 1:
            print(f"Saved figure: {fig_path}")
            
        # Also save as SVG
        svg_path = os.path.join(output_dir, "colortable_panel_comparison.svg")
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        if Plot_log_level >= 1:
            print(f"Saved figure: {svg_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    return output_dir

if __name__ == "__main__":
    # Example usage
    DataFrames_dir = [
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250607_2240236\20250607_2240246\20250612_1429370\20250614_1949188",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250608_0303173\20250608_0303173\20250612_1638247\20250614_2137355",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250609_0028398\20250609_0028408\20250612_1843092\20250614_2257342",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250610_0004544\20250610_0004569\20250612_2023463\20250612_2228583",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250610_0529590\20250610_0529590\20250615_1239319\20250615_1440242",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250610_0646347\20250610_0646347\20250615_1609401\20250615_1727535",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250610_0803025\20250610_0803025\20250615_1734526\20250615_1952036",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250610_0916439\20250610_0916439\20250615_2115060\20250615_2243023",
    ]
    
    text_list = [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        ""
    ]
    
    plotter_11_colortables_panel(
        DataFrames_dir=DataFrames_dir,
        text_list=text_list,
        show_plot=0,
        Plot_log_level=1,
        image_num_to_show=50,
        side_L_width=1,
        side_R_width=1.5,
        center_L_width=3,
        center_R_width=3,
        colortable_scale=1.0,  # Scale colortables to original size
        left_border_width=int(80*1),  # Add 80px white border on right side of left colortables
        right_border_width=int(80*1.6), # Add 80px white border on left side of right colortables
        image_height_ratio=0.55,  # Size multiplier for flame images (1.0 = standard size)
        image_h_pos=0.2,  # Horizontal position (0.1 = near left side)
        image_v_pos=0.75,  # Vertical position (0.8 = near top)
        split_legends=True,  # Split legends between left and right columns
        save_fig=True
    )
