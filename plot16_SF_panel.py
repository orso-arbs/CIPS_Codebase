import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re
import glob
import cv2
import Format_1 as F_1

# LaTeX settings
plt.rcParams['text.usetex'] = True
LATEX_FONT_SIZE = 16  # Global font size for LaTeX text
plt.rcParams['font.size'] = LATEX_FONT_SIZE
plt.rcParams['font.family'] = 'serif'

def plotter_16_SF_panel(
    input_dir,
    times_list,  # List of Time_VisIt values to include in the panel
    colortable_path=None,  # Path to the colortable image
    output_dir_manual="",
    output_dir_comment="",
    show_plot=0,
    Plot_log_level=1,
    ScaleFactor=1.5,  # Scale factor for zooming in on the spherical flame
    figsize=(16, 16),  # Figure size (width, height) in inches
    panel_rows=4,
    panel_cols=4,
    font_size_tau=14,
    font_size_rsf=12,
    text_y_offset_tau=0.05,
    text_y_offset_rsf=0.12,
    text_color='white',  # Color for annotation text
    text_bbox_facecolor='black',  # Background color for text box
    text_bbox_alpha=0.5,  # Transparency for text box
    text_bbox_pad=2,  # Padding for text box
    text_bbox_edgecolor='none',  # Border color for text box
    text_bbox_linewidth=0,  # Border width for text box
    vertical_line_color='black',  # Color for the vertical line
    vertical_line_style='-',  # Style for the vertical line
    vertical_line_width=1,  # Width of the vertical line
    vertical_line_height_ratio=0.9,  # Percentage of the image height for the vertical line
    colorbar_height=0.08,  # Height of colorbar relative to figure height
    colorbar_outline_color='black',  # Outline color for the colorbar
    colorbar_outline_width=1,  # Width of the colorbar outline
    dpi=300,  # DPI for the figure
    save_png=True,  # Whether to save as PNG
    save_svg=True   # Whether to save as SVG
):
    """
    Creates a panel of spherical flame images in a grid layout with masks overlay.
    
    Parameters
    ----------
    input_dir : str
        Directory containing the Analysis_A11_final_df.pkl file
    times_list : list
        List of Time_VisIt values for images to display
    colortable_path : str, optional
        Path to the colortable image file
    output_dir_manual : str, optional
        Manual output directory, by default ""
    output_dir_comment : str, optional
        Comment to append to the output directory name, by default ""
    show_plot : int, optional
        Whether to display the plot (1) or not (0), by default 0
    Plot_log_level : int, optional
        Logging level, by default 1
    ScaleFactor : float, optional
        Scale factor for zooming in on the spherical flame, by default 1.5
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (16, 16)
    panel_rows : int, optional
        Number of rows in the panel, by default 4
    panel_cols : int, optional
        Number of columns in the panel, by default 4
    font_size_tau : int, optional
        Font size for tau text, by default 14
    font_size_rsf : int, optional
        Font size for R_SF/delta_T text, by default 12
    text_y_offset_tau : float, optional
        Y offset for tau text, by default 0.05
    text_y_offset_rsf : float, optional
        Y offset for R_SF/delta_T text, by default 0.12
    text_color : str, optional
        Color for annotation text, by default 'white'
    text_bbox_facecolor : str, optional
        Background color for text box, by default 'black'
    text_bbox_alpha : float, optional
        Transparency for text box, by default 0.5
    text_bbox_pad : int, optional
        Padding for text box, by default 2
    text_bbox_edgecolor : str, optional
        Border color for text box, by default 'none'
    text_bbox_linewidth : int, optional
        Border width for text box, by default 0
    vertical_line_color : str, optional
        Color for the vertical line, by default 'black'
    vertical_line_style : str, optional
        Style for the vertical line, by default '-'
    vertical_line_width : int, optional
        Width of the vertical line, by default 1
    vertical_line_height_ratio : float, optional
        Percentage of the image height for the vertical line, by default 0.9
    colorbar_height : float, optional
        Height of colorbar relative to figure height, by default 0.08
    colorbar_outline_color : str, optional
        Outline color for the colorbar, by default 'black'
    colorbar_outline_width : int, optional
        Width of the colorbar outline, by default 1
    dpi : int, optional
        DPI for the figure, by default 300
    save_png : bool, optional
        Whether to save the figure as PNG, by default True
    save_svg : bool, optional
        Whether to save the figure as SVG, by default True
        
    Returns
    -------
    str
        Path to the output directory
    """
    # Create output directory
    output_dir = F_1.F_out_dir(input_dir=input_dir, script_path=__file__, output_dir_comment=output_dir_comment, output_dir_manual=output_dir_manual)
    
    if Plot_log_level >= 1:
        print(f"plotter_16_SF_panel: Output directory: {output_dir}")

    # Find the pickle file with the DataFrame
    pandas_wildcard_str = os.path.join(input_dir, "*.pkl")
    pkl_files = glob.glob(pandas_wildcard_str)
    
    if not pkl_files:
        print(f"No PKL files found in {input_dir}")
        return output_dir
    
    # Prioritize files with "Analysis_A11_final_df" in their name
    df_path_candidates = [f for f in pkl_files if "Analysis_A11_final_df" in f]
    if df_path_candidates:
        df_path = df_path_candidates[0]
    elif pkl_files:  # If no specific file found, take the first pkl file
        df_path = pkl_files[0]
    else:
        print(f"No suitable PKL file found in {input_dir}")
        return output_dir

    df = pd.read_pickle(df_path)
    
    if Plot_log_level >= 1:
        print(f"Loaded DataFrame from {df_path}")
    
    # Check if we have enough images for the panel
    if len(times_list) > panel_rows * panel_cols:
        print(f"Warning: More times provided ({len(times_list)}) than panel slots ({panel_rows * panel_cols}). Only first {panel_rows * panel_cols} will be used.")
        times_list = times_list[:panel_rows * panel_cols]
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Create the main grid for images
    gs = gridspec.GridSpec(panel_rows + 1, panel_cols, height_ratios=[1]*panel_rows + [colorbar_height])
    
    # Process each time value
    for idx, target_time in enumerate(times_list):
        if idx >= panel_rows * panel_cols:
            break
            
        # Find the closest time in the DataFrame
        closest_idx = (df['Time_VisIt'] - target_time).abs().idxmin()
        row = df.loc[closest_idx]
        
        if Plot_log_level >= 1:
            print(f"Processing image {idx+1}/{len(times_list)}: Target time {target_time}, closest time {row['Time_VisIt']}")
        
        try:
            # Get image info
            image_num = row['image_number']
            image_file_path = row['image_file_path']
            mask_from_df = row['masks']  # Get mask from DataFrame
            D_SF_px = row['D_SF_px']
            current_time = row['Time_VisIt']
            
            # Calculate R_SF_nonDim if available
            if 'R_SF_nonDim' in row:
                r_sf_nondim = row['R_SF_nonDim']
            else:
                # Calculate from D_SF_nonDim if available
                if 'D_SF_nonDim' in row:
                    r_sf_nondim = row['D_SF_nonDim'] / 2
                else:
                    r_sf_nondim = None
            
            # Calculate subplot position
            row_idx = idx // panel_cols
            col_idx = idx % panel_cols
            
            # Create subplot for this image
            ax = plt.subplot(gs[row_idx, col_idx])
            
            # Load and process image
            original_img = cv2.imread(image_file_path)
            if original_img is None:
                if Plot_log_level >= 1:
                    print(f"  Error: Could not read image file: {image_file_path}")
                raise IOError(f"Could not read image file: {image_file_path}")
                
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # Use the mask from the DataFrame
            mask = mask_from_df
            if mask is None:
                if Plot_log_level >= 1:
                    print(f"  Warning: Mask not found in DataFrame for image {image_num}")
            elif not isinstance(mask, np.ndarray):
                if Plot_log_level >= 1:
                    print(f"  Warning: Mask for image {image_num} is not a NumPy array. Type: {type(mask)}")
                mask = None
            
            # Get image dimensions
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
            if mask is not None:
                # Create colored mask for overlay
                color_mask = np.zeros_like(original_img)
                
                # Define a list of 10 distinct colors
                # Using common Matplotlib colors for variety
                distinct_colors = [
                    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 165, 0), (128, 0, 128),
                    (165, 42, 42), (255, 192, 203)
                ]

                # Get unique mask values greater than 0
                unique_mask_values = np.unique(mask[mask > 0])

                for mask_value in unique_mask_values:
                    color_index = int(mask_value) % 10  # Ensure mask_value is int for modulo
                    selected_color = distinct_colors[color_index]
                    color_mask[mask == mask_value] = selected_color
                    
                # Apply overlay to the right half with transparency
                alpha = 0.5  # Transparency factor
                mask_region = mask > 0
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
            
            # Display the zoomed image
            ax.imshow(zoom_img)
            ax.axis('off')
            
            # Add a vertical line at the center to separate original and segmented views, but only for a portion of the image height
            center_x = (right-left)//2
            height = bottom-top
            y_start = (1.0 - vertical_line_height_ratio) / 2 * height
            y_end = (1.0 - (1.0 - vertical_line_height_ratio) / 2) * height
            ax.plot([center_x, center_x], [y_start, y_end], 
                   color=vertical_line_color, linestyle=vertical_line_style, linewidth=vertical_line_width)
            
            # Add text annotations for Time_VisIt and R_SF_nonDim
            
            ax.text(0.5, text_y_offset_tau, f"$\\tau = {current_time:.2f}$", 
                    ha='center', va='top', transform=ax.transAxes, 
                    fontsize=font_size_tau, color=text_color,
                    bbox=dict(facecolor=text_bbox_facecolor, alpha=text_bbox_alpha, 
                             pad=text_bbox_pad, edgecolor=text_bbox_edgecolor, 
                             linewidth=text_bbox_linewidth))
            
            if r_sf_nondim is not None and r_sf_nondim != "":
                ax.text(0.5, text_y_offset_rsf, f"$R_{{SF}}/\\delta_T = {r_sf_nondim:.2f}$", 
                        ha='center', va='top', transform=ax.transAxes, 
                        fontsize=font_size_rsf, color=text_color,
                        bbox=dict(facecolor=text_bbox_facecolor, alpha=text_bbox_alpha, 
                                 pad=text_bbox_pad, edgecolor=text_bbox_edgecolor, 
                                 linewidth=text_bbox_linewidth))
            

        except Exception as e:
            if Plot_log_level >= 1:
                print(f"  Error processing image for time {target_time}: {e}")
            ax = plt.subplot(gs[row_idx, col_idx])
            ax.text(0.5, 0.5, f"Error loading\nimage for\n$\\tau = {target_time:.2f}$",
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # Add colortable at the bottom spanning the full width
    if colortable_path and os.path.exists(colortable_path):
        try:
            colorbar_img = cv2.imread(colortable_path)
            if colorbar_img is not None:
                colorbar_img = cv2.cvtColor(colorbar_img, cv2.COLOR_BGR2RGB)
                
                # Create subplot for colorbar across all columns
                colorbar_ax = plt.subplot(gs[-1, :])
                
                # Rotate the colorbar if it's vertical (height > width)
                h, w = colorbar_img.shape[:2]
                if h > w:
                    # Rotate the colorbar 90 degrees counter-clockwise
                    colorbar_img = cv2.rotate(colorbar_img, cv2.ROTATE_90_CLOCKWISE)
                
                # Display the colorbar
                colorbar_ax.imshow(colorbar_img)
                
                # Add a black outline around the colorbar
                colorbar_ax.spines['top'].set_visible(True)
                colorbar_ax.spines['bottom'].set_visible(True)
                colorbar_ax.spines['left'].set_visible(True)
                colorbar_ax.spines['right'].set_visible(True)
                
                for spine in colorbar_ax.spines.values():
                    spine.set_color(colorbar_outline_color)
                    spine.set_linewidth(colorbar_outline_width)
                
                colorbar_ax.axis('on')  # Turn on axis to show outline
                colorbar_ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)  # Hide ticks
                
                if Plot_log_level >= 1:
                    print(f"Added colortable from {colortable_path}")
            else:
                if Plot_log_level >= 1:
                    print(f"Could not read colortable image: {colortable_path}")
        except Exception as e:
            if Plot_log_level >= 1:
                print(f"Error adding colortable: {e}")
    elif Plot_log_level >= 1:
        print("Colortable path not provided or file does not exist")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    base_filename = os.path.join(output_dir, "SF_panel_plot")
    
    if save_png:
        png_path = f"{base_filename}.png"
        plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
        if Plot_log_level >= 1:
            print(f"Saved PNG figure: {png_path}")
    
    if save_svg:
        svg_path = f"{base_filename}.svg"
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        if Plot_log_level >= 1:
            print(f"Saved SVG figure: {svg_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return output_dir

if __name__ == "__main__":
    # Example usage
    plotter_16_SF_panel(
        input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187",
        times_list=[0.05, 0.50, 0.95, 1.40, 1.85, 2.30, 2.75, 3.20, 3.65, 4.10, 4.55, 5.00, 5.45, 5.90, 6.35, 6.80],
        colortable_path=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\colortables\PointWise_colorbar_unlabeled - no alpha border.png",
        show_plot=0,
        ScaleFactor=1.2,
        figsize=(20, 20),
        panel_rows=4,
        panel_cols=4,
        font_size_tau=25,
        font_size_rsf=25,
        text_y_offset_tau=-0.1,
        text_y_offset_rsf=0.00,
        text_color='black',
        text_bbox_facecolor='white',
        text_bbox_alpha=0.0,
        text_bbox_pad=2,
        text_bbox_edgecolor='none',
        text_bbox_linewidth=0,
        colorbar_height=0.2,
        colorbar_outline_color='black',
        colorbar_outline_width=2,
        dpi=300,
        save_png=True,
        save_svg=True,
        Plot_log_level=1,
        vertical_line_color='black',
        vertical_line_style='-',
        vertical_line_width=1,
        vertical_line_height_ratio=0.9
    )
