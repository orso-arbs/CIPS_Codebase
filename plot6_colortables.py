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

def plotter_6_colortables(
    input_dir,
    output_dir_manual="",
    output_dir_comment="",
    show_plot=0,
    Plot_log_level=1,
    image_width_ratio=0.5,     # Width ratio for combined image subplot
    plot_width_ratio=0.5,      # Width ratio for property plot subplot
    plot_spacing=0.0,          # Horizontal spacing between plots
    colorbar_width=0.1,       # Width of colorbar relative to subplot width
    colorbar_height=0.6,       # Height of colorbar relative to subplot height (reduced from 1.0)
    colorbar_x_pos=0.1,       # X position of colorbar relative to subplot width
    ScaleFactor=1.5,            # Scale factor for zooming in on the spherical flame
    figsize=(18, 6),            # Figure size (width, height) in inches
    FontSizeFactor_Legends = 1.4,        # Factor to adjust font size for subplots
    FontSizeFactor_Axis = 1.0,           # Factor to adjust font size for axes labels
    Legend_y_offset = 1.3,
    dpi=100,                    # DPI for the figure
    save_fig=True,              # Whether to save the figure
    video=False                 # Whether to create a video of the plots
):
    """
    Creates a plot with three horizontally aligned subplots:
    1. Left: A color bar from the PointWise color table
    2. Middle: A zoomed-in view of the spherical flame and its segmentation
    3. Right: A plot of important properties over image numbers
    
    Parameters
    ----------
    input_dir : str
        Directory containing the input data (same as plot1)
    output_dir_manual : str, optional
        Manual output directory, by default ""
    output_dir_comment : str, optional
        Comment to append to the output directory name, by default ""
    show_plot : int, optional
        Whether to display the plot (1) or not (0), by default 0
    Plot_log_level : int, optional
        Logging level, by default 1
    image_width_ratio : float, optional
        Width ratio for the image subplot, by default 0.45
    plot_width_ratio : float, optional
        Width ratio for the property plot subplot, by default 0.45
    plot_spacing : float, optional
        Horizontal spacing between plots, by default 0.2
    colorbar_width : float, optional
        Width of colorbar relative to subplot width, by default 0.15
    colorbar_height : float, optional
        Height of colorbar relative to subplot height, by default 1.0
    colorbar_x_pos : float, optional
        X position of colorbar relative to subplot width, by default 0.05
    ScaleFactor : float, optional
        Scale factor for zooming in on the spherical flame, by default 1.5
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (18, 6)
    dpi : int, optional
        DPI for the figure, by default 100
    save_fig : bool, optional
        Whether to save the figure, by default True
    video : bool, optional
        Whether to create a video of the plots, by default False
    
    Returns
    -------
    str
        Path to the output directory
    """
    # Create output directory

    output_dir = F_1.F_out_dir(input_dir=input_dir, script_path=__file__, output_dir_comment=output_dir_comment, output_dir_manual=output_dir_manual)
    
    if Plot_log_level >= 1:
        print(f"plotter_6_colortables: Output directory: {output_dir}")
    # print("started") # Can be removed or made conditional if desired, kept for now as a simple start marker

    # Find the PKL and DFs
    pandas_wildcard_str = os.path.join(input_dir, "*.pkl") # Changed from csv to pkl
    pkl_files = glob.glob(pandas_wildcard_str) # Changed from csv_files to pkl_files
    
    if not pkl_files: # Changed from csv_files
        print(f"No PKL files found in {input_dir}") # Changed from CSV
        return output_dir
    
    # Load the main DataFrame with image and segmentation data
    # Prioritize files with "DataFrame" in their name
    df_path_candidates = [f for f in pkl_files if "DataFrame" in f]
    if df_path_candidates:
        df_path = df_path_candidates[0]
    elif pkl_files: # If no "DataFrame" in name, take the first pkl file found
        df_path = pkl_files[0]
    else: # Should not happen due to the check above, but as a safeguard
        print(f"No suitable PKL file found in {input_dir}")
        return output_dir

    df = pd.read_pickle(df_path) # Changed from pd.read_csv
    
    if Plot_log_level >= 1:
        print(f"Loaded DataFrame from {df_path}")
    
    # Find the colortable directory
    # Expected to be exactly two levels up from input_dir and named "colortables"
    colortable_dir = None
    parent_dir = os.path.dirname(input_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    potential_colortable_dir = os.path.join(grandparent_dir, "colortables")

    if os.path.exists(potential_colortable_dir) and os.path.isdir(potential_colortable_dir):
        colortable_dir = potential_colortable_dir
        if Plot_log_level >= 1:
            print(f"Found colortables directory in grandparent_dir at: {colortable_dir}")
    else:
        if Plot_log_level >= 1: # This warning is important
            print(f"Colortables directory not found at expected location: {potential_colortable_dir}.")

    # Load colorbar image once before the loop
    colorbar_image = None
    if colortable_dir:
        labeled_path = os.path.join(colortable_dir, "PointWise_colorbar_labeled.png")
        unlabeled_path = os.path.join(colortable_dir, "PointWise_colorbar_unlabeled.png")
        
        if os.path.exists(labeled_path):
            colorbar_image = Image.open(labeled_path)
            if Plot_log_level >= 2: print(f"Loaded labeled colorbar from {labeled_path}")
        elif os.path.exists(unlabeled_path):
            colorbar_image = Image.open(unlabeled_path)
            if Plot_log_level >= 2: print(f"Loaded unlabeled colorbar from {unlabeled_path}")
        else:
            if Plot_log_level >= 1: print(f"Warning: No colorbar image found in {colortable_dir}.")


    # Process each image in the DataFrame
    for idx, row in df.iterrows():
        if Plot_log_level >= 1:
            print(f"Processing row {idx + 1}/{len(df)}: Image {row['image_number']}")
        try:
            # Get image info
            image_num = row['image_number']
            image_file_path = row['image_file_path']
            mask_from_df = row['masks'] # Get mask from DataFrame
            D_SF_px = row['D_SF_px']
            current_time = df.iloc[idx]['Time_VisIt'] # df.iloc[idx] is correct here for current_time based on overall df

            # Get min and max values for colorbar annotation from the current row
            min_val_colorbar_col = 'Min_Psuedocolored_variable_SF_VisIt'
            max_val_colorbar_col = 'Max_Psuedocolored_variable_SF_VisIt'
            
            if min_val_colorbar_col in row.index and max_val_colorbar_col in row.index: # Check if columns exist in the row's Series index
                min_val_colorbar = row[min_val_colorbar_col]
                max_val_colorbar = row[max_val_colorbar_col]
                if Plot_log_level >= 1: # Changed to level 2 to reduce verbosity for per-row print
                    print(f"  Using Min/Max from current row for colorbar: {min_val_colorbar}, {max_val_colorbar}")
            else:
                min_val_colorbar = 101
                max_val_colorbar = 102
                if Plot_log_level >= 1:
                    print(f"  Warning: Min/Max columns for colorbar not found in current row. Using defaults: {min_val_colorbar}, {max_val_colorbar}")


            if Plot_log_level >= 2: 
                print(f"  Image: {image_num}, File: {image_file_path}, D_SF_px: {D_SF_px}")
            
            # Create figure with 2 subplots
            fig = plt.figure(figsize=figsize, dpi=dpi)
            gs = gridspec.GridSpec(1, 2, width_ratios=[image_width_ratio, plot_width_ratio])
            gs.update(top=0.8, bottom=0.15, wspace=plot_spacing)  # Add horizontal spacing control

            # Combined left subplot for flame and colorbar
            ax_combined = plt.subplot(gs[0])
            
            # Display the flame image first
            if colorbar_image is not None:
                # Calculate vertical position to center the colorbar
                colorbar_y_pos = (1 - colorbar_height) / 2  # This centers the colorbar vertically
                
                # Calculate colorbar position using parameters
                colorbar_pos = [colorbar_x_pos, colorbar_y_pos, 
                              colorbar_width, colorbar_height]
                
                # Create a separate axes for colorbar that overlaps
                ax_colorbar = fig.add_axes(colorbar_pos)
                ax_colorbar.imshow(np.array(colorbar_image))
                #ax_colorbar.set_title("Color Table", fontsize=LATEX_FONT_SIZE)
                ax_colorbar.axis('off')

                # Add Min/Max annotations to the colorbar
                try:
                    max_text = f"max: {float(max_val_colorbar):.1f}" if pd.notna(max_val_colorbar) else f"max: {str(max_val_colorbar)}"
                    min_text = f"min: {float(min_val_colorbar):.1f}" if pd.notna(min_val_colorbar) else f"min: {str(min_val_colorbar)}"
                except (ValueError, TypeError): # Handle cases where conversion to float might fail or value is not numeric
                    max_text = f"max: {str(max_val_colorbar)}"
                    min_text = f"min: {str(min_val_colorbar)}"

                ax_colorbar.text(0.5, 1.02, max_text, 
                                 ha='center', va='bottom', transform=ax_colorbar.transAxes, 
                                 fontsize=LATEX_FONT_SIZE * 0.7, color='black')
                ax_colorbar.text(0.5, -0.02, min_text, 
                                 ha='center', va='top', transform=ax_colorbar.transAxes, 
                                 fontsize=LATEX_FONT_SIZE * 0.7, color='black')
            
            # Display the flame image in the main left subplot
            try:
                original_img = cv2.imread(image_file_path)
                if original_img is None:
                    print(f"  Error: Could not read image file: {image_file_path}") # Keep error
                    raise IOError(f"Could not read image file: {image_file_path}")
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                # Use the mask from the DataFrame
                mask = mask_from_df 
                if mask is None: 
                    if Plot_log_level >= 1: print(f"  Warning: Mask not found in DataFrame for image {image_num}")
                elif not isinstance(mask, np.ndarray): 
                    if Plot_log_level >= 1: print(f"  Warning: Mask for image {image_num} is not a NumPy array. Type: {type(mask)}")
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
                    # (Red, Green, Blue, Yellow, Cyan, Magenta, Orange, Purple, Brown, Pink)
                    distinct_colors = [
                        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                        (0, 255, 255), (255, 0, 255), (255, 165, 0), (128, 0, 128),
                        (165, 42, 42), (255, 192, 203)
                    ]

                    # Get unique mask values greater than 0
                    unique_mask_values = np.unique(mask[mask > 0])

                    for mask_value in unique_mask_values:
                        color_index = int(mask_value) % 10 # Ensure mask_value is int for modulo
                        selected_color = distinct_colors[color_index]
                        color_mask[mask == mask_value] = selected_color
                        
                    if Plot_log_level >= 3: # Made this very verbose log level 3
                        print(f"  Colored mask created for image {image_num}, unique values in mask: {unique_mask_values}")
                    
                    # Apply overlay to the right half with transparency
                    alpha = 0.5  # Transparency factor
                    mask_region = mask > 0
                    mask_region_right = np.zeros_like(mask_region)
                    mask_region_right[:, right_half_width:] = mask_region[:, right_half_width:]
                    
                    combined_img[mask_region_right] = (
                        alpha * color_mask[mask_region_right] + 
                        (1 - alpha) * combined_img[mask_region_right]
                    ).astype(np.uint8)
                    if Plot_log_level >= 2: print(f"Overlay applied to right half for image {image_num}")
                
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
                ax_combined.imshow(zoom_img)
                #ax_combined.set_title(f"Flame \\& Segmentation at $\\tau = {current_time:.2f}$", 
                #                    fontsize=LATEX_FONT_SIZE)
                ax_combined.axis('off')
                
                # Add a vertical line at the center to separate original and segmented views
                ax_combined.axvline(x=(right-left)//2, color='yellow', linestyle='-', linewidth=1)
                
            except Exception as e:
                print(f"  Error processing image content for {image_file_path}: {e}")
                ax_combined.text(0.5, 0.5, "Image\nloading\nerror", 
                               ha='center', va='center', 
                               transform=ax_combined.transAxes)
                ax_combined.axis('off')

            # Right subplot: Properties plot
            ax_1_12 = plt.subplot(gs[1])
            
            ax_1_12_R = ax_1_12.twinx()
            
            # Plot data using Time_VisIt as x-axis
            ax_1_12.plot(df['Time_VisIt'], df['diameter_mean_px'], label='Cell Mean Diameter [px]', color='green')
            #ax_1_12.plot(df['Time_VisIt'], df['diameter_median_px'], label='Cell Median Diameter [px]', color='darkgreen')
            ax_1_12.plot(df['Time_VisIt'], df['diameter_training_px'], label='Cellpose Training Diameter [px]', color='violet')
            ax_1_12.plot(df['Time_VisIt'], df['diameter_estimate_used_px'], label='Cellpose Estimate Diameter [px]', color='purple')
            
            S2 = 1e-1
            ax_1_12.plot(df['Time_VisIt'], df['D_SF_px'] * S2, label=f'Flame Ball Diameter [px] * {S2:.3f}', color='orange')
            ax_1_12_R.plot(df['Time_VisIt'], df['N_cells'], label='Number of cells', color='red')
            
            # Vertical line at current image
            ax_1_12.axvline(current_time, color='black', 
                           #label=f'τ = {current_time:.2f}', 
                           linestyle='dashed', linewidth=2)

            # Third y-axis for efficiency
            ax_1_12_RR = ax_1_12.twinx()
            ax_1_12_RR.spines["right"].set_position(("outward", 60))
            ax_1_12_RR.set_ylabel(r"SF Area coverage by segmented cells [\%]", fontsize=LATEX_FONT_SIZE / FontSizeFactor_Axis)
            ax_1_12_RR.set_ylim(0, 1)
            ax_1_12_RR.plot(df['Time_VisIt'], df['Ar_px2_CP_maskperSF'], 
                           label=r'SF Area coverage by segmented cells [\%]', 
                           color='gray')

            # Set axis limits and labels
            ax_1_12.set_xlim(df['Time_VisIt'].min(), df['Time_VisIt'].max())
            ax_1_12.set_ylim(0, max(df['diameter_mean_px'].max(), 
                                   df['diameter_median_px'].max(), 
                                   df['D_SF_px'].max() * S2) * 1.05)
            ax_1_12_R.set_ylim(df['N_cells'].min(), df['N_cells'].max() * 1.05)

            # Set labels
            ax_1_12.set_xlabel('$\\tau$', fontsize=LATEX_FONT_SIZE)
            ax_1_12.set_ylabel('Diameter [px]', fontsize=LATEX_FONT_SIZE / FontSizeFactor_Axis)
            ax_1_12_R.set_ylabel('Number of Cells', fontsize=LATEX_FONT_SIZE / FontSizeFactor_Axis)

            # Add legends with scaled font size - place above plot
            ax_1_12.legend(loc='upper left', frameon=False, 
                          fontsize=LATEX_FONT_SIZE/FontSizeFactor_Legends,
                          bbox_to_anchor=(0.0, Legend_y_offset))  # Moved up by adding y offset
            
            # Combine the right-side legends and place them above plot
            lines_r, labels_r = ax_1_12_R.get_legend_handles_labels()
            lines_rr, labels_rr = ax_1_12_RR.get_legend_handles_labels()
            ax_1_12_R.legend(lines_r + lines_rr, labels_r + labels_rr, 
                           loc='upper right', frameon=False,
                           bbox_to_anchor=(1.0, Legend_y_offset),  # Moved up by adding y offset
                           handlelength=2,
                           markerfirst=False,
                           fontsize=LATEX_FONT_SIZE/FontSizeFactor_Legends)
            # Remove the separate legend for ax_1_12_RR since it's combined
            
            # Adjust layout and save figure
            plt.tight_layout()
            
            if save_fig:
                fig_path = os.path.join(output_dir, f"colortable_plot_image_{int(image_num):04d}.png") # Cast image_num to int
                plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
                if Plot_log_level >= 1: 
                    print(f"  Saved figure: {fig_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
                
        except Exception as e:
            print(f"Error processing row {idx} for image {image_num}: {e}") # Keep error
    
    # Create a video if requested
    if video and save_fig:
        try:
            # Get all saved figures
            image_files = sorted(glob.glob(os.path.join(output_dir, "colortable_plot_image_*.png")))
            
            if not image_files:
                print("No images found for video creation.")
                return output_dir
            
            # Read the first image to get dimensions
            first_img = cv2.imread(image_files[0])
            height, width, layers = first_img.shape
            
            # Create video writer
            video_path = os.path.join(output_dir, "colortable_plot_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, 5, (width, height))  # 5 fps
            
            # Add each image to the video
            for img_path in image_files:
                img = cv2.imread(img_path)
                video.write(img)
            
            # Release the video writer
            video.release()
            
            if Plot_log_level >= 1:
                print(f"Created video: {video_path}")
                
        except Exception as e:
            print(f"Error creating video: {e}")
    
    return output_dir

if __name__ == "__main__":
    # Example usage
    
    plotter_6_colortables(
        # BBWW extract 
        input_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250610_0646347\20250610_0646347\20250615_1609401\20250615_1727535",
        output_dir_manual="",
        output_dir_comment="",
        show_plot=0,
        ScaleFactor=1.2,
        video=1,
        Plot_log_level=1,
    )
