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

def plotter_6_colortables(
    input_dir,
    output_dir_manual="",
    output_dir_comment="",
    show_plot=0,
    Plot_log_level=1,
    color_bar_width_ratio=0.1,  # Width ratio for colorbar subplot
    image_width_ratio=0.45,     # Width ratio for image subplot
    plot_width_ratio=0.45,      # Width ratio for property plot subplot
    ScaleFactor=1.5,            # Scale factor for zooming in on the spherical flame
    figsize=(18, 6),            # Figure size (width, height) in inches
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
    color_bar_width_ratio : float, optional
        Width ratio for the colorbar subplot, by default 0.1
    image_width_ratio : float, optional
        Width ratio for the image subplot, by default 0.45
    plot_width_ratio : float, optional
        Width ratio for the property plot subplot, by default 0.45
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
    
    # Find the CSV and DFs
    pandas_wildcard_str = os.path.join(input_dir, "*.csv")
    csv_files = glob.glob(pandas_wildcard_str)
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return output_dir
    
    # Load the main DataFrame with image and segmentation data
    df_path = [f for f in csv_files if "CP_" in f][0] if any("CP_" in f for f in csv_files) else csv_files[0]
    df = pd.read_csv(df_path)
    
    if Plot_log_level >= 1:
        print(f"Loaded DataFrame from {df_path}")
    
    # Find the colortable directory
    # Starting from input_dir, search upwards until we find a directory containing 'colortables'
    colortable_dir = None
    current_dir = input_dir
    
    for _ in range(5):  # Limit the search to 5 levels up
        potential_dir = os.path.join(current_dir, "colortables")
        if os.path.exists(potential_dir):
            colortable_dir = potential_dir
            break
        else:
            # Try to find a folder named colortables in the current directory
            for d in os.listdir(current_dir):
                if os.path.isdir(os.path.join(current_dir, d)) and "colortables" in d.lower():
                    colortable_dir = os.path.join(current_dir, d)
                    break
        
        if colortable_dir:
            break
            
        # Move up one directory
        current_dir = os.path.dirname(current_dir)
    
    if not colortable_dir:
        print("Could not find colortables directory. Looking for a PointWise_colorbar file directly.")
        # Try to find colorbar files directly
        colorbar_search = glob.glob(os.path.join(input_dir, "**/*PointWise_colorbar*.png"), recursive=True)
        if colorbar_search:
            colortable_dir = os.path.dirname(colorbar_search[0])
        else:
            print("Warning: Could not find colortables directory or colorbar files.")

    # Process each image in the DataFrame
    for idx, row in df.iterrows():
        try:
            # Get image info
            image_num = row['image_num']
            image_file_path = row['image_file_path']
            label_file_path = row['label_file_path']
            D_SF_px = row['D_SF_px']
            
            if Plot_log_level >= 2:
                print(f"Processing image {image_num}, file: {image_file_path}")
            
            # Create figure with 3 subplots of different widths
            fig = plt.figure(figsize=figsize, dpi=dpi)
            gs = gridspec.GridSpec(1, 3, width_ratios=[color_bar_width_ratio, image_width_ratio, plot_width_ratio])

            # 1. Left subplot: Colorbar
            ax_colorbar = plt.subplot(gs[0])
            
            # Try to load the labeled colorbar first, then the unlabeled one
            colorbar_image = None
            if colortable_dir:
                labeled_path = os.path.join(colortable_dir, "PointWise_colorbar_labeled.png")
                unlabeled_path = os.path.join(colortable_dir, "PointWise_colorbar_unlabeled.png")
                
                if os.path.exists(labeled_path):
                    colorbar_image = Image.open(labeled_path)
                elif os.path.exists(unlabeled_path):
                    colorbar_image = Image.open(unlabeled_path)
            
            if colorbar_image:
                ax_colorbar.imshow(np.array(colorbar_image))
                ax_colorbar.set_title("Color Table")
                ax_colorbar.axis('off')
            else:
                ax_colorbar.text(0.5, 0.5, "Colorbar\nnot found", 
                                ha='center', va='center', 
                                transform=ax_colorbar.transAxes)
                ax_colorbar.axis('off')

            # 2. Middle subplot: Flame and segmentation
            ax_image = plt.subplot(gs[1])
            
            # Load the original image and the label mask
            try:
                original_img = cv2.imread(image_file_path)
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                
                # Load the label mask if it exists
                if os.path.exists(label_file_path):
                    mask = cv2.imread(label_file_path, cv2.IMREAD_GRAYSCALE)
                else:
                    mask = None
                    print(f"Warning: Mask file not found: {label_file_path}")
                
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
                    # Set different colors for different mask values
                    for mask_value, color in zip([1, 2, 3], 
                                                [(255, 0, 0),    # Red
                                                 (0, 255, 0),    # Green
                                                 (0, 0, 255)]):  # Blue
                        color_mask[mask == mask_value] = color
                    
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
                ax_image.imshow(zoom_img)
                ax_image.set_title(f"Flame & Segmentation (Image {image_num})")
                ax_image.axis('off')
                
                # Add a vertical line at the center to separate original and segmented views
                ax_image.axvline(x=(right-left)//2, color='yellow', linestyle='-', linewidth=1)
                
            except Exception as e:
                print(f"Error processing image {image_file_path}: {e}")
                ax_image.text(0.5, 0.5, "Image\nloading\nerror", 
                             ha='center', va='center', 
                             transform=ax_image.transAxes)
                ax_image.axis('off')

            # 3. Right subplot: Properties plot (similar to plot1's ax_1_12)
            ax_props = plt.subplot(gs[2])
            
            # Plot key properties from the DataFrame
            props_to_plot = ['R_SF_px', 'D_SF_px', 'area_px2', 'perim_px']
            colors = ['blue', 'red', 'green', 'purple']
            
            # If we have multiple images, plot properties vs image_num
            if len(df) > 1:
                for prop, color in zip(props_to_plot, colors):
                    if prop in df.columns:
                        # Normalize values to make them comparable on the same scale
                        values = df[prop]
                        normalized = values / values.max() if values.max() > 0 else values
                        ax_props.plot(df['image_num'], normalized, '-o', color=color, label=prop)
                
                # Highlight current image
                for prop, color in zip(props_to_plot, colors):
                    if prop in df.columns:
                        values = df[prop]
                        normalized = values / values.max() if values.max() > 0 else values
                        current_val = normalized.iloc[idx]
                        ax_props.plot(image_num, current_val, 'o', color=color, markersize=10, markeredgecolor='black')
                
                ax_props.set_xlabel('Image Number')
                ax_props.set_ylabel('Normalized Value')
                ax_props.set_title('Key Properties')
                ax_props.legend()
                ax_props.grid(True, alpha=0.3)
            else:
                # If we only have one image, show property values as a bar chart
                valid_props = [prop for prop in props_to_plot if prop in df.columns]
                values = [df[prop].iloc[0] for prop in valid_props]
                ax_props.bar(valid_props, values, color=colors[:len(valid_props)])
                ax_props.set_title('Key Properties')
                ax_props.set_ylabel('Value')
                ax_props.grid(True, alpha=0.3)

            # Adjust layout and save figure
            plt.tight_layout()
            
            if save_fig:
                fig_path = os.path.join(output_dir, f"colortable_plot_image_{image_num:04d}.png")
                plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
                if Plot_log_level >= 1:
                    print(f"Saved figure: {fig_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
                
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
    
    # Create a video if requested
    if video and save_fig:
        try:
            import cv2
            
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
        input_dir="",
        output_dir_manual="",
        output_dir_comment="",
        show_plot=1,
        ScaleFactor=1.2,
        video=1
    )
