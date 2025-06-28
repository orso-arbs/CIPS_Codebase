import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from skimage import measure
import pickle
from pathlib import Path
import Format_1 as F_1

# Set up LaTeX font
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)

def plot_segmented_image(
    input_dir,
    output_dir_manual="",
    output_dir_comment="segmented_images",
    image_numbers=[], 
    show_masks=True, 
    show_outlines=True, 
    cells_to_color=[], # For the Panel describing the pipeline, the cell is [95]
    alpha=0.5, 
    zoom_factor=1.5, # Changed from 2 to 1.5
    # New text customization parameters
    title_text="",            # Title text (empty for no title)
    title_fontsize=16,        # Font size for title
    x_label="",               # x-axis label (empty for no label)
    y_label="",               # y-axis label (empty for no label)
    axis_label_fontsize=12,   # Font size for axis labels
    legend_text=[],           # List of legend entries
    legend_fontsize=10,       # Font size for legend text
    legend_position="lower right",  # Position for legend
    # Existing parameters
    label_text="", 
    label_size=12, 
    label_pos=(0.05, 0.95),
    contour_color='w',    
    contour_linestyle='-', 
    contour_linewidth=0.8, 
    show_radius=True, # Changed from False to True    
    radius_color='r',     
    radius_linestyle='--', 
    radius_linewidth=3, # Changed from 1.5 to 3
    show_plot=0):
    """
    Plot segmented images with masks and outlines.
    
    Parameters:
    -----------
    input_dir : str or Path
        Directory containing Analysis_A11_final_df.pkl
    output_dir_manual : str, optional
        Manual output directory, by default ""
    output_dir_comment : str, optional
        Comment to append to the output directory name, by default ""
    image_numbers : list
        List of image numbers to plot. If empty, all images are plotted.
    show_masks : bool
        Whether to show masks overlay
    show_outlines : bool
        Whether to show cell outlines
    cells_to_color : list, optional
        List of cell IDs to color. If empty or None, all cells are colored.
    alpha : float
        Transparency of the masks (0-1)
    zoom_factor : float
        Factor to zoom in to the center of the image
    title_text : str, optional
        Title text to display on the plot. Empty string for no title.
    title_fontsize : int, optional
        Font size for the title, by default 16
    x_label : str, optional
        Label for x-axis. Empty string for no label.
    y_label : str, optional
        Label for y-axis. Empty string for no label.
    axis_label_fontsize : int, optional
        Font size for axis labels, by default 12
    legend_text : list, optional
        List of strings for legend entries. Empty list for no legend.
    legend_fontsize : int, optional
        Font size for legend text, by default 10
    legend_position : str, optional
        Position for the legend (matplotlib position string), by default "lower right"
    contour_color : str
        Color of the cell outlines, by default 'w' (white)
    contour_linestyle : str
        Line style of the cell outlines, by default '-' (solid)
    contour_linewidth : float
        Line width of the cell outlines, by default 0.8
    show_radius : bool
        Whether to display reference radius circle, by default False
    radius_color : str
        Color for the radius circle, by default 'r' (red)
    radius_linestyle : str
        Line style for the radius circle, by default '--' (dashed)
    radius_linewidth : float
        Line width for the radius circle, by default 1.5
    show_plot : int, optional
        Whether to display the plot (1) or not (0), by default 0
    """
    # Create output directory using Format_1
    output_dir = F_1.F_out_dir(input_dir=input_dir, script_path=__file__, output_dir_comment=output_dir_comment, output_dir_manual=output_dir_manual)
    
    print(f"Output directory: {output_dir}")
    
    # Ensure directories are Path objects
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Load the dataframe from pickle
    df_path = input_dir / "Analysis_A11_final_df.pkl"
    with open(df_path, 'rb') as f:
        df = pickle.load(f)
    
    # If no image numbers provided, use all images
    if not image_numbers:
        image_numbers = df['image_number'].unique()
    
    print(f"Processing {len(image_numbers)} images...")
    
    # Process each row in the DataFrame
    for idx, row in df.iterrows():
        # Only process images in the image_numbers list
        image_num = row['image_number']
        
        # Handle the case where image_num might be a numpy array
        if isinstance(image_num, np.ndarray):
            if len(image_num) > 0:
                image_num = image_num[0]  # Take the first element if it's an array
            else:
                continue  # Skip if empty array
        
        if image_numbers and image_num not in image_numbers:
            continue
            
        print(f"Processing image number {image_num}...")
        
        # Get data for this image from the row
        try:
            image_file_path = row['image_file_path']
            mask_from_df = row['masks']  # Get mask from DataFrame
            D_SF_px = row['D_SF_px']
            R_SF_px = D_SF_px / 2  # Calculate radius from diameter
            current_time = df.iloc[idx]['Time_VisIt']
            
            # Read the image from file
            original_img = plt.imread(image_file_path)
            if original_img is None:
                print(f"Error: Could not read image file: {image_file_path}")
                continue
                
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Get image dimensions
            height, width = original_img.shape[:2]
            
            # Calculate zoom area centered on the image
            center_x, center_y = width // 2, height // 2
            new_width = int(width / zoom_factor)
            new_height = int(height / zoom_factor)
            x_min = max(0, center_x - new_width // 2)
            x_max = min(width, center_x + new_width // 2)
            y_min = max(0, center_y - new_height // 2)
            y_max = min(height, center_y + new_height // 2)
            
            # Display zoomed image
            zoomed_image = original_img[y_min:y_max, x_min:x_max]
            
            # Get the corresponding part of the mask
            if mask_from_df is not None and isinstance(mask_from_df, np.ndarray):
                zoomed_mask = mask_from_df[y_min:y_max, x_min:x_max]
            else:
                print(f"Warning: Mask not available for image {image_num}")
                zoomed_mask = None
            
            # Show the image
            if len(original_img.shape) == 3 and original_img.shape[2] == 3:  # RGB image
                ax.imshow(zoomed_image)
            else:  # Grayscale image
                ax.imshow(zoomed_image, cmap='gray')
            
            if show_masks and zoomed_mask is not None:
                # Define a list of 10 distinct colors (RGB format) like in plot6_colortables
                distinct_colors = [
                    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 165, 0), (128, 0, 128),
                    (165, 42, 42), (255, 192, 203)
                ]
                
                # Get unique cell IDs (excluding background)
                unique_cells = np.unique(zoomed_mask)
                unique_cells = unique_cells[unique_cells > 0]
                
                # Create overlay for all cells
                mask_overlay = np.zeros((*zoomed_mask.shape, 4))
                
                # Apply colors to each cell
                for cell_id in unique_cells:
                    # Skip cells not in cells_to_color if it's specified
                    if cells_to_color is not None and len(cells_to_color) > 0:
                        if cell_id not in cells_to_color:
                            continue
                    
                    cell_mask = zoomed_mask == cell_id
                    color_index = int(cell_id) % 10  # Use modulo to cycle through colors
                    color_rgb = distinct_colors[color_index]
                    # Convert RGB (0-255) to normalized values (0-1) for the overlay
                    normalized_color = (color_rgb[0]/255, color_rgb[1]/255, color_rgb[2]/255, alpha)
                    mask_overlay[cell_mask] = normalized_color
                
                ax.imshow(mask_overlay)
            
            if show_outlines and zoomed_mask is not None:
                # Draw outlines for each cell
                for cell_id in np.unique(zoomed_mask)[1:]:  # Skip background (0)
                    # Skip cells not in cells_to_color if it's specified
                    if cells_to_color is not None and len(cells_to_color) > 0:
                        if cell_id not in cells_to_color:
                            continue
                    
                    cell_mask = zoomed_mask == cell_id
                    contours = measure.find_contours(cell_mask, 0.5)
                    
                    for contour in contours:
                        ax.plot(contour[:, 1], contour[:, 0], 
                               color=contour_color, 
                               linestyle=contour_linestyle, 
                               linewidth=contour_linewidth)
            
            # Add reference radius circle if requested
            if show_radius:
                # Calculate center of zoomed image in original coordinates
                orig_center_x, orig_center_y = width // 2, height // 2
                
                # Calculate zoomed image center
                zoomed_center_x = orig_center_x - x_min
                zoomed_center_y = orig_center_y - y_min
                
                # Plot circle representing the flame radius
                theta = np.linspace(0, 2*np.pi, 200)
                # Scale radius based on zoom factor
                visible_radius = R_SF_px
                
                # Make sure circle is within zoomed boundaries
                if visible_radius > 0:
                    circle_x = zoomed_center_x + visible_radius * np.cos(theta)
                    circle_y = zoomed_center_y + visible_radius * np.sin(theta)
                    ax.plot(circle_x, circle_y, 
                           color=radius_color, 
                           linestyle=radius_linestyle, 
                           linewidth=radius_linewidth)
            
            # Add label if provided
            if label_text:
                ax.text(label_pos[0], label_pos[1], label_text, 
                        transform=ax.transAxes, fontsize=label_size,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Add title if provided
            if title_text:
                # You can include formatting like image number or time if needed
                formatted_title = title_text.format(image_num=image_num, time=current_time)
                ax.set_title(formatted_title, fontsize=title_fontsize)
            
            # Add axis labels if provided
            if x_label:
                ax.set_xlabel(x_label, fontsize=axis_label_fontsize)
            if y_label:
                ax.set_ylabel(y_label, fontsize=axis_label_fontsize)
            
            # Create legend handles if legend_text is provided
            if legend_text:
                from matplotlib.patches import Patch
                legend_handles = []
                
                # Create basic legend entries
                for i, text in enumerate(legend_text):
                    color = distinct_colors[i % len(distinct_colors)]
                    color_normalized = (color[0]/255, color[1]/255, color[2]/255)
                    legend_handles.append(Patch(facecolor=color_normalized, edgecolor='black', label=text))
                
                # Add legend
                ax.legend(handles=legend_handles, loc=legend_position, fontsize=legend_fontsize)
            
            # Remove axes if both x_label and y_label are empty
            if not x_label and not y_label:
                ax.set_axis_off()
            
            # Remove axes
            ax.set_axis_off()
            
            # Save figure
            fig.tight_layout()
            output_filename = f"plot13_segmented_image_{int(image_num):04d}"
            fig.savefig(output_dir / f"{output_filename}.png", format="png", dpi=300, bbox_inches='tight')
            fig.savefig(output_dir / f"{output_filename}.svg", format="svg", dpi=300, bbox_inches='tight')
            
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
            
            print(f"Saved plot for image number {image_num}")
            
        except Exception as e:
            print(f"Error processing image {image_num}: {e}")

if __name__ == "__main__":
    # Example usage
    plot_segmented_image(
        input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250628_1636311\20250628_1636322\20250628_1637345\20250628_1638434\20250628_1638484",
        output_dir_comment="segmented_images",
        image_numbers=[79],  # Use empty list [] to plot all images
        show_masks=True,
        show_outlines=True,
        cells_to_color=[],  # Only color cell IDs 1, 2, and 5 (use [] for all cells)
        alpha=0.5,           # Transparency of mask overlay
        zoom_factor=1.5,     # How much to zoom in to center (higher = more zoom)
        #label_text=r"Segmented cells",  # Use LaTeX formatting if needed
        label_size=12,
        label_pos=(0.05, 0.95),  # Position in relative coordinates (0-1)
        contour_color='w',    # White outlines
        contour_linestyle='-', # Solid line
        contour_linewidth=0.8, # Line width
        show_radius=True,     # Show the flame radius circle
        radius_color='r',     # Red circle
        radius_linestyle='--', # Dashed line
        radius_linewidth=3,    # Line width for radius
        show_plot=0
    )
