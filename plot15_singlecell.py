import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle
from pathlib import Path
import Format_1 as F_1
from skimage import measure
import matplotlib.patches as patches

# Set up LaTeX font
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)

def plot_single_cell(
    input_dir,
    output_dir_manual="",
    output_dir_comment="single_cell_plots",
    image_numbers=[79], 
    cell_id=95,
    zoom_factor=5.0,  # How much to zoom in on the cell
    # Display options
    show_mask=True,
    show_outline=True,
    show_diameter_circle=True,
    show_centroid_cross=True,  # New option to show centroid cross
    # Styling options
    mask_alpha=0.5,
    outline_color='w',
    outline_linestyle='-',
    outline_linewidth=0.8,
    diameter_circle_color='r',
    diameter_circle_linestyle='-',
    diameter_circle_linewidth=1.5,
    cross_color='black',        # Color of the centroid cross
    cross_size=10,              # Size of the cross in pixels
    cross_linewidth=2,          # Width of the cross lines
    # Label options
    x_label="",
    y_label="",
    axis_label_fontsize=12,
    # Tick options
    show_ticks=True,            # Whether to show axis ticks
    x_ticks=None,               # Custom x-axis tick positions
    y_ticks=None,               # Custom y-axis tick positions
    x_ticklabels=None,          # Custom x-axis tick labels
    y_ticklabels=None,          # Custom y-axis tick labels
    tick_fontsize=10,           # Font size for tick labels
    x_axis_position="top",      # Position of the x-axis: "top" or "bottom"
    # Legend options
    show_legend=False,  # New parameter to control whether legend is shown
    legend_text=[],
    legend_fontsize=10,
    legend_position="lower right",
    # Figure options
    fig_size=(8, 8),
    dpi=300,
    show_plot=0):
    """
    Plot a zoomed view of a single cell from segmentation data.
    
    Parameters:
    -----------
    input_dir : str or Path
        Directory containing Analysis_A11_final_df.pkl
    output_dir_manual : str, optional
        Manual output directory, by default ""
    output_dir_comment : str, optional
        Comment to append to the output directory name, by default "single_cell_plots"
    image_numbers : list, optional
        List of image numbers to plot. If empty, all images are plotted.
    cell_id : int, optional
        ID of the cell to plot, by default 96
    zoom_factor : float, optional
        How much to zoom in on the cell, by default 3.0
    show_mask : bool, optional
        Whether to show cell mask overlay, by default True
    show_outline : bool, optional
        Whether to show cell outline, by default True
    show_diameter_circle : bool, optional
        Whether to show circle representing cell diameter, by default True
    show_centroid_cross : bool, optional
        Whether to show a cross at the cell centroid, by default True
    mask_alpha : float, optional
        Transparency of the mask overlay (0-1), by default 0.5
    outline_color : str, optional
        Color of the cell outline, by default 'w'
    outline_linestyle : str, optional
        Line style of the cell outline, by default '-'
    outline_linewidth : float, optional
        Line width of the cell outline, by default 0.8
    diameter_circle_color : str, optional
        Color of the diameter circle, by default 'r'
    diameter_circle_linestyle : str, optional
        Line style of the diameter circle, by default '-'
    diameter_circle_linewidth : float, optional
        Line width of the diameter circle, by default 1.5
    cross_color : str, optional
        Color of the centroid cross, by default 'black'
    cross_size : int, optional
        Size of the cross in pixels, by default 10
    cross_linewidth : float, optional
        Width of the cross lines, by default 2
    x_label : str, optional
        Label for x-axis, by default ""
    y_label : str, optional
        Label for y-axis, by default ""
    axis_label_fontsize : int, optional
        Font size for axis labels, by default 12
    show_ticks : bool, optional
        Whether to show axis ticks, by default True
    x_ticks : list or None, optional
        Custom positions for x-axis ticks, by default None
    y_ticks : list or None, optional
        Custom positions for y-axis ticks, by default None
    x_ticklabels : list or None, optional
        Custom labels for x-axis ticks, by default None
    y_ticklabels : list or None, optional
        Custom labels for y-axis ticks, by default None
    tick_fontsize : int, optional
        Font size for tick labels, by default 10
    x_axis_position : str, optional
        Position of the x-axis, either "top" or "bottom", by default "top"
    show_legend : bool, optional
        Whether to show a legend on the plot, by default False
    legend_text : list, optional
        List of strings for legend entries, by default []
    legend_fontsize : int, optional
        Font size for legend text, by default 10
    legend_position : str, optional
        Position for the legend, by default "lower right"
    fig_size : tuple, optional
        Figure size (width, height) in inches, by default (8, 8)
    dpi : int, optional
        DPI for saving figures, by default 300
    show_plot : int, optional
        Whether to display the plot (1) or not (0), by default 0
    """
    # Create output directory using Format_1
    output_dir = F_1.F_out_dir(input_dir=input_dir, script_path=__file__, 
                             output_dir_comment=output_dir_comment, 
                             output_dir_manual=output_dir_manual)
    
    print(f"Output directory: {output_dir}")
    
    # Create subdirectories for PNG and SVG files
    png_dir = os.path.join(output_dir, "png")
    svg_dir = os.path.join(output_dir, "svg")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)
    
    # Ensure directories are Path objects
    input_dir = Path(input_dir)
    
    # Load the dataframe from pickle
    df_path = input_dir / "Analysis_A11_final_df.pkl"
    with open(df_path, 'rb') as f:
        df = pickle.load(f)
    
    # If no image numbers provided, use all images
    if not image_numbers:
        image_numbers = df['image_number'].unique()
    
    # Convert image_numbers to a standard Python list if it's a NumPy array
    if isinstance(image_numbers, np.ndarray):
        image_numbers = image_numbers.tolist()
    
    print(f"Processing {len(image_numbers)} images for cell ID {cell_id}...")
    
    # Define a list of 10 distinct colors (RGB format) similar to plot13_segmented_image.py
    distinct_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (255, 165, 0), (128, 0, 128),
        (165, 42, 42), (255, 192, 203)
    ]
    
    # Process each row in the DataFrame
    for idx, row in df.iterrows():
        # Only process images in the image_numbers list
        image_num = row['image_number']
        
        # Handle the case where image_num might be a numpy array
        if isinstance(image_num, np.ndarray):
            if len(image_num) > 0:
                image_num = int(image_num[0])  # Convert to int to ensure scalar
            else:
                continue  # Skip if empty array
                
        # Convert to int if it's a float to avoid comparison issues
        if isinstance(image_num, (float, np.float64, np.float32)):
            image_num = int(image_num)
        
        # Now check if we should process this image
        if image_numbers and image_num not in image_numbers:
            continue
            
        print(f"Processing image number {image_num}...")
        
        # Get data for this image from the row
        try:
            image_file_path = row['image_file_path']
            mask_from_df = row['masks']  # Get mask from DataFrame
            
            # Check if the specified cell_id exists in the mask
            if mask_from_df is None or cell_id not in np.unique(mask_from_df):
                print(f"  Cell ID {cell_id} not found in image {image_num}, skipping...")
                continue
            
            # Get cell-specific data
            cell_mask = mask_from_df == cell_id
            
            # Find the centroid of the cell
            props = measure.regionprops(cell_mask.astype(int))
            if not props:
                print(f"  No region found for cell ID {cell_id} in image {image_num}, skipping...")
                continue
            
            centroid_y, centroid_x = props[0].centroid
            
            # Get the cell diameter from the distribution lists
            # These lists should be in the DataFrame with each element corresponding to a cell ID
            try:
                # Check if the cell diameter distribution is available
                if ('d_cell_distribution_px' in row and 
                    isinstance(row['d_cell_distribution_px'], list) and 
                    len(row['d_cell_distribution_px']) > cell_id):
                    cell_diameter = row['d_cell_distribution_px'][cell_id]
                else:
                    # If not available directly, try to compute from the mask
                    cell_area = props[0].area
                    cell_diameter = 2 * np.sqrt(cell_area / np.pi)  # Approximation for diameter
            except (IndexError, TypeError) as e:
                print(f"  Error getting cell diameter for cell ID {cell_id}: {e}")
                cell_diameter = 20  # Default value
            
            # Read the image from file
            original_img = plt.imread(image_file_path)
            if original_img is None:
                print(f"Error: Could not read image file: {image_file_path}")
                continue
                
            # Create figure with inward-facing ticks
            fig, ax = plt.subplots(figsize=fig_size)
            plt.tick_params(axis='both', direction='in')
            
            # Calculate zoom area centered on the cell centroid
            height, width = original_img.shape[:2]
            
            # Determine the zoom window size based on the cell diameter and zoom factor
            zoom_window_size = int(cell_diameter * zoom_factor)
            
            # Calculate bounds for the zoom window
            x_min = max(0, int(centroid_x - zoom_window_size // 2))
            x_max = min(width, int(centroid_x + zoom_window_size // 2))
            y_min = max(0, int(centroid_y - zoom_window_size // 2))
            y_max = min(height, int(centroid_y + zoom_window_size // 2))
            
            # Extract the zoomed image region
            zoomed_image = original_img[y_min:y_max, x_min:x_max]
            zoomed_mask = mask_from_df[y_min:y_max, x_min:x_max]
            
            # Display the base image
            if len(original_img.shape) == 3 and original_img.shape[2] == 3:  # RGB image
                ax.imshow(zoomed_image)
            else:  # Grayscale image
                ax.imshow(zoomed_image, cmap='gray')
            
            # Apply cell mask overlay if requested
            if show_mask:
                # Create overlay for the specific cell
                mask_overlay = np.zeros((*zoomed_mask.shape, 4))
                cell_mask_zoomed = zoomed_mask == cell_id
                
                if np.any(cell_mask_zoomed):
                    color_index = cell_id % 10  # Use modulo to cycle through colors
                    color_rgb = distinct_colors[color_index]
                    # Convert RGB (0-255) to normalized values (0-1) for the overlay
                    normalized_color = (color_rgb[0]/255, color_rgb[1]/255, color_rgb[2]/255, mask_alpha)
                    mask_overlay[cell_mask_zoomed] = normalized_color
                    ax.imshow(mask_overlay)


            # Draw cell outline if requested
            if show_outline:
                cell_mask_zoomed = zoomed_mask == cell_id
                contours = measure.find_contours(cell_mask_zoomed, 0.5)
                
                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], 
                           color=outline_color, 
                           linestyle=outline_linestyle, 
                           linewidth=outline_linewidth)
            
            # Configure axes and ticks
            if show_ticks:
                # Set custom ticks if provided
                if x_ticks is not None:
                    ax.set_xticks(x_ticks)
                if y_ticks is not None:
                    ax.set_yticks(y_ticks)
                
                # Set custom tick labels if provided
                if x_ticklabels is not None:
                    ax.set_xticklabels(x_ticklabels, fontsize=tick_fontsize)
                else:
                    ax.tick_params(axis='x', labelsize=tick_fontsize)
                
                if y_ticklabels is not None:
                    ax.set_yticklabels(y_ticklabels, fontsize=tick_fontsize)
                else:
                    ax.tick_params(axis='y', labelsize=tick_fontsize)
                
                # Position the x-axis at the top of the plot if specified
                if x_axis_position.lower() == "top":
                    ax.xaxis.set_ticks_position('top')
                    ax.xaxis.set_label_position('top')
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['top'].set_visible(True)
            else:
                # Hide ticks and tick labels
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Add axis labels if provided
            if x_label:
                ax.set_xlabel(x_label, fontsize=axis_label_fontsize)
            if y_label:
                ax.set_ylabel(y_label, fontsize=axis_label_fontsize)
            
            # Draw centroid cross if requested
            if show_centroid_cross:
                # Calculate the cell centroid in the zoomed coordinate system
                zoomed_centroid_x = centroid_x - x_min
                zoomed_centroid_y = centroid_y - y_min
                
                # Draw horizontal line of the cross
                ax.plot([zoomed_centroid_x - cross_size/2, zoomed_centroid_x + cross_size/2],
                       [zoomed_centroid_y, zoomed_centroid_y],
                       color=cross_color, linewidth=cross_linewidth)
                
                # Draw vertical line of the cross
                ax.plot([zoomed_centroid_x, zoomed_centroid_x],
                       [zoomed_centroid_y - cross_size/2, zoomed_centroid_y + cross_size/2],
                       color=cross_color, linewidth=cross_linewidth)
            
            # Draw circle representing cell diameter AFTER drawing outline
            # (so it appears on top of the outline)
            if show_diameter_circle:
                # Calculate the cell centroid in the zoomed coordinate system
                zoomed_centroid_x = centroid_x - x_min
                zoomed_centroid_y = centroid_y - y_min
                
                # Create a circle patch with radius of half the cell diameter
                circle = plt.Circle((zoomed_centroid_x, zoomed_centroid_y), 
                                   cell_diameter / 2,
                                   fill=False, 
                                   edgecolor=diameter_circle_color,
                                   linestyle=diameter_circle_linestyle,
                                   linewidth=diameter_circle_linewidth)
                ax.add_patch(circle)
            
            # Add legend if show_legend is True and legend_text is provided
            if show_legend and legend_text:
                from matplotlib.patches import Patch
                legend_handles = []
                
                # Create legend entries
                for i, text in enumerate(legend_text):
                    color = distinct_colors[i % len(distinct_colors)]
                    color_normalized = (color[0]/255, color[1]/255, color[2]/255)
                    legend_handles.append(Patch(facecolor=color_normalized, edgecolor='black', label=text))
                
                # Add legend
                ax.legend(handles=legend_handles, loc=legend_position, fontsize=legend_fontsize)
            
            # Save figures
            fig.tight_layout()
            output_filename = f"plot15_singlecell_{int(image_num):04d}_cell{cell_id}"
            
            # Save as PNG
            png_path = os.path.join(png_dir, f"{output_filename}.png")
            fig.savefig(png_path, format="png", dpi=dpi, bbox_inches='tight')
            
            # Save as SVG
            svg_path = os.path.join(svg_dir, f"{output_filename}.svg")
            fig.savefig(svg_path, format="svg", dpi=dpi, bbox_inches='tight')
            
            print(f"  Saved plots for image {image_num}, cell {cell_id}")
            
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
                
        except Exception as e:
            print(f"Error processing image {image_num} for cell {cell_id}: {e}")

if __name__ == "__main__":
    # Example usage
    plot_single_cell(
        input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187",
        output_dir_comment="single_cell_plots",
        image_numbers=[79],  # Use empty list [] to plot all images
        cell_id=95,        # Default cell ID to plot
        zoom_factor=2.0,   # Zoom factor relative to cell diameter
        # Display options
        show_mask=True,
        show_outline=True,
        show_diameter_circle=True,
        show_centroid_cross=True,  # Show cross at centroid
        # Styling options
        mask_alpha=0.5,
        outline_color='w',
        outline_linestyle='-',
        outline_linewidth=3,
        diameter_circle_color='r',
        diameter_circle_linestyle='-',
        diameter_circle_linewidth=3,
        cross_color='black',      # Black cross at centroid
        cross_size=25,            # Size of cross in pixels
        cross_linewidth=3,        # Width of cross lines
        # Label options
        x_label="x [px]",
        y_label="y [px]",
        axis_label_fontsize=20,
        # Tick options
        show_ticks=True,
        tick_fontsize=20,
        x_axis_position="top",  # Position x-axis at the top of the plot
        # Legend options
        show_legend=False,
        legend_text=[""],
        legend_fontsize=20,
        legend_position="lower right",
        # Figure options
        fig_size=(8, 8),
        dpi=300,
        show_plot=0
    )
