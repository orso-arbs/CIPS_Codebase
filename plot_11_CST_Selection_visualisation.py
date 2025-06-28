import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import skimage.io as sk_io
from matplotlib import rcParams
import Format_1 as F_1
from CST_Selection_1 import plot_CST_selection_sanity_check, plot_CST_inclusion
from Spherical_Reconstruction_2 import Cubed_Sphere_Tile_Boundary, Coordinate_Transform_image_to_centered_Spherical

def setup_latex_fonts():
    """Configure matplotlib to use LaTeX fonts"""
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Computer Modern Roman']
    rcParams['text.usetex'] = True
    rcParams['font.size'] = 11
    rcParams['axes.titlesize'] = 12
    rcParams['axes.labelsize'] = 11
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10
    rcParams['figure.titlesize'] = 14

@F_1.ParameterLog(max_size=1024 * 10, log_level=0)
def plot_CST_Selection_visualisation(
    # input
    input_dir,
    Analysis_A11_df=None,  # DataFrame from previous processing, or None to load from input_dir
    image_number=79,  # Changed from 0 to 79
    
    # output and logging
    log_level=2,
    output_dir_manual="",
    output_dir_comment="",
    show_plots=True,  # Changed from False to True
    save_figure=True,
    figsize=(16, 8),
    dpi=300,
    Convert_to_grayscale_image=True,
    
    # Visualization parameters
    cst_boundary_linewidth=3.0,  # Changed from 2.0 to 3.0
    ref_circle_linewidth=2.0,    # Changed from 1.5 to 2.0
    legend_fontsize=20,  # Changed from 8 to 20
    legend_position='upper center',  # Changed from 'lower center' to 'upper center'
    legend_bbox_to_anchor=(0.5, 0.05),
    text_box_pos=(0.05, 0.9),  # Changed from (0.05, 0.95) to (0.05, 0.9)
    text_box_fontsize=25  # Changed from 14 to 25
):
    """
    Creates a 2x1 plot with CST selection sanity check and inclusion visualization for a specific image.
    
    Parameters
    ----------
    input_dir : str
        Path to the directory containing the Analysis_A11_df.pkl
    Analysis_A11_df : pd.DataFrame or None, optional
        DataFrame from previous processing or None to load from input_dir
    image_number : int, optional
        Index of the image to visualize (0-based). Default is 0.
    log_level : int, optional
        Controls verbosity of logging. Default is 2.
    output_dir_manual : str, optional
        If provided, specifies the output directory. Default is "".
    output_dir_comment : str, optional
        Comment to append to the output directory name. Default is "".
    show_plots : bool, optional
        Whether to display plots during processing. Default is False.
    save_figure : bool, optional
        Whether to save the visualization. Default is True.
    figsize : tuple, optional
        Figure dimensions (width, height) in inches. Default is (16, 8).
    dpi : int, optional
        DPI for saved figure. Default is 300.
    Convert_to_grayscale_image : bool, optional
        Whether to convert images to grayscale in plots. Default is True.
    cst_boundary_linewidth : float, optional
        Line width for CST boundary. Default is 2.0.
    ref_circle_linewidth : float, optional
        Line width for reference circle. Default is 1.5.
    legend_fontsize : int, optional
        Font size for the legend. Default is 8.
    legend_position : str, optional
        Position for the legend. Default is 'lower center'.
    legend_bbox_to_anchor : tuple, optional
        Fine-tuning for legend position (x, y). Default is (0.5, 0.05).
    text_box_pos : tuple, optional
        Position (x, y) for the (a)/(b) annotations in axes coordinates. Default is (0.05, 0.95).
    text_box_fontsize : int, optional
        Font size for the (a)/(b) annotations. Default is 14.
        
    Returns
    -------
    output_dir : str
        Path to the output directory.
    """
    #################################################### I/O
    # Create output directory
    output_dir = F_1.F_out_dir(
        input_dir=input_dir, 
        script_path=__file__, 
        output_dir_comment=output_dir_comment, 
        output_dir_manual=output_dir_manual
    )
    
    #################################################### Load Data
    if Analysis_A11_df is None:
        # Try to load from the input directory
        df_path = os.path.join(input_dir, 'Analysis_A11_final_df.pkl')
            
        print(f"\nLoading data from: {df_path}") if log_level >= 1 else None
        
        try:
            Analysis_A11_df = pd.read_pickle(df_path)
        except FileNotFoundError:
            print(f"Error: Could not find data file at {df_path}")
            return output_dir
    else:
        print("\nUsing provided Analysis_A11_df DataFrame") if log_level >= 1 else None
    
    # Get number of images/rows from loaded data
    N_images = len(Analysis_A11_df)
    print(f"Found {N_images} images in DataFrame") if log_level >= 1 else None
    
    # Make sure image_number is valid
    if image_number < 0 or image_number >= N_images:
        print(f"Error: image_number {image_number} is out of range [0, {N_images-1}]")
        return output_dir
    
    #################################################### Create Visualization
    # Setup LaTeX fonts
    setup_latex_fonts()
    
    # Create folder for visualizations
    viz_dir = os.path.join(output_dir, 'CST_Selection_Visualization')
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"\nGenerating CST visualization for image {image_number}") if log_level >= 1 else None
    
    # Extract data for this image
    masks = Analysis_A11_df.loc[image_number, 'masks']
    outlines = Analysis_A11_df.loc[image_number, 'outlines']
    R_SF_nonDim = Analysis_A11_df.loc[image_number, 'R_SF_nonDim']
    R_SF_px = Analysis_A11_df.loc[image_number, 'R_SF_px']
    nonDim_per_px = Analysis_A11_df.loc[image_number, 'nonDim_per_px']
    image_Nx_px = Analysis_A11_df.loc[image_number, 'image_Ny_px']  # These are swapped in the code
    image_Ny_px = Analysis_A11_df.loc[image_number, 'image_Nx_px']  # These are swapped in the code
    
    # Get cell classifications and centroids
    cell_classifications = {}
    cell_centroids_px = {}
    cell_inclusion = {}
    
    # Extract CST classifications from the DataFrame
    if 'CST_classification' in Analysis_A11_df.columns and 'CST_inclusion' in Analysis_A11_df.columns:
        classifications = Analysis_A11_df.loc[image_number, 'CST_classification']
        inclusions = Analysis_A11_df.loc[image_number, 'CST_inclusion']
        centroid_x = Analysis_A11_df.loc[image_number, 'centroid_xIm_distribution_px']
        centroid_y = Analysis_A11_df.loc[image_number, 'centroid_yIm_distribution_px']
        
        cell_ids = np.unique(masks)
        cell_ids = cell_ids[cell_ids > 0]  # Exclude background
        
        for i, cell_id in enumerate(cell_ids):
            if i < len(classifications) and i < len(inclusions) and i < len(centroid_x) and i < len(centroid_y):
                cell_classifications[cell_id] = classifications[i]
                cell_inclusion[cell_id] = inclusions[i]
                cell_centroids_px[cell_id] = (centroid_x[i], centroid_y[i])
    else:
        print("Warning: CST classification data not found in DataFrame. Skipping visualization.")
        return output_dir
    
    # Generate CST boundary for visualization
    CST_Boundary, CST_Boundary_combined = Cubed_Sphere_Tile_Boundary(R_SF_nonDim, N_pts=500)
    
    # Convert non-dimensional boundary to pixel coordinates
    # Step 1: Convert to centered pixel coordinates by dividing by nonDim_per_px
    centered_px_coords = np.zeros_like(CST_Boundary_combined)
    centered_px_coords[0] = CST_Boundary_combined[0] / nonDim_per_px
    centered_px_coords[1] = CST_Boundary_combined[1] / nonDim_per_px
    
    # Step 2: Transform to image coordinates
    CST_Boundary_combined_px = Coordinate_Transform_image_to_centered_Spherical(
        Coordinates=centered_px_coords, 
        centered_to_image=True,
        image_Nx_px=image_Nx_px,
        image_Ny_px=image_Ny_px
    )
    
    # Load the RGB image
    try:
        image_file_path = Analysis_A11_df.loc[image_number, 'image_file_path']
        image_RGB = sk_io.imread(image_file_path)
        if len(image_RGB.shape) > 2 and image_RGB.shape[2] > 3:
            image_RGB = image_RGB[..., :3]  # Take only the first 3 channels if more exist
    except Exception as e:
        print(f"Error loading image: {e}")
        print("Using a blank image instead.")
        image_RGB = np.zeros((image_Ny_px, image_Nx_px, 3), dtype=np.uint8)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    #fig.suptitle(f"CST Selection Visualization - Image {image_number}", fontsize=16)
    
    # Define custom function to handle plotting in each subplot
    def plot_on_axis(ax, plot_type='sanity_check', label=''):
        # Display base RGB image or convert to grayscale
        if Convert_to_grayscale_image:
            image_gray = np.mean(image_RGB, axis=2)  # Convert RGB to grayscale
            ax.imshow(image_gray, cmap='gray')
        else:
            ax.imshow(image_RGB)
        
        # Create colored mask image based on type
        colored_masks = np.zeros((*masks.shape, 4))  # RGBA
        
        # Set coloring based on plot type
        if plot_type == 'sanity_check':
            # Define colors for each classification type
            classification_colors = {
                'all_in_CST_Boundary': 'green',
                'center_in_CST_Boundary': 'blue',
                'center_out_CST_Boundary': 'yellow',
                'all_out_CST_Boundary': 'red'
            }
            
            for cell_id, classification in cell_classifications.items():
                if cell_id <= 0:  # Skip background
                    continue
                
                # Get the color for this classification
                color_name = classification_colors.get(classification, 'gray')
                color_rgba = plt.cm.colors.to_rgba(color_name)
                
                # Set alpha based on classification type
                alpha = 0.4 
                color_rgba = (*color_rgba[:3], alpha)
                
                # Apply color to the mask
                cell_mask = masks == cell_id
                colored_masks[cell_mask] = color_rgba
                
            # Add legend for classifications
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, 
                          label='All in', markeredgecolor='black'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, 
                          label='Center in', markeredgecolor='black'),
                plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='yellow', markersize=10, 
                          label='Center out', markeredgecolor='black'),
                plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, 
                          label='All out', markeredgecolor='black')
            ]
            
            # Plot cell centroids
            for cell_id, (x, y) in cell_centroids_px.items():
                classification = cell_classifications.get(cell_id)
                if classification:
                    color = classification_colors.get(classification, 'white')
                    marker = 'o' if '_in_CST_Boundary' in classification else '^'
                    ax.plot(x, y, marker=marker, color=color, markersize=8, 
                           markeredgecolor='black', markeredgewidth=1)
            
            title = "CST Cell Classification"
            
        else:  # inclusion plot
            # Define colors for inclusion status
            inclusion_colors = {
                True: 'green',  # Included in CST
                False: 'red'    # Excluded from CST
            }
            
            for cell_id, is_included in cell_inclusion.items():
                if cell_id <= 0:  # Skip background
                    continue
                
                # Get the color for this inclusion status
                color_name = inclusion_colors.get(is_included, 'gray')
                color_rgba = plt.cm.colors.to_rgba(color_name)
                
                # Set alpha based on inclusion
                alpha = 0.4 
                color_rgba = (*color_rgba[:3], alpha)
                
                # Apply color to the mask
                cell_mask = masks == cell_id
                colored_masks[cell_mask] = color_rgba
                
            # Add legend for inclusion status
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, 
                          label='Included in CST', markeredgecolor='black'),
                plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, 
                          label='Excluded from CST', markeredgecolor='black'),
            ]
            
            # Plot cell centroids
            for cell_id, (x, y) in cell_centroids_px.items():
                is_included = cell_inclusion.get(cell_id)
                if is_included is not None:
                    color = inclusion_colors.get(is_included, 'white')
                    marker = 'o' if is_included else '^'
                    ax.plot(x, y, marker=marker, color=color, markersize=8, 
                           markeredgecolor='black', markeredgewidth=1)
            
            title = "CST Cell Inclusion"
        
        # Display the colored masks
        ax.imshow(colored_masks)
        
        # Plot outlines 
        outlined = np.ma.masked_where(outlines == 0, outlines)
        ax.imshow(outlined, alpha=0.7, cmap='gray')
        
        # Add Cubed Sphere Tile Boundary
        ax.plot(CST_Boundary_combined_px[0], CST_Boundary_combined_px[1], 'r', 
               linewidth=cst_boundary_linewidth, label='CST Boundary')
        
        # Plot reference circle
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(R_SF_nonDim*np.cos(theta) / nonDim_per_px + image_Nx_px/2, 
               R_SF_nonDim*np.sin(theta) / nonDim_per_px + image_Ny_px/2, 
               'r--', label='Reference Circle', linewidth=ref_circle_linewidth)
        
        # Add text annotation (a) or (b) without a box
        ax.text(text_box_pos[0], text_box_pos[1], label, transform=ax.transAxes, 
                fontsize=text_box_fontsize, fontweight='bold')
        
        # Turn off axis
        ax.axis('off')
        
        # Create proper legend elements based on plot type
        if plot_type == 'sanity_check':
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, 
                          label='All in', markeredgecolor='black'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, 
                          label='Center in', markeredgecolor='black'),
                plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='yellow', markersize=10, 
                          label='Center out', markeredgecolor='black'),
                plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, 
                          label='All out', markeredgecolor='black')
            ]
        else:  # inclusion plot
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, 
                          label='Included in CST', markeredgecolor='black'),
                plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, 
                          label='Excluded from CST', markeredgecolor='black'),
            ]
        
        # Place legend with customizable position and font size
        ax.legend(handles=legend_elements, loc=legend_position, 
                 fontsize=legend_fontsize, frameon=False,
                 bbox_to_anchor=legend_bbox_to_anchor)
    
    # Plot in each subplot
    plot_on_axis(ax1, plot_type='sanity_check', label='(a)')
    plot_on_axis(ax2, plot_type='inclusion', label='(b)')
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_figure:
        # Debug statement
        print(f"Attempting to save figure to directory: {viz_dir}") if log_level >= 1 else None
        
        # Create directories if they don't exist
        try:
            os.makedirs(viz_dir, exist_ok=True)
            
            # Save both PNG and SVG formats with reduced figure size and resolution
            png_path = os.path.join(viz_dir, f'CST_Selection_Visualization_Image_{image_number}.png')
            svg_path = os.path.join(viz_dir, f'CST_Selection_Visualization_Image_{image_number}.svg')
            
            # Adjust figure size for saving to make images smaller
            fig.set_size_inches(figsize[0], figsize[1])
            plt.savefig(png_path, format='png', bbox_inches='tight', dpi=dpi)
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            print(f"Saved visualization to:") if log_level >= 1 else None
            print(f"  - {png_path}") if log_level >= 1 else None
            print(f"  - {svg_path}") if log_level >= 1 else None
        except Exception as e:
            print(f"Error saving figure: {e}")
    
    # Show figure if requested
    if show_plots:
        print(f"Attempting to show plot...") if log_level >= 1 else None
        try:
            plt.show(block=False)  # Use non-blocking mode to prevent freezing
            plt.pause(0.1)  # Small pause to render the figure
            input("Press Enter to continue...") if log_level >= 1 else None  # Wait for user input
        except Exception as e:
            print(f"Error displaying figure: {e}")
    
    # Always close the figure to free memory, but only after showing/saving
    try:
        plt.close(fig)
    except Exception as e:
        print(f"Error closing figure: {e}")
    
    print("\nVisualization complete!") if log_level >= 1 else None
    
    return output_dir

if __name__ == "__main__":
    # Example usage - these values can be modified directly in the code
    input_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250626_1706361"
    image_number = 79  # Change this to the desired image index
    
    output_dir = plot_CST_Selection_visualisation(
        input_dir=input_dir,
        image_number=image_number,
        save_figure=True,
        show_plots=True,
        Convert_to_grayscale_image=True,
        cst_boundary_linewidth=3.0,  # NEW: Thicker CST boundary line
        ref_circle_linewidth=2.0,    # NEW: Thicker reference circle
        log_level=2,  # Increase log level to see more output
        legend_fontsize=20,  # Set legend font size
        legend_position='upper center',  # Place legend position
        legend_bbox_to_anchor=(0.5, 0.05),  # Move legend slightly up from the bottom
        text_box_pos=(0.05, 0.9),  # Position for text annotations
        text_box_fontsize=25  # Font size for text annotations
    )
    
    print(f"Results saved to: {output_dir}")

    # Verify the files were created
    viz_dir = os.path.join(output_dir, 'CST_Selection_Visualization')
    png_path = os.path.join(viz_dir, f'CST_Selection_Visualization_Image_{image_number}.png')
    svg_path = os.path.join(viz_dir, f'CST_Selection_Visualization_Image_{image_number}.svg')
    
    print(f"Checking if files exist:")
    print(f"  PNG exists: {os.path.exists(png_path)}")
    print(f"  SVG exists: {os.path.exists(svg_path)}")
    print(f"  SVG exists: {os.path.exists(svg_path)}")
