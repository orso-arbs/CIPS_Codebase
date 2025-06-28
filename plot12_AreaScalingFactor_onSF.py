import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LogNorm
import skimage.io as sk_io
import Format_1 as F_1
from Spherical_Reconstruction_2 import (
    Cubed_Sphere_Tile_Boundary,
    Coordinate_Transform_image_to_centered_Spherical,
    detJ
)

def setup_latex_fonts(fontsize=11):
    """Configure matplotlib to use LaTeX fonts with specified size"""
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Computer Modern Roman']
    rcParams['text.usetex'] = True
    rcParams['font.size'] = fontsize
    rcParams['axes.titlesize'] = fontsize + 1
    rcParams['axes.labelsize'] = fontsize
    rcParams['xtick.labelsize'] = fontsize - 1
    rcParams['ytick.labelsize'] = fontsize - 1
    rcParams['legend.fontsize'] = fontsize - 1

def plot_area_scaling_factor(ax, R, nonDim_per_px, image_Nx_px, image_Ny_px, 
                            resolution=100, colormap='viridis', alpha=0.7):
    """
    Plot the area scaling factor (detJ) on an existing axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    R : float
        Radius of the sphere (non-dimensional)
    nonDim_per_px : float
        Conversion factor from non-dimensional to pixels
    image_Nx_px : int
        Image width in pixels
    image_Ny_px : int
        Image height in pixels
    resolution : int, optional
        Grid resolution for detJ calculation. Default is 100.
    colormap : str, optional
        Colormap to use. Default is 'viridis'.
    alpha : float, optional
        Transparency for the visualization. Default is 0.7.
        
    Returns
    -------
    dict
        Dictionary containing the mappable for colorbar and min/max values
    """
    # Create grid in centered non-dimensional coordinates
    x = np.linspace(-R, R, resolution)
    z = np.linspace(-R, R, resolution)
    X, Z = np.meshgrid(x, z)
    
    # Calculate detJ on the grid
    valid_points = X**2 + Z**2 < R**2
    J = np.zeros_like(X)
    J[valid_points] = detJ(R, X[valid_points], Z[valid_points])
    J[~valid_points] = np.nan
    
    # Get min and max values for later use in colorbar formatting
    min_val = J[valid_points].min()
    max_val = J[valid_points].max()
    
    # Convert grid to image pixel coordinates
    # Step 1: Convert to centered pixel coordinates
    X_px = X / nonDim_per_px
    Z_px = Z / nonDim_per_px
    
    # Step 2: Transform to image coordinates
    # Create coordinate arrays in the shape needed by Coordinate_Transform
    coords = np.vstack((X_px.flatten(), Z_px.flatten()))
    
    # Transform to image coordinates
    img_coords = Coordinate_Transform_image_to_centered_Spherical(
        Coordinates=coords,
        centered_to_image=True,
        image_Nx_px=image_Nx_px,
        image_Ny_px=image_Ny_px
    )
    
    # Reshape back to grid
    X_img = img_coords[0].reshape(X_px.shape)
    Z_img = img_coords[1].reshape(Z_px.shape)
    
    # Plot detJ as colormesh with logarithmic scale to highlight variations
    norm = LogNorm(vmin=min_val, vmax=max_val)
    pcm = ax.pcolormesh(X_img, Z_img, J, 
                      cmap=colormap, 
                      alpha=alpha,
                      norm=norm)
    
    # Return mappable for colorbar along with min/max values
    return {
        'mappable': pcm, 
        'min_val': min_val, 
        'max_val': max_val
    }

@F_1.ParameterLog(max_size=1024 * 10, log_level=0)
def plot_AreaScalingFactor_onSF(
    # input
    input_dir,
    Analysis_A11_df=None,  # DataFrame from previous processing, or None to load from input_dir
    image_number=100,
    
    # visualization parameters
    zoom_factor=2.0,   # Factor to multiply the flame radius R_SF_px for determining the zoom window size
    alpha_detJ=0.7,    # Transparency for the detJ overlay
    detJ_colormap='viridis',  # Colormap for the detJ visualization
    convert_to_grayscale=True,  # Whether to convert the base image to grayscale
    show_cst_boundary=True,    # Whether to show the CST boundary
    show_ref_circle=True,      # Whether to show the reference circle
    cst_boundary_linewidth=2.0,  # NEW: Line width for CST boundary
    ref_circle_linewidth=1.5,    # NEW: Line width for reference circle
    detJ_resolution=100,       # Resolution for detJ calculation grid
    
    # colorbar parameters
    colorbar_height=0.6,       # Height of the colorbar relative to the figure height
    colorbar_width=0.02,       # Width of the colorbar relative to the figure width
    colorbar_position='right',  # Position of the colorbar ('right', 'bottom', etc.)
    colorbar_pad=0.05,          # Padding between plot and colorbar
    colorbar_label=r"Area Scaling Factor $\det(J)$",  # LaTeX label for colorbar
    colorbar_fontsize=12,       # Font size for colorbar label
    colorbar_tick_fontsize=10,  # NEW: Font size for colorbar tick labels
    
    # arrow/annotation parameters
    show_area_ratio_arrows=True,   # Whether to show arrows pointing to CST boundaries
    arrow_fontsize=10,             # Font size for arrow annotations
    arrow_color='red',             # Color for the arrows
    arrow_textbox_alpha=0.5,       # NEW: Transparency of arrow text boxes
    arrow_textbox_color='white',   # NEW: Background color of arrow text boxes
    arrow_textbox_edgecolor='gray', # NEW: Edge color of arrow text boxes
    arrow_text_format="{:.2f}",    # NEW: Format string for area ratio values
    
    # output and logging
    log_level=2,
    output_dir_manual="",
    output_dir_comment="",
    show_plots=False,
    save_figure=True,
    figsize=(10, 10),
    dpi=300,
    
    # label parameters
    text_box_pos=(0.05, 0.95),  # Position for annotations (x, y) in axes coordinates
    text_box_fontsize=14,       # Font size for annotations
):
    """
    Creates a visualization of the area scaling factor (detJ) from the Cubed Sphere Tile
    transformation, overlaid on top of a flame image.
    
    Parameters
    ----------
    input_dir : str
        Path to the directory containing the Analysis_A11_df.pkl
    Analysis_A11_df : pd.DataFrame or None, optional
        DataFrame from previous processing or None to load from input_dir
    image_number : int, optional
        Index of the image to visualize. Default is 100.
    zoom_factor : float, optional
        Factor to multiply the flame radius R_SF_px for determining the zoom window size. Default is 2.0.
    alpha_detJ : float, optional
        Transparency for the detJ overlay. Default is 0.7.
    detJ_colormap : str, optional
        Colormap for the detJ visualization. Default is 'viridis'.
    convert_to_grayscale : bool, optional
        Whether to convert the base image to grayscale. Default is True.
    show_cst_boundary : bool, optional
        Whether to show the CST boundary. Default is True.
    show_ref_circle : bool, optional
        Whether to show the reference circle. Default is True.
    cst_boundary_linewidth : float, optional
        Line width for CST boundary. Default is 2.0.
    ref_circle_linewidth : float, optional
        Line width for reference circle. Default is 1.5.
    detJ_resolution : int, optional
        Resolution for detJ calculation grid. Default is 100.
    
    # colorbar parameters
    colorbar_height : float, optional
        Height of the colorbar relative to the figure height. Default is 0.6.
    colorbar_width : float, optional
        Width of the colorbar relative to the figure width. Default is 0.02.
    colorbar_position : str, optional
        Position of the colorbar ('right', 'bottom', etc.). Default is 'right'.
    colorbar_pad : float, optional
        Padding between plot and colorbar. Default is 0.05.
    colorbar_label : str, optional
        Label for the colorbar. Default is "Area Scaling Factor det(J)".
    colorbar_fontsize : int, optional
        Font size for colorbar label. Default is 12.
    colorbar_tick_fontsize : int, optional
        Font size for colorbar tick labels. Default is 10.
    
    # arrow/annotation parameters
    show_area_ratio_arrows : bool, optional
        Whether to show arrows pointing to CST boundaries with area ratio values. Default is True.
    arrow_fontsize : int, optional
        Font size for arrow annotations. Default is 10.
    arrow_color : str, optional
        Color for the arrows. Default is 'black'.
    arrow_textbox_alpha : float, optional
        Transparency of arrow text boxes. Default is 0.5.
    arrow_textbox_color : str, optional
        Background color of arrow text boxes. Default is 'white'.
    arrow_textbox_edgecolor : str, optional
        Edge color of arrow text boxes. Default is 'gray'.
    arrow_text_format : str, optional
        Format string for area ratio values. Default is "{:.2f}".
    
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
        Figure dimensions (width, height) in inches. Default is (10, 10).
    dpi : int, optional
        DPI for saved figure. Default is 300.
    text_box_pos : tuple, optional
        Position (x, y) for annotations in axes coordinates. Default is (0.05, 0.95).
    text_box_fontsize : int, optional
        Font size for annotations. Default is 14.
        
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
    
    # Get number of images/rows in the DataFrame
    N_images = len(Analysis_A11_df)
    print(f"Found {N_images} images in DataFrame") if log_level >= 1 else None
    
    # Make sure image_number is valid
    if image_number < 0 or image_number >= N_images:
        print(f"Error: image_number {image_number} is out of range [0, {N_images-1}]")
        return output_dir
    
    #################################################### Setup Visualization
    # Setup LaTeX fonts
    setup_latex_fonts()
    
    # Create folder for visualizations
    viz_dir = os.path.join(output_dir, 'AreaScalingFactor_Visualization')
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"\nGenerating Area Scaling Factor visualization for image {image_number}") if log_level >= 1 else None
    
    # Extract data for this image
    row = Analysis_A11_df.iloc[image_number]
    
    # Extract necessary parameters
    R_SF_nonDim = row['R_SF_nonDim']
    R_SF_px = row['R_SF_px']
    nonDim_per_px = row['nonDim_per_px']
    image_Nx_px = row['image_Ny_px']  # These are swapped in the code
    image_Ny_px = row['image_Nx_px']  # These are swapped in the code
    
    # Try to load the image
    try:
        image_file_path = row['image_file_path']
        print(f"Loading image from: {image_file_path}") if log_level >= 2 else None
        image_RGB = sk_io.imread(image_file_path)
        
        # Handle images with more than 3 channels
        if len(image_RGB.shape) > 2 and image_RGB.shape[2] > 3:
            image_RGB = image_RGB[..., :3]  # Take only the first 3 channels
            
        # Convert to grayscale if requested
        if convert_to_grayscale:
            image_gray = np.mean(image_RGB, axis=2).astype(np.uint8)
    except Exception as e:
        print(f"Error loading image: {e}")
        print("Using a blank image instead.")
        image_RGB = np.zeros((image_Ny_px, image_Nx_px, 3), dtype=np.uint8)
        if convert_to_grayscale:
            image_gray = np.zeros((image_Ny_px, image_Nx_px), dtype=np.uint8)
    
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
    
    #################################################### Create Visualization
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate zoom region to focus on the spherical flame - using same approach as plot6_colortables
    center_x, center_y = image_Nx_px // 2, image_Ny_px // 2
    zoom_half_size = int(R_SF_px * zoom_factor / 2)
    
    # Ensure zoom region is within image bounds
    left = max(0, center_x - zoom_half_size)
    right = min(image_Nx_px, center_x + zoom_half_size)
    top = max(0, center_y - zoom_half_size)
    bottom = min(image_Ny_px, center_y + zoom_half_size)
    
    # Convert to integers to ensure valid array indexing
    left = int(left)
    right = int(right)
    top = int(top)
    bottom = int(bottom)
    
    print(f"Zoom region: left={left}, right={right}, top={top}, bottom={bottom}") if log_level >= 2 else None
    print(f"R_SF_px = {R_SF_px}, zoom window = {zoom_half_size*2} pixels (zoom_factor={zoom_factor})") if log_level >= 2 else None
    
    # Extract the zoom region if we have an image
    if convert_to_grayscale:
        try:
            print(f"Image gray shape: {image_gray.shape}") if log_level >= 2 else None
            zoom_img = image_gray[top:bottom, left:right]
            # Use extent to make sure coordinates match the original image - this is key for overlay alignment
            ax.imshow(zoom_img, cmap='gray', extent=[left, right, bottom, top])
        except Exception as e:
            print(f"Error displaying grayscale image: {e}. Using placeholder.")
            ax.add_patch(plt.Rectangle((left, top), right-left, bottom-top, color='black'))
    else:
        try:
            print(f"Image RGB shape: {image_RGB.shape}") if log_level >= 2 else None
            zoom_img = image_RGB[top:bottom, left:right, :]
            # Use extent to make sure coordinates match the original image
            ax.imshow(zoom_img, extent=[left, right, bottom, top])
        except Exception as e:
            print(f"Error displaying RGB image: {e}. Using placeholder.")
            ax.add_patch(plt.Rectangle((left, top), right-left, bottom-top, color='black'))
    
    # Overlay the detJ visualization using our custom function
    detJ_result = plot_area_scaling_factor(
        ax=ax,
        R=R_SF_nonDim,
        nonDim_per_px=nonDim_per_px,
        image_Nx_px=image_Nx_px,
        image_Ny_px=image_Ny_px,
        resolution=detJ_resolution,
        colormap=detJ_colormap,
        alpha=alpha_detJ
    )
    
    # Add CST boundary
    if show_cst_boundary:
        ax.plot(CST_Boundary_combined_px[0], CST_Boundary_combined_px[1], 'r-', 
               linewidth=cst_boundary_linewidth, label='CST Boundary')
    
    # Add reference circle
    if show_ref_circle:
        theta = np.linspace(0, 2*np.pi, 200)
        circle_x = R_SF_nonDim*np.cos(theta) / nonDim_per_px + image_Nx_px/2
        circle_y = R_SF_nonDim*np.sin(theta) / nonDim_per_px + image_Ny_px/2
        ax.plot(circle_x, circle_y, 'r--', linewidth=ref_circle_linewidth, label='Reference Circle')
    
    # Set the axis limits to the zoom region
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    
    # Add colorbar with custom settings
    if detJ_result and 'mappable' in detJ_result:
        # Create a colorbar with logarithmic scale ticks
        min_val = detJ_result['min_val']
        max_val = detJ_result['max_val']
        
        # Create colorbar with adjusted height and width
        if colorbar_position == 'right':
            # For right-side colorbar, adjust height and width
            cax = plt.axes([
                0.92, (1 - colorbar_height) / 2, colorbar_width, colorbar_height
            ])
            cbar = plt.colorbar(detJ_result['mappable'], cax=cax)
        else:
            # For other positions, use standard colorbar
            cbar = plt.colorbar(detJ_result['mappable'], ax=ax, 
                             location=colorbar_position, pad=colorbar_pad)
        
        # Generate logarithmically spaced ticks (similar to plot_boundary_and_detJ)
        ticks = np.logspace(np.log10(min_val), np.log10(max_val), 5)
        cbar.set_ticks(ticks)
        
        # Format tick labels to show simpler numbers
        cbar.ax.yaxis.set_major_formatter(plt.ScalarFormatter())
        cbar.ax.minorticks_off()  # Turn off minor ticks for cleaner look
        
        # Set the fontsize for tick labels
        cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
        
        # Set the colorbar label with custom fontsize
        cbar.set_label(colorbar_label, fontsize=colorbar_fontsize)
    
    # Add arrows pointing to northern CST boundary with area ratio values
    if show_area_ratio_arrows and CST_Boundary is not None:
        # Get northern boundary points from CST_Boundary
        north_points = CST_Boundary.at[0, 'N']
        
        # Calculate area ratios at specific points on the northern boundary
        # 1. Center of northern boundary
        mid_idx = len(north_points[0]) // 2
        mid_x_nonDim = north_points[0][mid_idx]
        mid_z_nonDim = north_points[1][mid_idx]
        mid_area_ratio = detJ(R_SF_nonDim, mid_x_nonDim, mid_z_nonDim)
        
        # 2. End of northern boundary (right side)
        end_idx = len(north_points[0]) - 1
        end_x_nonDim = north_points[0][end_idx]
        end_z_nonDim = north_points[1][end_idx]
        end_area_ratio = detJ(R_SF_nonDim, end_x_nonDim, end_z_nonDim)
        
        # Convert to image coordinates for plotting
        # Center point
        mid_centered_px = np.array([[mid_x_nonDim / nonDim_per_px], [mid_z_nonDim / nonDim_per_px]])
        mid_img_px = Coordinate_Transform_image_to_centered_Spherical(
            Coordinates=mid_centered_px,
            centered_to_image=True,
            image_Nx_px=image_Nx_px,
            image_Ny_px=image_Ny_px
        )
        mid_x_img, mid_y_img = mid_img_px[0][0], mid_img_px[1][0]
        
        # End point
        end_centered_px = np.array([[end_x_nonDim / nonDim_per_px], [end_z_nonDim / nonDim_per_px]])
        end_img_px = Coordinate_Transform_image_to_centered_Spherical(
            Coordinates=end_centered_px,
            centered_to_image=True,
            image_Nx_px=image_Nx_px,
            image_Ny_px=image_Ny_px
        )
        end_x_img, end_y_img = end_img_px[0][0], end_img_px[1][0]
        
        # Format the area ratio text using the provided format string
        mid_text = arrow_text_format.format(mid_area_ratio)
        end_text = arrow_text_format.format(end_area_ratio)
        
        # Add arrows and text boxes
        # 1. Center point arrow
        # Calculate arrow direction - pointing upward and slightly outward
        mid_arrow_dx = 0
        mid_arrow_dy = 170  # Point upward
        ax.annotate(
            mid_text,
            xy=(mid_x_img, mid_y_img), 
            xytext=(mid_x_img + mid_arrow_dx, mid_y_img + mid_arrow_dy),
            arrowprops=dict(facecolor=arrow_color, shrink=0.05, width=1.5, headwidth=8),
            fontsize=arrow_fontsize,
            bbox=dict(boxstyle="round,pad=0.3", fc=arrow_textbox_color, 
                     ec=arrow_textbox_edgecolor, alpha=arrow_textbox_alpha),
            ha='center'
        )
        
        # 2. End point arrow
        # Calculate arrow direction - pointing outward from the center
        end_arrow_dx = -170  # Point to the right
        end_arrow_dy = 170  # Point slightly upward
        ax.annotate(
            end_text,
            xy=(end_x_img, end_y_img), 
            xytext=(end_x_img + end_arrow_dx, end_y_img + end_arrow_dy),
            arrowprops=dict(facecolor=arrow_color, shrink=0.05, width=1.5, headwidth=8),
            fontsize=arrow_fontsize,
            bbox=dict(boxstyle="round,pad=0.3", fc=arrow_textbox_color, 
                     ec=arrow_textbox_edgecolor, alpha=arrow_textbox_alpha),
            ha='center'
        )
    
    # Remove axes for cleaner look
    ax.set_axis_off()

    plt.tight_layout()
    
    # Save figure if requested
    if save_figure:
        try:
            # Save both PNG and SVG formats
            png_path = os.path.join(viz_dir, f'AreaScalingFactor_Image_{image_number}.png')
            svg_path = os.path.join(viz_dir, f'AreaScalingFactor_Image_{image_number}.svg')
            
            plt.savefig(png_path, format='png', bbox_inches='tight', dpi=dpi)
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            
            print(f"Saved visualization to:") if log_level >= 1 else None
            print(f"  - {png_path}") if log_level >= 1 else None
            print(f"  - {svg_path}") if log_level >= 1 else None
        except Exception as e:
            print(f"Error saving figure: {e}")
    
    # Show figure if requested
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
    
    print("\nVisualization complete!") if log_level >= 1 else None
    
    return output_dir

if __name__ == "__main__":
    # Example usage
    input_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250621_1325208\20250621_1325282\20250621_1621429\20250621_2245499\20250621_2253452"
    image_number = 100  # Change to desired image index
    
    output_dir = plot_AreaScalingFactor_onSF(
        input_dir=input_dir,
        image_number=image_number,
        zoom_factor=3.0,            # Zoom to 3.0 * flame radius
        alpha_detJ=0.7,             # 70% transparency for detJ overlay
        detJ_colormap='hsv',    # Use viridis colormap
        convert_to_grayscale=True,  # Convert image to grayscale
        show_cst_boundary=True,     # Show CST boundary
        show_ref_circle=True,       # Show reference circle
        cst_boundary_linewidth=3.0, # NEW: Thicker CST boundary line
        ref_circle_linewidth=3.0,   # NEW: Thicker reference circle
        detJ_resolution=100,        # Grid resolution for detJ calculation
        
        # Colorbar customization
        colorbar_height=0.6,        # Height of colorbar (60% of figure height)
        colorbar_width=0.06,        # Width of colorbar (6% of figure width)
        colorbar_position='right',  # Position on right side
        colorbar_pad=0.05,          # Padding between plot and colorbar
        colorbar_label=r"Area rescaling factor $dS/dxdz$", # LaTeX label
        colorbar_fontsize=20,       # Font size for colorbar label
        colorbar_tick_fontsize=20,  # Font size for colorbar ticks
        
        # Arrow annotations customization
        show_area_ratio_arrows=True, # Show arrows pointing to area ratio values
        arrow_fontsize=20,          # Font size for arrow annotations
        arrow_color='red',          # Arrow color
        arrow_textbox_alpha=0.7,    # Textbox transparency
        arrow_textbox_color='white', # Textbox background color
        arrow_text_format="{:.2f}", # Format for area ratio values
        
        # General settings
        save_figure=True,           # Save output figures
        show_plots=False,            # Show interactive plot
        figsize=(10, 10),           # Square figure
        dpi=300,                    # High resolution output
        text_box_pos=(0.05, 0.95),  # Text in top-left corner
        text_box_fontsize=14,       # Font size for text
        log_level=2                 # Verbose output
    )
    
    print(f"Results saved to: {output_dir}")
