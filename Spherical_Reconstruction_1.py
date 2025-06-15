import numpy as np
import pandas as pd
import os
import pickle
import Format_1 as F_1
import skimage.io as sk_io
import matplotlib.pyplot as plt
from skimage import color
import time

from cellpose import utils # Needed for utils.diameters

@F_1.ParameterLog(max_size = 1024 * 10, log_level = 0)
def Spherical_Reconstruction_Auxillary_1(
    # input
    input_dir, # Should be the output directory of CP_dimentionalise
    dimentionalised_df=None,  # Add new parameter to accept DataFrame

    # output and logging
    Spherical_Reconstruction_log_level = 1,
    output_dir_manual = "", output_dir_comment = "",
    show_plots = False, # New argument to control plt.show()
    plot_CST_detJ = False, # New argument to control detJ plot generation
    ):


    #################################################### I/O
    # Use the input_dir (output of CP_extract) as the base for the new output dir
    output_dir = F_1.F_out_dir(input_dir, __file__, output_dir_comment = output_dir_comment, output_dir_manual = output_dir_manual) # Format_1 required definition of output directory

    #################################################### Load Extracted Data

    # If DataFrame is provided, use it directly
    if dimentionalised_df is not None:
        print(f"Using provided DataFrame") if Spherical_Reconstruction_log_level >= 1 else None
    else:
        # Load from file as before
        dimentionalised_data_path = os.path.join(input_dir, 'dimentionalised_DataFrame.pkl')
        print(f"\n Loading extracted data from: {dimentionalised_data_path} \n") if Spherical_Reconstruction_log_level >= 1 else None
        try:
            dimentionalised_df = pd.read_pickle(dimentionalised_data_path)
        except FileNotFoundError:
            print(f"Error: Could not find extracted data file at {dimentionalised_data_path}")
            print("Ensure CP_dimentionalise ran successfully and produced 'dimentionalised_DataFrame.pkl' in the specified input directory.")
            return None # Or raise an error
    
    # Get number of images/rows from loaded data
    N_images = len(dimentionalised_df)

    # Use the first image instead of a hardcoded index (40)
    i = 0  # Use the first image for visualization
    if N_images == 0:
        print("No images found in the DataFrame")
        return output_dir

    print(f"Image {i} of {N_images}")
    R = dimentionalised_df.loc[i, 'R_SF_nonDim']
    image_Nx_px = dimentionalised_df.loc[i, 'image_Ny_px']
    image_Ny_px = dimentionalised_df.loc[i, 'image_Nx_px']
    d_T_per_px = dimentionalised_df.loc[i, 'd_T_per_px']
    
    CST_Boundary_nonDim, CST_Boundary_combined_nonDim = Cubed_Sphere_Tile_Boundary(R, N_pts=100)
    CST_Boundary_combined_px = Affine_image_px_and_NonDim(
        Coordinates = CST_Boundary_combined_nonDim,
        nonDim_to_px=True, 
        image_Nx_px=image_Nx_px,
        image_Ny_px=image_Ny_px,
        d_T_per_px=d_T_per_px,
        )

    # Load image without converting to grayscale
    image_RGB = sk_io.imread(dimentionalised_df.loc[i, 'image_file_path'])[..., :3]
    outlines = dimentionalised_df.loc[i, 'outlines']
    masks = dimentionalised_df.loc[i, 'masks']


    if 1==1: # plot image + masks + outlines + CST_Boundary + reference cirle
        # Create figure with white background
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Display base RGB image
        im_rgb = ax.imshow(image_RGB)
        
        # Plot masks with transparent zeros
        masked = np.ma.masked_where(masks == 0, masks)
        im_masks = ax.imshow(masked, alpha=0.9)
        # fig.colorbar(im_masks, label='Mask Values')  # Hide colorbar
        #
        ## Plot outlines with transparent zeros
        outlined = np.ma.masked_where(outlines == 0, outlines)
        im_outlines = ax.imshow(outlined, alpha=1)
        # fig.colorbar(im_outlines, label='Outline Values')  # Hide colorbar
        
        # add Cubed Sphere Tile Boundary
        boundary_plot = ax.plot(CST_Boundary_combined_px[0], CST_Boundary_combined_px[1], 'r', linewidth=2, label='Cubed Sphere Tile Boundary')

        # Plot reference circle
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(R*np.cos(theta) / d_T_per_px + image_Nx_px/2, R*np.sin(theta) / d_T_per_px + image_Ny_px/2, 'r--', 
        label='Reference Circle', linewidth=2)

        ax.set_title(f"Image {i} Cubic Sphere Tile Boundary and Reference Circle", fontsize=16)
        ax.axis('off')
        
        # Add cursor info
        def format_coord(x, y):
            x, y = int(x), int(y)
            if 0 <= x < image_RGB.shape[1] and 0 <= y < image_RGB.shape[0]:
                mask_val = masks[y, x]
                outline_val = outlines[y, x]
                return f'x={x:d}, y={y:d}, mask={mask_val:.2f}, outline={outline_val:.2f}'
            return 'x=, y='
            
        ax.format_coord = format_coord
        
        # Save the figure
        output_filename_svg = f"image_{i}_reconstruction.svg"
        output_path_svg = os.path.join(output_dir, output_filename_svg)
        plt.savefig(output_path_svg, format='svg', bbox_inches='tight')
        print(f"Plot saved to {output_path_svg}") if Spherical_Reconstruction_log_level >= 1 else None

        output_filename_png = f"image_{i}_reconstruction.png"
        output_path_png = os.path.join(output_dir, output_filename_png)
        plt.savefig(output_path_png, format='png', bbox_inches='tight', dpi=300) # Added dpi for PNG
        print(f"Plot saved to {output_path_png}") if Spherical_Reconstruction_log_level >= 1 else None

        if show_plots:
            plt.show()
        plt.close(fig) # Close the figure


    if plot_CST_detJ: # to plot: Cubed Sphere Tile Boundary with det(J)
        R_detJ_plot = 1.0 # Using a fixed R for this illustrative plot
        CST_Boundary_detJ, _ = Cubed_Sphere_Tile_Boundary(R_detJ_plot) # CST_Boundary_combined is not needed here
        fig_detJ, ax_detJ = plot_boundary_and_detJ(CST_Boundary_detJ, R_detJ_plot)
        
        # Save the detJ figure
        output_filename_detJ_svg = "CST_boundary_detJ_plot.svg"
        output_path_detJ_svg = os.path.join(output_dir, output_filename_detJ_svg)
        fig_detJ.savefig(output_path_detJ_svg, format='svg', bbox_inches='tight')
        print(f"DetJ plot saved to {output_path_detJ_svg}") if Spherical_Reconstruction_log_level >= 1 else None

        output_filename_detJ_png = "CST_boundary_detJ_plot.png"
        output_path_detJ_png = os.path.join(output_dir, output_filename_detJ_png)
        fig_detJ.savefig(output_path_detJ_png, format='png', bbox_inches='tight', dpi=300) # Added dpi for PNG
        print(f"DetJ plot saved to {output_path_detJ_png}") if Spherical_Reconstruction_log_level >= 1 else None
        
        if show_plots:
            plt.show()
        plt.close(fig_detJ) # Close the detJ figure

    return output_dir



def detJ(R, x, z):
    """Calculate the Jacobian determinant at point (x,z)"""
    return R/np.sqrt(R**2 - x**2 - z**2)

def Affine_image_px_and_NonDim(Coordinates, px_to_nonDim=False, nonDim_to_px=False, 
                                image_Nx_px=None, image_Ny_px=None, d_T_per_px=None):
    """
    Transforms coordinates between pixel and non-dimensional space.
    
    Args:
        Coordinates (numpy.ndarray): 2xN array of coordinates
        px_to_nonDim (bool): Convert from pixel to non-dimensional
        nonDim_to_px (bool): Convert from non-dimensional to pixel
        image_Nx_px (int): Image width in pixels
        image_Ny_px (int): Image height in pixels
        d_T_per_px (float): Conversion factor
    
    Returns:
        numpy.ndarray: 2xN array of transformed coordinates
    """
    if px_to_nonDim and nonDim_to_px:
        raise ValueError("Cannot set both px_to_nonDim and nonDim_to_px to True")
    
    if not px_to_nonDim and not nonDim_to_px:
        raise ValueError("Must set either px_to_nonDim or nonDim_to_px to True")
        
    if image_Nx_px is None or image_Ny_px is None or d_T_per_px is None:
        raise ValueError("Must provide image_Nx_px, image_Ny_px, and d_T_per_px")

    # Debug: Check input coordinates
    if Coordinates.shape[0] != 2:
        print(f"WARNING: Input coordinates shape: {Coordinates.shape}")
        print(f"First few coordinates: {Coordinates[:5] if len(Coordinates) > 5 else Coordinates}")
    
    Transformed_Coordinates = np.zeros_like(Coordinates)
    
    if px_to_nonDim:
        x_px = Coordinates[0]
        y_px = Coordinates[1]
        Transformed_Coordinates[0] = ((x_px + 1/2) - image_Nx_px/2) * d_T_per_px # x coord
        Transformed_Coordinates[1] = (image_Ny_px/2 - (y_px + 1/2)) * d_T_per_px # z coord
        
    if nonDim_to_px:
        x_nonDim = Coordinates[0]
        z_nonDim = Coordinates[1]
        Transformed_Coordinates[0] = x_nonDim / d_T_per_px + image_Nx_px/2 - 1/2 # x coord 
        Transformed_Coordinates[1] = image_Ny_px/2 - z_nonDim / d_T_per_px - 1/2 # y coord
    
    # Debug: Check output coordinates
    if Transformed_Coordinates.shape[0] != 2:
        print(f"WARNING: Output coordinates shape: {Transformed_Coordinates.shape}")
        print(f"First few transformed coordinates: {Transformed_Coordinates[:5] if len(Transformed_Coordinates) > 5 else Transformed_Coordinates}")

    return Transformed_Coordinates

def Cubed_Sphere_Tile_Boundary(R, N_pts=500):
    """
    Calculates the boundary points of a cubed sphere tile.
    
    Args:
        R (float): Radius of the sphere
        N_pts (int): Number of points for discretization (default=100)
    
    Returns:
        pd.DataFrame: DataFrame containing the boundary points (N,W,S,E)
        Each column contains a 2xN_pts array with x and z coordinates centered in the sphere
    """
    # Calculate L (side length of the cube)
    L = R / np.sqrt(3)
    
    # Create empty DataFrame to store the boundaries
    CST_Boundary = pd.DataFrame(columns=['N', 'W', 'S', 'E'])
    
    # North boundary
    x_N = np.linspace(-L, L, N_pts)
    z_N = np.sqrt((R**2 - x_N**2) / 2)
    CST_Boundary.at[0, 'N'] = np.vstack((x_N, z_N))

    # South boundary
    x_S = np.linspace(L, -L, N_pts)
    z_S = -np.sqrt((R**2 - x_S**2) / 2)
    CST_Boundary.at[0, 'S'] = np.vstack((x_S, z_S))

    # East boundary
    z_E = np.linspace(L, -L, N_pts)
    x_E = np.sqrt((R**2 - z_E**2) / 2)
    CST_Boundary.at[0, 'E'] = np.vstack((x_E, z_E))
    
    # West boundary
    z_W = np.linspace(-L, L, N_pts)
    x_W = -np.sqrt((R**2 - z_W**2) / 2)
    CST_Boundary.at[0, 'W'] = np.vstack((x_W, z_W))
    
    # Concatenate all boundaries into one array
    boundaries = []
    for b in ['N', 'E', 'S', 'W']:  # Order matters for continuous path
        boundaries.append(CST_Boundary.at[0, b])
    CST_Boundary_combined = np.hstack(boundaries)
    CST_Boundary_combined = np.hstack([CST_Boundary_combined, CST_Boundary_combined[:, [0]]]) # Add first point again to close the loop
    
    return CST_Boundary, CST_Boundary_combined


def plot_boundary_and_detJ(CST_Boundary, R, plot_resolution=200, cmap='viridis', alpha=1.0):
    """
    Plots the boundaries of the cubed sphere tile with detJ pseudocolor field.
    
    Args:
        CST_Boundary (pd.DataFrame): DataFrame containing the boundary points
        R (float): Radius of the sphere for reference circle
        plot_resolution (int): Number of points for the pseudocolor field
        cmap (str): Colormap for the pseudocolor field
        alpha (float): Transparency of the pseudocolor field

    Run with:
    R = 1.0
    boundary = Cubed_Sphere_Tile_Boundary(R, N_pts=100)
    plot_boundary_and_detJ(boundary, R)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    
    # Create figure and axis with equal aspect ratio
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, aspect='equal')
    
    # Create equispaced grid for pseudocolor plot
    x = np.linspace(-R, R, plot_resolution)
    z = np.linspace(-R, R, plot_resolution)
    X, Z = np.meshgrid(x, z)
    
    # Calculate detJ on the grid
    valid_points = X**2 + Z**2 < R**2
    J = np.zeros_like(X)
    J[valid_points] = detJ(R, X[valid_points], Z[valid_points])
    J[~valid_points] = np.nan

    # Create colorbar with more levels
    levels = np.logspace(np.log10(J[valid_points].min()), 
                        np.log10(J[valid_points].max()), 
                        50)  # Increase number of levels
    
    # Plot pseudocolor field with specified levels
    pcm = ax.pcolormesh(Z, X, J, 
                        cmap=cmap, 
                        alpha=alpha,
                        norm=LogNorm(vmin=J[valid_points].min(), 
                        vmax=J[valid_points].max()))
    
    # Add colorbar with single set of ticks
    min_val = J[valid_points].min()
    max_val = J[valid_points].max()
    ticks = np.logspace(np.log10(min_val), np.log10(max_val), 10)
    cb = fig.colorbar(pcm, ax=ax, label='det(J)', ticks=ticks)
    cb.ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    # Remove the minor ticks to avoid overlapping
    cb.ax.minorticks_off()
    
    # Plot boundaries with improved visibility
    boundaries = ['N', 'S', 'W', 'E']
    boundaries = ['N', 'S', 'W', 'E']
    colors = ['red', 'blue', 'green', 'orange']
    labels = ['North', 'South', 'West', 'East']
    
    for boundary, color, label in zip(boundaries, colors, labels):
        points = CST_Boundary.at[0, boundary]
        ax.plot(points[0], points[1], color=color, label=label, 
                linewidth=2.5, linestyle='-')
    
    # Calculate and plot extrema points
    extrema_points = []
    for boundary, points in CST_Boundary.iloc[0].items():
        # Get middle and end points
        mid_idx = len(points[1]) // 2
        x_mid, z_mid = points[0][mid_idx], points[1][mid_idx]
        x_end, z_end = points[0][-1], points[1][-1]
        
        # Calculate detJ values
        detJ_mid = detJ(R, x_mid, z_mid)
        detJ_end = detJ(R, x_end, z_end)
        
        # Plot markers
        ax.plot(z_mid, x_mid, 'k+', markersize=10, linewidth=2)
        ax.plot(z_end, x_end, 'k+', markersize=10, linewidth=2)
        
        # Store points and values for North boundary arrow annotations
        if boundary == 'N':
            # Arrow to middle point (minimum)
            ax.annotate(f'{detJ_mid:.2f}',
                        xy=(x_mid, z_mid), xycoords='data',
                        xytext=(x_mid, z_mid - R*0.2), textcoords='data',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
                        bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8))
            
            # Arrow to end point (maximum)
            ax.annotate(f'{detJ_end:.2f}',
                        xy=(x_end, z_end), xycoords='data',
                        xytext=(x_end - R*0.2, z_end - R*0.2), textcoords='data',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"),
                        bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8))

    # Plot reference circle
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(R*np.cos(theta), R*np.sin(theta), 'k--', 
            label='Reference Circle', linewidth=1.5)
    
    # Improve grid and labels
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('z', fontsize=12)
    ax.set_title('Cubed Sphere Tile Boundary with det(J)', 
                fontsize=14, pad=20)
    
    # Set equal axis limits
    ax.set_xlim(-R*1.1, R*1.1)
    ax.set_ylim(-R*1.1, R*1.1)
    
    # Improve legend
    #ax.legend(bbox_to_anchor=(0.7, 1.0), fontsize=10)
    
    # Adjust layout to prevent cutting off
    plt.tight_layout()
    
    #plt.show()
    return fig, ax



def point_in_CST_check(x_nonDim, z_nonDim, R_SF_nonDim):
    """
    Determines if a point is inside the CST (Cubed Sphere Tile) boundary using analytical geometry.
    
    By considering the CST symmetry, only the first quadrant in x-z is checked 
    by taking the absolute values of coordinates.
    
    Args:
        x_nonDim: x-coordinate in non-dimensional space
        z_nonDim: z-coordinate in non-dimensional space
        R_SF_nonDim: Radius of the sphere in non-dimensional units
        
    Returns:
        bool: True if point is inside the CST, False otherwise
    """
    # Calculate the CST Boundary extrema
    L = R_SF_nonDim / np.sqrt(3) 
    
    min_CST_Boundary = L
    max_CST_Boundary = np.sqrt(R_SF_nonDim**2 / 2)
    
    # Take absolute values to exploit symmetry
    x_abs = abs(x_nonDim)
    z_abs = abs(z_nonDim)
    
    if x_abs > max_CST_Boundary or z_abs > max_CST_Boundary:
        return False
    
    if x_abs > min_CST_Boundary and z_abs > min_CST_Boundary:
        return False

    if x_abs <= min_CST_Boundary and z_abs <= min_CST_Boundary:
        return True
    
    if x_abs <= max_CST_Boundary and z_abs <= max_CST_Boundary:
        if x_abs >= z_abs and x_abs <= np.sqrt((R_SF_nonDim**2 - z_abs**2)/2):
            return True
        if x_abs >= z_abs and x_abs > np.sqrt((R_SF_nonDim**2 - z_abs**2)/2):
            return False
        
        if z_abs >= x_abs and z_abs <= np.sqrt((R_SF_nonDim**2 - x_abs**2)/2):
            return True
        if z_abs >= x_abs and z_abs > np.sqrt((R_SF_nonDim**2 - x_abs**2)/2):
            return False

    # If we reach here, point is not classified correctly
    print(f"Warning: ({x_nonDim}, {z_nonDim}) point_in_CST_check unclassified. max_CST_Boundary={max_CST_Boundary}, min_CST_Boundary={min_CST_Boundary}")
    
    return False

def CST_selection(SRec_df, output_dir, Spherical_Reconstruction_log_level=2, show_plots=False):
    """
    Classifies cells based on their position relative to the CST boundary.
    
    Args:
        SRec_df: DataFrame containing spherical reconstruction data
        output_dir: Directory to save plots
        Spherical_Reconstruction_log_level: Verbosity level
        show_plots: Whether to display plots
    
    Returns:
        DataFrame: Updated DataFrame with CST classification columns
    """
    # Initialize new columns for the CST selection data
    new_columns = [
        'CST_classification',
        'A_cell_distribution_CST_px2',
        'A_cell_distribution_CST_nonDim2',
        'd_cell_distribution_CST_px',
        'd_cell_distribution_CST_nonDim',
        'A_cell_SRec_distribution_CST_nonDim2',
        'A_cell_SRec_distribution_CST_px2',
        'd_cell_SRec_distribution_CST_nonDim',
        'd_cell_SRec_distribution_CST_px',
        'centroid_x_distribution_CST_px',
        'centroid_y_distribution_CST_px',
        'centroid_x_distribution_CST_nonDim',
        'centroid_z_distribution_CST_nonDim'
    ]
    
    for col in new_columns:
        SRec_df[col] = None
    
    # Create directory for classification plots
    plots_dir = os.path.join(output_dir, 'CST_classification_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Process each image
    N_images = len(SRec_df)
    for i in range(N_images):
        if Spherical_Reconstruction_log_level >= 1:
            print(f"\rProcessing CST selection for image {i+1}/{N_images}", end='', flush=True)
        
        # Extract data for this image
        masks = SRec_df.loc[i, 'masks']
        R_SF_nonDim = SRec_df.loc[i, 'R_SF_nonDim']
        R_SF_px = SRec_df.loc[i, 'R_SF_px']
        image_Nx_px = SRec_df.loc[i, 'image_Ny_px']  # These are swapped in the code
        image_Ny_px = SRec_df.loc[i, 'image_Nx_px']  # These are swapped in the code
        d_T_per_px = SRec_df.loc[i, 'd_T_per_px']
        
        # Get cell IDs
        cell_ids = np.unique(masks)
        cell_ids = cell_ids[cell_ids > 0]  # Exclude background
        
        # Generate CST boundary for visualization purposes only
        CST_Boundary, CST_Boundary_combined = Cubed_Sphere_Tile_Boundary(R_SF_nonDim, N_pts=500)
        CST_Boundary_combined_px = Affine_image_px_and_NonDim(
            Coordinates=CST_Boundary_combined,
            nonDim_to_px=True,
            image_Nx_px=image_Nx_px,
            image_Ny_px=image_Ny_px,
            d_T_per_px=d_T_per_px,
        )
        
        # Extract cell properties for this image. The diameter Pixel based values are already output by Cellpos. They are recalculated here via the areas to have asserted control of how they are generated.
        A_cell_distribution_px2 = SRec_df.loc[i, 'A_cell_distribution_px2']
        A_cell_distribution_nonDim2 = SRec_df.loc[i, 'A_cell_distribution_nonDim2']
        d_cell_distribution_px = SRec_df.loc[i, 'd_cell_distribution_px']
        d_cell_distribution_nonDim = SRec_df.loc[i, 'd_cell_distribution_nonDim']
        A_cell_SRec_distribution_nonDim2 = SRec_df.loc[i, 'A_cell_SRec_distribution_nonDim2']
        A_cell_SRec_distribution_px2 = SRec_df.loc[i, 'A_cell_SRec_distribution_px2']
        d_cell_SRec_distribution_nonDim = SRec_df.loc[i, 'd_cell_SRec_distribution_nonDim']
        d_cell_SRec_distribution_px = SRec_df.loc[i, 'd_cell_SRec_distribution_px']
        centroid_x_distribution_px = SRec_df.loc[i, 'centroid_x_distribution_px']
        centroid_y_distribution_px = SRec_df.loc[i, 'centroid_y_distribution_px']
        centroid_x_distribution_nonDim = SRec_df.loc[i, 'centroid_x_distribution_nonDim']
        centroid_z_distribution_nonDim = SRec_df.loc[i, 'centroid_z_distribution_nonDim']
        
        # Lists for filtered cells
        CST_classification = []
        A_cell_distribution_CST_px2 = []
        A_cell_distribution_CST_nonDim2 = []
        d_cell_distribution_CST_px = []
        d_cell_distribution_CST_nonDim = []
        A_cell_SRec_distribution_CST_nonDim2 = []
        A_cell_SRec_distribution_CST_px2 = []
        d_cell_SRec_distribution_CST_nonDim = []
        d_cell_SRec_distribution_CST_px = []
        centroid_x_distribution_CST_px = []
        centroid_y_distribution_CST_px = []
        centroid_x_distribution_CST_nonDim = []
        centroid_z_distribution_CST_nonDim = []
        
        # Dictionary for cell classifications and centroids for plotting
        cell_classifications = {}
        cell_centroids_px = {}
        
        # Process each cell
        for idx, cell_id in enumerate(cell_ids):
            print(f"\rProcessing CST selection for image {i+1}/{N_images} - cell {idx+1}/{len(cell_ids)}", end='', flush=True) if Spherical_Reconstruction_log_level >= 2 else None
            # Get cell mask and centroid
            cell_mask = masks == cell_id
            y_coords, x_coords = np.where(cell_mask)
            if len(y_coords) == 0:
                continue
                
            # Get centroid in pixel and non-dimensional coordinates
            centroid_x_px = centroid_x_distribution_px[idx]
            centroid_y_px = centroid_y_distribution_px[idx]
            centroid_x_nonDim = centroid_x_distribution_nonDim[idx]
            centroid_z_nonDim = centroid_z_distribution_nonDim[idx]
                        
            # Check if centroid is inside CST boundary using the analytical function
            centroid_in_CST = point_in_CST_check(
                x_nonDim=centroid_x_nonDim,
                z_nonDim=centroid_z_nonDim,
                R_SF_nonDim=R_SF_nonDim
            )
                        
            ######## Check if all mask points are inside CST boundary
            
            # Ensure mask coordinates are correctly processed
            
            # Better debug information about the coordinates
            if Spherical_Reconstruction_log_level >= 4:
                print(f"\nCell {cell_id} has {len(x_coords)} pixels")
                if len(x_coords) > 0:
                    print(f"Sample pixel coordinates (x, y): ({x_coords[0]}, {y_coords[0]})")
            
            # Convert mask points to non-dimensional coordinates
            # Fix: Ensure proper stacking of coordinates
            mask_coords_px = np.vstack((x_coords, y_coords)).astype(float)
            
            # Debug to verify mask_coords_px
            if Spherical_Reconstruction_log_level >= 4:
                print(f"mask_coords_px shape: {mask_coords_px.shape}")
                if mask_coords_px.shape[1] > 0:
                    print(f"First few mask_coords_px: {mask_coords_px[:, :5] if mask_coords_px.shape[1] > 5 else mask_coords_px}")
            

            mask_coords_nonDim = Affine_image_px_and_NonDim(
                Coordinates=mask_coords_px,	
                px_to_nonDim=True,
                image_Nx_px=masks.shape[1],
                image_Ny_px=masks.shape[0],
                d_T_per_px=d_T_per_px
            )

            # Debug to verify mask_coords_nonDim
            if Spherical_Reconstruction_log_level >= 4:
                print(f"mask_coords_nonDim shape: {mask_coords_nonDim.shape}")
                if mask_coords_nonDim.shape[1] > 0:
                    print(f"First few mask_coords_nonDim: {mask_coords_nonDim[:, :5] if mask_coords_nonDim.shape[1] > 5 else mask_coords_nonDim}")
            
            # Check each mask point against the CST boundary
            all_points_in = True
            any_point_in = False
            n_points_checked = 0
            n_points_in_CST = 0
            
            # Check a sample of points for optimization (especially for large masks)
            max_points_to_check = 1000  # Limit for very large masks
            check_stride = max(1, len(mask_coords_nonDim[0]) // max_points_to_check)
            
            # First check if any point is in and if all points might be in
            for j in range(0, len(mask_coords_nonDim[0]), check_stride):
                px = mask_coords_nonDim[0][j]
                pz = mask_coords_nonDim[1][j]
                
                if Spherical_Reconstruction_log_level >= 4 and n_points_checked < 5:
                    print(f"Testing point #{n_points_checked}: ({px:.4f}, {pz:.4f})")
                    
                n_points_checked += 1
                point_in = point_in_CST_check(
                    x_nonDim=px,
                    z_nonDim=pz,
                    R_SF_nonDim=R_SF_nonDim,
                )
                
                if point_in:
                    n_points_in_CST += 1
                    any_point_in = True
                else:
                    all_points_in = False
                
            
            # Classify cell
            if centroid_in_CST and all_points_in:
                classification = "all_in_CST"
            elif centroid_in_CST and any_point_in:
                classification = "center_in_CST"
            elif not centroid_in_CST and any_point_in:
                classification = "center_out_CST"
            else:
                classification = "all_out_CST"
            
            # Additional debug check if we got all_out_CST but have points in CST
            if classification == "all_out_CST" and n_points_in_CST > 0:
                print(f"WARNING: Cell {cell_id} classified as all_out_CST but has {n_points_in_CST} points in CST")
            
            # Store classification for plotting
            cell_classifications[cell_id] = classification
            cell_centroids_px[cell_id] = (centroid_x_px, centroid_y_px)
            print(f"\rClassification: {classification} with centroid_in_CST: {centroid_in_CST} and any_point_in: {any_point_in}") if Spherical_Reconstruction_log_level >= 2 else None
            
            # Only include cells that are not completely outside the CST
            if classification != "all_out_CST":
                CST_classification.append(classification)
                A_cell_distribution_CST_px2.append(A_cell_distribution_px2[idx])
                A_cell_distribution_CST_nonDim2.append(A_cell_distribution_nonDim2[idx])
                d_cell_distribution_CST_px.append(d_cell_distribution_px[idx])
                d_cell_distribution_CST_nonDim.append(d_cell_distribution_nonDim[idx])
                A_cell_SRec_distribution_CST_nonDim2.append(A_cell_SRec_distribution_nonDim2[idx])
                A_cell_SRec_distribution_CST_px2.append(A_cell_SRec_distribution_px2[idx])
                d_cell_SRec_distribution_CST_nonDim.append(d_cell_SRec_distribution_nonDim[idx])
                d_cell_SRec_distribution_CST_px.append(d_cell_SRec_distribution_px[idx])
                centroid_x_distribution_CST_px.append(centroid_x_px)
                centroid_y_distribution_CST_px.append(centroid_y_px)
                centroid_x_distribution_CST_nonDim.append(centroid_x_nonDim)
                centroid_z_distribution_CST_nonDim.append(centroid_z_nonDim)
        
        # Store the filtered lists in the DataFrame
        SRec_df.at[i, 'CST_classification'] = np.array(CST_classification)
        SRec_df.at[i, 'A_cell_distribution_CST_px2'] = np.array(A_cell_distribution_CST_px2)
        SRec_df.at[i, 'A_cell_distribution_CST_nonDim2'] = np.array(A_cell_distribution_CST_nonDim2)
        SRec_df.at[i, 'd_cell_distribution_CST_px'] = np.array(d_cell_distribution_CST_px)
        SRec_df.at[i, 'd_cell_distribution_CST_nonDim'] = np.array(d_cell_distribution_CST_nonDim)
        SRec_df.at[i, 'A_cell_SRec_distribution_CST_nonDim2'] = np.array(A_cell_SRec_distribution_CST_nonDim2)
        SRec_df.at[i, 'A_cell_SRec_distribution_CST_px2'] = np.array(A_cell_SRec_distribution_CST_px2)
        SRec_df.at[i, 'd_cell_SRec_distribution_CST_nonDim'] = np.array(d_cell_SRec_distribution_CST_nonDim)
        SRec_df.at[i, 'd_cell_SRec_distribution_CST_px'] = np.array(d_cell_SRec_distribution_CST_px)
        SRec_df.at[i, 'centroid_x_distribution_CST_px'] = np.array(centroid_x_distribution_CST_px)
        SRec_df.at[i, 'centroid_y_distribution_CST_px'] = np.array(centroid_y_distribution_CST_px)
        SRec_df.at[i, 'centroid_x_distribution_CST_nonDim'] = np.array(centroid_x_distribution_CST_nonDim)
        SRec_df.at[i, 'centroid_z_distribution_CST_nonDim'] = np.array(centroid_z_distribution_CST_nonDim)
        
        # Generate classification plot
        print("Generating classification plot...") if Spherical_Reconstruction_log_level >= 2 else None
        if len(cell_classifications) > 0:  # Only plot if we have cells to classify
            try:
                # Load the image for plotting
                image_RGB = sk_io.imread(SRec_df.loc[i, 'image_file_path'])[..., :3]
                outlines = SRec_df.loc[i, 'outlines']
                
                # Create and save the classification plot
                output_path = os.path.join(plots_dir, f"image_{i}_CST_classification.png")
                plot_CST_selection_sanity_check(
                    image_RGB=image_RGB,
                    masks=masks,
                    outlines=outlines,
                    CST_Boundary_combined_px=CST_Boundary_combined_px,
                    R=R_SF_nonDim,
                    d_T_per_px=d_T_per_px,
                    image_Nx_px=image_Nx_px,
                    image_Ny_px=image_Ny_px,
                    cell_classifications=cell_classifications,
                    cell_centroids_px=cell_centroids_px,
                    output_path=output_path,
                    show_plot=show_plots,
                    title_prefix=f"Image {i+1}"
                )
            except Exception as e:
                print(f"\nError generating plot for image {i}: {e}")
    
    # Print newline after progress updates
    if Spherical_Reconstruction_log_level >= 1:
        print()
        
    return SRec_df

def plot_CST_selection_sanity_check(image_RGB, masks, outlines, CST_Boundary_combined_px, R, d_T_per_px, 
                                   image_Nx_px, image_Ny_px, cell_classifications, cell_centroids_px,
                                   output_path, show_plot=False, title_prefix="Image", 
                                   Convert_to_grayscale_image=True):
    """
    Create a visualization of cell classifications based on the CST boundary.
    
    Args:
        image_RGB: The RGB image
        masks: The cell masks
        outlines: The cell outlines
        CST_Boundary_combined_px: The CST boundary coordinates in pixel space
        R: Radius of the sphere
        d_T_per_px: Conversion factor from pixels to non-dimensional units
        image_Nx_px: Image width in pixels
        image_Ny_px: Image height in pixels
        cell_classifications: Dict mapping cell IDs to classification strings
        cell_centroids_px: Dict mapping cell IDs to (x, y) centroids in pixel space
        output_path: Path to save the figure
        show_plot: Whether to display the plot
        title_prefix: Prefix for the plot title
    
    Returns:
        None
    """
    # Define colors for each classification type
    classification_colors = {
        'all_in_CST': 'green',
        'center_in_CST': 'blue',
        'center_out_CST': 'yellow',
        'all_out_CST': 'red'
    }
    
    # Create figure with white background
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Display base RGB image or convert to grayscale
    if Convert_to_grayscale_image:
        image_gray = np.mean(image_RGB, axis=2)  # Convert RGB to grayscale
        ax.imshow(image_gray, cmap='gray')
    else:
        ax.imshow(image_RGB)
    
    # Plot masks with color coding by classification
    # Create a colored mask image
    colored_masks = np.zeros((*masks.shape, 4))  # RGBA
    
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
    
    # Display the colored masks
    ax.imshow(colored_masks)
    
    # Plot outlines 
    outlined = np.ma.masked_where(outlines == 0, outlines)
    ax.imshow(outlined, alpha=0.7, cmap='gray')
    
    # Add Cubed Sphere Tile Boundary
    ax.plot(CST_Boundary_combined_px[0], CST_Boundary_combined_px[1], 'r', linewidth=2, label='CST Boundary')
    
    # Plot reference circle
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(R*np.cos(theta) / d_T_per_px + image_Nx_px/2, R*np.sin(theta) / d_T_per_px + image_Ny_px/2, 'r--', 
            label='Reference Circle', linewidth=1.5)
    
    # Plot cell centroids
    for cell_id, (x, y) in cell_centroids_px.items():
        classification = cell_classifications.get(cell_id)
        if classification:
            color = classification_colors.get(classification, 'white')
            marker = 'o' if 'in_CST' in classification else 'x'
            ax.plot(x, y, marker=marker, color=color, markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    # Add legend for classifications
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='All in CST'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Center in CST'),
        plt.Line2D([0], [0], marker='x', color='yellow', markersize=10, label='Center out CST'),
        plt.Line2D([0], [0], marker='x', color='red', markersize=10, label='All out CST')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Set title and turn off axis
    ax.set_title(f"{title_prefix} - CST Cell Classification", fontsize=16)
    ax.axis('off')
    
    # Save the figure
    plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
    print(f"CST classification plot saved to {output_path}")
    
    if show_plot:
        plt.show()
    plt.close(fig)

def Spherical_Reconstruction_1(
    # input
    input_dir, # Should be the output directory of CP_dimentionalise

    # output and logging
    Spherical_Reconstruction_log_level = 2,
    output_dir_manual = "", output_dir_comment = "",
    show_plots = False,
    plot_CST_detJ = False,
    plot_diameter_sanity_check_imagewise = False,  # Control per-image diameter plots
    plot_diameter_sanity_check_summary = True,    # Control summary diameter plot
    ):
    """
    Performs spherical reconstruction on segmented cell data.
    
    This function processes the output from CP_dimentionalise, applies spherical
    reconstruction to each cell, and calculates various cell properties in both
    pixel and non-dimensional space.
    
    Parameters
    ----------
    input_dir : str
        Path to the directory containing the dimentionalised_DataFrame.pkl
        generated by the CP_dimentionalise function.
    Spherical_Reconstruction_log_level : int, optional
        Controls the verbosity of logging messages. Defaults to 2.
    output_dir_manual : str, optional
        If provided, specifies the exact output directory path. Defaults to "".
    output_dir_comment : str, optional
        A comment to append to the default output directory name. Defaults to "".
    show_plots : bool, optional
        Whether to display plots during processing. Defaults to True.
    plot_CST_detJ : bool, optional
        Whether to plot the Cubed Sphere Tile boundary with det(J) visualization.
        Defaults to False.
    plot_diameter_sanity_check_imagewise : bool, optional
        Whether to generate individual sanity check plots comparing original vs reconstructed 
        diameters for each image. Plots will be saved in 'diameter_sanity_check_plots' subdirectory.
        Defaults to True.
    plot_diameter_sanity_check_summary : bool, optional
        Whether to generate a summary plot comparing original vs reconstructed diameters
        across all images. Plot will be saved in 'diameter_sanity_check_plots' subdirectory.
        Defaults to True.
    
    Returns
    -------
    output_dir : str
        The path to the directory where the reconstructed data is saved.
    """

    #################################################### I/O
    # Use the input_dir (output of CP_extract) as the base for the new output dir
    output_dir = F_1.F_out_dir(input_dir, __file__, output_dir_comment = output_dir_comment, output_dir_manual = output_dir_manual) # Format_1 required definition of output directory

    #################################################### Load Extracted Data

    dimentionalised_data_path = os.path.join(input_dir, 'dimentionalised_DataFrame.pkl')
    print(f"\n Loading extracted data from: {dimentionalised_data_path} \n") if Spherical_Reconstruction_log_level >= 1 else None
    try:
        dimentionalised_df = pd.read_pickle(dimentionalised_data_path)
    except FileNotFoundError:
        print(f"Error: Could not find extracted data file at {dimentionalised_data_path}")
        print("Ensure CP_dimentionalise ran successfully and produced 'dimentionalised_DataFrame.pkl' in the specified input directory.")
        return None # Or raise an error
    
    # Get number of images/rows from loaded data
    N_images = len(dimentionalised_df)
    print(f"Loaded data for {N_images} images") if Spherical_Reconstruction_log_level >= 1 else None
    
    #################################################### Perform Spherical Reconstruction

    # Create a copy of the dimensionalized DataFrame to add our new columns
    SRec_df = dimentionalised_df.copy()
    
    # Initialize lists for all new columns we'll add
    new_columns = [
        'A_cell_distribution_px2', 'A_cell_distribution_nonDim2',  # Added new column
        'd_cell_distribution_px', 'd_cell_distribution_nonDim', 
        'A_cell_SRec_distribution_nonDim2', 'A_cell_SRec_distribution_px2', 
        'd_cell_SRec_distribution_nonDim', 'd_cell_SRec_distribution_px', 
        'centroid_x_distribution_px', 'centroid_y_distribution_px', 
        'centroid_x_distribution_nonDim', 'centroid_z_distribution_nonDim'
    ]
    
    for col in new_columns:
        SRec_df[col] = None
    
    # Process each image
    for i in range(N_images):
        # Extract required data for this image
        diameter_distribution_px = dimentionalised_df.loc[i, 'diameter_distribution_px']
        masks = dimentionalised_df.loc[i, 'masks']
        R_SF_px = dimentionalised_df.loc[i, 'R_SF_px']
        R_SF_nonDim = dimentionalised_df.loc[i, 'R_SF_nonDim']
        d_T_per_px = dimentionalised_df.loc[i, 'd_T_per_px']
        
        # Get unique cell IDs (excluding background = 0)
        cell_ids = np.unique(masks)
        cell_ids = cell_ids[cell_ids > 0]  # Exclude background (ID=0)
        
        # Print progress update that overwrites itself
        if Spherical_Reconstruction_log_level >= 1:
            print(f"\rProcessing image {i+1}/{N_images} ({len(cell_ids)} cells found)", end='', flush=True)

        # If no cells found, continue to next image
        if len(cell_ids) == 0:
            continue

        # Initialize lists for each cell property
        A_cell_distribution_px2 = []
        A_cell_distribution_nonDim2 = []
        d_cell_distribution_px = []
        d_cell_distribution_nonDim = []
        A_cell_SRec_distribution_nonDim2 = []
        A_cell_SRec_distribution_px2 = []
        d_cell_SRec_distribution_nonDim = []
        d_cell_SRec_distribution_px = []
        centroid_x_distribution_px = []
        centroid_y_distribution_px = []
        centroid_x_distribution_nonDim = []
        centroid_z_distribution_nonDim = []
        
        # Process each cell in the image
        for cell_id in cell_ids:
            # Create a binary mask for this specific cell
            cell_mask = masks == cell_id
            
            # Calculate centroid of the cell
            y_coords, x_coords = np.where(cell_mask)
            if len(y_coords) == 0:
                continue
                
            # 8. Find cell centroid in pixel coordinates
            centroid_y_px = np.mean(y_coords)
            centroid_x_px = np.mean(x_coords)
            
            # Find each pixel in non-dimensional space
            cell_coords_px = np.vstack((x_coords, y_coords))
            cell_coords_nonDim = Affine_image_px_and_NonDim(
                Coordinates=cell_coords_px,
                px_to_nonDim=True,
                image_Nx_px=masks.shape[1],
                image_Ny_px=masks.shape[0],
                d_T_per_px=d_T_per_px
            )

            # 1. Calculate cell area in pixels and in nonDimensional units
            A_cell_px2 = len(y_coords)
            A_cell_nonDim = A_cell_px2 * (d_T_per_px ** 2)  # Convert area to non-dimensional units
            
            # 2. Calculate cell diameter in pixels using area
            d_cell_px = 2 * np.sqrt(A_cell_px2 / np.pi)
            
            # 3. Calculate cell diameter in non-dimensional units
            d_cell_nonDim = d_cell_px * d_T_per_px
            
            # 4. Calculate spherically reconstructed area

            # Calculate pixel coords in pixel units but centered around the image center. Pixel units are chosen as the acuracy is improoved for numerical processing with larger numbers.
            cell_coords_px = np.vstack((x_coords, y_coords))
            cell_coords_px_centered = Affine_image_px_and_NonDim(
                Coordinates=cell_coords_px,
                px_to_nonDim=True,
                image_Nx_px=masks.shape[1],
                image_Ny_px=masks.shape[0],
                d_T_per_px=1
            )

            # Calculate spherically reconstructed area in pixel units centered in the shpere using the Jacobian determinant for each pixel
            A_cell_SRec_px2 = 0
            for j in range(len(x_coords)):
                x = cell_coords_px_centered[0][j]
                z = cell_coords_px_centered[1][j]
                # Calculate pixel area contribution using Jacobian from spherical coordinates to orthographic projection
                detJ_val = detJ(R_SF_px, x, z)
                A_cell_SRec_px2 += detJ_val * 1 # Each pixel contributes 1 px^2 of area

            # calculate spherically reconstructed area in non-dimensional units
            A_cell_SRec_nonDim2 = A_cell_SRec_px2 * (d_T_per_px ** 2)

            print(f"Cell {cell_id} - A_cell_SRec_nonDim2/A_cell_nonDim: {A_cell_SRec_nonDim2 / A_cell_nonDim} ") if Spherical_Reconstruction_log_level >= 4 else None


            # 6. Calculate spherically reconstructed diameter in non-dimensional units
            d_cell_SRec_nonDim = 2 * np.sqrt(A_cell_SRec_nonDim2 / np.pi)
            
            # 7. Calculate spherically reconstructed diameter in pixels
            d_cell_SRec_px = 2 * np.sqrt(A_cell_SRec_px2 / np.pi)
            
            # 9. Convert centroid to non-dimensional coordinates
            centroid_coords_px = np.array([[centroid_x_px], [centroid_y_px]])
            centroid_coords_nonDim = Affine_image_px_and_NonDim(
                Coordinates=centroid_coords_px,
                px_to_nonDim=True,
                image_Nx_px=masks.shape[1],
                image_Ny_px=masks.shape[0],
                d_T_per_px=d_T_per_px
            )
            centroid_x_nonDim = centroid_coords_nonDim[0][0]
            centroid_z_nonDim = centroid_coords_nonDim[1][0]
            
            # Append values to respective lists
            A_cell_distribution_px2.append(A_cell_px2)
            A_cell_distribution_nonDim2.append(A_cell_nonDim)
            d_cell_distribution_px.append(d_cell_px)
            d_cell_distribution_nonDim.append(d_cell_nonDim)
            A_cell_SRec_distribution_nonDim2.append(A_cell_SRec_nonDim2)
            A_cell_SRec_distribution_px2.append(A_cell_SRec_px2)
            d_cell_SRec_distribution_nonDim.append(d_cell_SRec_nonDim)
            d_cell_SRec_distribution_px.append(d_cell_SRec_px)
            centroid_x_distribution_px.append(centroid_x_px)
            centroid_y_distribution_px.append(centroid_y_px)
            centroid_x_distribution_nonDim.append(centroid_x_nonDim)
            centroid_z_distribution_nonDim.append(centroid_z_nonDim)
        
            print(f"Processed cell {cell_id}: "
                  f"length A_cell_distribution_px2={len(A_cell_distribution_px2)}, "
                  f"A_cell_distribution_px2={A_cell_distribution_px2[-1]}, ") if Spherical_Reconstruction_log_level >= 4 else None

        # Store lists in DataFrame - Convert to numpy arrays first
        SRec_df.at[i, 'A_cell_distribution_px2'] = np.array(A_cell_distribution_px2)
        SRec_df.at[i, 'A_cell_distribution_nonDim2'] = np.array(A_cell_distribution_nonDim2)
        SRec_df.at[i, 'd_cell_distribution_px'] = np.array(d_cell_distribution_px)
        SRec_df.at[i, 'd_cell_distribution_nonDim'] = np.array(d_cell_distribution_nonDim)
        SRec_df.at[i, 'A_cell_SRec_distribution_nonDim2'] = np.array(A_cell_SRec_distribution_nonDim2)
        SRec_df.at[i, 'A_cell_SRec_distribution_px2'] = np.array(A_cell_SRec_distribution_px2)
        SRec_df.at[i, 'd_cell_SRec_distribution_nonDim'] = np.array(d_cell_SRec_distribution_nonDim)
        SRec_df.at[i, 'd_cell_SRec_distribution_px'] = np.array(d_cell_SRec_distribution_px)
        SRec_df.at[i, 'centroid_x_distribution_px'] = np.array(centroid_x_distribution_px)
        SRec_df.at[i, 'centroid_y_distribution_px'] = np.array(centroid_y_distribution_px)
        SRec_df.at[i, 'centroid_x_distribution_nonDim'] = np.array(centroid_x_distribution_nonDim)
        SRec_df.at[i, 'centroid_z_distribution_nonDim'] = np.array(centroid_z_distribution_nonDim)

        # Create directory for image-wise sanity check plots if needed
        if plot_diameter_sanity_check_imagewise:
            plots_dir = os.path.join(output_dir, 'image_wise_diameter_sanity_check_plots')
            os.makedirs(plots_dir, exist_ok=True)

        # Create sanity check plots in dedicated folder - per image
        if plot_diameter_sanity_check_imagewise:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot original diameters versus reconstructed diameters
            ax.scatter(d_cell_distribution_px, d_cell_SRec_distribution_px, alpha=0.7)
            
            # Add diagonal line (perfect match)
            min_val = min(min(d_cell_distribution_px), min(d_cell_SRec_distribution_px))
            max_val = max(max(d_cell_distribution_px), max(d_cell_SRec_distribution_px))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Match')
            
            ax.set_xlabel('Original Cell Diameter (pixels)', fontsize=12)
            ax.set_ylabel('Spherical_Reconstruction_1() Cell Diameter (pixels)', fontsize=12)
            ax.set_title(f'Image {i+1}: Original vs Spherically Reconstructed Cell Diameters', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Save the figure in the dedicated folder
            output_filename = f"image_{i}_diameter_comparison.png"
            output_path = os.path.join(plots_dir, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_path}") if Spherical_Reconstruction_log_level >= 1 else None
            
            if show_plots:
                plt.show()
            plt.close(fig)

    #################################################### Perform CST Selection
    
    print("\nPerforming CST Selection...") if Spherical_Reconstruction_log_level >= 1 else None
    SRec_df = CST_selection(
        SRec_df=SRec_df, 
        output_dir=output_dir,
        Spherical_Reconstruction_log_level=Spherical_Reconstruction_log_level,
        show_plots=show_plots
    )
    print("CST Selection complete.") if Spherical_Reconstruction_log_level >= 1 else None

    #################################################### Save Results
    
    # Save the extended DataFrame in both pickle and CSV formats
    srec_df_path = os.path.join(output_dir, 'SRec_DataFrame.pkl')
    srec_csv_path = os.path.join(output_dir, 'SRec_DataFrame.csv')
    
    # Save as pickle (preserves all data types)
    SRec_df.to_pickle(srec_df_path)
    print(f"\nSaved spherical reconstruction data to: {srec_df_path}") if Spherical_Reconstruction_log_level >= 1 else None
    
    # Save as CSV (for easier manual inspection)
    SRec_df.to_csv(srec_csv_path, sep='\t', index=False)
    print(f"Saved spherical reconstruction data to: {srec_csv_path}\n") if Spherical_Reconstruction_log_level >= 1 else None

    # Generate summary comparison plot for all images
    all_orig_diams = []
    all_recon_diams = []
    
    if plot_diameter_sanity_check_summary:
        for i in range(N_images):
            orig_diams = SRec_df.loc[i, 'd_cell_distribution_px']
            recon_diams = SRec_df.loc[i, 'd_cell_SRec_distribution_px']
            
            if orig_diams is not None and recon_diams is not None:
                all_orig_diams.extend(orig_diams)
                all_recon_diams.extend(recon_diams)
        
        if all_orig_diams and all_recon_diams:  # Make sure lists are not empty
            fig, ax = plt.subplots(figsize=(12, 10))
            
            ax.scatter(all_orig_diams, all_recon_diams, alpha=0.5)
            
            min_val = min(min(all_orig_diams), min(all_recon_diams))
            max_val = max(max(all_orig_diams), max(all_recon_diams))
            
            # Add linear regression
            z = np.polyfit(all_orig_diams, all_recon_diams, 1)
            p = np.poly1d(z)
            ax.plot(np.array([min_val, max_val]), p(np.array([min_val, max_val])), 
                    'r-', label=f'Fit: y = {z[0]:.3f}x + {z[1]:.3f}')
            
            # Add diagonal line
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Match')

            ax.set_xlabel('Original Cell Diameter (pixels)', fontsize=14)
            ax.set_ylabel('Reconstructed Cell Diameter (pixels)', fontsize=14)
            ax.set_title('All Images: Original vs Spherically Reconstructed Cell Diameters', fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            
            # Save the summary plot in main output directory
            output_filename = "all_images_diameter_comparison.png"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Summary plot saved to {output_path}") if Spherical_Reconstruction_log_level >= 1 else None
            
            if show_plots:
                plt.show()
            plt.close(fig)

    # Add a newline after the loop is done to move to next line
    if Spherical_Reconstruction_log_level >= 1:
        print()  # This will move to the next line after the progress updates

    Spherical_Reconstruction_Auxillary_1(
        input_dir=output_dir,
        dimentionalised_df=dimentionalised_df,  # Pass the DataFrame
        Spherical_Reconstruction_log_level=Spherical_Reconstruction_log_level,
        output_dir_manual=output_dir_manual,
        output_dir_comment=output_dir_comment,
        show_plots=show_plots,
        plot_CST_detJ=plot_CST_detJ,
    )

    return output_dir

def plot_sanitycheck_CST_selection_pixel_wise(R_SF_nonDim, output_dir, resolution=500, show_plot=False):
    """
    Creates a visualization of the CST boundary check function.
    
    Args:
        R_SF_nonDim: Radius of the sphere in non-dimensional units
        output_dir: Directory to save the plot
        resolution: Resolution of the test grid
        show_plot: Whether to display the plot
        
    Returns:
        None
    """
    # Create directory for sanity check plots
    plots_dir = os.path.join(output_dir, 'CST_boundary_sanity_checks')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create test grid in non-dimensional coordinates
    range_limit = R_SF_nonDim * 1.1  # Just beyond the sphere radius
    x_range = np.linspace(-range_limit, range_limit, resolution)
    z_range = np.linspace(-range_limit, range_limit, resolution)
    X, Z = np.meshgrid(x_range, z_range)
    
    # Calculate CST boundary
    CST_Boundary, CST_Boundary_combined = Cubed_Sphere_Tile_Boundary(R_SF_nonDim, N_pts=500)
    
    # Test each point in the grid
    results = np.zeros((resolution, resolution))
    for i in range(resolution):
        print(f"\rProcessing row {i+1}/{resolution}", end='', flush=True)
        for j in range(resolution):
            x = X[i, j]
            z = Z[i, j]
            # First check if point is within sphere
            if x**2 + z**2 <= R_SF_nonDim**2:
                # Then check if it's in CST
                results[i, j] = 1 if point_in_CST_check(x, z, R_SF_nonDim) else 0.5
            else:
                results[i, j] = 0  # Outside sphere
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Create custom colormap: red for outside sphere, yellow for outside CST but inside sphere, green for inside CST
    cmap = plt.cm.colors.ListedColormap(['red', 'yellow', 'green'])
    bounds = [0, 0.25, 0.75, 1.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot the results
    im = ax.pcolormesh(X, Z, results, cmap=cmap, norm=norm)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0.125, 0.5, 1.0])
    cbar.set_ticklabels(['Outside Sphere', 'Outside CST, Inside Sphere', 'Inside CST'])
    
    # Plot CST boundary
    for boundary, color, label in zip(['N', 'E', 'S', 'W'], ['lime', 'cyan', 'magenta', 'blue'], ['North', 'East', 'South', 'West']):
        points = CST_Boundary.at[0, boundary]
        ax.plot(points[0], points[1], color=color, label=label, linewidth=2)
    
    # Plot sphere boundary
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(R_SF_nonDim*np.cos(theta), R_SF_nonDim*np.sin(theta), 'k--', linewidth=1.5, label='Sphere')
    
    # Mark cube corners
    L = R_SF_nonDim / np.sqrt(3)
    corners = [
        (L, L), (L, -L), (-L, -L), (-L, L)
    ]
    for x, z in corners:
        ax.plot(x, z, 'ko', markersize=8)
    
    # Set title and labels
    ax.set_title(f'CST Boundary Check Sanity Plot (R={R_SF_nonDim:.2f})', fontsize=14)
    ax.set_xlabel('x (non-dimensional)', fontsize=12)
    ax.set_ylabel('z (non-dimensional)', fontsize=12)
    ax.legend(loc='upper right')
    
    # Set aspect ratio to equal
    ax.set_aspect('equal')
    
    # Save the figure
    output_path = os.path.join(plots_dir, f"CST_boundary_check_sanity_R{R_SF_nonDim:.2f}.png")
    plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
    print(f"CST boundary check sanity plot saved to {output_path}")
    
    if show_plot:
        plt.show()
    plt.close(fig)

    return None

# Example usage:
if __name__ == "__main__":
    print("Running Spherical Reconstruction...")

    # plot_sanitycheck_CST_selection_pixel_wise(
    #     R_SF_nonDim = 1, 
    #     output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\20250604_1311111\20250604_1312276\20250604_1312276\20250604_1313140\20250604_1314474\20250604_1314484\20250612_2041591", 
    #     resolution=500, 
    #     show_plot=True)

    Spherical_Reconstruction_1(
        # Dimentionalised output of BBWW 2000px 
        input_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\20250607_2240236\20250608_0303173\20250608_0303173\20250608_0409296\20250608_0643128\20250608_0645118",
        
        # Dimentionalised output_dir of hot 3000px States 79 and 100
        #input_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\20250604_1311111\20250604_1312276\20250604_1312276\20250604_1313140\20250604_1314474\20250604_1314484",
        
        Spherical_Reconstruction_log_level = 2,
        show_plots = False,
        plot_CST_detJ = True,
        plot_diameter_sanity_check_imagewise = False,
        plot_diameter_sanity_check_summary = True
    )