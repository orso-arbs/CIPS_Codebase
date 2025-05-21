import numpy as np
import pandas as pd
import os
import pickle
import Format_1 as F_1
import skimage.io as sk_io
import matplotlib.pyplot as plt
from skimage import color

from cellpose import utils # Needed for utils.diameters

@F_1.ParameterLog(max_size = 1024 * 10, log_level = 0)
def Spherical_Reconstruction_1(
    # input
    input_dir, # Should be the output directory of CP_dimentionalise

    # output and logging
    Spherical_Reconstruction_log_level = 0,
    output_dir_manual = "", output_dir_comment = "",
    ):


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

    i = 40

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


    if 1==0: # plot image + masks + outlines + CST_Boundary + reference cirle
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
        
        plt.show()

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

    Transformed_Coordinates = np.zeros_like(Coordinates)
    
    if px_to_nonDim:
        x_px = Coordinates[0]
        y_px = Coordinates[1]
        Transformed_Coordinates[0] = (x_px - image_Nx_px/2) * d_T_per_px
        Transformed_Coordinates[1] = (image_Ny_px/2 - y_px) * d_T_per_px
        
    if nonDim_to_px:
        x_nonDim = Coordinates[0]
        z_nonDim = Coordinates[1]
        Transformed_Coordinates[0] = x_nonDim / d_T_per_px + image_Nx_px/2
        Transformed_Coordinates[1] = image_Ny_px/2 - z_nonDim / d_T_per_px
        
    return Transformed_Coordinates

def Cubed_Sphere_Tile_Boundary(R, N_pts=500):
    """
    Calculates the boundary points of a cubed sphere tile.
    
    Args:
        R (float): Radius of the sphere
        N_pts (int): Number of points for discretization (default=100)
    
    Returns:
        pd.DataFrame: DataFrame containing the boundary points (N,W,S,E)
        Each column contains a 2xN_pts array with x and z coordinates
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



    return CST_Boundary

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
                        xytext=(x_end + R*0.2, z_end - R*0.2), textcoords='data',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"),
                        bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8))

    # Plot reference circle
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(R*np.cos(theta), R*np.sin(theta), 'k--', 
            label='Reference Circle', linewidth=1.5)
    
    # Improve grid and labels
    ax.grid(True, alpha=0.3, linestyle='--')
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
    
    plt.show()
    return fig, ax


# Example usage:
if __name__ == "__main__":
    print("Running Spherical Reconstruction...")

    if 1==0: # to plot: Cubed Sphere Tile Boundary with det(J)
        R = 1.0
        CST_Boundary, CST_Boundary_combined = Cubed_Sphere_Tile_Boundary(R)
        plot_boundary_and_detJ(CST_Boundary, R)


    Spherical_Reconstruction_1(
    input_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\SF_CP_analysis_pipeline_data\Visit_Projector_1_2025-05-10_14-46-34_A11_T-3_VM-hot\CP_segment_1_2025-05-10_15-40-15_cyto3\CP_extract_1_2025-05-10_15-46-46\CP_DIM~1"
    )