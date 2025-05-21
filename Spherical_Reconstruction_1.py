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
    print(f"Image file path: {dimentionalised_df.loc[i, 'image_file_path']}")
    image_RGB = color.rgb2gray(sk_io.imread(dimentionalised_df.loc[i, 'image_file_path'])[..., :3])
    R = dimentionalised_df.loc[i, 'R_SF_nonDim']
    print(f"R = {R}")

    # # Example usage in your main function:
    # for i in range(N_images):
    #     image_RGB = color.rgb2gray(sk_io.imread(dimentionalised_df.loc[i, 'image_file_path'])[..., :3])

    #     R = dimentionalised_df.loc[i, 'R_SF_nonDim']
    #     print(f"R = {R}")






def detJ(R, x, z):
    """Calculate the Jacobian determinant at point (x,z)"""
    return R/np.sqrt(R**2 - x**2 - z**2)

def Cubed_Sphere_Tile_Boundary(R, N_pts=100):
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
    z_N = np.linspace(L, -L, N_pts)
    x_N = np.sqrt((R**2 - z_N**2) / 2)
    CST_Boundary.at[0, 'N'] = np.vstack((x_N, z_N))
    
    # South boundary
    z_S = np.linspace(L, -L, N_pts)
    x_S = -np.sqrt((R**2 - z_S**2) / 2)
    CST_Boundary.at[0, 'S'] = np.vstack((x_S, z_S))
    
    # West boundary
    x_W = np.linspace(-L, L, N_pts)
    z_W = np.sqrt((R**2 - x_W**2) / 2)
    CST_Boundary.at[0, 'W'] = np.vstack((x_W, z_W))
    
    # East boundary
    x_E = np.linspace(-L, L, N_pts)
    z_E = -np.sqrt((R**2 - x_E**2) / 2)
    CST_Boundary.at[0, 'E'] = np.vstack((x_E, z_E))
    
    return CST_Boundary # CST_Boundary.at[0

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
        ax.plot(points[1], points[0], color=color, label=label, 
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
                        xy=(z_mid, x_mid), xycoords='data',
                        xytext=(z_mid, x_mid - R*0.2), textcoords='data',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
                        bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8))
            
            # Arrow to end point (maximum)
            ax.annotate(f'{detJ_end:.2f}',
                        xy=(z_end, x_end), xycoords='data',
                        xytext=(z_end + R*0.2, x_end - R*0.2), textcoords='data',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"),
                        bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8))

    # Plot reference circle
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(R*np.cos(theta), R*np.sin(theta), 'k--', 
            label='Reference Circle', linewidth=1.5)
    
    # Improve grid and labels
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('z', fontsize=12)
    ax.set_ylabel('x', fontsize=12)
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


    #R = 1.0
    #boundary = Cubed_Sphere_Tile_Boundary(R, N_pts=100)
    #plot_boundary_and_detJ(boundary, R)


    Spherical_Reconstruction_1(
    input_dir = r"C:\Users\obs\Desktop\VCL_Pipe_1_2025-05-16_01-02-47_HRR\Visit_Projector_1_2025-05-16_01-02-49\CP_segment_1_2025-05-16_02-02-47\CP_extract_1_2025-05-16_02-10-02\dim2_VisIt_R_1_2025-05-16_02-10-14"    )