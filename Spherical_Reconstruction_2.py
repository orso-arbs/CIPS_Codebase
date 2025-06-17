import numpy as np
import pandas as pd
import os
import pickle
import Format_1 as F_1
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import skimage.io as sk_io

@F_1.ParameterLog(max_size=1024 * 10, log_level=0)
def Spherical_Reconstruction_2(
    # input
    input_dir,
    Analysis_A11_df=None,  # DataFrame from previous processing, or None to load from input_dir
    
    # output and logging
    SR2_log_level=2,
    output_dir_manual="",
    output_dir_comment="",
    show_plots=False,
    plot_CST_detJ=False
):
    """
    Performs spherical reconstruction on cell data in the Analysis_A11_df DataFrame.
    
    This function calculates spherical reconstruction values for each cell using
    the Jacobian determinant method for orthographic projections.
    
    Parameters
    ----------
    input_dir : str
        Path to the directory containing the dimensionalized DataFrame
    Analysis_A11_df : pd.DataFrame or None, optional
        DataFrame from previous processing or None to load from input_dir
    SR2_log_level : int, optional
        Controls verbosity of logging. Default is 2.
    output_dir_manual : str, optional
        If provided, specifies the output directory. Default is "".
    output_dir_comment : str, optional
        Comment to append to the output directory name. Default is "".
    show_plots : bool, optional
        Whether to display plots during processing. Default is False.
    plot_CST_detJ : bool, optional
        Whether to generate a plot of the CST boundary with det(J). Default is False.
        
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
        # Try to load from the dimensionalized DataFrame (from dim3_A11)
        dim_data_path = os.path.join(input_dir, 'Analysis_A11_df.pkl')
        
        # Fall back to the original directory from dim3_A11
        if not os.path.exists(dim_data_path):
            # Could also look for dimensionalized_DataFrame.pkl
            dim_data_path = os.path.join(input_dir, 'dimensionalized_DataFrame.pkl')
            
        print(f"\nLoading dimensionalized data from: {dim_data_path}") if SR2_log_level >= 1 else None
        
        try:
            dim_df = pd.read_pickle(dim_data_path)
            Analysis_A11_df = dim_df.copy()
        except FileNotFoundError:
            print(f"Error: Could not find dimensionalized data file at {dim_data_path}")
            print("Ensure dim3_A11 ran successfully or dimensionalized data is available.")
            return output_dir
    else:
        print("\nUsing provided Analysis_A11_df DataFrame") if SR2_log_level >= 1 else None
    
    # Get number of images/rows from loaded data
    N_images = len(Analysis_A11_df)
    print(f"Processing {N_images} images for spherical reconstruction") if SR2_log_level >= 1 else None
    
    #################################################### Initialize New Columns
    # Add new columns for spherical reconstruction results
    srec_columns = [
        'A_cell_distribution_px2', 'A_cell_distribution_nonDim2',
        'd_cell_distribution_px', 'd_cell_distribution_nonDim', 
        'A_cell_SRec_distribution_nonDim2', 'A_cell_SRec_distribution_px2', 
        'd_cell_SRec_distribution_nonDim', 'd_cell_SRec_distribution_px', 
        'centroid_x_distribution_px', 'centroid_y_distribution_px', 
        'centroid_x_distribution_nonDim', 'centroid_z_distribution_nonDim'
    ]
    
    # Initialize columns to hold lists/arrays
    for col in srec_columns:
        if col not in Analysis_A11_df.columns:
            Analysis_A11_df[col] = pd.Series(dtype='object')
    
    #################################################### Process Each Image
    print("\nCalculating spherical reconstruction values...") if SR2_log_level >= 1 else None
    
    for i in range(N_images):
        print(f"\rProcessing image {i+1}/{N_images}", end='', flush=True) if SR2_log_level >= 1 else None
        
        # Extract required data for this image
        masks = Analysis_A11_df.loc[i, 'masks']
        R_SF_px = Analysis_A11_df.loc[i, 'R_SF_px']
        R_SF_nonDim = Analysis_A11_df.loc[i, 'R_SF_nonDim']
        d_T_per_px = Analysis_A11_df.loc[i, 'd_T_per_px']
        
        # Get unique cell IDs (excluding background = 0)
        cell_ids = np.unique(masks)
        cell_ids = cell_ids[cell_ids > 0]  # Exclude background (ID=0)
        
        # If no cells found, continue to next image
        if len(cell_ids) == 0:
            print(f"\nNo cells found in image {i+1}") if SR2_log_level >= 2 else None
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
            
            # Calculate centroid and area of the cell
            y_coords, x_coords = np.where(cell_mask)
            if len(y_coords) == 0:
                continue
            
            # Find cell centroid in pixel coordinates
            centroid_y_px = np.mean(y_coords)
            centroid_x_px = np.mean(x_coords)
            
            # Calculate cell area in pixels and non-dimensional units
            A_cell_px2 = len(y_coords)
            A_cell_nonDim2 = A_cell_px2 * (d_T_per_px ** 2)
            
            # Calculate cell diameter based on area
            d_cell_px = 2 * np.sqrt(A_cell_px2 / np.pi)
            d_cell_nonDim = d_cell_px * d_T_per_px
            
            # Find each pixel in non-dimensional space (centered coordinates)
            cell_coords_px = np.vstack((x_coords, y_coords))
            cell_coords_px_centered = Affine_image_px_and_NonDim(
                Coordinates=cell_coords_px,
                px_to_nonDim=True,
                image_Nx_px=masks.shape[1],
                image_Ny_px=masks.shape[0],
                d_T_per_px=1  # Use 1 to keep in pixel units but centered
            )
            
            # Calculate spherically reconstructed area using Jacobian determinant
            A_cell_SRec_px2 = 0
            for j in range(len(x_coords)):
                x = cell_coords_px_centered[0][j]
                z = cell_coords_px_centered[1][j]
                # Calculate pixel area contribution using Jacobian
                detJ_val = detJ(R_SF_px, x, z)
                A_cell_SRec_px2 += detJ_val * 1  # Each pixel contributes 1 px^2 of area
            
            # Convert to non-dimensional units
            A_cell_SRec_nonDim2 = A_cell_SRec_px2 * (d_T_per_px ** 2)
            
            # Calculate spherically reconstructed diameters
            d_cell_SRec_nonDim = 2 * np.sqrt(A_cell_SRec_nonDim2 / np.pi)
            d_cell_SRec_px = 2 * np.sqrt(A_cell_SRec_px2 / np.pi)
            
            # Convert centroid to non-dimensional coordinates
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
            A_cell_distribution_nonDim2.append(A_cell_nonDim2)
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
        
        # Store lists as numpy arrays in DataFrame
        Analysis_A11_df.at[i, 'A_cell_distribution_px2'] = np.array(A_cell_distribution_px2)
        Analysis_A11_df.at[i, 'A_cell_distribution_nonDim2'] = np.array(A_cell_distribution_nonDim2)
        Analysis_A11_df.at[i, 'd_cell_distribution_px'] = np.array(d_cell_distribution_px)
        Analysis_A11_df.at[i, 'd_cell_distribution_nonDim'] = np.array(d_cell_distribution_nonDim)
        Analysis_A11_df.at[i, 'A_cell_SRec_distribution_nonDim2'] = np.array(A_cell_SRec_distribution_nonDim2)
        Analysis_A11_df.at[i, 'A_cell_SRec_distribution_px2'] = np.array(A_cell_SRec_distribution_px2)
        Analysis_A11_df.at[i, 'd_cell_SRec_distribution_nonDim'] = np.array(d_cell_SRec_distribution_nonDim)
        Analysis_A11_df.at[i, 'd_cell_SRec_distribution_px'] = np.array(d_cell_SRec_distribution_px)
        Analysis_A11_df.at[i, 'centroid_x_distribution_px'] = np.array(centroid_x_distribution_px)
        Analysis_A11_df.at[i, 'centroid_y_distribution_px'] = np.array(centroid_y_distribution_px)
        Analysis_A11_df.at[i, 'centroid_x_distribution_nonDim'] = np.array(centroid_x_distribution_nonDim)
        Analysis_A11_df.at[i, 'centroid_z_distribution_nonDim'] = np.array(centroid_z_distribution_nonDim)
    
    print("\nSpherical reconstruction complete!") if SR2_log_level >= 1 else None
    
    #################################################### Generate Visualization Plots
    
    # Create demo visualization using the first image
    if N_images > 0:
        if N_images > 100:
            i = 100  # Use the 100th image if available
        else:
            i = N_images -1  # Use the last image if less than 100
        R = Analysis_A11_df.loc[i, 'R_SF_nonDim']
        image_Nx_px = Analysis_A11_df.loc[i, 'image_Ny_px']  # Note: These are swapped in the code
        image_Ny_px = Analysis_A11_df.loc[i, 'image_Nx_px']  # Note: These are swapped in the code
        d_T_per_px = Analysis_A11_df.loc[i, 'd_T_per_px']
        
        # Calculate CST boundary
        CST_Boundary_nonDim, CST_Boundary_combined_nonDim = Cubed_Sphere_Tile_Boundary(R, N_pts=100)
        CST_Boundary_combined_px = Affine_image_px_and_NonDim(
            Coordinates=CST_Boundary_combined_nonDim,
            nonDim_to_px=True,
            image_Nx_px=image_Nx_px,
            image_Ny_px=image_Ny_px,
            d_T_per_px=d_T_per_px
        )
        
        # Load image
        image_RGB = sk_io.imread(Analysis_A11_df.loc[i, 'image_file_path'])[..., :3]
        outlines = Analysis_A11_df.loc[i, 'outlines']
        masks = Analysis_A11_df.loc[i, 'masks']
        
        # Plot image with CST boundary and reference circle
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image_RGB)
        
        # Plot masks with transparency
        masked = np.ma.masked_where(masks == 0, masks)
        ax.imshow(masked, alpha=0.5)
        
        # Plot outlines
        outlined = np.ma.masked_where(outlines == 0, outlines)
        ax.imshow(outlined, alpha=1)
        
        # Add CST boundary
        ax.plot(CST_Boundary_combined_px[0], CST_Boundary_combined_px[1], 'r', 
                linewidth=2, label='CST Boundary')
        
        # Add reference circle
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(R*np.cos(theta) / d_T_per_px + image_Nx_px/2, 
                R*np.sin(theta) / d_T_per_px + image_Ny_px/2, 'r--',
                label='Reference Circle', linewidth=2)
        
        ax.set_title(f"Image {i+1} with CST Boundary and Reference Circle", fontsize=16)
        ax.axis('off')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Save the figure
        output_path = os.path.join(output_dir, f"image_{i+1}_CST_boundary.png")
        plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
        print(f"\nSaved CST boundary visualization to: {output_path}") if SR2_log_level >= 1 else None
        
        if show_plots:
            plt.show()
        plt.close(fig)
        
        # Generate detJ plot if requested
        if plot_CST_detJ:
            R_detJ_plot = 1.0  # Fixed R for demonstration
            CST_Boundary_detJ, _ = Cubed_Sphere_Tile_Boundary(R_detJ_plot, N_pts=100)
            fig_detJ, ax_detJ = plot_boundary_and_detJ(CST_Boundary_detJ, R_detJ_plot)
            
            # Save detJ plot
            output_path_detJ = os.path.join(output_dir, "CST_boundary_detJ_plot.png")
            fig_detJ.savefig(output_path_detJ, format='png', dpi=300, bbox_inches='tight')
            print(f"Saved detJ plot to: {output_path_detJ}") if SR2_log_level >= 1 else None
            
            if show_plots:
                plt.show()
            plt.close(fig_detJ)
    
    #################################################### Save Results
    # Save the processed DataFrame
    output_pkl_path = os.path.join(output_dir, 'Analysis_A11_df.pkl')
    output_csv_path = os.path.join(output_dir, 'Analysis_A11_df.csv')
    
    Analysis_A11_df.to_pickle(output_pkl_path)
    Analysis_A11_df.to_csv(output_csv_path, sep='\t', index=False)
    
    print(f"\nSaved spherical reconstruction data to:") if SR2_log_level >= 1 else None
    print(f"  - {output_pkl_path}") if SR2_log_level >= 1 else None
    print(f"  - {output_csv_path}") if SR2_log_level >= 1 else None
    
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
    
    # Check input shape
    if Coordinates.shape[0] != 2:
        print(f"WARNING: Input coordinates shape: {Coordinates.shape}")
        print(f"First few coordinates: {Coordinates[:5] if len(Coordinates) > 5 else Coordinates}")
    
    Coordinates = Coordinates.astype(float)  # Ensure float type for calculations
    Transformed_Coordinates = np.zeros_like(Coordinates)
    
    if px_to_nonDim:
        x_px = Coordinates[0]
        y_px = Coordinates[1]
        Transformed_Coordinates[0] = ((x_px + 1/2) - image_Nx_px/2) * d_T_per_px  # x coord
        Transformed_Coordinates[1] = (image_Ny_px/2 - (y_px + 1/2)) * d_T_per_px  # z coord
        
    if nonDim_to_px:
        x_nonDim = Coordinates[0]
        z_nonDim = Coordinates[1]
        Transformed_Coordinates[0] = x_nonDim / d_T_per_px + image_Nx_px/2 - 1/2  # x coord 
        Transformed_Coordinates[1] = image_Ny_px/2 - z_nonDim / d_T_per_px - 1/2  # y coord
    
    return Transformed_Coordinates

def Cubed_Sphere_Tile_Boundary(R, N_pts=500):
    """
    Calculates the boundary points of a cubed sphere tile.
    
    Args:
        R (float): Radius of the sphere
        N_pts (int): Number of points for discretization (default=500)
    
    Returns:
        pd.DataFrame: DataFrame containing the boundary points (N,W,S,E)
        numpy.ndarray: Combined boundary points as a 2xN array
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
    # Add first point again to close the loop
    CST_Boundary_combined = np.hstack([CST_Boundary_combined, CST_Boundary_combined[:, [0]]])
    
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
    
    Returns:
        (matplotlib.figure.Figure, matplotlib.axes.Axes): Figure and axes objects
    """
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

    # Plot pseudocolor field
    pcm = ax.pcolormesh(Z, X, J, 
                       cmap=cmap, 
                       alpha=alpha,
                       norm=LogNorm(vmin=J[valid_points].min(), 
                                   vmax=J[valid_points].max()))
    
    # Add colorbar
    min_val = J[valid_points].min()
    max_val = J[valid_points].max()
    ticks = np.logspace(np.log10(min_val), np.log10(max_val), 10)
    cb = fig.colorbar(pcm, ax=ax, label='det(J)', ticks=ticks)
    cb.ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    cb.ax.minorticks_off()
    
    # Plot boundaries
    boundaries = ['N', 'S', 'W', 'E']
    colors = ['red', 'blue', 'green', 'orange']
    labels = ['North', 'South', 'West', 'East']
    
    for boundary, color, label in zip(boundaries, colors, labels):
        points = CST_Boundary.at[0, boundary]
        ax.plot(points[0], points[1], color=color, label=label, 
               linewidth=2.5, linestyle='-')
    
    # Plot reference circle
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(R*np.cos(theta), R*np.sin(theta), 'k--', 
           label='Reference Circle', linewidth=1.5)
    
    # Add labels and grid
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('z', fontsize=12)
    ax.set_title('Cubed Sphere Tile Boundary with det(J)', 
               fontsize=14, pad=20)
    
    # Set equal axis limits
    ax.set_xlim(-R*1.1, R*1.1)
    ax.set_ylim(-R*1.1, R*1.1)
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax

# Example usage when script is run directly
if __name__ == "__main__":
    print("Running Spherical_Reconstruction_2 as standalone module...")
    
    # When used in pipeline, input_dir is the Analysis_A11 output dir
    a11_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\20250604_1311111\20250604_1312276\20250604_1312276\20250604_1313140\20250615_1635229\20250615_1815446"
    
    # This path would be to the dim3_A11 output subdirectory in the pipeline
    dim3_output_dir = os.path.join(a11_output_dir, "20250615_1815476_dim3_A11")
    
    # Load the dimensionalized DataFrame 
    try:
        df_path = os.path.join(dim3_output_dir, "Analysis_A11_df.pkl")
        analysis_df = pd.read_pickle(df_path)
        print(f"Loaded Analysis_A11_df from: {df_path}")
    except FileNotFoundError:
        print(f"DataFrame not found at {df_path}. Using the direct input directory instead.")
        # In a real situation, we would need to handle this case properly
        # For this example, we'll just proceed without a DataFrame (it would fail)
        analysis_df = None
    
    output_dir = Spherical_Reconstruction_2(
        input_dir=a11_output_dir,  # Main A11 output dir as input_dir for Format_1
        Analysis_A11_df=analysis_df,  # DataFrame from previous step
        SR2_log_level=2,
        show_plots=True,
        plot_CST_detJ=True,
        output_dir_comment="test_standalone"
    )
    
    print(f"Spherical reconstruction complete. Results in: {output_dir}")
