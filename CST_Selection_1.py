import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import Format_1 as F_1
import skimage.io as sk_io
from Spherical_Reconstruction_2 import Cubed_Sphere_Tile_Boundary, Affine_image_px_and_NonDim

@F_1.ParameterLog(max_size=1024 * 10, log_level=0)
def CST_Selection_1(
    # input
    input_dir,
    Analysis_A11_df=None,  # DataFrame from previous processing, or None to load from input_dir
    
    # output and logging
    CST_log_level=2,
    output_dir_manual="",
    output_dir_comment="",
    show_plots=False,
    plot_CST_selection=True,
    Convert_to_grayscale_image=True
):
    """
    Selects cells within the Cubed Sphere Tile (CST) boundary.
    
    Parameters
    ----------
    input_dir : str
        Path to the directory containing the spherically reconstructed DataFrame
    Analysis_A11_df : pd.DataFrame or None, optional
        DataFrame from previous processing or None to load from input_dir
    CST_log_level : int, optional
        Controls verbosity of logging. Default is 2.
    output_dir_manual : str, optional
        If provided, specifies the output directory. Default is "".
    output_dir_comment : str, optional
        Comment to append to the output directory name. Default is "".
    show_plots : bool, optional
        Whether to display plots during processing. Default is False.
    plot_CST_selection : bool, optional
        Whether to generate classification plots. Default is True.
    Convert_to_grayscale_image : bool, optional
        Whether to convert images to grayscale in plots. Default is True.
        
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
        # Try to load from the spherically reconstructed DataFrame
        srec_data_path = os.path.join(input_dir, 'Analysis_A11_df.pkl')
            
        print(f"\nLoading spherically reconstructed data from: {srec_data_path}") if CST_log_level >= 1 else None
        
        try:
            srec_df = pd.read_pickle(srec_data_path)
            Analysis_A11_df = srec_df.copy()
        except FileNotFoundError:
            print(f"Error: Could not find spherically reconstructed data file at {srec_data_path}")
            print("Ensure Spherical_Reconstruction_2 ran successfully.")
            return output_dir
    else:
        print("\nUsing provided Analysis_A11_df DataFrame") if CST_log_level >= 1 else None
    
    # Get number of images/rows from loaded data
    N_images = len(Analysis_A11_df)
    print(f"Processing {N_images} images for CST selection") if CST_log_level >= 1 else None
    
    #################################################### Initialize CST Selection Columns
    # Add new columns for CST selection results
    cst_columns = [
        'CST_classification',
        'CST_inclusion',  # New column to track if cell is included in CST
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
    
    # Initialize columns to hold lists/arrays
    for col in cst_columns:
        if col not in Analysis_A11_df.columns:
            Analysis_A11_df[col] = None
    
    #################################################### Create Plot Directory
    # Create directory for classification plots
    if plot_CST_selection:
        plots_dir = os.path.join(output_dir, 'CST_classification_plots')
        os.makedirs(plots_dir, exist_ok=True)
    
    #################################################### Process Each Image
    print("\nPerforming CST Selection...") if CST_log_level >= 1 else None
    
    # Process each image
    for i in range(N_images):
        if CST_log_level >= 1:
            print(f"\rProcessing CST selection for image {i+1}/{N_images}", end='', flush=True)
        
        # Extract data for this image
        masks = Analysis_A11_df.loc[i, 'masks']
        R_SF_nonDim = Analysis_A11_df.loc[i, 'R_SF_nonDim']
        R_SF_px = Analysis_A11_df.loc[i, 'R_SF_px']
        image_Nx_px = Analysis_A11_df.loc[i, 'image_Ny_px']  # These are swapped in the code
        image_Ny_px = Analysis_A11_df.loc[i, 'image_Nx_px']  # These are swapped in the code
        d_T_per_px = Analysis_A11_df.loc[i, 'd_T_per_px']
        
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
            d_T_per_px=d_T_per_px
        )
        
        # Extract cell properties for this image
        A_cell_distribution_px2 = Analysis_A11_df.loc[i, 'A_cell_distribution_px2']
        A_cell_distribution_nonDim2 = Analysis_A11_df.loc[i, 'A_cell_distribution_nonDim2']
        d_cell_distribution_px = Analysis_A11_df.loc[i, 'd_cell_distribution_px']
        d_cell_distribution_nonDim = Analysis_A11_df.loc[i, 'd_cell_distribution_nonDim']
        A_cell_SRec_distribution_nonDim2 = Analysis_A11_df.loc[i, 'A_cell_SRec_distribution_nonDim2']
        A_cell_SRec_distribution_px2 = Analysis_A11_df.loc[i, 'A_cell_SRec_distribution_px2']
        d_cell_SRec_distribution_nonDim = Analysis_A11_df.loc[i, 'd_cell_SRec_distribution_nonDim']
        d_cell_SRec_distribution_px = Analysis_A11_df.loc[i, 'd_cell_SRec_distribution_px']
        centroid_x_distribution_px = Analysis_A11_df.loc[i, 'centroid_x_distribution_px']
        centroid_y_distribution_px = Analysis_A11_df.loc[i, 'centroid_y_distribution_px']
        centroid_x_distribution_nonDim = Analysis_A11_df.loc[i, 'centroid_x_distribution_nonDim']
        centroid_z_distribution_nonDim = Analysis_A11_df.loc[i, 'centroid_z_distribution_nonDim']
        
        # Lists for filtered cells
        CST_classification = []
        CST_inclusion = []  # New list to track if cell is included in CST
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
        cell_inclusion = {}  # New dictionary to track cell inclusion status
        
        # Process each cell
        for idx, cell_id in enumerate(cell_ids):
            print(f"\rProcessing CST selection for image {i+1}/{N_images} - cell {idx+1}/{len(cell_ids)}", end='', flush=True) if CST_log_level >= 2 else None
            
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
                        
            # Check if centroid is inside CST boundary
            centroid_in_CST = point_in_CST_check(
                x_nonDim=centroid_x_nonDim,
                z_nonDim=centroid_z_nonDim,
                R_SF_nonDim=R_SF_nonDim
            )
                        
            # Convert mask points to non-dimensional coordinates
            mask_coords_px = np.vstack((x_coords, y_coords))
            mask_coords_nonDim = Affine_image_px_and_NonDim(
                Coordinates=mask_coords_px,
                px_to_nonDim=True,
                image_Nx_px=masks.shape[1],
                image_Ny_px=masks.shape[0],
                d_T_per_px=d_T_per_px
            )
            
            # Check if all points are inside CST boundary
            all_points_in = True
            any_point_in = False
            n_points_checked = 0
            n_points_in_CST = 0
            
            # Check a sample of points for optimization
            max_points_to_check = 1000  # Limit for very large masks
            check_stride = max(1, len(mask_coords_nonDim[0]) // max_points_to_check)
            
            for j in range(0, len(mask_coords_nonDim[0]), check_stride):
                px = mask_coords_nonDim[0][j]
                pz = mask_coords_nonDim[1][j]
                
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
            
            # Classify cell with new names
            if centroid_in_CST and all_points_in:
                classification = "all_in_CST_Boundary"
            elif centroid_in_CST and any_point_in:
                classification = "center_in_CST_Boundary"
            elif not centroid_in_CST and any_point_in:
                classification = "center_out_CST_Boundary"
            else:
                classification = "all_out_CST_Boundary"
                
            # Determine if cell should be included in CST based on new condition
            is_included = (classification == "all_in_CST_Boundary") or \
                         (classification == "center_in_CST_Boundary" and centroid_z_nonDim >= -centroid_x_nonDim)
            
            # Store classification for plotting
            cell_classifications[cell_id] = classification
            cell_centroids_px[cell_id] = (centroid_x_px, centroid_y_px)
            cell_inclusion[cell_id] = is_included
            
            # Add all cells to the classification
            CST_classification.append(classification)
            CST_inclusion.append(is_included)
            
            # Only include cells that meet the new condition
            if is_included:
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
        Analysis_A11_df.at[i, 'CST_classification'] = np.array(CST_classification)
        Analysis_A11_df.at[i, 'CST_inclusion'] = np.array(CST_inclusion)
        Analysis_A11_df.at[i, 'A_cell_distribution_CST_px2'] = np.array(A_cell_distribution_CST_px2)
        Analysis_A11_df.at[i, 'A_cell_distribution_CST_nonDim2'] = np.array(A_cell_distribution_CST_nonDim2)
        Analysis_A11_df.at[i, 'd_cell_distribution_CST_px'] = np.array(d_cell_distribution_CST_px)
        Analysis_A11_df.at[i, 'd_cell_distribution_CST_nonDim'] = np.array(d_cell_distribution_CST_nonDim)
        Analysis_A11_df.at[i, 'A_cell_SRec_distribution_CST_nonDim2'] = np.array(A_cell_SRec_distribution_CST_nonDim2)
        Analysis_A11_df.at[i, 'A_cell_SRec_distribution_CST_px2'] = np.array(A_cell_SRec_distribution_CST_px2)
        Analysis_A11_df.at[i, 'd_cell_SRec_distribution_CST_nonDim'] = np.array(d_cell_SRec_distribution_CST_nonDim)
        Analysis_A11_df.at[i, 'd_cell_SRec_distribution_CST_px'] = np.array(d_cell_SRec_distribution_CST_px)
        Analysis_A11_df.at[i, 'centroid_x_distribution_CST_px'] = np.array(centroid_x_distribution_CST_px)
        Analysis_A11_df.at[i, 'centroid_y_distribution_CST_px'] = np.array(centroid_y_distribution_CST_px)
        Analysis_A11_df.at[i, 'centroid_x_distribution_CST_nonDim'] = np.array(centroid_x_distribution_CST_nonDim)
        Analysis_A11_df.at[i, 'centroid_z_distribution_CST_nonDim'] = np.array(centroid_z_distribution_CST_nonDim)
        
        # Generate classification plots
        if plot_CST_selection and len(cell_classifications) > 0:
            try:
                # Load the image for plotting
                image_RGB = sk_io.imread(Analysis_A11_df.loc[i, 'image_file_path'])[..., :3]
                outlines = Analysis_A11_df.loc[i, 'outlines']
                
                print(f"\nGenerating CST classification plots for image {i+1}") if CST_log_level >= 1 else None
                
                # Plot 1: Classification by boundary position
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
                    CST_log_level=CST_log_level,
                    title_prefix=f"Image {i+1}",
                    Convert_to_grayscale_image=Convert_to_grayscale_image
                )
                
                # Plot 2: CST inclusion status
                inclusion_output_path = os.path.join(plots_dir, f"image_{i}_CST_inclusion.png")
                plot_CST_inclusion(
                    image_RGB=image_RGB,
                    masks=masks,
                    outlines=outlines,
                    CST_Boundary_combined_px=CST_Boundary_combined_px,
                    R=R_SF_nonDim,
                    d_T_per_px=d_T_per_px,
                    image_Nx_px=image_Nx_px,
                    image_Ny_px=image_Ny_px,
                    cell_inclusion=cell_inclusion,
                    cell_centroids_px=cell_centroids_px,
                    output_path=inclusion_output_path,
                    show_plot=show_plots,
                    CST_log_level=CST_log_level,
                    title_prefix=f"Image {i+1}",
                    Convert_to_grayscale_image=Convert_to_grayscale_image
                )
                
            except Exception as e:
                print(f"\nError generating plot for image {i}: {e}")
    
    # Print newline after progress updates
    if CST_log_level >= 1:
        print()
        
    #################################################### Save Results
    # Save the processed DataFrame
    output_pkl_path = os.path.join(output_dir, 'Analysis_A11_df.pkl')
    output_csv_path = os.path.join(output_dir, 'Analysis_A11_df.csv')
    
    Analysis_A11_df.to_pickle(output_pkl_path)
    Analysis_A11_df.to_csv(output_csv_path, sep='\t', index=False)
    
    print(f"\nSaved CST selection data to:") if CST_log_level >= 1 else None
    print(f"  - {output_pkl_path}") if CST_log_level >= 1 else None
    print(f"  - {output_csv_path}") if CST_log_level >= 1 else None
    
    return output_dir

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
    
    # Default case if we reach here
    return False

def plot_CST_selection_sanity_check(image_RGB, masks, outlines, CST_Boundary_combined_px, R, d_T_per_px, 
                                   image_Nx_px, image_Ny_px, cell_classifications, cell_centroids_px,
                                   output_path, show_plot=False, CST_log_level=1, title_prefix="Image", 
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
        Convert_to_grayscale_image: Whether to convert RGB to grayscale
    
    Returns:
        None
    """
    # Define colors for each classification type (with updated names)
    classification_colors = {
        'all_in_CST_Boundary': 'green',
        'center_in_CST_Boundary': 'blue',
        'center_out_CST_Boundary': 'yellow',
        'all_out_CST_Boundary': 'red'
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
            marker = 'o' if '_in_CST_Boundary' in classification else '^'
            ax.plot(x, y, marker=marker, color=color, markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    # Add legend for classifications
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='All in CST Boundary'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Center in CST Boundary'),
        plt.Line2D([0], [0], marker='^', color='yellow', markersize=10, label='Center out CST Boundary'),
        plt.Line2D([0], [0], marker='^', color='red', markersize=10, label='All out CST Boundary')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Set title and turn off axis
    ax.set_title(f"{title_prefix} - CST Cell Classification", fontsize=16)
    ax.axis('off')
    
    # Save the figure
    plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
    print(f"CST classification plot saved to {output_path}") if CST_log_level >= 2 else None
    
    if show_plot:
        plt.show()
    plt.close(fig)

def plot_CST_inclusion(image_RGB, masks, outlines, CST_Boundary_combined_px, R, d_T_per_px, 
                       image_Nx_px, image_Ny_px, cell_inclusion, cell_centroids_px,
                       output_path, show_plot=False, CST_log_level=1, title_prefix="Image", 
                       Convert_to_grayscale_image=True):
    """
    Create a visualization of cell inclusion status in the CST.
    
    Args:
        image_RGB: The RGB image
        masks: The cell masks
        outlines: The cell outlines
        CST_Boundary_combined_px: The CST boundary coordinates in pixel space
        R: Radius of the sphere
        d_T_per_px: Conversion factor from pixels to non-dimensional units
        image_Nx_px: Image width in pixels
        image_Ny_px: Image height in pixels
        cell_inclusion: Dict mapping cell IDs to boolean inclusion status
        cell_centroids_px: Dict mapping cell IDs to (x, y) centroids in pixel space
        output_path: Path to save the figure
        show_plot: Whether to display the plot
        title_prefix: Prefix for the plot title
        Convert_to_grayscale_image: Whether to convert RGB to grayscale
    
    Returns:
        None
    """
    # Define colors for inclusion status
    inclusion_colors = {
        True: 'green',  # Included in CST
        False: 'red'    # Excluded from CST
    }
    
    # Create figure with white background
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Display base RGB image or convert to grayscale
    if Convert_to_grayscale_image:
        image_gray = np.mean(image_RGB, axis=2)  # Convert RGB to grayscale
        ax.imshow(image_gray, cmap='gray')
    else:
        ax.imshow(image_RGB)
    
    # Plot masks with color coding by inclusion status
    # Create a colored mask image
    colored_masks = np.zeros((*masks.shape, 4))  # RGBA
    
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
        is_included = cell_inclusion.get(cell_id)
        if is_included is not None:
            color = inclusion_colors.get(is_included, 'white')
            marker = 'o' if is_included else '^'
            ax.plot(x, y, marker=marker, color=color, markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    # Add legend for inclusion status
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Included in CST'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Excluded from CST'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Set title and turn off axis
    ax.set_title(f"{title_prefix} - CST Cell Inclusion", fontsize=16)
    ax.axis('off')
    
    # Save the figure
    plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
    print(f"CST inclusion plot saved to {output_path}") if CST_log_level >= 2 else None
    
    if show_plot:
        plt.show()
    plt.close(fig)

# Example usage when script is run directly
if __name__ == "__main__":
    print("Running CST_Selection_1 as standalone module...")
    
    # When used in pipeline, input_dir is the Analysis_A11 output dir
    a11_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250610_0004544\20250610_0004569\20250612_2023463\20250615_1957072\20250615_2002477"
    
    # This path would be to the Spherical_Reconstruction_2 output subdirectory in the pipeline
    sr2_output_dir = os.path.join(a11_output_dir, "20250615_2002532") 
    
    # Load the spherically reconstructed DataFrame
    try:
        df_path = os.path.join(sr2_output_dir, "Analysis_A11_df.pkl")
        analysis_df = pd.read_pickle(df_path)
        print(f"Loaded Analysis_A11_df from: {df_path}")
    except FileNotFoundError:
        print(f"DataFrame not found at {df_path}. Using the direct input directory instead.")
        # In a real situation, we would need to handle this case properly
        # For this example, we'll just proceed without a DataFrame (it would fail)
        analysis_df = None
    
    output_dir = CST_Selection_1(
        input_dir=a11_output_dir,  # Main A11 output dir as input_dir for Format_1
        Analysis_A11_df=analysis_df,  # DataFrame from previous step
        CST_log_level=2,
        show_plots=False,
        plot_CST_selection=True,
        Convert_to_grayscale_image=True,
        output_dir_comment="test_standalone"
    )
    
    print(f"CST selection complete. Results in: {output_dir}")

