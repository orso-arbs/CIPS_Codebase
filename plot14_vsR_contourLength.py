import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import Format_1 as F_1
from skimage import measure

# LaTeX settings
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

def plotter_14_vsR_contourLength(
    input_dir,
    x_column='R_SF_nonDim',  # Default to non-dimensional radius
    x_scaling_factor=1,  
    y_column='contour_length_nonDim',  # This will be calculated
    y_scaling_factor=1,  
    output_dir_manual="",
    output_dir_comment="",
    image_list=[],
    omit_image_list=[],  # New parameter to omit specific images
    connect_with_lines=True,
    marker_style='o',
    marker_size=6,
    line_style='-',
    line_width=1.5,
    line_color='black',
    marker_color='blue',
    show_grid=True,      # New parameter to control grid display
    grid_style='--',     # New parameter for grid line style
    grid_width=0.5,      # New parameter for grid line width
    grid_color='gray',   # New parameter for grid color
    grid_alpha=0.5,      # New parameter for grid transparency
    x_label=None,
    y_label=None,
    legend_label=None,   # New parameter for custom legend label
    x_label_fontsize=20,
    y_label_fontsize=20,
    tick_label_fontsize=20,
    legend_fontsize=12,
    legend_loc='upper left',
    figsize=(10, 6),
    dpi=100,
    show_plot=0,
    Plot_log_level=2
):
    """
    Creates a plot of the total contour length of cell outlines vs. non-dimensional radius.
    
    Parameters
    ----------
    input_dir : str
        Directory containing the Analysis_A11_final_df.pkl file
    x_column : str, optional
        Column name to plot on x-axis, by default 'R_SF_nonDim'
    x_scaling_factor : float, optional
        Factor to multiply x-axis values by, by default 1
    y_column : str, optional
        Column name to plot on y-axis, by default 'contour_length_nonDim'
    y_scaling_factor : float, optional
        Factor to multiply y-axis values by, by default 1
    output_dir_manual : str, optional
        Manual output directory, by default ""
    output_dir_comment : str, optional
        Comment to append to the output directory name, by default ""
    image_list : list, optional
        List of image numbers to include in plot, if empty all images are used, by default []
    omit_image_list : list, optional
        List of image numbers to exclude from plot (applied after image_list filter), by default []
    connect_with_lines : bool, optional
        Whether to connect points with lines, by default True
    marker_style : str, optional
        Style of markers, by default 'o'
    marker_size : int, optional
        Size of markers, by default 6
    line_style : str, optional
        Style of lines, by default '-'
    line_width : float, optional
        Width of lines, by default 1.5
    line_color : str, optional
        Color of lines, by default 'blue'
    marker_color : str, optional
        Color of markers, by default 'blue'
    show_grid : bool, optional
        Whether to show grid lines, by default True
    grid_style : str, optional
        Style of grid lines, by default '--'
    grid_width : float, optional
        Width of grid lines, by default 0.5
    grid_color : str, optional
        Color of grid lines, by default 'gray'
    grid_alpha : float, optional
        Transparency of grid lines (0-1), by default 0.5
    x_label : str, optional
        Label for x-axis, by default None (will use x_column)
    y_label : str, optional
        Label for y-axis, by default None (will use y_column)
    legend_label : str, optional
        Custom text for the legend label. If None, no legend is shown.
    x_label_fontsize : int, optional
        Font size for x-axis label, by default 16
    y_label_fontsize : int, optional
        Font size for y-axis label, by default 16
    tick_label_fontsize : int, optional
        Font size for tick labels, by default 12
    legend_fontsize : int, optional
        Font size for legend, by default 12
    legend_loc : str, optional
        Location of legend, by default 'upper left'
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (10, 6)
    dpi : int, optional
        DPI for the figure, by default 100
    show_plot : int, optional
        Whether to display the plot (1) or not (0), by default 0
    Plot_log_level : int, optional
        Logging level, by default 1
    
    Returns
    -------
    str
        Path to the output directory
    """
    # Create output directory
    output_dir = F_1.F_out_dir(input_dir=input_dir, script_path=__file__, 
                             output_dir_comment=output_dir_comment, 
                             output_dir_manual=output_dir_manual)
    
    if Plot_log_level >= 1:
        print(f"plotter_14_vsR_contourLength: Output directory: {output_dir}")

    # Create directories for PNG and SVG outputs
    png_dir = os.path.join(output_dir, "png")
    svg_dir = os.path.join(output_dir, "svg")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    # Find the PKL file
    pandas_wildcard_str = os.path.join(input_dir, "Analysis_A11_final_df.pkl")
    pkl_files = glob.glob(pandas_wildcard_str)
    
    if not pkl_files:
        print(f"No Analysis_A11_final_df.pkl file found in {input_dir}")
        return output_dir
    
    # Load the DataFrame
    df_path = pkl_files[0]
    df = pd.read_pickle(df_path)
    
    if Plot_log_level >= 1:
        print(f"Loaded DataFrame from {df_path}")
    
    # Check if the required columns exist for x-axis
    if x_column not in df.columns:
        print(f"Column '{x_column}' not found in DataFrame. Available columns: {df.columns.tolist()}")
        return output_dir
    
    # Filter DataFrame based on image_list if provided (do this BEFORE calculating contour lengths)
    if image_list:
        df = df[df['image_number'].isin(image_list)]
        if df.empty:
            print(f"No matching images found for the provided image_list: {image_list}")
            return output_dir
        if Plot_log_level >= 1:
            print(f"Processing only {len(df)} images from image_list")
    
    # Exclude images in omit_image_list if provided
    if omit_image_list:
        df = df[~df['image_number'].isin(omit_image_list)]
        if df.empty:
            print(f"No images remaining after applying omit_image_list: {omit_image_list}")
            return output_dir
        if Plot_log_level >= 1:
            print(f"Excluded {len(omit_image_list)} images: {omit_image_list}")
    
    # Calculate contour length for each image (now only for filtered images)
    contour_length_px = []
    contour_length_nonDim = []
    contour_length_CST_px = []
    contour_length_CST_nonDim = []
    contour_length_CSTx6_px = []
    contour_length_CSTx6_nonDim = []
    
    for i, row in df.iterrows():
        # Get the masks and nonDim_per_px from the DataFrame
        masks = row['masks']
        nonDim_per_px = row['nonDim_per_px']
        
        # Get CST inclusion array if it exists
        has_cst_inclusion = 'CST_inclusion' in row and row['CST_inclusion'] is not None and len(row['CST_inclusion']) > 0
        if has_cst_inclusion:
            cst_inclusion = row['CST_inclusion']
            if Plot_log_level >= 2:
                print(f"Found CST inclusion data with {len(cst_inclusion)} entries")
        else:
            if Plot_log_level >= 1:
                print(f"Warning: No CST inclusion data found for image {i+1}, will calculate total contour length only")
        
        if Plot_log_level >= 2:
            print(f"Processing image {i+1}/{len(df)} for contour length calculation")
        
        # Calculate total contour length in pixels
        total_length_px = 0
        cst_length_px = 0
        
        # Get unique cell IDs (excluding background)
        cell_ids = np.unique(masks)
        cell_ids = cell_ids[cell_ids > 0]
        
        for idx, cell_id in enumerate(cell_ids):
            # Check if we have valid CST inclusion data for this cell
            is_in_cst = False
            if has_cst_inclusion and idx < len(cst_inclusion):
                is_in_cst = cst_inclusion[idx]
            
            # Create binary mask for this cell
            print(f"Processing cell ID {cell_id}/{len(cell_ids)} in image {i+1}", end='\r', flush=True)
            cell_mask = masks == cell_id
            
            # Find contours for this cell
            contours = measure.find_contours(cell_mask, 0.5)
            
            # Calculate length of this cell's contour
            cell_length_px = 0
            
            for contour in contours:
                for j in range(len(contour)-1):
                    dy = contour[j+1, 0] - contour[j, 0]
                    dx = contour[j+1, 1] - contour[j, 1]
                    segment_length = np.sqrt(dx**2 + dy**2)
                    cell_length_px += segment_length
            
            # Add to total length for all cells
            total_length_px += cell_length_px
            
            # If the cell is in the CST, add to CST length
            if is_in_cst:
                cst_length_px += cell_length_px
        
        # Convert to non-dimensional units
        total_length_nonDim = total_length_px * nonDim_per_px
        cst_length_nonDim = cst_length_px * nonDim_per_px
        
        # Calculate CST x6 (total sphere) values
        cst_length_x6_px = cst_length_px * 6
        cst_length_x6_nonDim = cst_length_nonDim * 6
        
        # Append to lists
        contour_length_px.append(total_length_px)
        contour_length_nonDim.append(total_length_nonDim)
        contour_length_CST_px.append(cst_length_px)
        contour_length_CST_nonDim.append(cst_length_nonDim)
        contour_length_CSTx6_px.append(cst_length_x6_px)
        contour_length_CSTx6_nonDim.append(cst_length_x6_nonDim)
        
        if Plot_log_level >= 2:
            print(f"\nImage {i+1} contour length calculations:")
            print(f"  Total contour length = {total_length_px:.2f} px = {total_length_nonDim:.4f} non-dim")
            print(f"  CST contour length = {cst_length_px:.2f} px = {cst_length_nonDim:.4f} non-dim")
            print(f"  CST x6 contour length = {cst_length_x6_px:.2f} px = {cst_length_x6_nonDim:.4f} non-dim")
    
    # Add the calculated values to the DataFrame
    df['contour_length_px'] = contour_length_px
    df['contour_length_nonDim'] = contour_length_nonDim
    df['contour_length_CST_px'] = contour_length_CST_px
    df['contour_length_CST_nonDim'] = contour_length_CST_nonDim
    df['contour_length_CSTx6_px'] = contour_length_CSTx6_px
    df['contour_length_CSTx6_nonDim'] = contour_length_CSTx6_nonDim
    
    if Plot_log_level >= 1:
        print(f"Calculated contour lengths for {len(df)} images")
    
    # Create figure
    plt.figure(figsize=figsize, dpi=dpi)
    
    # Set font size for all elements
    plt.rcParams['font.size'] = tick_label_fontsize
    
    # Create plot
    if connect_with_lines:
        plt.plot(df[x_column] * x_scaling_factor, df[y_column] * y_scaling_factor, 
                marker=marker_style, markersize=marker_size, 
                linestyle=line_style, linewidth=line_width,
                color=line_color, markerfacecolor=marker_color, 
                markeredgecolor='black', label=legend_label)
    else:
        plt.scatter(df[x_column] * x_scaling_factor, df[y_column] * y_scaling_factor, 
                  s=marker_size**2, marker=marker_style,
                  color=marker_color, edgecolors='black', 
                  label=legend_label)
    
    # Set default labels if not provided
    if x_label is None:
        x_label = f"Non-dimensional Radius (R)"
    
    if y_label is None:
        y_label = f"Total Contour Length (non-dim)"
    
    # Set labels
    plt.xlabel(x_label, fontsize=x_label_fontsize)
    plt.ylabel(y_label, fontsize=y_label_fontsize)
    
    # Set tick parameters for inward facing ticks
    plt.tick_params(axis='both', direction='in', which='both', labelsize=tick_label_fontsize)
    
    # Add grid if requested
    if show_grid:
        plt.grid(True, linestyle=grid_style, linewidth=grid_width, 
                alpha=grid_alpha, color=grid_color)
    
    # Add legend only if a legend_label is provided
    if legend_label is not None:
        plt.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figures - include both column names in filename
    base_filename = f"{x_column.replace(' ', '_')}_vs_{y_column.replace(' ', '_')}"
    png_path = os.path.join(png_dir, f"{base_filename}.png")
    svg_path = os.path.join(svg_dir, f"{base_filename}.svg")
    
    plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    
    if Plot_log_level >= 1:
        print(f"Saved figures to:\n  {png_path}\n  {svg_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return output_dir

if __name__ == "__main__":
    # Example usage
    plotter_14_vsR_contourLength(
        input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_1647163",
        output_dir_comment="contour_length_plot",
        x_column="R_SF_nonDim",
        # Use the CST x6 contour length for y-axis
        y_column="contour_length_CSTx6_nonDim",
        image_list=[],
        omit_image_list=[106],
        connect_with_lines=True,
        marker_style='',
        marker_size=8,
        line_color='black',
        marker_color='black',
        show_grid=True,
        grid_alpha=0.3,
        legend_label="Total Interface Length",
        x_label=r'$R_{SF}/\delta_T$',
        y_label=r'Total Contour Length $/\delta_T$',
        show_plot=1,
        Plot_log_level=1
    )