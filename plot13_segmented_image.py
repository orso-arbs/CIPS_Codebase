import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib import rcParams
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as PathEffects
from scipy.ndimage import binary_dilation
import pickle

# Set LaTeX font
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

def format_1(input_dir):
    """
    Format directory paths based on Format_1 convention.
    
    Parameters:
    -----------
    input_dir : str
        Input directory path.
        
    Returns:
    --------
    tuple
        (input_dir, output_dir)
    """
    if input_dir[-1] != '/':
        input_dir = input_dir + '/'
    output_dir = input_dir + "Plots/"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    return input_dir, output_dir

def load_data(input_dir, image_number=100):
    """
    Load the dataframe and extract data for the specified image number.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the analysis dataframe.
    image_number : int, optional
        Image number to visualize. Default is 100.
        
    Returns:
    --------
    dict
        Dictionary containing image data, masks, and metadata.
    """
    # Load the dataframe
    df_path = os.path.join(input_dir, 'Analysis_A11_final_df.pkl')
    with open(df_path, 'rb') as f:
        df = pickle.load(f)
    
    # Extract data for the specified image number
    img_data = df[df['image_number'] == image_number]
    
    if img_data.empty:
        raise ValueError(f"Image number {image_number} not found in the dataframe.")
    
    # Get the first row (should be metadata for the image)
    img_row = img_data.iloc[0]
    
    # Extract image, mask, outlines
    img = img_row.get('img', None)
    masks = img_row.get('masks', None)
    outlines = img_row.get('outlines', None)
    
    return {
        'image': img,
        'masks': masks,
        'outlines': outlines,
        'metadata': img_row,
        'df_data': img_data
    }

def plot_segmented_image(data, show_masks=True, show_outlines=True, mask_alpha=0.5, 
                         zoom_factor=2, label_content=None, label_position=(0.05, 0.95),
                         label_size=12, figsize=(8, 8)):
    """
    Plot the segmented image with masks and outlines.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing image data and masks from load_data function.
    show_masks : bool, optional
        Whether to show cell masks. Default is True.
    show_outlines : bool, optional
        Whether to show cell outlines. Default is True.
    mask_alpha : float, optional
        Transparency of the masks. Default is 0.5.
    zoom_factor : float, optional
        Factor to zoom in to the center. Default is 2.
    label_content : str, optional
        Content of the label to display. Default is None (no label).
    label_position : tuple, optional
        Position of the label (x, y) in figure coordinates. Default is (0.05, 0.95).
    label_size : int, optional
        Size of the label font. Default is 12.
    figsize : tuple, optional
        Figure size in inches. Default is (8, 8).
        
    Returns:
    --------
    tuple
        (figure, axis) matplotlib objects
    """
    # Extract data
    img = data['image']
    masks = data['masks']
    
    if img is None:
        raise ValueError("Image data not found.")
    if masks is None and show_masks:
        print("Warning: Mask data not found, proceeding without masks.")
        show_masks = False
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate zoom window
    if zoom_factor > 1:
        h, w = img.shape[:2]
        center_h, center_w = h // 2, w // 2
        new_h, new_w = h // zoom_factor, w // zoom_factor
        y_start = center_h - new_h // 2
        y_end = y_start + new_h
        x_start = center_w - new_w // 2
        x_end = x_start + new_w
        
        # Ensure bounds are within image dimensions
        y_start = max(0, y_start)
        y_end = min(h, y_end)
        x_start = max(0, x_start)
        x_end = min(w, x_end)
        
        # Crop image and masks
        img = img[y_start:y_end, x_start:x_end]
        if masks is not None:
            masks = masks[y_start:y_end, x_start:x_end]
    
    # Display the base image
    if len(img.shape) == 2:  # Grayscale image
        ax.imshow(img, cmap='gray')
    else:  # Color image
        ax.imshow(img)
    
    # Overlay masks if requested
    if show_masks and masks is not None:
        # Create colormap for masks
        unique_cells = np.unique(masks)
        unique_cells = unique_cells[unique_cells > 0]  # Skip background (0)
        
        # Random colors for each cell
        np.random.seed(42)  # For reproducibility
        colors = np.random.rand(len(unique_cells), 4)
        colors[:, -1] = mask_alpha  # Set alpha
        
        # Create and apply mask overlay
        mask_overlay = np.zeros((*masks.shape, 4))
        for i, cell_id in enumerate(unique_cells):
            cell_mask = masks == cell_id
            for c in range(4):
                mask_overlay[cell_mask, c] = colors[i % len(colors), c]
        
        ax.imshow(mask_overlay)
    
    # Add outlines if requested
    if show_outlines and data['outlines'] is not None:
        outlines = data['outlines']
        if zoom_factor > 1:
            outlines = outlines[y_start:y_end, x_start:x_end]
        
        # Create a binary mask for outlines
        outline_mask = outlines > 0
        
        # Display outlines in white
        y_coords, x_coords = np.where(outline_mask)
        ax.scatter(x_coords, y_coords, s=0.5, color='white', alpha=1)
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add label if content is provided
    if label_content:
        ax.text(label_position[0], label_position[1], label_content,
                transform=ax.transAxes, fontsize=label_size, 
                verticalalignment='top', color='white',
                bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    return fig, ax

def save_plot(fig, output_dir, image_number, dpi=300):
    """
    Save the plot as SVG and PNG files.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure object to save.
    output_dir : str
        Directory to save the plot.
    image_number : int
        Image number for filename.
    dpi : int, optional
        Resolution for PNG export. Default is 300.
    """
    base_filename = f"plot13_segmented_image_{image_number}"
    
    # Save as SVG
    svg_path = os.path.join(output_dir, f"{base_filename}.svg")
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    # Save as PNG
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    fig.savefig(png_path, format='png', dpi=dpi, bbox_inches='tight')
    
    print(f"Plots saved to {svg_path} and {png_path}")

def main(input_dir, image_number=100, show_masks=True, show_outlines=True, 
         mask_alpha=0.5, zoom_factor=2, label_content=None, 
         label_position=(0.05, 0.95), label_size=12, figsize=(8, 8)):
    """
    Main function to create and save the segmented image plot.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the analysis dataframe.
    image_number : int, optional
        Image number to visualize. Default is 100.
    show_masks : bool, optional
        Whether to show cell masks. Default is True.
    show_outlines : bool, optional
        Whether to show cell outlines. Default is True.
    mask_alpha : float, optional
        Transparency of the masks. Default is 0.5.
    zoom_factor : float, optional
        Factor to zoom in to the center. Default is 2.
    label_content : str, optional
        Content of the label to display. Default is None (no label).
    label_position : tuple, optional
        Position of the label (x, y) in figure coordinates. Default is (0.05, 0.95).
    label_size : int, optional
        Size of the label font. Default is 12.
    figsize : tuple, optional
        Figure size in inches. Default is (8, 8).
    """
    # Format directories
    input_dir, output_dir = format_1(input_dir)
    
    # Load data
    data = load_data(input_dir, image_number)
    
    # Create plot
    fig, ax = plot_segmented_image(
        data,
        show_masks=show_masks,
        show_outlines=show_outlines,
        mask_alpha=mask_alpha,
        zoom_factor=zoom_factor,
        label_content=label_content,
        label_position=label_position,
        label_size=label_size,
        figsize=figsize
    )
    
    # Save plot
    save_plot(fig, output_dir, image_number)
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    # Set parameters here for VSCode execution
    INPUT_DIR = r"C:\your\path\to\data\directory"  # Replace with your actual path
    IMAGE_NUMBER = 100
    SHOW_MASKS = True
    SHOW_OUTLINES = True
    MASK_ALPHA = 0.5
    ZOOM_FACTOR = 2
    LABEL_CONTENT = "Segmented Cells"  # Set to None for no label
    LABEL_POSITION = (0.05, 0.95)  # (x, y) in figure coordinates
    LABEL_SIZE = 12
    FIGSIZE = (8, 8)  # inches
    
    main(
        input_dir=INPUT_DIR,
        image_number=IMAGE_NUMBER,
        show_masks=SHOW_MASKS,
        show_outlines=SHOW_OUTLINES,
        mask_alpha=MASK_ALPHA,
        zoom_factor=ZOOM_FACTOR,
        label_content=LABEL_CONTENT,
        label_position=LABEL_POSITION,
        label_size=LABEL_SIZE,
        figsize=FIGSIZE
    )
