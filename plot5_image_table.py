import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import Format_1 as F_1

# LaTeX settings
plt.rcParams['text.usetex'] = True
LATEX_FONT_SIZE = 20  # Global font size for LaTeX text
plt.rcParams['font.size'] = LATEX_FONT_SIZE

@F_1.ParameterLog(max_size = 1024 * 10) # 10KB 
def plot_image_table(
    input_dir,
    image_paths,  # List of image file paths
    n_rows,       # Number of rows
    n_cols,       # Number of columns
    row_labels,   # List of row labels (can contain LaTeX math expressions)
    col_labels,   # List of column labels (can contain LaTeX math expressions)
    subcaptions=None,  # Add subcaptions parameter, list of strings for each image
    figsize=(15, 10),  # Figure size in inches
    output_dir_manual="",
    output_dir_comment="",
    show_plot=0,
    Plot_log_level=0,
    ):
    """
    Creates a table of images with row and column labels.
    
    Parameters:
    -----------
    input_dir : str
        Input directory for F_1.F_out_dir
    image_paths : list of str
        List of paths to images. Length must be n_rows * n_cols
    n_rows, n_cols : int
        Number of rows and columns in the grid
    row_labels, col_labels : list of str
        Lists of labels for rows and columns. Can include LaTeX math expressions
        between $ symbols, e.g., "$\\omega_1$"
    """
    print("Running plot_image_table") if Plot_log_level >= 1 else None
    
    # Create output directory
    output_dir = F_1.F_out_dir(input_dir, __file__, 
                              output_dir_manual=output_dir_manual,
                              output_dir_comment=output_dir_comment)

    # Validate inputs
    if len(image_paths) != n_rows * n_cols:
        raise ValueError(f"Number of images ({len(image_paths)}) must equal n_rows * n_cols ({n_rows * n_cols})")
    if len(row_labels) != n_rows:
        raise ValueError(f"Number of row labels ({len(row_labels)}) must equal n_rows ({n_rows})")
    if len(col_labels) != n_cols:
        raise ValueError(f"Number of column labels ({len(col_labels)}) must equal n_cols ({n_cols})")

    # Create figure with enough space for labels
    # Calculate figure size to maintain aspect ratio
    sample_img = mpimg.imread(image_paths[0])
    img_aspect = sample_img.shape[1] / sample_img.shape[0]  # width/height
    base_width = figsize[0] / (n_cols + 0.2)  # Account for label column
    fig_width = figsize[0]
    fig_height = base_width * n_rows / img_aspect + figsize[0] * 0.1  # Reduced from 0.2 to 0.1 for labels

    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Calculate grid spacing to accommodate labels and subcaptions
    grid = plt.GridSpec(n_rows*2 + 1, n_cols + 1, figure=fig,
                       hspace=0.3,  # Increased from 0.05 to 0.3 for more vertical spacing
                       wspace=0.3,  # Increased from 0.1 to 0.3 for more horizontal spacing
                       height_ratios=[0.3] + [1, 0.15]*n_rows,  # Increased top label height from 0.15 to 0.3
                       width_ratios=[0.4] + [1]*n_cols)   # Increased left label width from 0.15 to 0.4

    # Add column labels
    for j, label in enumerate(col_labels):
        ax = fig.add_subplot(grid[0, j+1])
        ax.text(0.5, 0.5, label, ha='center', va='center', fontsize=LATEX_FONT_SIZE)
        ax.axis('off')

    # Add row labels and images
    for i in range(n_rows):
        # Row label
        ax = fig.add_subplot(grid[i*2+1, 0])
        ax.text(0.5, 0.5, row_labels[i], 
               ha='center',  # Keep horizontally centered in the cell
               va='center', 
               fontsize=LATEX_FONT_SIZE,
               linespacing=0.8,  # Adjust line spacing between rows
               multialignment='left')  # Left-align multiple lines within the text
        ax.axis('off')
        
        # Images in this row
        for j in range(n_cols):
            # Image subplot
            ax_img = fig.add_subplot(grid[i*2+1, j+1])  # Adjust grid position for images
            img_idx = i * n_cols + j
            try:
                img = mpimg.imread(image_paths[img_idx])
                ax_img.imshow(img, aspect='equal')  # Force equal aspect ratio
                # Add letter label ((a), (b), etc.) in top left corner using LaTeX
                letter = chr(ord('a') + img_idx)  # Convert 0,1,2,... to a,b,c,...
                ax_img.text(0.05, 0.95, f"$({letter})$", 
                       transform=ax_img.transAxes,  # Use axis coordinates (0-1)
                       fontsize=LATEX_FONT_SIZE,
                       color='black',
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.0),
                       verticalalignment='top')
                ax_img.axis('off')

                # Add subcaption if provided
                if subcaptions and img_idx < len(subcaptions):
                    ax_sub = fig.add_subplot(grid[i*2+2, j+1])
                    ax_sub.text(0.5, 0.8, subcaptions[img_idx],  # Moved up from 0.5 to 0.8
                              ha='center', va='center', 
                              fontsize=LATEX_FONT_SIZE*0.8,
                              transform=ax_sub.transAxes)
                    ax_sub.axis('off')

            except Exception as e:
                print(f"Error loading image {image_paths[img_idx]}: {e}")
                ax_img.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
                ax_img.axis('off')

    # Save plot
    plt.savefig(os.path.join(output_dir, 'image_table.png'), 
                bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, 'image_table.pdf'), 
                bbox_inches='tight')
    print(f"Saved plots to {output_dir}") if Plot_log_level >= 1 else None
    
    if show_plot:
        plt.show()
    else:
        plt.close()

    return output_dir

# Example usage
if __name__ == "__main__":
    # Example data
    example_images = [
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Manuscript\Images\Resolutions and diameters crops with disk\S0-1000-10disk.png",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Manuscript\Images\Resolutions and diameters crops with disk\S0-1000-21disk.png",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Manuscript\Images\Resolutions and diameters crops with disk\S0-1000-42disk.png",

        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Manuscript\Images\Resolutions and diameters crops with disk\S0-3000-15disk.png",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Manuscript\Images\Resolutions and diameters crops with disk\S0-3000-29disk.png",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Manuscript\Images\Resolutions and diameters crops with disk\S0-3000-60disk.png",


        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Manuscript\Images\Resolutions and diameters crops with disk\S79-1000-20disk.png",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Manuscript\Images\Resolutions and diameters crops with disk\S79-1000-40disk.png",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Manuscript\Images\Resolutions and diameters crops with disk\S79-1000-80disk.png",

        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Manuscript\Images\Resolutions and diameters crops with disk\S79-3000-55disk.png",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Manuscript\Images\Resolutions and diameters crops with disk\S79-3000-111disk.png",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Manuscript\Images\Resolutions and diameters crops with disk\S79-3000-222disk.png",
    ]
    
    row_labels = [
        # Changed row label format to ensure alignment using array
        r"$\begin{array}{l}1000^2px^2\\[3pt] R=65px\end{array}$",
        r"$\begin{array}{l}3000^2px^2\\[3pt] R=193px\end{array}$",
        r"$\begin{array}{l}1000^2px^2\\[3pt] R=306px\end{array}$",
        r"$\begin{array}{l}3000^2px^2\\[3pt] R=896px\end{array}$",
    ]
    col_labels = [r"$\frac{1}{2}d_{auto\ estimate}$", r"$d_{auto\ estimate}$", r"$2d_{auto\ estimate}$"]
    
    # Fix subcaption formatting
    subcaptions = [
        r"$d_{estimate}=10px$",
        r"$d_{estimate}=21px$",
        r"$d_{estimate}=42px$",
        
        r"$d_{estimate}=15px$",
        r"$d_{estimate}=29px$",
        r"$d_{estimate}=60px$",
        
        r"$d_{estimate}=20px$",
        r"$d_{estimate}=40px$",
        r"$d_{estimate}=80px$",
        
        r"$d_{estimate}=55px$",
        r"$d_{estimate}=111px$",
        r"$d_{estimate}=222px$",
    ]

    plot_image_table(
        input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_misc",
        image_paths=example_images,
        n_rows=4,
        n_cols=3,
        row_labels=row_labels,
        col_labels=col_labels,
        subcaptions=subcaptions,
        show_plot=0,
        Plot_log_level=1
    )
