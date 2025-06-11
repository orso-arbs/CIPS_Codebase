import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Set image dimensions as parameters
width = 4
height = 4

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Function to create the coordinate grid plot
def create_coordinate_grid(ax, w, h):
    # Create simple image with specified dimensions
    img = np.zeros((h, w))
    
    # Display the image with white background
    cmap = ListedColormap(['white'])  # Use white color only
    im = ax.imshow(img, cmap=cmap)
    
    # Add gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='black', linewidth=2)
    ax.set_xticks(np.arange(-.5, w, 1))
    ax.set_yticks(np.arange(-.5, h, 1))
    
    # Remove ticks completely
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Add coordinates in each pixel
    for i in range(h):
        for j in range(w):
            # Calculate a and b coordinates
            a = (j + 1/2) - w/2
            b = h/2 - (i + 1/2)
            
            # Create text labels for both coordinate systems
            text = f"px: ({j},{i})\ncentered: ({a:.1f},{b:.1f})"
            
            ax.text(j, i, text, ha='center', va='center', 
                    color='black', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.0, pad=1))
    
    ax.set_title(f"{w}x{h} Pixel Image")

# Create first subplot with original dimensions
create_coordinate_grid(ax1, width, height)

# Create second subplot with dimensions+1
create_coordinate_grid(ax2, width+1, height+1)

plt.tight_layout()
plt.show()
