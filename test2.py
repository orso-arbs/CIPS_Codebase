import glob
import os
import matplotlib.pyplot as plt
from skimage import io, color
import numpy as np

# Define the directory where images are stored
visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop Small First few"

# Get all PNG image files in the directory (you can also use '*.jpg' or other formats if needed)
image_files = glob.glob(os.path.join(visit_images_dir, "*.png"))

# Create a figure to plot the images and histograms
fig, axes = plt.subplots(2, len(image_files), figsize=(15, 10))  # 2 rows: one for images and one for histograms

# Iterate through the image files and plot them
for i, image_file in enumerate(image_files):
    image = io.imread(image_file)
    
    # Convert to grayscale by ignoring the alpha channel
    grayscale_image = color.rgb2gray(image[..., :3])  # Use only the RGB channels
    
    # Print the min and max values of the grayscale image
    min_val = np.min(grayscale_image)
    max_val = np.max(grayscale_image)
    print(f"Image {i+1} - Min value: {min_val}, Max value: {max_val}")
    
    # Plot the grayscale image
    axes[0, i].imshow(grayscale_image, cmap='gray')
    axes[0, i].set_title(f"Grayscale Image {i+1}")
    axes[0, i].axis('off')  # Hide the axis

    # Calculate the histogram of pixel values
    hist, bins = np.histogram(grayscale_image, bins=256, range=(0, 1))
    
    # Plot the histogram
    axes[1, i].plot(bins[:-1], hist, color='black')  # Plot the histogram (excluding last bin)
    axes[1, i].set_title(f"Histogram {i+1}")
    axes[1, i].set_xlim(0, 1)
    axes[1, i].set_xlabel('Pixel Intensity')
    axes[1, i].set_ylabel('Frequency')

# Show the plot
plt.tight_layout()
plt.show()
