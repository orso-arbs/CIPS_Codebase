import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# 1. Define the control points for the grayscale colormap.
# This dictionary defines the value (0=black, 1=white) at specific points
# along the colormap (from 0.0 to 1.0).
# The format for each entry is (position, value_at_position, value_at_position).
# For grayscale, 'red', 'green', and 'blue' channels are identical.

cdict = {
    'red':   [(0.0,  0.0, 0.0),   # At point 0.0, the color is black (0)
              (0.25, 1.0, 1.0),   # At point 0.25, the color is white (1)
              (0.75, 0.5, 0.5),   # At point 0.75, the color is mid-gray (0.5)
              (1.0,  0.0, 0.0)],  # At point 1.0, the color is black (0)

    'green': [(0.0,  0.0, 0.0),
              (0.25, 1.0, 1.0),
              (0.75, 0.5, 0.5),
              (1.0,  0.0, 0.0)],

    'blue':  [(0.0,  0.0, 0.0),
              (0.25, 1.0, 1.0),
              (0.75, 0.5, 0.5),
              (1.0,  0.0, 0.0)]
}


# 2. Create the colormap object from the dictionary
cmap_name = "my_custom_bw_colormap"
custom_cmap = mcolors.LinearSegmentedColormap(cmap_name, cdict)

# 3. Create some data to visualize
# Here, we'll create a simple 2D gradient
data = np.random.rand(20, 20)

# 4. Plot the data using the custom colormap
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(data, cmap=custom_cmap)

# Add a colorbar to show the colormap
fig.colorbar(im, ax=ax)
ax.set_title("Plot with a Custom Grayscale Colormap")

# Display the plot
plt.show()
