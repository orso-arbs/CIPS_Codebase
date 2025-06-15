import numpy as np
from Spherical_Reconstruction_1 import Affine_image_px_and_NonDim

# Create a small set of test points
# Format: 2xN array where first row is x-coordinates, second row is y-coordinates
test_points = np.array([
[100.0, 200.0, 150.0],  # x coordinates
[100.0, 150.0, 200.0]   # y coordinates
])

print("Original test points shape:", test_points.shape)
print("Original test points:\n", test_points)

# Test parameters
image_Nx_px = 3000.0
image_Ny_px = 3000.0
d_T_per_px = 0.0004

# Test pixel to non-dimensional transformation
nonDim_points = Affine_image_px_and_NonDim(
    Coordinates=test_points,
    px_to_nonDim=True,
    nonDim_to_px=False,
    image_Nx_px=image_Nx_px,
    image_Ny_px=image_Ny_px,
    d_T_per_px=d_T_per_px
)
print("\nTransformed to non-dimensional:")
print("Shape:", nonDim_points.shape)
print("Points:\n", nonDim_points)

# Test transform back to pixels
px_points = Affine_image_px_and_NonDim(
    Coordinates=nonDim_points,
    px_to_nonDim=False,
    nonDim_to_px=True,
    image_Nx_px=image_Nx_px,
    image_Ny_px=image_Ny_px,
    d_T_per_px=d_T_per_px
)
print("\nTransformed back to pixels:")
print("Shape:", px_points.shape)
print("Points:\n", px_points)

# Calculate and print the difference between original and transformed back points
diff = test_points - px_points
print("\nDifference between original and transformed back points:")
print("Max absolute difference:", np.max(np.abs(diff)))
print("Difference matrix:\n", diff)


