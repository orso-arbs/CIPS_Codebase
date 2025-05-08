import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Constants
S = 1.5
a = 1.0

# Sweep settings
n_x_deviaiton = 0.5
n_x_min = -n_x_deviaiton
n_x_max = n_x_deviaiton
n_x_N = 3
n_x_values = np.linspace(n_x_min, n_x_max, n_x_N)

# Grid
Ngrid = 200
D_X = 3.0
D_Y = 2.0
x = np.linspace(-D_X, D_X, Ngrid)
y = np.linspace(-D_Y, D_Y, Ngrid)
X, Y = np.meshgrid(x, y)
Y_safe = np.where(np.abs(Y) < 1e-3, 1e-3, Y)

# Plot
fig, axes = plt.subplots(1, n_x_N, figsize=(4 * n_x_N, 4), sharex=True, sharey=True)

# We will store the last streamplot for the colorbar
last_strm = None

for ax, n_x in zip(axes, n_x_values):
    T_X = - (S / (a * Y_safe) + n_x * a * X)
    T_Y = np.full_like(T_X, n_x)

    U_X = a * X
    U_Y = - a * Y

    # Streamplot
    ax.streamplot(X, Y, U_X, U_Y, color='lightblue', density=0.8, linewidth=0.4, zorder=0)


    magnitude = np.hypot(T_X, T_Y)
    # Split point
    color_max = 4.0  # threshold value for first colormap

    # Define total range
    data_max = magnitude.max()
    n_colors1 = 256   # number of colors in first colormap
    n_colors2 = 64    # number in second

    # Create boundaries with a breakpoint at color_max
    boundaries1 = np.linspace(0, color_max, n_colors1 + 1)
    boundaries2 = np.linspace(color_max, data_max, n_colors2 + 1)[1:]  # skip duplicate
    boundaries = np.concatenate((boundaries1, boundaries2))
    norm = BoundaryNorm(boundaries, n_colors1 + n_colors2)

    # Combine colormaps
    cmap1 = plt.cm.plasma(np.linspace(0, 1, n_colors1))
    cmap2 = plt.cm.gray(np.linspace(0.3, 1, n_colors2))
    combined_colors = np.vstack((cmap1, cmap2))
    combined_cmap = ListedColormap(combined_colors)

    strm = ax.streamplot(X, Y, T_X, T_Y, color=np.hypot(T_X, T_Y), cmap=combined_cmap, norm=norm, density=0.8)
    last_strm = strm  # save for colorbar

    ax.axhline( S / a, color='red', linewidth=1)
    ax.axhline( - S / a, color='red', linewidth=1)

    # Axis lines
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)

    ax.set_title(f'$n_x={n_x:.2f}$')
    ax.set_aspect('equal')
    ax.set_xlim(-D_X, D_X)
    ax.set_ylim(-D_Y, D_Y)
    ax.grid(True)

# Adjust layout
plt.tight_layout(rect=[0, 0, 0.90, 0.90])  # Leave space

# Add colorbar manually
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(last_strm.lines, cax=cbar_ax, label='|$(T_X, T_Y)$|')

fig.suptitle('Streamlines for various $n_x$', y=0.95)
plt.show()
