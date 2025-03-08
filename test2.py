import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Create some example data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = np.exp(-x)

# Create a figure with a custom GridSpec layout
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 3, figure=fig)

# Create subplots with the specified grid layout
ax1 = fig.add_subplot(gs[0, 0])   # (1,1)
ax2 = fig.add_subplot(gs[0, 1])   # (1,2)
ax3 = fig.add_subplot(gs[1, 0])   # (2,1)
ax4 = fig.add_subplot(gs[1, 1:3])   # (2,2)
#ax5 = fig.add_subplot(gs[1, 2])   # (2,3)

# Plot: Sine curve in ax1 (1,1)
ax1.plot(x, y1, label='Sine Wave', color='blue')
ax1.set_title("Sine Wave")
ax1.set_xlabel("x")
ax1.set_ylabel("sin(x)")
ax1.legend()

# Plot: Cosine curve in ax2 (1,2)
ax2.plot(x, y2, label='Cosine Wave', color='green')
ax2.set_title("Cosine Wave")
ax2.set_xlabel("x")
ax2.set_ylabel("cos(x)")
ax2.legend()

# Combine ax1 and ax2 by merging them into one horizontally elongated subplot
ax1.plot(x, y1, label='Sine Wave', color='blue')
ax1.set_title("Sine Wave and Cosine Wave")
ax1.set_xlabel("x")
ax1.set_ylabel("Function Value")
ax1.legend()
ax2.axis('off')  # Hide ax2 since it's now merged into ax1

# Plot: Tangent curve in ax3 (2,1)
ax3.plot(x, y3, label='Tangent Wave', color='red')
ax3.set_title("Tangent Wave")
ax3.set_xlabel("x")
ax3.set_ylabel("tan(x)")
ax3.set_ylim(-10, 10)  # Limit y to avoid large values of tan(x)

# Plot: Exponential decay in ax4 (2,2)
ax4.plot(x, y4, label='Exponential Decay', color='purple')
ax4.set_title("Exponential Decay")
ax4.set_xlabel("x")
ax4.set_ylabel("exp(-x)")
ax4.legend()

# Plot: Leave empty in ax5 (2,3)
#ax5.axis('off')

# Adjust the layout to avoid overlap
plt.tight_layout()

# Show the plot
plt.show()
