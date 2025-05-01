import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def test_ax1_and_twin():
    # Mock data similar to what your dataframe likely contains
    time = np.linspace(0, 1, 100)
    R_SF_nonDim = np.sin(2 * np.pi * time) * 0.5 + 1
    R_SF_px = np.cos(2 * np.pi * time) * 10 + 50

    dimentionalised_df = pd.DataFrame({
        'Time_VisIt': time,
        'R_SF_nonDim': R_SF_nonDim,
        'R_SF_px': R_SF_px,
    })

    fig, (ax_1, ax_2, ax_3, ax_4) = plt.subplots(4, 1, figsize=(8, 10))

    # Subplot 1: R_SF_nonDim and R_SF_px vs time
    ax_1.plot(dimentionalised_df['Time_VisIt'], dimentionalised_df['R_SF_nonDim'], label="R_SF_nonDim", color='orange', linestyle='solid')
    ax_1.set_xlabel('Time')
    ax_1.set_ylabel("R_SF_nonDim", color='orange')
    ax_1.tick_params(axis='y', labelcolor='orange')
    ax_1.spines["left"].set_position(("outward", 0))
    ax_1.set_title('Spherical Flame Radius Comparison of nondimensionalised ($d_L$) vs px\nof VisIt')

    # Twin axis
    ax_1_twin = ax_1.twinx()
    ax_1_twin.plot(dimentionalised_df['Time_VisIt'], dimentionalised_df['R_SF_px'],
                   label="R_SF_px", color='orange', linestyle='dashed')
    ax_1.spines['right'].set_visible(False)
    ax_1_twin.set_ylabel("R_SF_px", color='orange')
    ax_1_twin.tick_params(axis='y', labelcolor='orange', colors='orange')
    ax_1_twin.spines["right"].set_linestyle('dashed')
    ax_1_twin.spines["right"].set_color('orange')
    ax_1_twin.spines["right"].set_position(("outward", 0))

    fig.tight_layout()
    plt.show()

# Run the test
test_ax1_and_twin()
