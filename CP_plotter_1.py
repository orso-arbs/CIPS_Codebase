import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, plot, utils, io
import datetime
import glob
import os
import time
import pandas as pd
import matplotlib.gridspec as gridspec
import numpy as np
from skimage import io as sk_io, color, measure

import sys
import os
sys.path.append(os.path.abspath(r"C:/Users/obs/OneDrive/ETH/ETH_MSc/Masters Thesis/Python Code/Python_Orso_Utility_Scripts_MscThesis")) # dir containing Format_1 
import Format_1 as F_1

import video_maker_1 as vm1


@F_1.ParameterLog(max_size = 1024 * 10) # 10KB 
def CP_plotter_1(input_dir, # Format_1 requires input_dir
    df,
    output_dir_manual = "", output_dir_comment = "",
    video = 1,
    ):

    ### output 
    output_dir = F_1.F_out_dir(input_dir, __file__, output_dir_comment = "") # Format_1 required definition of output directory


    # auxillary function to plot the data

    # Number of rows in the DataFrame
    N_images = len(df)

    # Find the maximum frequency for all histograms
    bin_size = 2
    max_frequency = 0
    for i in range(N_images):
        unique_diameters, counts_diameters = np.unique(df.loc[i, 'diameter_distribution'], return_counts=True)
        max_diameter = max(unique_diameters) if unique_diameters.size > 0 else 0
        bins = np.arange(0, max_diameter + bin_size, bin_size)    
        hist, _ = np.histogram(df.loc[i, 'diameter_distribution'], bins=bins)
        if hist.size > 0:
            max_frequency = max(max_frequency, hist.max())
        else:
            max_frequency = 0
    
    print(f"\nPlotting data for image:")
    for i in range(N_images): # Plot the data for each row
        print(f"\r{os.path.basename(df.loc[i, 'image_file_name'])} \t {i+1}/{N_images}", end='', flush=True)


        # Create the figure with a custom GridSpec layout
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)

        # Get the image, outlines, and masks for the current row
        image = color.rgb2gray(sk_io.imread(df.loc[i, 'image_file_name'])[..., :3]) # grayscale
        outlines = df.loc[i, 'outlines']
        masks = df.loc[i, 'masks']

        # plot panel
        ax_0_0 = fig.add_subplot(gs[0, 0])
        ax_0_1 = fig.add_subplot(gs[0, 1])
        ax_0_2 = fig.add_subplot(gs[0, 2])
        ax_1_0 = fig.add_subplot(gs[1, 0])
        ax_1_12 = fig.add_subplot(gs[1, 1:3]) # spanning across two columns

        # Plot: original image
        ax_0_0.imshow(image, cmap='gray')
        ax_0_0.set_title(f"Original Image {i+1}")
        ax_0_0.axis('off')

        
        # Plot: image with outlines
        ax_0_1.imshow(image, cmap='gray', alpha=1)  # First overlay of original image
        ax_0_1.imshow(plot.mask_overlay(image, outlines), alpha=1, cmap='gist_rainbow')
        ax_0_1.set_title(f"Image {i+1} with Outlines")
        ax_0_1.axis('off')
        

        # Plot: image with masks
        ax_0_2.imshow(image, cmap='gray')
        ax_0_2.imshow(plot.mask_overlay(image, masks), alpha=0.5, cmap='gist_rainbow')
        ax_0_2.set_title(f"Image {i+1} with Masks")
        ax_0_2.axis('off')

        # Plot: Diameter distribution vs. diameter frequency (bin count histogram)
        unique_diameters, counts_diameters = np.unique(df.loc[i, 'diameter_distribution'], return_counts=True)
        max_diameter = max(unique_diameters) if unique_diameters.size > 0 else 0
        bins = np.arange(0, max_diameter + bin_size, bin_size)

        ax_1_0.hist(df.loc[i, 'diameter_distribution'], bins=bins)
        ax_1_0.set_title("Diameter Distribution")
        ax_1_0.set_xlabel("Diameter")
        ax_1_0.set_ylabel("Frequency")

        mean_diameter = df.loc[i, 'diameter_mean']
        median_diameter = df.loc[i, 'diameter_median']
        diameter_training = df.iloc[i]['diameter_training']
        ax_1_0.axvline(mean_diameter, color='blue', linewidth=1)
        ax_1_0.text(mean_diameter, ax_1_0.get_ylim()[1] * 0.9, f'Mean: {mean_diameter:05.2f}', color='blue')
        ax_1_0.axvline(median_diameter, color='green', linewidth=1)
        ax_1_0.text(median_diameter, ax_1_0.get_ylim()[1] * 0.8, f'Median: {median_diameter:05.2f}', color='green')
        ax_1_0.axvline(diameter_training, color='violet', linewidth=1)
        ax_1_0.text(diameter_training, ax_1_0.get_ylim()[1] * 0.7, f"Training: {df.iloc[i]['diameter_training'] if pd.notna(df.iloc[i]['diameter_training']) else 0.00:05.2f}", color='violet')

        ax_1_0.set_xlim(0, df['diameter_distribution'].apply(lambda x: np.max(x) if x.size > 0 else 0).max() * 1.05)
        ax_1_0.set_ylim(0, max_frequency*1.05)

        # Plot: Image number vs. median diameter, mean diameter, and amount of cells (up to current image)
        #ax_1_12 = ax_1_12_auxilliary
        ax_1_12_R = ax_1_12.twinx()
        ax_1_12.plot(range(N_images), df['diameter_mean'], label=f"{df.iloc[i]['diameter_mean']:05.2f} = Cell Mean Diameter", color='blue')
        ax_1_12.plot(range(N_images), df['diameter_median'], label=f"{df.iloc[i]['diameter_median']:05.2f} = Cell Median Diameter", color='green')
        ax_1_12.plot(range(N_images), df['diameter_training'], label=f"{df.iloc[i]['diameter_training'] if pd.notna(df.iloc[i]['diameter_training']) else 0.00:05.2f} = Cellpose Training Diameter", color='violet')
        #S = max(df['diameter_mean'].max(), df['diameter_median'].max()) / df['D_FB'].max()
        #ax_1_12.plot(range(N_images), df['D_FB'] * S, label=f"{(df.iloc[i]['D_FB']*S):.2f} = Flame Ball Diameter * {S:.3f}", color='orange')
        S2 = 1e-1
        ax_1_12.plot(range(N_images), df['D_FB'] * S2, label=f"{(df.iloc[i]['D_FB']*S2):05.2f} = Flame Ball Diameter * {S2:.3f}", color='orange')
        ax_1_12_R.plot(range(N_images), df['N_cells'], label=f"{df.iloc[i]['N_cells']:05.2f} = Cell Number", color='red')
        ax_1_12.axvline(i, color='blue', label=f'{i+1:.2f} = shown image', linestyle='dashed', linewidth=3)

        # Create a third y-axis for the dotted line plots
        ax_1_12_RR = ax_1_12.twinx()  # Second twin axis
        ax_1_12_RR.spines["right"].set_position(("outward", 60))  # Move third axis further right
        ax_1_12_RR.set_ylabel("A/A Values")  # Label for third y-axis
        ax_1_12_RR.set_ylim(0, 1)  # Set limits for third y-axis
        ax_1_12_RR.plot(range(N_images), df['Ar_FBperimage'], label=f"{df.iloc[i]['Ar_FBperimage']:05.2f}" + ' = $A_{Flame Ball}/A_{Image}$', color='gray', linestyle='dotted')
        ax_1_12_RR.plot(range(N_images), df['Ar_CP_maskperImage'], label=f"{df.iloc[i]['Ar_CP_maskperImage']:05.2f}" + " = $A_{Cell masks}/A_{Image}$", color='gray', linestyle='dashed')
        ax_1_12_RR.plot(range(N_images), df['Ar_CP_maskperFB'], label=f"{df.iloc[i]['Ar_CP_maskperFB']:05.2f}" + " = $A_{Cell masks}/A_{Flame Ball}$", color='gray')

        ax_1_12.set_xlim(0, N_images - 1)
        ax_1_12.set_ylim(min(df['diameter_mean'].min(), df['diameter_median'].min(), df['D_FB'].max()*S2), max(df['diameter_mean'].max(), df['diameter_median'].max(), df['D_FB'].max() * S2)*1.05)
        ax_1_12_R.set_ylim(df['N_cells'].min(), df['N_cells'].max()*1.05)

        ax_1_12.set_title("Diameter and Cell Count")
        ax_1_12.set_xlabel("Image Number")
        ax_1_12.set_ylabel("Diameter")
        ax_1_12_R.set_ylabel("Number of Cells")
        ax_1_12.legend(loc='upper left')
        ax_1_12_R.legend(loc='upper right')
        ax_1_12_RR.legend(loc='lower right')


        # Adjust layout and save the figure as a PNG file
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f'plot_{i+1:04d}.png')
        plt.savefig(plot_filename)
        #plt.show()
        plt.close(fig)
    print("\n") # new line

    if video == 1:
        vm1.create_video_from_images(
            plot_image_folder = output_dir,
            video_output_dir = output_dir, 
            fps=5,
            )


    ### ToDO

    # add numbers to masks or outlines





    ### return
    return output_dir # Format_1 requires outpu_dir as first return
