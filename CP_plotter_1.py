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
from skimage import io as skimage_io  # skimage io as an alias

import sys
import os
sys.path.append(os.path.abspath(r"C:/Users/obs/OneDrive/ETH/ETH_MSc/Masters Thesis/Python Code/Python_Orso_Utility_Scripts_MscThesis")) # dir containing Format_1 
import Format_1 as F_1

import video_maker_1 as vm1


@F_1.ParameterLog(max_size = 1024 * 10) # 10KB 
def CP_plotter_1(input_dir, # Format_1 requires input_dir
    df,
    output_dir_manual = "", output_dir_comment = ""
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
        print(f"{os.path.basename(df.loc[i, 'image_file_name'])} \t {i+1}/{N_images}", end='', flush=True)
        
        # Create a new figure for each row
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Get the image, outlines, and masks for the current row
        image = io.imread(df.loc[i,'image_file_name'])
        outlines = df.loc[i, 'outlines']
        masks = df.loc[i, 'masks']

        # Plot: original image
        axes[0,0].imshow(image, cmap='gray')
        axes[0,0].set_title(f"Original Image {i+1}")
        axes[0,0].axis('off')

        # Plot: image with outlines
        axes[0,1].imshow(image, cmap='gray', alpha = 1) #  interpolation='none'
        axes[0,1].imshow(plot.mask_overlay(image, outlines), alpha=1, cmap='gist_rainbow')

        #axes[0,1].contour(outlines, colors='red', levels=[1],linewidths=2)
        axes[0,1].set_title(f"Image {i+1} with Outlines")
        axes[0,1].axis('off')

        # Plot: image with masks
        axes[0,2].imshow(image, cmap='gray')
        axes[0,2].imshow(plot.mask_overlay(image, masks), alpha=0.5, cmap='gist_rainbow')
        axes[0,2].set_title(f"Image {i+1} with Masks")
        axes[0,2].axis('off')

        # Plot: Diameter distribution vs. diameter frequency (bin count histogram)
        unique_diameters, counts_diameters = np.unique(df.loc[i, 'diameter_distribution'], return_counts=True)
        max_diameter = max(unique_diameters) if unique_diameters.size > 0 else 0
        bins = np.arange(0, max_diameter + bin_size, bin_size)

        axes[1, 0].hist(df.loc[i, 'diameter_distribution'], bins=bins)
        axes[1, 0].set_title("Diameter Distribution")
        axes[1, 0].set_xlabel("Diameter")
        axes[1, 0].set_ylabel("Frequency")

        mean_diameter = df.loc[i, 'diameter_mean']
        median_diameter = df.loc[i, 'diameter_median']
        axes[1, 0].axvline(mean_diameter, color='blue', linestyle='dashed', linewidth=1)
        axes[1, 0].text(mean_diameter, axes[1, 0].get_ylim()[1] * 0.9, f'Mean: {mean_diameter:.2f}', color='blue')
        axes[1, 0].axvline(median_diameter, color='green', linestyle='dashed', linewidth=1)
        axes[1, 0].text(median_diameter, axes[1, 0].get_ylim()[1] * 0.8, f'Median: {median_diameter:.2f}', color='green')

        axes[1, 0].set_xlim(0, df['diameter_distribution'].apply(lambda x: np.max(x) if x.size > 0 else 0).max() * 1.05)
        axes[1, 0].set_ylim(0, max_frequency*1.05)

        # Plot 1: Image number vs. median diameter, mean diameter, and amount of cells (up to current image)
        ax1 = axes[1, 1]
        ax2 = ax1.twinx()
        ax1.plot(range(N_images), df['diameter_mean'], label='Mean Diameter', color='blue')
        ax1.plot(range(N_images), df['diameter_median'], label='Median Diameter', color='green')
        ax2.plot(range(N_images), df['N_cells'], label='Number of Cells', color='red')
        axes[1, 1].axvline(i, color='blue', label=f'shown image: {i+1:.2f}', linestyle='dashed', linewidth=3)
        #axes[1, 1].text(i, axes[1, 1].get_ylim()[1] * 0.9, f'shown image: {i:.2f}', color='blue')

        ax1.set_xlim(0, N_images - 1)
        ax1.set_ylim(min(df['diameter_mean'].min(), df['diameter_median'].min()), max(df['diameter_mean'].max(), df['diameter_median'].max())*1.05)
        ax2.set_ylim(df['N_cells'].min(), df['N_cells'].max()*1.05)

        ax1.set_title("Diameter and Cell Count")
        ax1.set_xlabel("Image Number")
        ax1.set_ylabel("Diameter")
        ax2.set_ylabel("Number of Cells")
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # Plot: Leave empty
        axes[1, 2].axis('off')






        # Adjust layout and save the figure as a PNG file
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f'plot_{i+1:04d}.png')
        plt.savefig(plot_filename)
        plt.close(fig)
    print("") # new line


    vm1.create_video_from_images(
        plot_image_folder = output_dir,
        video_output_dir = output_dir, 
        fps=5,
        )


    ### ToDO

    # add numbers to masks or outlines





    ### return
    return output_dir # Format_1 requires outpu_dir as first return
