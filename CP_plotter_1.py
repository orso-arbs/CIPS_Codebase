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
    CP_extract_df = None, # if None a .pkl file has to be in the input_dir. otherwise no CP_extract data is provided.
    output_dir_manual = "", output_dir_comment = "",
    video = 1,
    ):

    ### output 
    output_dir = F_1.F_out_dir(input_dir, __file__, output_dir_comment = "") # Format_1 required definition of output directory

    pkl_files = glob.glob(os.path.join(input_dir, "*.pkl"))

    ### Load CP extraxct data
    if pkl_files:
        CP_extract_df_pkl = pkl_files[0] # If a .pkl file exists, use it as the pickle file path. If multiple .pkl files exist the first is used.
    else:
        CP_extract_df_pkl = None

    if CP_extract_df is None and CP_extract_df_pkl is None:
        raise ValueError("No CP_extract data provided. Provide Data.")
    elif CP_extract_df is None and CP_extract_df_pkl is not None:
        print(f"Loading CP_extract data from pickle file {os.path.basename(CP_extract_df_pkl)}")
        CP_extract_df = pd.read_pickle(CP_extract_df_pkl)
    elif CP_extract_df is not None and CP_extract_df_pkl is None:
        print("Loading CP_extract data from passed DataFrame in function argument")
        # No action needed since CP_extract_df is already passed
    elif CP_extract_df is not None and CP_extract_df_pkl is not None:
        print("Both CP_extract_df and CP_extract_df_pkl provided. Using data from passed DataFrame in function argument")
        # No action needed since CP_extract_df is already passed
    else:
        raise ValueError("Loading CP_extract data disambiguation failed. Check CP_extract_df and CP_extract_df_pkl")



    # Load A11 data
    # Define the file paths
    file_paths = [
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_K_mean_as_mean_stretch_rate_vs_time_manual_extraction.txt",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_N_c_as_number_of_cells_vs_time_manual_extraction.txt",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_R_mean_as_average_radius_of_the_wrinkled_flame_fron_vs_time_manual_extraction.txt",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_R_mean_dot_as_first_time_derivative_of_the_average_radius_of_the_wrinkled_flame_front_vs_time_manual_extraction.txt",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_s_a_as_average_normal_component_of_the_absolute_propagation_velocity_vs_time_manual_extraction.txt",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_s_d_as_average_density_weighted_displacement_speed_vs_time_manual_extraction.txt",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_A_as_flame_surface_area_of_the_wrinkled_spherical_front_vs_time_manual_extraction.txt",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_a_t_as_average_total_aerodynamic_strain_vs_time_manual_extraction.txt",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_iHRR_as_integral_heat_release_rate_vs_time_manual_extraction.txt",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_K_geom_as_geometric_stretch_rate_vs_time_manual_extraction.txt"
    ]

    # Load all files into a dictionary of DataFrames and create variables dynamically
    for file_path in file_paths:
        # Extract file name without path and extension
        file_name = file_path.split('\\')[-1].replace('.txt', '')
        
        # Simplify the variable name by removing 'A11_SF_' and '_as' parts
        variable_name = file_name.split('_as')[0]
        
        # Load the CSV file into a DataFrame and assign it dynamically
        globals()[variable_name] = pd.read_csv(file_path)
    ''' A11 dataframes:
    A11_SF_K_mean
    A11_SF_N_c
    A11_SF_R_mean
    A11_SF_R_mean_dot
    A11_SF_s_a
    A11_SF_s_d
    A11_SF_A
    A11_SF_a_t
    A11_SF_iHRR
    A11_SF_K_geom
    '''


    # auxillary function to plot the data

    # Number of rows in the DataFrame
    N_images = len(CP_extract_df)

    # Find the maximum frequency for all histograms
    bin_size = 2
    max_frequency = 0
    for i in range(N_images):
        unique_diameters, counts_diameters = np.unique(CP_extract_df.loc[i, 'diameter_distribution_px'], return_counts=True)
        max_diameter = max(unique_diameters) if unique_diameters.size > 0 else 0
        bins = np.arange(0, max_diameter + bin_size, bin_size)    
        hist, _ = np.histogram(CP_extract_df.loc[i, 'diameter_distribution_px'], bins=bins)
        if hist.size > 0:
            max_frequency = max(max_frequency, hist.max())
        else:
            max_frequency = 0
    
    print(f"\nPlotting data for image:")
    for i in range(N_images): # Plot the data for each row
        print(f"\r{os.path.basename(CP_extract_df.loc[i, 'image_file_name'])} \t {i+1}/{N_images}", end='', flush=True)


        # Create the figure with a custom GridSpec layout
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)

        # Get the image, outlines, and masks for the current row
        image = color.rgb2gray(sk_io.imread(CP_extract_df.loc[i, 'image_file_name'])[..., :3]) # grayscale
        outlines = CP_extract_df.loc[i, 'outlines']
        masks = CP_extract_df.loc[i, 'masks']

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
        unique_diameters, counts_diameters = np.unique(CP_extract_df.loc[i, 'diameter_distribution_px'], return_counts=True)
        max_diameter = max(unique_diameters) if unique_diameters.size > 0 else 0
        bins = np.arange(0, max_diameter + bin_size, bin_size)

        ax_1_0.hist(CP_extract_df.loc[i, 'diameter_distribution_px'], bins=bins)
        ax_1_0.set_title("Diameter Distribution")
        ax_1_0.set_xlabel("Diameter")
        ax_1_0.set_ylabel("Frequency")

        mean_diameter = CP_extract_df.loc[i, 'diameter_mean_px']
        median_diameter = CP_extract_df.loc[i, 'diameter_median_px']
        diameter_training_px = CP_extract_df.iloc[i]['diameter_training_px']
        ax_1_0.axvline(mean_diameter, color='blue', linewidth=1)
        ax_1_0.text(mean_diameter, ax_1_0.get_ylim()[1] * 0.9, f'Mean: {mean_diameter:05.2f}', color='blue')
        ax_1_0.axvline(median_diameter, color='green', linewidth=1)
        ax_1_0.text(median_diameter, ax_1_0.get_ylim()[1] * 0.8, f'Median: {median_diameter:05.2f}', color='green')
        if pd.notna(diameter_training_px): # make sure diameter training is available
            ax_1_0.axvline(diameter_training_px, color='violet', linewidth=1)
            ax_1_0.text(diameter_training_px, ax_1_0.get_ylim()[1] * 0.7, f"Training: {diameter_training_px:05.2f}", color='violet')
        
        ax_1_0.set_xlim(0, CP_extract_df['diameter_distribution_px'].apply(lambda x: np.max(x) if x.size > 0 else 0).max() * 1.05)
        ax_1_0.set_ylim(0, max_frequency*1.05)

        # Plot: Image number vs. median diameter, mean diameter, and amount of cells (up to current image)
        #ax_1_12 = ax_1_12_auxilliary
        ax_1_12_R = ax_1_12.twinx()
        ax_1_12.plot(range(N_images), CP_extract_df['diameter_mean_px'], label=f"{CP_extract_df.iloc[i]['diameter_mean_px']:05.2f} = Cell Mean Diameter", color='blue')
        ax_1_12.plot(range(N_images), CP_extract_df['diameter_median_px'], label=f"{CP_extract_df.iloc[i]['diameter_median_px']:05.2f} = Cell Median Diameter", color='green')
        ax_1_12.plot(range(N_images), CP_extract_df['diameter_training_px'], label=f"{CP_extract_df.iloc[i]['diameter_training_px'] if pd.notna(CP_extract_df.iloc[i]['diameter_training_px']) else 0.00:05.2f} = Cellpose Training Diameter", color='violet')
        #S = max(CP_extract_df['diameter_mean_px'].max(), CP_extract_df['diameter_median_px'].max()) / CP_extract_df['D_FB_px'].max()
        #ax_1_12.plot(range(N_images), CP_extract_df['D_FB_px'] * S, label=f"{(CP_extract_df.iloc[i]['D_FB_px']*S):.2f} = Flame Ball Diameter * {S:.3f}", color='orange')
        S2 = 1e-1
        ax_1_12.plot(range(N_images), CP_extract_df['D_FB_px'] * S2, label=f"{(CP_extract_df.iloc[i]['D_FB_px']*S2):05.2f} = Flame Ball Diameter * {S2:.3f}", color='orange')
        ax_1_12_R.plot(range(N_images), CP_extract_df['N_cells'], label=f"{CP_extract_df.iloc[i]['N_cells']:05.2f} = Number of cells", color='red')
        ax_1_12.axvline(i, color='blue', label=f'{i+1:.2f} = shown image', linestyle='dashed', linewidth=3)

        # Create a third y-axis for the dotted line plots
        ax_1_12_RR = ax_1_12.twinx()  # Second twin axis
        ax_1_12_RR.spines["right"].set_position(("outward", 60))  # Move third axis further right
        ax_1_12_RR.set_ylabel("A/A Values")  # Label for third y-axis
        ax_1_12_RR.set_ylim(0, 1)  # Set limits for third y-axis
        #ax_1_12_RR.plot(range(N_images), CP_extract_df['Ar_px2_FBperimage'], label=f"{CP_extract_df.iloc[i]['Ar_px2_FBperimage']:05.2f}" + ' = $A_{Flame Ball}/A_{Image}$', color='gray', linestyle='dotted')
        #ax_1_12_RR.plot(range(N_images), CP_extract_df['Ar_px2_CP_maskperImage'], label=f"{CP_extract_df.iloc[i]['Ar_px2_CP_maskperImage']:05.2f}" + " = $A_{Cell masks}/A_{Image}$", color='gray', linestyle='dashed')
        ax_1_12_RR.plot(range(N_images), CP_extract_df['Ar_px2_CP_maskperFB'], label=f"{CP_extract_df.iloc[i]['Ar_px2_CP_maskperFB']:05.2f}" + " = $A_{Cell masks}/A_{Flame Ball}$", color='gray')

        ax_1_12.set_xlim(0, N_images - 1)
        ax_1_12.set_ylim(min(CP_extract_df['diameter_mean_px'].min(), CP_extract_df['diameter_median_px'].min(), CP_extract_df['D_FB_px'].max()*S2), max(CP_extract_df['diameter_mean_px'].max(), CP_extract_df['diameter_median_px'].max(), CP_extract_df['D_FB_px'].max() * S2)*1.05)
        ax_1_12_R.set_ylim(CP_extract_df['N_cells'].min(), CP_extract_df['N_cells'].max()*1.05)

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
