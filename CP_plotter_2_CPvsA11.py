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
def CP_plotter_2_CPvsA11(input_dir, # Format_1 requires input_dir
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
    max_diameter = max([diameter for sublist in CP_extract_df['diameter_distribution_nonDim'] for diameter in sublist])
    bin_size = max_diameter/15
    max_frequency = 0
    for i in range(N_images):
        unique_diameters, counts_diameters = np.unique(CP_extract_df.loc[i, 'diameter_distribution_nonDim'], return_counts=True)
        max_diameter = max(unique_diameters) if unique_diameters.size > 0 else 0
        bins = np.arange(0, max_diameter + bin_size, bin_size)    
        hist, _ = np.histogram(CP_extract_df.loc[i, 'diameter_distribution_nonDim'], bins=bins)
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
        unique_diameters, counts_diameters = np.unique(CP_extract_df.loc[i, 'diameter_distribution_nonDim'], return_counts=True)
        max_diameter_i = max(unique_diameters) if unique_diameters.size > 0 else 0
        bins = np.arange(0, max_diameter_i + bin_size, bin_size)

        ax_1_0.hist(CP_extract_df.loc[i, 'diameter_distribution_nonDim'], bins=bins, orientation='horizontal', edgecolor='turquoise', color='white')
        ax_1_0.set_title("Diameter Distribution")
        ax_1_0.set_xlabel("Frequency")
        ax_1_0.set_ylabel("Diameter")

        mean_diameter = CP_extract_df.loc[i, 'diameter_mean_nonDim']
        median_diameter = CP_extract_df.loc[i, 'diameter_median_nonDim']
        diameter_training_nonDim = CP_extract_df.iloc[i]['diameter_training_nonDim']
        ax_1_0.axvline(mean_diameter, color='blue', linewidth=1)
        ax_1_0.text(mean_diameter, ax_1_0.get_ylim()[1] * 0.9, f'Mean: {mean_diameter:05.2f}', color='green')
        if pd.notna(diameter_training_nonDim): # make sure diameter training is available
            ax_1_0.axvline(diameter_training_nonDim, color='aquamarine', linewidth=1)
            ax_1_0.text(diameter_training_nonDim, ax_1_0.get_ylim()[1] * 0.7, f"Training: {diameter_training_nonDim:05.2f}", color='violet')
        
        ax_1_0.set_xlim(0, max_frequency*1.05)
        ax_1_0.set_ylim(0, CP_extract_df['diameter_distribution_nonDim'].apply(lambda x: np.max(x) if x.size > 0 else 0).max() * 1.05)


        # Plot: Image number vs. median diameter, mean diameter, and amount of cells (up to current image)
        #ax_1_12 = ax_1_12_auxilliary
        ax_1_12.plot(CP_extract_df['time'], CP_extract_df['diameter_mean_nonDim'], label=f"{CP_extract_df.iloc[i]['diameter_mean_nonDim']:05.2f} = Cell Mean Diameter", color='green')
        ax_1_12.plot(CP_extract_df['time'], CP_extract_df['diameter_training_nonDim'], label=f"{CP_extract_df.iloc[i]['diameter_training_nonDim'] if pd.notna(CP_extract_df.iloc[i]['diameter_training_nonDim']) else 0.00:05.2f} = Cellpose Training Diameter", color='aquamarine')
        
        #S = max(CP_extract_df['diameter_mean_nonDim'].max(), CP_extract_df['diameter_median_nonDim'].max()) / CP_extract_df['D_FB_nonDim'].max()
        #ax_1_12.plot(range(N_images), CP_extract_df['D_FB_nonDim'] * S, label=f"{(CP_extract_df.iloc[i]['D_FB_nonDim']*S):.2f} = Spherical Flame Diameter * {S:.3f}", color='orange')
            
        cp_time = CP_extract_df.iloc[i]['time']
        closest_index_R_mean = (np.abs(A11_SF_R_mean['time'] - cp_time)).argmin()
        closest_index_iHRR = (np.abs(A11_SF_iHRR['time'] - cp_time)).argmin()
        if A11_SF_R_mean['time'].iloc[closest_index_R_mean] < A11_SF_R_mean['time'].min() or A11_SF_R_mean['time'].iloc[closest_index_R_mean] > A11_SF_R_mean['time'].max():
            print(f"Time {cp_time} in CP_extract_df is out of range in A11_SF_R_mean, R_mean: NaN")
        else: # Get the R_mean at the closest time in A11_SF_R_mean if its in range
            closest_A11_r_mean = A11_SF_R_mean['R_mean'].iloc[closest_index_R_mean]
        if A11_SF_iHRR['time'].iloc[closest_index_iHRR] < A11_SF_iHRR['time'].min() or A11_SF_iHRR['time'].iloc[closest_index_iHRR] > A11_SF_iHRR['time'].max():
            print(f"Time {cp_time} in CP_extract_df is out of range in A11_SF_iHRR, iHRR: NaN")
        else: # Get the iHRR at the closest time in A11_SF_iHRR if its in range
            closest_A11_iHRR = A11_SF_iHRR['iHRR'].iloc[closest_index_iHRR]
        
        S2 = 1e-1
        ax_1_12.plot(CP_extract_df['time'], CP_extract_df['R_FB_nonDim'] * S2, label=f"{(CP_extract_df.iloc[i]['R_FB_nonDim']*S2):05.2f} = Image deduced Spherical Flame Radius * {S2:.3f}", color='olive')
        ax_1_12.plot(A11_SF_R_mean['time'], A11_SF_R_mean['R_mean'] * S2, label=f"{(closest_A11_r_mean*S2):05.2f} = A11 Spherical Flame Radius * {S2:.3f}", color='olive', linestyle='dashed')
        
        S3 = 1
        ax_1_12_L = ax_1_12.twinx() 
        ax_1_12_L.spines["left"].set_position(("outward", 2)) 
        ax_1_12_L.plot(A11_SF_iHRR['time'], A11_SF_iHRR['iHRR'] * S3, label=f"{(closest_A11_iHRR*S2):05.2f} = A11 integral heat release rate * {S2:.3f}", color='orange', linestyle='dashed')

        ax_1_12_R = ax_1_12.twinx()
        ax_1_12_R.plot(CP_extract_df['time'], CP_extract_df['N_cells'], label=f"{CP_extract_df.iloc[i]['N_cells']:05.2f} = Number of cells", color='red')
        
        ax_1_12.axvline(CP_extract_df.iloc[i]['time'], color='black', label=f'{i+1:.2f} = shown image', linestyle='dashed', linewidth=3)

        # Create a third y-axis for the dotted line plots
        ax_1_12_RR = ax_1_12.twinx()  # Second twin axis
        ax_1_12_RR.spines["right"].set_position(("outward", 60))  # Move third axis further right
        ax_1_12_RR.set_ylabel("A_CP/A_SF")  # Label for third y-axis
        ax_1_12_RR.set_ylim(0, 1)  # Set limits for third y-axis
        ax_1_12_RR.plot(CP_extract_df['time'], CP_extract_df['Ar_px2_CP_maskperFB'], label=f"{CP_extract_df.iloc[i]['Ar_px2_CP_maskperFB']:05.2f}" + " = $A_{Cell masks}/A_{Spherical Flame}$", color='gray')

        ax_1_12.set_xlim(0, 7)
        ax_1_12.set_ylim(0, max_diameter*1.05)
        ax_1_12_L.set_ylim(0, A11_SF_iHRR['iHRR'].max()*1.05)
        ax_1_12_R.set_ylim(CP_extract_df['N_cells'].min(), CP_extract_df['N_cells'].max()*1.05)

        ax_1_12.set_title("Diameter and Cell Count")
        ax_1_12.set_xlabel("time")
        ax_1_12.set_ylabel("Diameter", color='green')
        ax_1_12_L.set_ylabel("Heat Release Rate", color='orange')
        ax_1_12_R.set_ylabel("Number of Cells", color='red')

        ax_1_12.legend(loc='upper left')
        ax_1_12_L.legend(loc='lower left')
        ax_1_12_R.legend(loc='upper right')
        ax_1_12_RR.legend(loc='lower right')

        ax_1_12_L.tick_params(axis='y', labelcolor='orange')
        ax_1_12_R.tick_params(axis='y', labelcolor='red')


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
