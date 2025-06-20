import numpy as np
import matplotlib.pyplot as plt
from cellpose import plot
import glob
import os
import pandas as pd
import matplotlib.gridspec as gridspec
import numpy as np
from skimage import io as sk_io, color
import re

import sys
import os
import Format_1 as F_1

import video_maker_1 as vm1
import Add_whitespace_text_to_png as AWTTP


@F_1.ParameterLog(max_size = 1024 * 10) # 10KB 
def plotter_1(input_dir, # Format_1 requires input_dir
    CP_data_df = None, # if None a .pkl file has to be in the input_dir. otherwise no CP_data_ data is provided.
    output_dir_manual = "", output_dir_comment = "",
    video = 1,
    Plot_log_level=1, # Added Plot_log_level argument
    ):

    """
    Generates plots visualizing Cellpose segmentation results and optionally creates a video.

    This function takes the data generated by `CP_extract_1` or a pre-loaded DataFrame. For each image
    in the sequence, it creates a multi-panel plot showing:
    1. The original grayscale image.
    2. The image overlaid with Cellpose outlines.
    3. The image overlaid with Cellpose masks.
    4. A histogram of detected cell diameters (in pixels) with mean, median,
       training, and estimated diameters marked.
    5. A time-series plot showing the evolution of mean/median cell diameter,
       estimated flame ball diameter (scaled), number of cells, and CP efficiency
       up to the current image.

    The generated plots are saved as individual PNG files in the output directory.
    Optionally, these plots can be compiled into a video. Text banners with
    Cellpose settings and metadata are added to the top and bottom of each plot image.

    Parameters
    ----------
    input_dir : str
        Path to the directory containing the `.pkl` file
        with data generated by `CP_extract_1`.
    CP_data_df : pandas.DataFrame or None, optional
        A pre-loaded DataFrame containing the segmentation data. If provided,
        it overrides loading from the `.pkl` file in `input_dir`. If None, the
        function attempts to load `segmentation_DataFrame.pkl` from `input_dir`.
        Defaults to None.
    output_dir_manual : str, optional
        If provided, specifies the exact output directory path for the plots/video.
        Overrides the default naming convention managed by `Format_1.F_out_dir`.
        Defaults to "".
    output_dir_comment : str, optional
        A comment to append to the default output directory name if `output_dir_manual`
        is not provided. Defaults to "".
    video : int, optional
        If 0, only saves the individual plot images.
        If 1, it also creates a video from the generated plot images using `video_maker_1.create_video_from_images`.
        Defaults to 1.
    Plot_log_level : int, optional
        Controls the verbosity of logging for this plotting function.
        Currently not implemented beyond accepting the parameter. Defaults to 1.

    Returns
    -------
    output_dir : str
        The path to the directory where the output plots (and optionally video)
        are saved. This is always the first return value as required by Format_1.

    Notes
    -----
    - Relies on `Format_1.py` for output directory creation (`F_out_dir`) and
      parameter logging (`@F_1.ParameterLog`).
    - Uses `video_maker_1.py` to create the video if `video=1`.
    - Uses `Add_whitespace_text_to_png.py` to add text banners to the plots.
    """
    #################################################### I/O
    output_dir = F_1.F_out_dir(input_dir, __file__, output_dir_comment = output_dir_comment, output_dir_manual = output_dir_manual) # Format_1 required definition of output directory
    print(f"Output directory: {output_dir}")
    print(f"type Output directory: {type(output_dir)}")


    pkl_files = glob.glob(os.path.join(input_dir, "*.pkl"))

    #################################################### Load CP extraxct data
    if pkl_files:
        CP_data_df_pkl = pkl_files[0] # If a .pkl file exists, use it as the pickle file path. If multiple .pkl files exist the first is used.
    else:
        CP_data_df_pkl = None

    if CP_data_df is None and CP_data_df_pkl is None:
        raise ValueError("No CP_data data provided. Provide Data.")
    elif CP_data_df is None and CP_data_df_pkl is not None:
        print(f"Loading CP_data data from pickle file {os.path.basename(CP_data_df_pkl)}")
        CP_data_df = pd.read_pickle(CP_data_df_pkl)
    elif CP_data_df is not None and CP_data_df_pkl is None:
        print("Loading CP_data data from passed DataFrame in function argument")
        # No action needed since CP_data_df is already passed
    elif CP_data_df is not None and CP_data_df_pkl is not None:
        print("Both CP_data_df and CP_data_df_pkl provided. Using data from passed DataFrame in function argument")
        # No action needed since CP_data_df is already passed
    else:
        raise ValueError("Loading CP_data data disambiguation failed. Check CP_data_df and CP_data_df_pkl")
    
    if CP_data_df['Time_VisIt'] is not None:
        print("Time_VisIt column found in CP_data_df. Using it as time.")
        CP_data_df['time'] = CP_data_df['Time_VisIt']
    else: 
        print("No Time_VisIt column found in CP_data_df. Using index as time.")
        CP_data_df['time'] = CP_data_df.index

    
    #################################################### Plotting
    
    # auxillary function to plot the data

    # Number of rows in the DataFrame
    N_images = len(CP_data_df)

    # Find the maximum frequency for all histograms
    bin_size = 2
    max_frequency = 0
    for i in range(N_images):
        unique_diameters, counts_diameters = np.unique(CP_data_df.loc[i, 'd_cell_distribution_px'], return_counts=True)
        max_diameter = max(unique_diameters) if unique_diameters.size > 0 else 0
        bins = np.arange(0, max_diameter + bin_size, bin_size)    
        hist, _ = np.histogram(CP_data_df.loc[i, 'd_cell_distribution_px'], bins=bins)
        if hist.size > 0:
            max_frequency = max(max_frequency, hist.max())
        else:
            max_frequency = 0
    
    print(f"\nPlotting data for image:")
    for i in range(N_images): # Plot the data for each row
        print(f"\r{os.path.basename(CP_data_df.loc[i, 'image_file_name'])} \t {i+1}/{N_images}", end='', flush=True)


        # Create the figure with a custom GridSpec layout
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)




        # Get the image, outlines, and masks for the current row
        image = color.rgb2gray(sk_io.imread(CP_data_df.loc[i, 'image_file_path'])[..., :3]) # grayscale
        outlines = CP_data_df.loc[i, 'outlines']
        masks = CP_data_df.loc[i, 'masks']

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
        unique_diameters, counts_diameters = np.unique(CP_data_df.loc[i, 'd_cell_distribution_px'], return_counts=True)
        max_diameter = max(unique_diameters) if unique_diameters.size > 0 else 0
        bins = np.arange(0, max_diameter + bin_size, bin_size)

        ax_1_0.hist(CP_data_df.loc[i, 'd_cell_distribution_px'], bins=bins)
        ax_1_0.set_title("Diameter Distribution")
        ax_1_0.set_xlabel("Diameter [px]")
        ax_1_0.set_ylabel("Frequency")

        mean_diameter = CP_data_df.loc[i, 'd_cell_mean_px'] if np.isfinite(CP_data_df.loc[i, 'd_cell_mean_px']) else 0
        median_diameter = CP_data_df.loc[i, 'd_cell_median_px'] if np.isfinite(CP_data_df.loc[i, 'd_cell_median_px']) else 0
        diameter_training_px = CP_data_df.iloc[i]['diameter_training_px'] if np.isfinite(CP_data_df.iloc[i]['diameter_training_px']) else 0
        diameter_estimate_used_px = CP_data_df.iloc[i]['diameter_estimate_used_px'] if np.isfinite(CP_data_df.iloc[i]['diameter_estimate_used_px']) else 0
        ax_1_0.axvline(mean_diameter, color='blue', linewidth=1)
        ax_1_0.text(mean_diameter, ax_1_0.get_ylim()[1] * 0.9, f'Mean: {mean_diameter:05.2f}', color='blue')
        ax_1_0.axvline(median_diameter, color='green', linewidth=1)
        ax_1_0.text(median_diameter, ax_1_0.get_ylim()[1] * 0.8, f'Median: {median_diameter:05.2f}', color='green')
        if pd.notna(diameter_training_px): # make sure diameter training is available
            ax_1_0.axvline(diameter_training_px, color='violet', linewidth=1)
            ax_1_0.text(diameter_training_px, ax_1_0.get_ylim()[1] * 0.7, f"Training: {diameter_training_px:05.2f}", color='violet')
        ax_1_0.axvline(diameter_estimate_used_px, color='purple', linewidth=1)
        ax_1_0.text(diameter_estimate_used_px, ax_1_0.get_ylim()[1] * 0.6, f'Estimate: {diameter_estimate_used_px:05.2f}', color='purple')

        ax_1_0.set_xlim(0, CP_data_df['d_cell_distribution_px'].apply(lambda x: np.max(x) if x.size > 0 else 0).max() * 1.05)
        ax_1_0.set_ylim(0, max_frequency*1.05)

        # Plot: Image number vs. median diameter, mean diameter, and amount of cells (up to current image)
        x_array = range(N_images)
        #ax_1_12 = ax_1_12_auxilliary
        ax_1_12_R = ax_1_12.twinx()
        ax_1_12.plot(range(N_images), CP_data_df['d_cell_mean_px'], label=f"{CP_data_df.iloc[i]['d_cell_mean_px']:05.2f} = Cell Mean Diameter [px]", color='blue')
        ax_1_12.plot(range(N_images), CP_data_df['d_cell_median_px'], label=f"{CP_data_df.iloc[i]['d_cell_median_px']:05.2f} = Cell Median Diameter [px]", color='green')
        ax_1_12.plot(range(N_images), CP_data_df['diameter_training_px'], label=f"{CP_data_df.iloc[i]['diameter_training_px'] if pd.notna(CP_data_df.iloc[i]['diameter_training_px']) else 'N/A' :05.2f} = Cellpose Training Diameter [px]", color='violet')
        ax_1_12.plot(range(N_images), CP_data_df['diameter_estimate_used_px'], label=f"{CP_data_df.iloc[i]['diameter_estimate_used_px'] if pd.notna(CP_data_df.iloc[i]['diameter_estimate_used_px']) else 'N/A' :05.2f} = Cellpose Estimate Diameter [px]", color='purple')
        #S = max(CP_data_df['d_cell_mean_px'].max(), CP_data_df['d_cell_median_px'].max()) / CP_data_df['D_SF_px'].max()
        #ax_1_12.plot(range(N_images), CP_data_df['D_SF_px'] * S, label=f"{(CP_data_df.iloc[i]['D_SF_px']*S):.2f} = Flame Ball Diameter * {S:.3f}", color='orange')
        S2 = 1e-1
        ax_1_12.plot(range(N_images), CP_data_df['D_SF_px'] * S2, label=f"{(CP_data_df.iloc[i]['D_SF_px']*S2):05.2f} = Flame Ball Diameter [px] * {S2:.3f}", color='orange')
        ax_1_12_R.plot(range(N_images), CP_data_df['N_cells'], label=f"{CP_data_df.iloc[i]['N_cells']:05.2f} = Number of cells", color='red')
        ax_1_12.axvline(i, color='blue', label=f'{i+1:.2f} = shown image. {CP_data_df.iloc[i]["image_Nx_px"]:.0f}px_x x {CP_data_df.iloc[i]["image_Ny_px"]:.0f}px_y', linestyle='dashed', linewidth=3)

        # Create a third y-axis for the dotted line plots
        ax_1_12_RR = ax_1_12.twinx()  # Second twin axis
        ax_1_12_RR.spines["right"].set_position(("outward", 60))  # Move third axis further right
        ax_1_12_RR.set_ylabel("CP efficiency $\mu_{CP} = A_{CP}/A_{SF}$")  # Label for third y-axis
        ax_1_12_RR.set_ylim(0, 1)  # Set limits for third y-axis
        #ax_1_12_RR.plot(range(N_images), CP_data_df['Ar_px2_SFperimage'], label=f"{CP_data_df.iloc[i]['Ar_px2_SFperimage']:05.2f}" + ' = $A_{Flame Ball}/A_{Image}$', color='gray', linestyle='dotted')
        #ax_1_12_RR.plot(range(N_images), CP_data_df['Ar_px2_CP_maskperImage'], label=f"{CP_data_df.iloc[i]['Ar_px2_CP_maskperImage']:05.2f}" + " = $A_{Cell masks}/A_{Image}$", color='gray', linestyle='dashed')
        ax_1_12_RR.plot(range(N_images), CP_data_df['Ar_px2_CP_maskperSF'], label=f"{CP_data_df.iloc[i]['Ar_px2_CP_maskperSF']:05.2f}" + " = CP efficiency $\mu_{CP} = A_{CP}/A_{SF}$", color='gray')

        ax_1_12.set_xlim(0, N_images - 1)
        #ax_1_12.set_ylim(min(CP_data_df['d_cell_mean_px'].min(), CP_data_df['d_cell_median_px'].min(), CP_data_df['D_SF_px'].max()*S2), max(CP_data_df['d_cell_mean_px'].max(), CP_data_df['d_cell_median_px'].max(), CP_data_df['D_SF_px'].max() * S2)*1.05)
        ax_1_12.set_ylim(0, max(CP_data_df['d_cell_mean_px'].max(), CP_data_df['d_cell_median_px'].max(), CP_data_df['D_SF_px'].max() * S2)*1.05)
        ax_1_12_R.set_ylim(CP_data_df['N_cells'].min(), CP_data_df['N_cells'].max()*1.05)

        ax_1_12.set_xticks(range(N_images))  # Keep original ticks
        ax_1_12.set_xticklabels(range(1, N_images + 1))  # Display shifted labels

        ax_1_12.set_title("Diameter and Cell Count")
        ax_1_12.set_xlabel("Image Number")
        ax_1_12.set_ylabel("Diameter [px]")
        ax_1_12_R.set_ylabel("Number of Cells")
        ax_1_12.legend(loc='upper left')
        ax_1_12_R.legend(loc='upper right')
        ax_1_12_RR.legend(loc='lower right')

        #fig.suptitle("Your Figure Title", fontsize=16, fontweight='bold')
        # Adjust layout and save the figure as a PNG file
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f'plot_{i+1:04d}_{output_dir_comment}.png')
        plt.savefig(plot_filename)
        #plt.show()
        plt.close(fig)







        # text to add to top and bottom of the plot:
        matches = re.findall(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', output_dir)
        DateAndTime = matches[-1] if matches else None
        input_image_path = plot_filename
        output_image_path = plot_filename

        top_text = "" \
        "Cellpose Settings:\n" \
        f"model_type_segmentation = {CP_data_df.iloc[i].get('CP_model_type_for_segmentation', 'N/A')}\n" \
        f"segmentation_model_path = {CP_data_df.iloc[i].get('CP_segmentation_model_actual_path', 'N/A')}\n" \
        f"model_type_diam_est_input = {CP_data_df.iloc[i].get('CP_model_type_for_diameter_estimation_input', 'N/A')}\n" \
        f"diam_est_model_type_used = {CP_data_df.iloc[i].get('CP_size_model_type_in_wrapper', 'N/A')}\n" \
        f"augment (tiling) =    {CP_data_df.iloc[i].get('augment', 'N/A')}\n" \
        f"resample =            {CP_data_df.iloc[i].get('resample', 'N/A')}\n" \
        f"niter =               {CP_data_df.iloc[i].get('niter', 'N/A')}\n" \
        f"flow_threshold =      {CP_data_df.iloc[i].get('flow_threshold', 'N/A')}\n" \
        f"cellprob_threshold =  {CP_data_df.iloc[i].get('cellprob_threshold', 'N/A')}\n" \
        f"CP_segment_output_dir_comment =  {CP_data_df.iloc[i].get('CP_segment_output_dir_comment', 'N/A')}\n" \
        f"channels =            {CP_data_df.iloc[i].get('channels', 'N/A')}\n" \
        f"Image directory =     {os.path.dirname(CP_data_df.iloc[i]['image_file_path'])}\n" \
        f"Image =               {os.path.basename(CP_data_df.iloc[i]['image_file_path'])}\n" \
        
        bottom_text = "" \
        "Orso Birelli Schmid\n" \
        "Masters Thesis ETH Zurich\n" \
        f"{DateAndTime}" \
        

        AWTTP.add_white_space_with_banners(input_image_path, output_image_path, top_text, bottom_text, font_size = 18)


    print("\n") # new line

    if video == 1:
        vm1.create_video_from_images(
            plot_image_folder = output_dir,
            video_output_dir = output_dir, 
            fps=5,
            output_dir_comment = output_dir_comment,
            )

    ####################################################### return

    return output_dir # Format_1 requires outpu_dir as first return

# Example usage:
if __name__ == "__main__":
    print("Running plotter...")
    
    plotter_1(
        # Extraction output_dir of hot 3000px States 79 and 100
        input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\20250604_1311111\20250604_1312276\20250604_1312276\20250604_1313140\20250615_1635229",
        output_dir_comment="test_plotting",
        Plot_log_level=1,
        video=1
    )
