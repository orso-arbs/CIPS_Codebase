import numpy as np
import pandas as pd
import os
import math
import pickle
import sys

# Assuming Format_1 is needed for logging or other utilities, import it
# import Format_1 as F_1

# Assuming utils from cellpose is needed for diameters calculation, import it
from cellpose import utils

# Add other necessary imports based on the copied code
import re
from skimage import io as sk_io, color
import glob


# @F_1.ParameterLog(max_size = 1024 * 10, log_level = 0) # Add if ParameterLog is desired for this function
def CP_dimentionalise(
    # input
    segmentation_extracted_DataFrame, # Input dataframe from CP_extract

    # parameters (add any necessary parameters here, e.g., A11 data paths if they should be configurable)
    CP_dimentionalise_log_level = 0,
    A11_data_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction", # Default A11 data directory

    ):

    """
    Dimensionalises segmentation data using A11 simulation results.

    This function takes a pandas DataFrame containing extracted segmentation data,
    loads external A11 simulation results, and calculates non-dimensional metrics
    by interpolating A11 data based on the image sequence number. The non-dimensional
    metrics are appended as new columns to the input DataFrame.

    Parameters
    ----------
    segmentation_extracted_DataFrame : pandas.DataFrame
        The DataFrame containing extracted segmentation data from CP_extract.
    CP_dimentionalise_log_level : int, optional
        Controls the verbosity of logging messages printed to the console.
        0: Minimal logging.
        1: Basic loading and processing steps.
        Defaults to 0.
    t_max : float, optional
        Maximum simulation time for linear mapping of image sequence numbers.
        Defaults to 6.81.
    d_T : float, optional
        Reference flame thickness for non-dimensionalization. Defaults to 7.516e-3.
    S_L : float, optional
        Reference laminar flame speed for non-dimensionalization. Defaults to 51.44.
    R0 : float, optional
        Initial spherical flame radius for non-dimensionalization. Defaults to 10 * d_T.
    A11_data_dir : str, optional
        Path to the directory containing the A11 manual extraction data files.
        Defaults to "C:\\Users\\obs\\OneDrive\\ETH\\ETH_MSc\\Masters Thesis\\Data\\A11_manual_extraction".


    Returns
    -------
    segmentation_dimentionalised_DataFrame : pandas.DataFrame
        The input DataFrame with appended non-dimensionalised columns.
    """

    segmentation_dimentionalised_DataFrame = segmentation_extracted_DataFrame.copy() # Work on a copy

    ####################################################  Images and A11 data

    print("\n Non Dimentionalissing and matching CP and A11 data \n")  if CP_dimentionalise_log_level >= 1 else None

    # Load A11 data
    # Construct full paths using A11_data_dir
    A11_SF_K_mean_path = os.path.join(A11_data_dir, "A11_SF_K_mean_as_mean_stretch_rate_vs_time_manual_extraction.txt")
    A11_SF_N_c_path = os.path.join(A11_data_dir, "A11_SF_N_c_as_number_of_cells_vs_time_manual_extraction.txt")
    A11_SF_R_mean_path = os.path.join(A11_data_dir, "A11_SF_R_mean_as_average_radius_of_the_wrinkled_flame_fron_vs_time_manual_extraction.txt")
    A11_SF_R_mean_dot_path = os.path.join(A11_data_dir, "A11_SF_R_mean_dot_as_first_time_derivative_of_the_average_radius_of_the_wrinkled_flame_front_vs_time_manual_extraction.txt")
    A11_SF_s_a_path = os.path.join(A11_data_dir, "A11_SF_s_a_as_average_normal_component_of_the_absolute_propagation_velocity_vs_time_manual_extraction.txt")
    A11_SF_s_d_path = os.path.join(A11_data_dir, "A11_SF_s_d_as_average_density_weighted_displacement_speed_vs_time_manual_extraction.txt")
    A11_SF_A_path = os.path.join(A11_data_dir, "A11_SF_A_as_flame_surface_area_of_the_wrinkled_spherical_front_vs_time_manual_extraction.txt")
    A11_SF_a_t_path = os.path.join(A11_data_dir, "A11_SF_a_t_as_average_total_aerodynamic_strain_vs_time_manual_extraction.txt")
    A11_SF_iHRR_path = os.path.join(A11_data_dir, "A11_SF_iHRR_as_integral_heat_release_rate_vs_time_manual_extraction.txt")
    A11_SF_K_geom_path = os.path.join(A11_data_dir, "A11_SF_K_geom_as_geometric_stretch_rate_vs_time_manual_extraction.txt")


    A11_SF_K_mean = pd.read_csv(A11_SF_K_mean_path)
    A11_SF_N_c = pd.read_csv(A11_SF_N_c_path)
    A11_SF_R_mean = pd.read_csv(A11_SF_R_mean_path)
    A11_SF_R_mean_dot = pd.read_csv(A11_SF_R_mean_dot_path)
    A11_SF_s_a = pd.read_csv(A11_SF_s_a_path)
    A11_SF_s_d = pd.read_csv(A11_SF_s_d_path)
    A11_SF_A = pd.read_csv(A11_SF_A_path)
    A11_SF_a_t = pd.read_csv(A11_SF_a_t_path)
    A11_SF_iHRR = pd.read_csv(A11_SF_iHRR_path)
    A11_SF_K_geom = pd.read_csv(A11_SF_K_geom_path)


    # reference vales
    ''' A11 p.36
    "...reference values from detailed description of chemistry is based on the mech-
    anism of Li et al. [2004], consisting of 9 species and 19 non-duplicate
    elementary reactions. The planar premixed flame structure calculated us-
    ing PREMIX [Rupley et al., 1995]..."
    '''
    t_min = 0.0 # initial time
    t_max = 6.81, # max time estimated from plots
    d_T = 7.516 * 1e-3, # flame thickness
    S_L = 51.44, # laminar flame speed
    R0 = 10 * d_T, # initial Spherical flame radius
    t_ref = d_T/S_L # flame time scale


    # Calculate the time values for each DataFrame
    segmentation_dimentionalised_DataFrame['time'] = np.nan # initialize time collumn filled with nan
    min_image_num = segmentation_dimentionalised_DataFrame["image_number"].min()
    max_image_num = segmentation_dimentionalised_DataFrame["image_number"].max()
    if min_image_num == max_image_num:
        segmentation_dimentionalised_DataFrame["time"] = 0  # Assign 0 to all if there's no range
        print("Warning in dimetionalisation: min_image_num == max_image_num")
    else:
        # Linear mapping from [min_image_num, max_image_num] to [t_min, t_max]
        segmentation_dimentionalised_DataFrame["time"] = (segmentation_dimentionalised_DataFrame["image_number"] - min_image_num) / (max_image_num - min_image_num) * t_max
        print("segmentation_dimentionalised_DataFrame[time]", segmentation_dimentionalised_DataFrame["time"]) if CP_dimentionalise_log_level >= 1 else None


    # Add non-dimensional columns to the DataFrame
    nonDim_columns = [
        'd_T_per_px', 'image_Nx_nonDim', 'image_Ny_nonDim', 'diameter_training_nonDim', 'diameter_estimate_used_nonDim',
        'diameter_mean_nonDim', 'diameter_median_nonDim', 'diameter_distribution_nonDim',
        'A_image_nonDim2', 'A_empty_nonDim2', 'A_FB_nonDim2', 'D_FB_nonDim', 'R_FB_nonDim', 'A_CP_mask_nonDim',
    ]
    # Add these columns to the existing DataFrame
    for col in nonDim_columns:
        segmentation_dimentionalised_DataFrame[col] = np.nan  # Initialize them with NaN
    # Add non-dimensional columns to the DataFrame
    columns_with_lists = [
        'diameter_distribution_nonDim',
    ]
    for column in columns_with_lists:
        segmentation_dimentionalised_DataFrame[column] = pd.Series(dtype="object")  # Explicitly set dtype to 'object'


    # extract pixel to nonDimentionalised length scaling.
    # This loop seems to be calculating the same values for each row based on the whole dataframe.
    # It can be optimized by calculating these outside the loop and assigning to the column.
    # However, to match the original logic structure, I will keep the loop for now,
    # but note that it's inefficient.

    A11_SF_R_mean = A11_SF_R_mean.sort_values(by='time')
    segmentation_dimentionalised_DataFrame = segmentation_dimentionalised_DataFrame.sort_values(by='time')
    segmentation_dimentionalised_DataFrame['R_mean_interpolated'] = np.interp(segmentation_dimentionalised_DataFrame['time'], A11_SF_R_mean['time'], A11_SF_R_mean['R_mean']) # Interpolate 'R_mean' at the 'time' values in segmentation_dimentionalised_DataFrame
    segmentation_dimentionalised_DataFrame['d_T_per_px'] = segmentation_dimentionalised_DataFrame['R_mean_interpolated'] / segmentation_dimentionalised_DataFrame['R_FB_px']


    for i in range(len(segmentation_dimentionalised_DataFrame)):
        print("i = ", i) if CP_dimentionalise_log_level >= 1 else None


        # find d_T_per_px - already calculated above, just retrieve
        d_T_per_px_i = segmentation_dimentionalised_DataFrame.loc[i, 'd_T_per_px']

        # Calculate non-dimensionalised values
        image_Nx_nonDim = segmentation_dimentionalised_DataFrame.loc[i, "image_Nx_px"] * d_T_per_px_i
        image_Ny_nonDim = segmentation_dimentionalised_DataFrame.loc[i, "image_Ny_px"] * d_T_per_px_i

        diameter_training_px_i = segmentation_dimentionalised_DataFrame.loc[i, 'diameter_training_px']
        diameter_estimate_used_px_i = segmentation_dimentionalised_DataFrame.loc[i, 'diameter_estimate_used_px']
        diameter_median_px_i = segmentation_dimentionalised_DataFrame.loc[i, 'diameter_median_px']
        diameter_mean_px_i = segmentation_dimentionalised_DataFrame.loc[i, 'diameter_mean_px']

        # Need to re-calculate diameters from masks as the original dataframe stores masks
        masks_i = segmentation_dimentionalised_DataFrame.loc[i, 'masks']
        # Check if masks_i is not None and is a numpy array before processing
        if masks_i is not None and isinstance(masks_i, np.ndarray):
            diameters_tuple_i_placeholder = utils.diameters(masks_i)
            diameter_array_px_i_placeholder = diameters_tuple_i_placeholder[1]
            diameter_distribution_px_i = diameter_array_px_i_placeholder
        else:
            diameter_distribution_px_i = np.array([]) # Handle cases with no masks


        diameter_training_nonDim = diameter_training_px_i * d_T_per_px_i if diameter_training_px_i is not None and d_T_per_px_i is not None else None
        diameter_estimate_used_nonDim_i = diameter_estimate_used_px_i * d_T_per_px_i if diameter_estimate_used_px_i is not None and d_T_per_px_i is not None else None
        diameter_median_nonDim_i = diameter_median_px_i * d_T_per_px_i if diameter_median_px_i is not None and d_T_per_px_i is not None else None
        diameter_mean_nonDim_i = diameter_mean_px_i * d_T_per_px_i if diameter_mean_px_i is not None and d_T_per_px_i is not None else None # Added None check
        diameter_distribution_nonDim_i = diameter_distribution_px_i * d_T_per_px_i if diameter_distribution_px_i is not None and d_T_per_px_i is not None else np.array([]) # Added None check


        A_image_nonDim = segmentation_dimentionalised_DataFrame.loc[i, "A_image_px2"] * d_T_per_px_i**2 if d_T_per_px_i is not None else None
        A_empty_nonDim = segmentation_dimentionalised_DataFrame.loc[i, "A_empty_px2"] * d_T_per_px_i**2 if d_T_per_px_i is not None else None
        A_FB_nonDim = segmentation_dimentionalised_DataFrame.loc[i, "A_FB_px2"] * d_T_per_px_i**2 if d_T_per_px_i is not None else None
        D_FB_nonDim = segmentation_dimentionalised_DataFrame.loc[i, "D_FB_px"] * d_T_per_px_i if d_T_per_px_i is not None else None
        R_FB_nonDim = segmentation_dimentionalised_DataFrame.loc[i, "R_FB_px"] * d_T_per_px_i if d_T_per_px_i is not None else None
        A_CP_mask_nonDim = segmentation_dimentionalised_DataFrame.loc[i, "A_CP_mask_px"] * d_T_per_px_i**2 if d_T_per_px_i is not None else None


        segmentation_dimentionalised_DataFrame.at[i, 'd_T_per_px_i'] = d_T_per_px_i
        segmentation_dimentionalised_DataFrame.at[i, 'image_Nx_nonDim'] = image_Nx_nonDim
        segmentation_dimentionalised_DataFrame.at[i, 'image_Ny_nonDim'] = image_Ny_nonDim
        segmentation_dimentionalised_DataFrame.at[i, 'diameter_training_nonDim'] = diameter_training_nonDim
        segmentation_dimentionalised_DataFrame.at[i, 'diameter_estimate_used_nonDim'] = diameter_estimate_used_nonDim_i
        segmentation_dimentionalised_DataFrame.at[i, 'diameter_mean_nonDim'] = diameter_mean_nonDim_i
        segmentation_dimentionalised_DataFrame.at[i, 'diameter_median_nonDim'] = diameter_median_nonDim_i
        segmentation_dimentionalised_DataFrame.at[i, 'diameter_distribution_nonDim'] = diameter_distribution_nonDim_i
        segmentation_dimentionalised_DataFrame.at[i, 'A_image_nonDim2'] = A_image_nonDim
        segmentation_dimentionalised_DataFrame.at[i, 'A_empty_nonDim2'] = A_empty_nonDim
        segmentation_dimentionalised_DataFrame.at[i, 'A_FB_nonDim2'] = A_FB_nonDim
        segmentation_dimentionalised_DataFrame.at[i, 'D_FB_nonDim'] = D_FB_nonDim
        segmentation_dimentionalised_DataFrame.at[i, 'R_FB_nonDim'] = R_FB_nonDim
        segmentation_dimentionalised_DataFrame.at[i, 'A_CP_mask_nonDim'] = A_CP_mask_nonDim

        # clean diameter_distribution_nonDim
        segmentation_dimentionalised_DataFrame['diameter_distribution_nonDim'] = segmentation_dimentionalised_DataFrame['diameter_distribution_nonDim'].apply(
            lambda x: np.array([x]) if isinstance(x, np.ndarray) and x.ndim == 0 else x
        )


    # Print columns that have None values
    none_columns = segmentation_dimentionalised_DataFrame.columns[segmentation_dimentionalised_DataFrame.isnull().any()].tolist()
    print(f"NB: Columns with None values: {none_columns}") if CP_dimentionalise_log_level >= 1 else None


    #################################################### return
    return segmentation_dimentionalised_DataFrame