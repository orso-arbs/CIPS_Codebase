import numpy as np
import pandas as pd
import os
import pickle
import Format_1 as F_1
import warnings
from cellpose import utils # Needed for utils.diameters

warnings.simplefilter(action='ignore', category=FutureWarning)

@F_1.ParameterLog(max_size = 1024 * 10, log_level = 0)
def CP_dimentionalise_1(
    # input
    input_dir, # Should be the output directory of CP_extract_1

    # output and logging
    CP_dimentionalise_log_level = 0,
    output_dir_manual = "", output_dir_comment = "",
    ):
    """
    Loads extracted segmentation data, non-dimensionalizes metrics using A11 data,
    and saves the results to a pandas DataFrame.

    This function reads the DataFrame produced by CP_extract_1 (expected to be
    'extracted_DataFrame.pkl' in the input_dir). It then loads external A11
    simulation data, calculates a time mapping based on image numbers, interpolates
    A11 flame radius data (R_mean) to determine a scaling factor (d_T_per_px)
    for each image/time step. Using this scaling factor, it calculates
    non-dimensional versions of various metrics (diameters, areas) and appends
    these as new columns to the DataFrame. Finally, the dimensionalized DataFrame
    is saved in CSV, Pickle, and Excel formats.

    Parameters
    ----------
    input_dir : str
        Path to the directory containing the output from CP_extract_1, specifically
        the 'extracted_DataFrame.pkl' file.
    CP_dimentionalise_log_level : int, optional
        Controls the verbosity of logging messages printed to the console.
        0: Minimal logging.
        1: Basic processing steps.
        Defaults to 0.
    output_dir_manual : str, optional
        If provided, specifies the exact output directory path. Overrides the
        default naming convention managed by `Format_1.F_out_dir`. Defaults to "".
    output_dir_comment : str, optional
        A comment to append to the default output directory name if `output_dir_manual`
        is not provided. Defaults to "".

    Returns
    -------
    output_dir : str
        The path to the directory where the output DataFrame files
        (`dimentionalised_DataFrame.csv`, `dimentionalised_DataFrame.pkl`,
        `dimentionalised_DataFrame.xlsx`) and logs (`_log.json`) are saved.
        This is always the first return value as required by Format_1.

    Notes
    -----
    - This function relies on `Format_1.py` for output directory creation (`F_out_dir`)
      and parameter logging (`@F_1.ParameterLog`).
    - Requires external data files containing A11 simulation results located at
      fixed paths (e.g., "C:\\Users\\obs\\OneDrive\\ETH\\ETH_MSc\\Masters Thesis\\Data\\A11_manual_extraction\\...").
    - Assumes the input DataFrame contains necessary columns like 'image_number',
      'R_FB_px', 'diameter_training_px', 'diameter_estimate_used_px', etc.
    - The non-dimensionalization process maps image sequence numbers linearly to a
      simulation time range [0, t_max=6.81] and uses interpolated A11 data (specifically
      `A11_SF_R_mean`) to determine the scaling factor `d_T_per_px`.
    """
    #################################################### I/O
    # Use the input_dir (output of CP_extract) as the base for the new output dir
    output_dir = F_1.F_out_dir(input_dir, __file__, output_dir_comment = output_dir_comment) # Format_1 required definition of output directory

    #################################################### Load Extracted Data
    extracted_data_path = os.path.join(input_dir, 'extracted_DataFrame.pkl')
    print(f"\n Loading extracted data from: {extracted_data_path} \n") if CP_dimentionalise_log_level >= 1 else None
    try:
        extracted_df = pd.read_pickle(extracted_data_path)
    except FileNotFoundError:
        print(f"Error: Could not find extracted data file at {extracted_data_path}")
        print("Ensure CP_extract_1 ran successfully and produced 'extracted_DataFrame.pkl' in the specified input directory.")
        return None # Or raise an error

    N_images = len(extracted_df) # Get number of images/rows from loaded data

    #################################################### Images and A11 data for Non-Dimensionalization

    print("\n Non Dimentionalising and matching CP and A11 data \n") if CP_dimentionalise_log_level >= 1 else None

    # Load A11 data (Consider making these paths configurable or relative if possible)
    a11_base_path = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction"
    try:
        A11_SF_K_mean = pd.read_csv(os.path.join(a11_base_path, "A11_SF_K_mean_as_mean_stretch_rate_vs_time_manual_extraction.txt"))
        A11_SF_N_c = pd.read_csv(os.path.join(a11_base_path, "A11_SF_N_c_as_number_of_cells_vs_time_manual_extraction.txt"))
        A11_SF_R_mean = pd.read_csv(os.path.join(a11_base_path, "A11_SF_R_mean_as_average_radius_of_the_wrinkled_flame_fron_vs_time_manual_extraction.txt"))
        A11_SF_R_mean_dot = pd.read_csv(os.path.join(a11_base_path, "A11_SF_R_mean_dot_as_first_time_derivative_of_the_average_radius_of_the_wrinkled_flame_front_vs_time_manual_extraction.txt"))
        A11_SF_s_a = pd.read_csv(os.path.join(a11_base_path, "A11_SF_s_a_as_average_normal_component_of_the_absolute_propagation_velocity_vs_time_manual_extraction.txt"))
        A11_SF_s_d = pd.read_csv(os.path.join(a11_base_path, "A11_SF_s_d_as_average_density_weighted_displacement_speed_vs_time_manual_extraction.txt"))
        A11_SF_A = pd.read_csv(os.path.join(a11_base_path, "A11_SF_A_as_flame_surface_area_of_the_wrinkled_spherical_front_vs_time_manual_extraction.txt"))
        A11_SF_a_t = pd.read_csv(os.path.join(a11_base_path, "A11_SF_a_t_as_average_total_aerodynamic_strain_vs_time_manual_extraction.txt"))
        A11_SF_iHRR = pd.read_csv(os.path.join(a11_base_path, "A11_SF_iHRR_as_integral_heat_release_rate_vs_time_manual_extraction.txt"))
        A11_SF_K_geom = pd.read_csv(os.path.join(a11_base_path, "A11_SF_K_geom_as_geometric_stretch_rate_vs_time_manual_extraction.txt"))
    except FileNotFoundError as e:
        print(f"Error loading A11 data file: {e}")
        print("Please ensure A11 data files are present at the expected location:", a11_base_path)
        return None # Or raise an error


    # reference values
    ''' A11 p.36
    "...reference values from detailed description of chemistry is based on the mech-
    anism of Li et al. [2004], consisting of 9 species and 19 non-duplicate
    elementary reactions. The planar premixed flame structure calculated us-
    ing PREMIX [Rupley et al., 1995]..."
    '''
    d_T = 7.516 * 1e-3 # flame thickness [m]
    S_L = 51.44 # laminar flame speed [m/s]
    T_b = 1843.5 # burned gas temperature [K]

    t_ref = d_T/S_L # flame time scale [s]

    t_max = 6.81 # max time estimated from plots [s] - Check if this is non-dim or dimensional time
    R0 = 10 * d_T # initial Spherical flame radius [m]

    # Create a copy to avoid modifying the original DataFrame loaded
    dimentionalised_df = extracted_df.copy()

    # Calculate the time values for each DataFrame row
    dimentionalised_df['time'] = np.nan # initialize time collumn filled with nan
    min_image_num = dimentionalised_df["image_number"].min()
    max_image_num = dimentionalised_df["image_number"].max()
    if pd.isna(min_image_num) or pd.isna(max_image_num):
         print("Warning: Cannot calculate time mapping due to missing image numbers.")
         # Handle this case appropriately, maybe skip time calculation or fill with a default
    elif min_image_num == max_image_num:
        dimentionalised_df["time"] = 0  # Assign 0 to all if there's no range
        print("Warning in dimensionalisation: min_image_num == max_image_num. Setting time to 0.")
    else:
        # Linear mapping from [min_image_num, max_image_num] to [0, t_max]
        dimentionalised_df["time"] = (dimentionalised_df["image_number"] - min_image_num) / (max_image_num - min_image_num) * t_max
        print("dimentionalised_df['time']:", dimentionalised_df["time"].to_string()) if CP_dimentionalise_log_level >= 1 else None


    # Add non-dimensional columns to the DataFrame
    nonDim_columns = [
        'd_T_per_px', 'image_Nx_nonDim', 'image_Ny_nonDim', 'diameter_training_nonDim', 'diameter_estimate_used_nonDim',
        'diameter_mean_nonDim', 'diameter_median_nonDim', 'diameter_distribution_nonDim',
        'A_image_nonDim2', 'A_empty_nonDim2', 'A_FB_nonDim2', 'D_FB_nonDim', 'R_FB_nonDim', 'A_CP_mask_nonDim',
    ]
    # Add these columns to the existing DataFrame
    for col in nonDim_columns:
        dimentionalised_df[col] = np.nan  # Initialize them with NaN

    # Ensure 'diameter_distribution_nonDim' column exists and has dtype 'object'
    if 'diameter_distribution_nonDim' not in dimentionalised_df.columns:
        dimentionalised_df['diameter_distribution_nonDim'] = pd.Series(dtype="object")
    else:
        # Ensure existing column can hold lists/arrays if it wasn't already object type
        if dimentionalised_df['diameter_distribution_nonDim'].dtype != 'object':
             dimentionalised_df['diameter_distribution_nonDim'] = dimentionalised_df['diameter_distribution_nonDim'].astype(object)


    # Calculate pixel to nonDimensionalised length scaling (d_T_per_px)
    # Sort both dataframes by time for interpolation
    A11_SF_R_mean = A11_SF_R_mean.sort_values(by='time')
    dimentionalised_df = dimentionalised_df.sort_values(by='time')

    # Interpolate 'R_mean' from A11 data at the 'time' values in dimentionalised_df
    # Ensure times are within the range of A11 data or handle extrapolation if needed
    dimentionalised_df['R_mean_interpolated_nonDim'] = np.interp(
        dimentionalised_df['time'],
        A11_SF_R_mean['time'],
        A11_SF_R_mean['R_mean'] # Assuming A11 R_mean is already non-dimensionalized by d_T
    )

    # Calculate d_T_per_px = (R_mean_nonDim * d_T) / R_FB_px
    # Or if R_mean in A11 is dimensional: d_T_per_px = R_mean_interpolated_dim / R_FB_px
    # Assuming A11 R_mean is non-dimensionalized by d_T as per typical flame analysis
    dimentionalised_df['d_T_per_px'] = (dimentionalised_df['R_mean_interpolated_nonDim'] * d_T) / dimentionalised_df['R_FB_px']
    print("Calculated d_T_per_px:", dimentionalised_df['d_T_per_px'].to_string()) if CP_dimentionalise_log_level >= 1 else None

    # Calculate non-dimensionalised values row by row (vectorization might be faster if needed)
    for i in dimentionalised_df.index: # Use index after sorting
        print("Processing row index =", i) if CP_dimentionalise_log_level >= 2 else None

        d_T_per_px_i = dimentionalised_df.loc[i, 'd_T_per_px']

        if pd.isna(d_T_per_px_i):
            print(f"Warning: Skipping non-dim calculation for index {i} due to missing d_T_per_px.")
            continue # Skip calculations if scaling factor is NaN

        # Calculate non-dimensional lengths (multiply px value by d_T_per_px)
        image_Nx_nonDim = dimentionalised_df.loc[i, "image_Nx_px"] * d_T_per_px_i
        image_Ny_nonDim = dimentionalised_df.loc[i, "image_Ny_px"] * d_T_per_px_i

        diameter_training_px_i = dimentionalised_df.loc[i, 'diameter_training_px']
        diameter_estimate_used_px_i = dimentionalised_df.loc[i, 'diameter_estimate_used_px']
        diameter_median_px_i = dimentionalised_df.loc[i, 'diameter_median_px']
        diameter_mean_px_i = dimentionalised_df.loc[i, 'diameter_mean_px']
        diameter_distribution_px_i = dimentionalised_df.loc[i, 'diameter_distribution_px'] # This should be an array

        diameter_training_nonDim = diameter_training_px_i * d_T_per_px_i if pd.notna(diameter_training_px_i) else np.nan
        diameter_estimate_used_nonDim_i = diameter_estimate_used_px_i * d_T_per_px_i if pd.notna(diameter_estimate_used_px_i) else np.nan
        diameter_median_nonDim_i = diameter_median_px_i * d_T_per_px_i if pd.notna(diameter_median_px_i) else np.nan
        diameter_mean_nonDim_i = diameter_mean_px_i * d_T_per_px_i if pd.notna(diameter_mean_px_i) else np.nan

        # Handle potential issues with diameter_distribution_px (e.g., if it's NaN or not an array)
        if isinstance(diameter_distribution_px_i, np.ndarray):
            diameter_distribution_nonDim_i = diameter_distribution_px_i * d_T_per_px_i
        else:
            diameter_distribution_nonDim_i = np.nan # Or handle as appropriate

        # Calculate non-dimensional areas (multiply px^2 value by d_T_per_px^2)
        A_image_nonDim = dimentionalised_df.loc[i, "A_image_px2"] * d_T_per_px_i**2
        A_empty_nonDim = dimentionalised_df.loc[i, "A_empty_px2"] * d_T_per_px_i**2
        A_FB_nonDim = dimentionalised_df.loc[i, "A_FB_px2"] * d_T_per_px_i**2
        A_CP_mask_nonDim = dimentionalised_df.loc[i, "A_CP_mask_px"] * d_T_per_px_i**2

        # Calculate non-dimensional lengths derived from areas
        D_FB_nonDim = dimentionalised_df.loc[i, "D_FB_px"] * d_T_per_px_i
        R_FB_nonDim = dimentionalised_df.loc[i, "R_FB_px"] * d_T_per_px_i


        # Assign calculated values back to the DataFrame
        dimentionalised_df.at[i, 'image_Nx_nonDim'] = image_Nx_nonDim
        dimentionalised_df.at[i, 'image_Ny_nonDim'] = image_Ny_nonDim
        dimentionalised_df.at[i, 'diameter_training_nonDim'] = diameter_training_nonDim
        dimentionalised_df.at[i, 'diameter_estimate_used_nonDim'] = diameter_estimate_used_nonDim_i
        dimentionalised_df.at[i, 'diameter_mean_nonDim'] = diameter_mean_nonDim_i
        dimentionalised_df.at[i, 'diameter_median_nonDim'] = diameter_median_nonDim_i
        dimentionalised_df.at[i, 'diameter_distribution_nonDim'] = diameter_distribution_nonDim_i # Assign the array
        dimentionalised_df.at[i, 'A_image_nonDim2'] = A_image_nonDim
        dimentionalised_df.at[i, 'A_empty_nonDim2'] = A_empty_nonDim
        dimentionalised_df.at[i, 'A_FB_nonDim2'] = A_FB_nonDim
        dimentionalised_df.at[i, 'D_FB_nonDim'] = D_FB_nonDim
        dimentionalised_df.at[i, 'R_FB_nonDim'] = R_FB_nonDim
        dimentionalised_df.at[i, 'A_CP_mask_nonDim'] = A_CP_mask_nonDim

    # Clean diameter_distribution_nonDim again after loop if necessary (e.g., ensure arrays)
    # This might not be needed if handled correctly within the loop
    dimentionalised_df['diameter_distribution_nonDim'] = dimentionalised_df['diameter_distribution_nonDim'].apply(
         lambda x: np.array([x]) if isinstance(x, np.ndarray) and x.ndim == 0 else x
    )


    #################################################### Save Dimensionalized Data

    print(f"\n Saving dimensionalized data to: {output_dir} \n") if CP_dimentionalise_log_level >= 1 else None

    # Define output filenames
    csv_filename = 'dimentionalised_DataFrame.csv'
    pickle_filename = 'dimentionalised_DataFrame.pkl'
    excel_filename = 'dimentionalised_DataFrame.xlsx'

    # Save DataFrame to CSV
    dimentionalised_df.to_csv(os.path.join(output_dir, csv_filename), sep='\t', index=False)

    # Save DataFrame to Pickle
    dimentionalised_df.to_pickle(os.path.join(output_dir, pickle_filename))

    # # Save DataFrame to Excel (useful for manual inspection)
    # try:
    #     # Ensure complex objects like numpy arrays are handled if needed, or select subset of columns
    #     # excel_safe_df = dimentionalised_df.select_dtypes(exclude=['object']) # Example: exclude object columns
    #     # excel_safe_df.to_excel(os.path.join(output_dir, excel_filename), index=False)
    #     # Or convert arrays to strings/lists if Excel export is desired with them
    #     df_for_excel = dimentionalised_df.copy()
    #     for col in df_for_excel.columns:
    #          if df_for_excel[col].apply(lambda x: isinstance(x, np.ndarray)).any():
    #              df_for_excel[col] = df_for_excel[col].apply(lambda x: str(x.tolist()) if isinstance(x, np.ndarray) else x) # Convert arrays to string representation of list
    #     df_for_excel.to_excel(os.path.join(output_dir, excel_filename), index=False)

    # except Exception as e:
    #     print(f"Warning: Could not save to Excel format. Error: {e}")
    #     print("Excel format might not support complex data types like numpy arrays stored in the DataFrame.")


    #################################################### return
    return output_dir # Format_1 requires output_dir as first return

# Example usage (if run directly)
if __name__ == '__main__':
    # This is placeholder logic and needs actual paths
    print("Running CP_dimentionalise_1 as main script (Example Usage)")
    # Define a dummy input directory where CP_extract_1 output would be
    # You would replace this with the actual output directory from a CP_extract_1 run
    example_input_dir = r"C:\path\to\your\cp_extract_output_directory" # MUST BE SET MANUALLY FOR TESTING

    if not os.path.exists(example_input_dir):
         print(f"Error: Example input directory does not exist: {example_input_dir}")
         print("Please set 'example_input_dir' to a valid directory containing 'extracted_DataFrame.pkl'")
    else:
        # Create a dummy extracted_DataFrame.pkl for testing if it doesn't exist
        dummy_pkl_path = os.path.join(example_input_dir, 'extracted_DataFrame.pkl')
        if not os.path.exists(dummy_pkl_path):
            print(f"Creating dummy 'extracted_DataFrame.pkl' in {example_input_dir} for testing.")
            # Create a minimal DataFrame structure similar to what CP_extract would produce
            dummy_data = {
                'image_number': [1, 2, 3],
                'image_Nx_px': [100, 100, 100], 'image_Ny_px': [100, 100, 100],
                'diameter_training_px': [30, 30, 30], 'diameter_estimate_used_px': [28, 29, 31],
                'diameter_mean_px': [27.5, 28.5, 30.5], 'diameter_median_px': [27, 28, 30],
                'diameter_distribution_px': [np.array([26, 27, 28, 29]), np.array([27, 28, 29, 30]), np.array([29, 30, 31, 32])],
                'A_image_px2': [10000, 10000, 10000], 'A_empty_px2': [1000, 1100, 900],
                'A_FB_px2': [9000, 8900, 9100], 'D_FB_px': [106.9, 106.3, 107.5], 'R_FB_px': [53.5, 53.2, 53.8],
                'A_CP_mask_px': [5000, 5100, 4900],
                # Add other necessary columns if the function depends on them
            }
            dummy_df = pd.DataFrame(dummy_data)
            # Ensure object dtype for array column
            dummy_df['diameter_distribution_px'] = dummy_df['diameter_distribution_px'].astype(object)
            dummy_df.to_pickle(dummy_pkl_path)

        # Create dummy A11 files if they don't exist (replace with actual paths if available)
        a11_test_path = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction" # Use the same path as in the function
        os.makedirs(a11_test_path, exist_ok=True)
        a11_r_mean_path = os.path.join(a11_test_path, "A11_SF_R_mean_as_average_radius_of_the_wrinkled_flame_fron_vs_time_manual_extraction.txt")
        if not os.path.exists(a11_r_mean_path):
             print(f"Creating dummy A11 R_mean file at {a11_r_mean_path}")
             a11_r_mean_data = {'time': [0, 2, 4, 6, 7], 'R_mean': [10, 15, 25, 40, 50]} # Example non-dim R_mean/d_T
             pd.DataFrame(a11_r_mean_data).to_csv(a11_r_mean_path, index=False)
        # Create other dummy A11 files if needed by the function logic

        # Call the function
        output_directory = CP_dimentionalise_1(
            input_dir=example_input_dir,
            CP_dimentionalise_log_level=1,
            output_dir_comment="_test_run"
        )

        if output_directory:
            print(f"CP_dimentionalise_1 completed successfully.")
            print(f"Output saved to: {output_directory}")
        else:
            print("CP_dimentionalise_1 failed.")