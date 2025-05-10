import numpy as np
import pandas as pd
import os
import pickle
import Format_1 as F_1
import warnings
from cellpose import utils # Needed for utils.diameters

warnings.simplefilter(action='ignore', category=FutureWarning)

@F_1.ParameterLog(max_size = 1024 * 10, log_level = 0)
def dimentionalise_2_from_VisIt_R_Average(
    # input
    input_dir, # Should be the output directory of CP_extract_1

    # output and logging
    CP_dimentionalise_log_level = 0,
    output_dir_manual = "", output_dir_comment = "",
    ):

    """
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
    
    # Get number of images/rows from loaded data
    N_images = len(extracted_df)


    #################################################### Reference Values

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


    #################################################### Non-Dimensionalization from A11 manually extracted data

    # length -> length / flame thickness
    # speed -> speed / flame speed










    #################### Initialise DataFrame for non-dimensionalisation

    # Create a copy to avoid modifying the loaded CP_exctract DataFrame 
    dimentionalised_df = extracted_df.copy()
    # Add non-dimensional columns to the new DataFrame
    nonDim_columns = [
        'd_T_per_px', 'image_Nx_nonDim', 'image_Ny_nonDim', 'diameter_training_nonDim', 'diameter_estimate_used_nonDim',
        'diameter_mean_nonDim', 'diameter_median_nonDim', 'diameter_distribution_nonDim',
        'A_image_nonDim2', 'A_empty_nonDim2', 'A_SF_nonDim2', 'D_SF_nonDim', 'R_SF_nonDim', 'A_CP_mask_nonDim',
    ]
    for col in nonDim_columns:
        dimentionalised_df[col] = np.nan  # Initialize them with NaN

    #################### Calculate ratio of nonDimensionalised length per pixel (d_T_per_px)

    print("NOTE: assuming VisIt data is in nonDImentionalised scaling d_T to calculate ratio of nonDimensionalised length per pixel (d_T_per_px)")
    # d_T_per_px = (R_mean_nonDim * d_T) / R_SF_px
    dimentionalised_df['d_T_per_px'] = (dimentionalised_df['R_SF_Average_VisIt'] * d_T) / dimentionalised_df['R_SF_px']
    print("Calculated d_T_per_px:", dimentionalised_df['d_T_per_px'].to_string()) if CP_dimentionalise_log_level >= 1 else None




    #################### Calculate non-dimensionalised values 

    # row by row (by Image) (vectorization might be faster if needed)
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

        # Ensure 'diameter_distribution_nonDim' column exists and has dtype 'object'
        if 'diameter_distribution_nonDim' not in dimentionalised_df.columns:
            dimentionalised_df['diameter_distribution_nonDim'] = pd.Series(dtype="object")
        else:
            # Ensure existing column can hold lists/arrays if it wasn't already object type
            if dimentionalised_df['diameter_distribution_nonDim'].dtype != 'object':
                dimentionalised_df['diameter_distribution_nonDim'] = dimentionalised_df['diameter_distribution_nonDim'].astype(object)
        # Handle potential issues with diameter_distribution_px (e.g., if it's NaN or not an array)
        if isinstance(diameter_distribution_px_i, np.ndarray):
            diameter_distribution_nonDim_i = diameter_distribution_px_i * d_T_per_px_i
        else:
            diameter_distribution_nonDim_i = np.nan # Or handle as appropriate

        # Calculate non-dimensional areas (multiply px^2 value by d_T_per_px^2)
        A_image_nonDim = dimentionalised_df.loc[i, "A_image_px2"] * d_T_per_px_i**2
        A_empty_nonDim = dimentionalised_df.loc[i, "A_empty_px2"] * d_T_per_px_i**2
        A_SF_nonDim = dimentionalised_df.loc[i, "A_SF_px2"] * d_T_per_px_i**2
        A_CP_mask_nonDim = dimentionalised_df.loc[i, "A_CP_mask_px"] * d_T_per_px_i**2

        # Calculate non-dimensional lengths derived from areas
        D_SF_nonDim = dimentionalised_df.loc[i, "D_SF_px"] * d_T_per_px_i
        R_SF_nonDim = dimentionalised_df.loc[i, "R_SF_px"] * d_T_per_px_i


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
        dimentionalised_df.at[i, 'A_SF_nonDim2'] = A_SF_nonDim
        dimentionalised_df.at[i, 'D_SF_nonDim'] = D_SF_nonDim
        dimentionalised_df.at[i, 'R_SF_nonDim'] = R_SF_nonDim
        dimentionalised_df.at[i, 'A_CP_mask_nonDim'] = A_CP_mask_nonDim

    # Clean diameter_distribution_nonDim again after loop if necessary (e.g., ensure arrays)
    # This might not be needed if handled correctly within the loop
    dimentionalised_df['diameter_distribution_nonDim'] = dimentionalised_df['diameter_distribution_nonDim'].apply(
        lambda x: np.array([x]) if isinstance(x, np.ndarray) and x.ndim == 0 else x
    )


    #################################################### Save Dimensionalized Data

    print(f"\n Saving dimensionalized data to: {output_dir} \n") if CP_dimentionalise_log_level >= 1 else None

    # Save DataFrame to CSV
    csv_filename = 'dimentionalised_DataFrame.csv'
    dimentionalised_df.to_csv(os.path.join(output_dir, csv_filename), sep='\t', index=False)

    # Save DataFrame to Pickle
    pickle_filename = 'dimentionalised_DataFrame.pkl'
    dimentionalised_df.to_pickle(os.path.join(output_dir, pickle_filename))

    # # Save DataFrame to Excel (useful for manual inspection)
    #excel_filename = 'dimentionalised_DataFrame.xlsx'
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