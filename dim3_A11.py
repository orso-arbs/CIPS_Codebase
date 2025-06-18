import numpy as np
import pandas as pd
import os
import pickle
import Format_1 as F_1
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

@F_1.ParameterLog(max_size=1024 * 10, log_level=0)
def dim3_A11(
    # input
    input_dir,  # Should be the output directory of CP_extract_1
    Analysis_A11_df=None,  # DataFrame from previous processing, or None to load from input_dir
    
    # Reference values for dimensionalization
    ref_values=None,  # Dictionary with 'd_T', 'S_L', 'T_b', etc.
    
    # output and logging
    dim3_A11_log_level=2,
    output_dir_manual="", 
    output_dir_comment=""
):
    """
    Dimensionalizes pixel data using physical parameters from Altantzis2011.
    
    Parameters
    ----------
    input_dir : str
        Path to the directory containing the extracted_DataFrame.pkl from CP_extract_1
    Analysis_A11_df : pd.DataFrame or None, optional
        The DataFrame from previous processing, or None to load from input_dir.
    ref_values : dict, optional
        Dictionary containing reference values for dimensionalization.
        Must include: 'd_T' (flame thickness), 'S_L' (laminar flame speed),
        'T_b' (burned gas temperature), and 't_ref' (flame time scale).
    dim3_A11_log_level : int, optional
        Controls the verbosity of logging. Default is 2.
    output_dir_manual : str, optional
        If provided, specifies the output directory. Default is "".
    output_dir_comment : str, optional
        Comment to append to the output directory name. Default is "".
    
    Returns
    -------
    output_dir : str
        Path to the output directory.
    """
    #################################################### I/O
    # Create output directory
    output_dir = F_1.F_out_dir(
        input_dir=input_dir, 
        script_path=__file__, 
        output_dir_comment=output_dir_comment, 
        output_dir_manual=output_dir_manual
    )
    
    #################################################### Load Extracted Data
    if Analysis_A11_df is None:
        # Look for extracted_DataFrame.pkl one level up from input_dir
        parent_dir = os.path.dirname(input_dir)
        extracted_data_path = os.path.join(parent_dir, 'extracted_DataFrame.pkl')
        
        # If not found, try in the input_dir as fallback
        if not os.path.exists(extracted_data_path):
            extracted_data_path = os.path.join(input_dir, 'extracted_DataFrame.pkl')
            
        print(f"\nLoading extracted data from: {extracted_data_path}") if dim3_A11_log_level >= 1 else None
        
        try:
            extracted_df = pd.read_pickle(extracted_data_path)
            Analysis_A11_df = extracted_df.copy()
        except FileNotFoundError:
            print(f"Error: Could not find extracted data file at {extracted_data_path}")
            print("Ensure CP_extract_1 ran successfully and produced 'extracted_DataFrame.pkl'")
            return output_dir
    else:
        print("\nUsing provided Analysis_A11_df DataFrame") if dim3_A11_log_level >= 1 else None
    
    # Get number of images/rows from loaded data
    N_images = len(Analysis_A11_df)
    print(f"Processing {N_images} images for dimensionalization") if dim3_A11_log_level >= 1 else None
    
    print(Analysis_A11_df["image_number"])
    #################################################### Reference Values
    # Default reference values from A11 if not provided
    if ref_values is None:
        ref_values = {
            'd_T': 7.516e-3,  # flame thickness [m]
            'S_L': 51.44,     # laminar flame speed [m/s]
            'T_b': 1843.5,    # burned gas temperature [K]
            't_ref': 7.516e-3/51.44  # flame time scale [s]
        }
    
    # Make sure all required values exist
    required_keys = ['d_T', 'S_L', 'T_b', 't_ref']
    for key in required_keys:
        if key not in ref_values:
            print(f"Error: Reference value '{key}' not provided")
            if key == 't_ref' and 'd_T' in ref_values and 'S_L' in ref_values:
                ref_values['t_ref'] = ref_values['d_T'] / ref_values['S_L']
                print(f"Calculated 't_ref' as d_T/S_L = {ref_values['t_ref']} s")
            else:
                return output_dir
    
    # Add reference values to the DataFrame as metadata
    for key, value in ref_values.items():
        Analysis_A11_df[f'ref_{key}'] = value
    
    #################################################### Add Non-Dimensional Columns
    # Add new columns for dimensionalization
    dim_columns = [
        'd_T_per_px', 'image_Nx_nonDim', 'image_Ny_nonDim', 
        'diameter_training_nonDim', 'diameter_estimate_used_nonDim', 
        'd_cell_mean_nonDim', 'd_cell_median_nonDim',
        'A_image_nonDim2', 'A_empty_nonDim2', 'A_SF_nonDim2', 
        'D_SF_nonDim', 'R_SF_nonDim', 'A_CP_mask_nonDim',
    ]
    
    # Array-valued columns that need object dtype
    array_columns = [
        'diameter_distribution_nonDim',
        'd_cell_distribution_nonDim', 
        'centroid_xIm_distribution_nonDim', 
        'centroid_yIm_distribution_nonDim',
        'A_cell_distribution_nonDim2',
    ]
    
    # Initialize regular columns with NaN
    for col in dim_columns:
        if col not in Analysis_A11_df.columns:
            Analysis_A11_df[col] = np.nan
    
    # Initialize array columns with object dtype
    for col in array_columns:
        if col not in Analysis_A11_df.columns:
            Analysis_A11_df[col] = pd.Series(dtype='object')
    
    #################################################### Calculate Dimensionalization Factor
    # Calculate d_T_per_px from VisIt data
    print("\nCalculating dimensionalization factor d_T_per_px...") if dim3_A11_log_level >= 1 else None
    
    # Method: Use R_SF_Average_VisIt (non-dimensional) and R_SF_px (pixels)
    # d_T_per_px = (R_SF_Average_VisIt * d_T) / R_SF_px
    Analysis_A11_df['d_T_per_px'] = (Analysis_A11_df['R_SF_Average_VisIt'] * ref_values['d_T']) / Analysis_A11_df['R_SF_px']
    
    # Print the d_T_per_px values for verification
    if dim3_A11_log_level >= 2:
        print("\nDimensionalization factors (d_T_per_px):")
        # Corrected loop: Iterate with enumerate to get both index and value
        for i, value in enumerate(Analysis_A11_df['d_T_per_px']):
            print(f"  Image {i + 1}: {value:.6e} d_T/px")

    #################################################### Calculate Non-Dimensionalized Values
    print("\nCalculating dimensionalized quantities...") if dim3_A11_log_level >= 1 else None
    
    # Process each image - using DataFrame index rather than range(N_images)
    for idx, (i, row) in enumerate(Analysis_A11_df.iterrows()):
        print(f"\rProcessing image {idx+1}/{N_images}", end='', flush=True) if dim3_A11_log_level >= 1 else None
        
        d_T_per_px_i = row['d_T_per_px']
        
        if pd.isna(d_T_per_px_i):
            print(f"\nWarning: Skipping dimensionalization for index {i} due to missing d_T_per_px.")
            continue
            
        # Image dimensions in non-dimensional units
        Analysis_A11_df.at[i, 'image_Nx_nonDim'] = row['image_Nx_px'] * d_T_per_px_i
        Analysis_A11_df.at[i, 'image_Ny_nonDim'] = row['image_Ny_px'] * d_T_per_px_i
        
        # Length values in non-dimensional units (scalar values)
        for dim_field, px_field in [
            ('diameter_training_nonDim', 'diameter_training_px'),
            ('diameter_estimate_used_nonDim', 'diameter_estimate_used_px'),
            ('d_cell_mean_nonDim', 'd_cell_mean_px'),
            ('d_cell_median_nonDim', 'd_cell_median_px'),
        ]:
            if px_field in Analysis_A11_df.columns:
                px_value = row[px_field]
                Analysis_A11_df.at[i, dim_field] = px_value * d_T_per_px_i if pd.notna(px_value) else np.nan
        
        # Handle array-valued length distributions
        array_fields = [
            ('d_cell_distribution_nonDim', 'd_cell_distribution_px'),
            ('centroid_xIm_distribution_nonDim', 'centroid_xIm_distribution_px'),
            ('centroid_yIm_distribution_nonDim', 'centroid_yIm_distribution_px'),
        ]
        
        for dim_field, px_field in array_fields:
            if px_field in Analysis_A11_df.columns:
                px_array = row[px_field]
                # Check if array exists and has elements
                if isinstance(px_array, (np.ndarray, list)) and len(px_array) > 0:
                    # Create new numpy array directly with scaling applied
                    Analysis_A11_df.at[i, dim_field] = np.array(px_array) * d_T_per_px_i
                else:
                    # Consistent empty array format
                    Analysis_A11_df.at[i, dim_field] = np.array([])
        
        # Area values in non-dimensional units (scalar values)
        for dim_field, px_field in [
            ('A_image_nonDim2', 'A_image_px2'),
            ('A_empty_nonDim2', 'A_empty_px2'),
            ('A_SF_nonDim2', 'A_SF_px2'),
            ('A_CP_mask_nonDim', 'A_CP_mask_px'),
        ]:
            if px_field in Analysis_A11_df.columns:
                px_value = row[px_field]
                Analysis_A11_df.at[i, dim_field] = px_value * (d_T_per_px_i**2) if pd.notna(px_value) else np.nan
        
        # Handle array-valued area distributions
        if 'A_cell_distribution_px2' in Analysis_A11_df.columns:
            px_array = row['A_cell_distribution_px2']
            # Check if array exists and has elements
            if isinstance(px_array, (np.ndarray, list)) and len(px_array) > 0:
                # Create new numpy array directly with scaling applied
                Analysis_A11_df.at[i, 'A_cell_distribution_nonDim2'] = np.array(px_array) * (d_T_per_px_i**2)
            else:
                # Consistent empty array format
                Analysis_A11_df.at[i, 'A_cell_distribution_nonDim2'] = np.array([])
        
        # Length values derived from areas
        Analysis_A11_df.at[i, 'D_SF_nonDim'] = row['D_SF_px'] * d_T_per_px_i
        Analysis_A11_df.at[i, 'R_SF_nonDim'] = row['R_SF_px'] * d_T_per_px_i
    
    print("\nDimensionalization complete!") if dim3_A11_log_level >= 1 else None
    
    # Make sure the index is properly reset before saving
    Analysis_A11_df = Analysis_A11_df.reset_index(drop=True)
    
    #################################################### Save Results
    # Save the dimensionalized DataFrame
    output_pkl_path = os.path.join(output_dir, 'Analysis_A11_df.pkl')
    output_csv_path = os.path.join(output_dir, 'Analysis_A11_df.csv')
    
    Analysis_A11_df.to_pickle(output_pkl_path)
    Analysis_A11_df.to_csv(output_csv_path, sep='\t', index=False)
    
    print(f"\nSaved dimensionalized data to:") if dim3_A11_log_level >= 1 else None
    print(f"  - {output_pkl_path}") if dim3_A11_log_level >= 1 else None
    print(f"  - {output_csv_path}") if dim3_A11_log_level >= 1 else None
    
    return output_dir

# Example usage when script is run directly
if __name__ == "__main__":
    print("Running dim3_A11 as standalone module...")
    
    # Input should be the CP_extract output directory
    input_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\20250604_1311111\20250604_1312276\20250604_1312276\20250604_1313140\20250615_1635229"
    
    # Define reference values (same as in Analysis_Altantzis2011)
    ref_values = {
        'd_T': 7.516e-3,    # flame thickness [m]
        'S_L': 51.44,     # laminar flame speed [m/s]
        'T_b': 1843.5,    # burned gas temperature [K]
        't_ref': 7.516e-3/51.44  # flame time scale [s]
    }
    
    output_dir = dim3_A11(
        input_dir=input_dir,
        Analysis_A11_df=None,  # When running standalone, there is no DataFrame yet
        ref_values=ref_values,
        dim3_A11_log_level=2,
        output_dir_comment="test_standalone"
    )
    
    print(f"Dimensionalization complete. Results in: {output_dir}")
