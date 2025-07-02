import os
import sys
import pandas as pd
import numpy as np
import glob
import Format_1 as F_1

def append_roundness_metrics(
    input_dir,
    output_dir_manual="",
    output_dir_comment="",
    log_level=1
):
    """
    Calculate and append roundness metrics to the Analysis_A11_final_df DataFrame.
    
    Roundness is calculated as: contour_length^2 / (4 * pi * area)
    A perfect circle has roundness = 1, higher values indicate more irregular shapes.
    
    Parameters
    ----------
    input_dir : str
        Directory containing the Analysis_A11_final_df.pkl file
    output_dir_manual : str, optional
        Manual output directory, by default ""
    output_dir_comment : str, optional
        Comment to append to the output directory name, by default ""
    log_level : int, optional
        Logging level, by default 1
    
    Returns
    -------
    str
        Path to the output directory
    """
    # Create output directory
    output_dir = F_1.F_out_dir(input_dir=input_dir, script_path=__file__, 
                             output_dir_comment=output_dir_comment, 
                             output_dir_manual=output_dir_manual)
    
    if log_level >= 1:
        print(f"append_roundness_metrics: Output directory: {output_dir}")

    # Find the PKL file
    pandas_wildcard_str = os.path.join(input_dir, "Analysis_A11_final_df.pkl")
    pkl_files = glob.glob(pandas_wildcard_str)
    
    if not pkl_files:
        print(f"No Analysis_A11_final_df.pkl file found in {input_dir}")
        return output_dir
    
    # Load the DataFrame
    df_path = pkl_files[0]
    df = pd.read_pickle(df_path)
    
    if log_level >= 1:
        print(f"Loaded DataFrame from {df_path}")
        print(f"Original DataFrame shape: {df.shape}")
    
    # Check if required columns exist
    required_columns = [
        'contour_length_SRec_distribution_CSTx6_nonDim',
        'A_cell_SRec_distribution_CSTx6_nonDim2'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return output_dir
    
    # Calculate roundness distribution for each image
    if log_level >= 1:
        print("Calculating roundness metrics...")
    
    # Process each row (image) individually
    for idx, row in df.iterrows():
        # Extract arrays from the row
        contour_lengths = row['contour_length_SRec_distribution_CSTx6_nonDim']
        areas = row['A_cell_SRec_distribution_CSTx6_nonDim2']
        
        # Check if arrays exist and have matching lengths
        if isinstance(contour_lengths, np.ndarray) and isinstance(areas, np.ndarray) and len(contour_lengths) == len(areas):
            # Calculate roundness for each cell in this image
            roundness_values = (4 * np.pi * areas) / np.power(contour_lengths, 2)
            
            # Store the distribution in the DataFrame
            df.at[idx, 'Roundness_distribution_SRec_CSTx6_nonDim'] = roundness_values
            
            # Calculate mean roundness for this image using np.nanmean to handle NaN values
            mean_roundness = np.nanmean(roundness_values) if len(roundness_values) > 0 else np.nan
            df.at[idx, 'Roundness_mean_SRec_CSTx6_nonDim'] = mean_roundness
            
            if log_level >= 2:
                print(f"Image {row['image_number']}: Calculated roundness for {len(roundness_values)} cells, mean: {mean_roundness:.4f}")
        else:
            # Handle case where arrays don't exist or have mismatched lengths
            df.at[idx, 'Roundness_distribution_SRec_CSTx6_nonDim'] = np.array([])
            df.at[idx, 'Roundness_mean_SRec_CSTx6_nonDim'] = np.nan
            
            if log_level >= 2:
                print(f"Image {row['image_number']}: No valid data for roundness calculation")
    
    if log_level >= 1:
        print(f"Updated DataFrame shape: {df.shape}")
        print(f"Added columns: 'Roundness_distribution_SRec_CSTx6_nonDim', 'Roundness_mean_SRec_CSTx6_nonDim'")
    
    # Save the updated DataFrame back to the original location
    df.to_pickle(df_path)

    # Also save as CSV
    csv_path = os.path.join(input_dir, "Analysis_A11_final_df.csv")
    df.to_csv(csv_path, sep='\t', index=False)
    
    if log_level >= 1:
        print(f"Saved updated DataFrame to:")
        print(f"  {df_path}")
        print(f"  {csv_path}")
    
    return output_dir

if __name__ == "__main__":
    # Example usage
    input_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187"
    
    append_roundness_metrics(
        input_dir=input_dir,
        output_dir_comment="append_roundness_metrics",
        log_level=2
    )
