import pandas as pd
from pathlib import Path

def update_dataframes_in_dirs(target_dirs: list, source_file_path: str):
    """
    Updates pickle and CSV files in a list of directories by adding two new
    columns from a source pickle file.

    Args:
        target_dirs (list): A list of strings, where each string is a path to a
                            directory to be processed.
        source_file_path (str): The full path to the .pkl file that contains
                                the columns to be added.
    """
    # --- Configuration ---
    # The name of the file to be updated in each directory
    PICKLE_FILENAME = "Visit_projector_1_data.pkl"
    CSV_FILENAME = "Visit_projector_1_data.csv"
    # The names of the columns to add
    COLUMNS_TO_ADD = [
        "Min_Psuedocolored_variable_SF_VisIt",
        "Max_Psuedocolored_variable_SF_VisIt"
    ]

    # --- 1. Load the Source Data ---
    source_path = Path(source_file_path)
    if not source_path.is_file():
        print(f"Error: Source file not found at {source_path}")
        return

    try:
        print(f"Reading source data from: {source_path}")
        source_df = pd.read_pickle(source_path)
        # Isolate the columns you want to add
        new_columns_data = source_df[COLUMNS_TO_ADD]
        print("Source data loaded successfully.")
    except Exception as e:
        print(f"Error reading source file {source_path}: {e}")
        return

    # --- 2. Process Each Target Directory ---
    for dir_path_str in target_dirs:
        target_dir = Path(dir_path_str)
        target_pickle_file = target_dir / PICKLE_FILENAME

        print("-" * 50)
        print(f"Processing: {target_pickle_file}")

        if not target_pickle_file.is_file():
            print(f"Warning: Pickle file not found in this directory. Skipping.")
            continue

        try:
            # Read the existing dataframe in the target directory
            target_df = pd.read_pickle(target_pickle_file)

            # Check if BOTH columns already exist. If so, skip modification.
            if all(col in target_df.columns for col in COLUMNS_TO_ADD):
                print("Columns already exist. No action taken.")
                continue

            # Add the new columns.
            # IMPORTANT: This assumes the rows in the source file and the target
            # file correspond one-to-one in the same order.
            print("Columns not found. Adding new columns...")
            for col in COLUMNS_TO_ADD:
                if col not in target_df.columns:
                    # Assign the series from the source data to the new column
                    target_df[col] = new_columns_data[col]

            # Save the modified dataframe, overwriting the old pickle file
            target_df.to_pickle(target_pickle_file)
            print(f"Successfully updated and saved file: {target_pickle_file}")

            # Also save the updated dataframe to a CSV file, overwriting it
            target_csv_file = target_dir / CSV_FILENAME
            target_df.to_csv(target_csv_file, sep='\t', index=False)
            print(f"Successfully updated and saved CSV file: {target_csv_file}")


        except Exception as e:
            print(f"An error occurred while processing {target_pickle_file}: {e}")

    print("-" * 50)
    print("Script finished.")


if __name__ == '__main__':
    # --- User Setup ---
    # 1. Define the full path to your source file
    #    (the one with Min_ and Max_ columns)
    source_file = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250610_0004544\20250610_0004569\Visit_projector_1_data.pkl"

    # 2. List all the directories you want to update
    #    Use raw strings (r"...") for Windows paths to avoid issues with backslashes
    directories_to_process = [
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250607_2240236\20250607_2240246",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250608_0303173\20250608_0303173",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250609_0028398\20250609_0028408",
        
        # Add as many directory paths as you need
    ]

    # --- Run the Function ---
    update_dataframes_in_dirs(target_dirs=directories_to_process, source_file_path=source_file)
