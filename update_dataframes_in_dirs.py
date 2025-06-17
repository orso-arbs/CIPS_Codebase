import pandas as pd
from pathlib import Path

def update_dataframes_in_dirs(target_dirs: list, source_file_path: str):
    """
    Updates pickle and CSV files in a list of directories. It adds or
    overwrites specified columns by matching rows with a source file based
    on a key column.

    Args:
        target_dirs (list): A list of strings, where each string is a path to a
                            directory to be processed.
        source_file_path (str): The full path to the .pkl file that contains
                                the columns to be added.
    """
    # --- Configuration ---
    PICKLE_FILENAME = "Visit_projector_1_data.pkl"
    CSV_FILENAME = "Visit_projector_1_data.csv"
    KEY_COLUMN = "Image_filename_VisIt"
    COLUMNS_TO_ADD = [
        "Min_Psuedocolored_variable_SF_VisIt",
        "Max_Psuedocolored_variable_SF_VisIt"
    ]

    # --- 1. Load the Source Data and Validate ---
    source_path = Path(source_file_path)
    if not source_path.is_file():
        print(f"Error: Source file not found at {source_path}")
        return

    try:
        print(f"Reading source data from: {source_path}")
        source_df = pd.read_pickle(source_path)
        # Validate that the necessary columns exist in the source file
        if KEY_COLUMN not in source_df.columns:
            print(f"Error: Key column '{KEY_COLUMN}' not found in source file. Aborting.")
            return
        if not all(col in source_df.columns for col in COLUMNS_TO_ADD):
            print(f"Error: Not all required data columns {COLUMNS_TO_ADD} found in source file. Aborting.")
            return
        print("Source data loaded and validated successfully.")
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

            # Check if the key column exists in the target file
            if KEY_COLUMN not in target_df.columns:
                print(f"Warning: Key column '{KEY_COLUMN}' not found in target file. Skipping.")
                continue

            # If all columns already exist, we assume it's up-to-date.
            if all(col in target_df.columns for col in COLUMNS_TO_ADD):
                print("Columns already exist. No action taken.")
                continue

            print("Columns missing or incomplete. Updating by matching rows...")

            # To avoid creating duplicate columns (e.g., 'col_x', 'col_y') during the merge,
            # we first identify which of the columns-to-add already exist in the target...
            existing_cols_to_overwrite = [col for col in COLUMNS_TO_ADD if col in target_df.columns]
            if existing_cols_to_overwrite:
                print(f"Will overwrite existing columns: {existing_cols_to_overwrite}")
                # ...and drop them. The merge will add them back with the new values from source.
                target_df = target_df.drop(columns=existing_cols_to_overwrite)

            # Prepare the data from the source file, containing only the key and the columns to add.
            source_subset = source_df[[KEY_COLUMN] + COLUMNS_TO_ADD]

            # Perform a left merge. This keeps all rows from the target dataframe
            # and adds data from the source dataframe where the key column matches.
            updated_df = pd.merge(
                target_df,
                source_subset,
                on=KEY_COLUMN,
                how="left"
            )

            # Save the modified dataframe, overwriting the old pickle file
            updated_df.to_pickle(target_pickle_file)
            print(f"Successfully updated and saved file: {target_pickle_file}")

            # Also save the updated dataframe to a CSV file, overwriting it
            target_csv_file = target_dir / CSV_FILENAME
            updated_df.to_csv(target_csv_file, sep='\t', index=False)
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
        r"C:\Users\obs\Desktop\test_col\20250604_1312276\20250604_1312276",
    ]

    # --- Run the Function ---
    update_dataframes_in_dirs(target_dirs=directories_to_process, source_file_path=source_file)
