import os
import shutil

def move_nested_folders_to_base(base_directory_path):
    """
    Recursively moves all nested folders inside a base directory into the
    base directory itself. Handles name conflicts by renaming.

    Args:
        base_directory_path (str): The absolute path to the base directory.
    """
    if not os.path.isdir(base_directory_path):
        print(f"Error: Base directory '{base_directory_path}' not found.")
        return

    moved_something = False
    folders_to_move = []

    # First pass: Collect all nested directories to move
    # We iterate from deeper levels upwards to avoid issues with changing directory structures
    for root, dirs, files in os.walk(base_directory_path, topdown=False):
        # Skip the base directory itself and its immediate children during collection
        # as we only want to move *nested* directories.
        # A directory is nested if its parent is not the base_directory_path
        # or if it's deeper than an immediate child.
        # The `root` path will be the parent of the `dirs` found in the current iteration.
        if root == base_directory_path:
            continue # Don't process immediate children of the base in this loop iteration

        for dir_name in dirs:
            source_path = os.path.join(root, dir_name)
            # Ensure the source_path is not an immediate child of base_directory_path
            # This check is a bit more explicit
            if os.path.dirname(source_path) != base_directory_path:
                 folders_to_move.append(source_path)


    # Second pass: Move collected directories
    for source_path in folders_to_move:
        dir_name = os.path.basename(source_path)
        destination_path = os.path.join(base_directory_path, dir_name)

        # Handle potential name conflicts
        counter = 1
        original_destination_path = destination_path
        while os.path.exists(destination_path):
            print(f"Warning: Directory '{os.path.basename(original_destination_path)}' already exists in base. Trying to rename...")
            destination_path = os.path.join(base_directory_path, f"{dir_name}_{counter}")
            counter += 1
            if counter > 100: # Safety break to prevent infinite loop in extreme cases
                print(f"Error: Too many conflicts for directory '{dir_name}'. Skipping '{source_path}'.")
                destination_path = None # Mark as not to move
                break
        
        if destination_path is None:
            continue

        try:
            print(f"Moving '{source_path}' to '{destination_path}'")
            shutil.move(source_path, destination_path)
            moved_something = True
        except Exception as e:
            print(f"Error moving '{source_path}' to '{destination_path}': {e}")

    if not moved_something and not folders_to_move: # If no folders were identified to move
        print("No nested folders found to move.")


def cleanup_empty_dirs(directory_path):
    """
    Removes all empty subdirectories recursively.
    """
    removed_any = False
    for root, dirs, files in os.walk(directory_path, topdown=False): # topdown=False is crucial here
        for dirname in dirs:
            dirpath = os.path.join(root, dirname)
            # Check if directory is empty
            if not os.listdir(dirpath):
                try:
                    os.rmdir(dirpath)
                    print(f"Removed empty directory: '{dirpath}'")
                    removed_any = True
                except OSError as e:
                    print(f"Error removing directory '{dirpath}': {e}")
    if not removed_any:
        print("No empty directories found to clean up (or none were left after moving).")


if __name__ == "__main__":
    # --- IMPORTANT: SET YOUR LIST OF BASE DIRECTORY PATHS HERE ---
    base_dirs_list = [
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-11_19-12-06_Colortable_Vmag\variation_plots_combined",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-11_19-12-06_Colortable_Vmag\VCL_Pipeline_2025-05-11_19-12-06_PsCol_CustomBW1",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-11_19-12-06_Colortable_Vmag\VCL_Pipeline_2025-05-11_19-53-42_PsCol_orangehot_inverted",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-11_19-12-06_Colortable_Vmag\VCL_Pipeline_2025-05-11_20-24-15_PsCol_Purples",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-11_19-12-06_Colortable_Vmag\VCL_Pipeline_2025-05-11_20-55-02_PsCol_hot_and_cold",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-11_19-12-06_Colortable_Vmag\VCL_Pipeline_2025-05-11_21-28-31_PsCol_BrBG",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-11_19-12-06_Colortable_Vmag\VCL_Pipeline_2025-05-11_21-58-19_PsCol_difference",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-11_19-12-06_Colortable_Vmag\VCL_Pipeline_2025-05-11_22-30-27_PsCol_plasma",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-11_19-12-06_Colortable_Vmag\VCL_Pipeline_2025-05-11_23-03-05_PsCol_turbo",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-11_19-12-06_Colortable_Vmag\VCL_Pipeline_2025-05-11_23-33-14_PsCol_hot_invC_default",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-11_19-12-06_Colortable_Vmag\VCL_Pipeline_2025-05-12_00-11-19_PsCol_Accent_invC_defaul",   
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-16_13-40-24_Colortable_HRR\variation_plots_combined",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-16_13-40-24_Colortable_HRR\VCL_Pipe_1_2025-05-16_13-40-24_CustomBW",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-16_13-40-24_Colortable_HRR\VCL_Pipe_1_2025-05-16_14-56-38_orangehot_invC_1",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-16_13-40-24_Colortable_HRR\VCL_Pipe_1_2025-05-16_15-36-39_Purples",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-16_13-40-24_Colortable_HRR\VCL_Pipe_1_2025-05-16_16-27-05_hot_and_cold",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-16_13-40-24_Colortable_HRR\VCL_Pipe_1_2025-05-16_17-07-22_BrBG",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-16_13-40-24_Colortable_HRR\VCL_Pipe_1_2025-05-16_17-48-50_difference",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-16_13-40-24_Colortable_HRR\VCL_Pipe_1_2025-05-16_18-26-57_plasma",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-16_13-40-24_Colortable_HRR\VCL_Pipe_1_2025-05-16_20-25-21_Accent",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-16_23-20-42_Colortable_OH\variation_plots_combined",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-16_23-20-42_Colortable_OH\VCL_Pipe_1_2025-05-16_23-20-42_CustomBW",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-16_23-20-42_Colortable_OH\VCL_Pipe_1_2025-05-17_00-35-11_orangehot_invC_1",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-16_23-20-42_Colortable_OH\VCL_Pipe_1_2025-05-17_01-09-52_Purples",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-16_23-20-42_Colortable_OH\VCL_Pipe_1_2025-05-17_01-47-06_hot_and_cold",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-16_23-20-42_Colortable_OH\VCL_Pipe_1_2025-05-17_02-26-26_BrBG",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-16_23-20-42_Colortable_OH\VCL_Pipe_1_2025-05-17_03-04-23_difference",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-16_23-20-42_Colortable_OH\VCL_Pipe_1_2025-05-17_03-40-22_plasma",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-16_23-20-42_Colortable_OH\VCL_Pipe_1_2025-05-17_04-20-55_turbo",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-16_23-20-42_Colortable_OH\VCL_Pipe_1_2025-05-17_04-57-07_hot",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-16_23-20-42_Colortable_OH\VCL_Pipe_1_2025-05-17_05-33-49_Accent",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-17_13-32-40_Colortable_PeriodicBW_vars\variation_plots_combined",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-17_13-32-40_Colortable_PeriodicBW_vars\VCL_Pipe_1_2025-05-17_13-32-40_Periods1_ww1bb1",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-17_13-32-40_Colortable_PeriodicBW_vars\VCL_Pipe_1_2025-05-17_15-01-19_Periods1_ww2bb2",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-17_13-32-40_Colortable_PeriodicBW_vars\VCL_Pipe_1_2025-05-17_15-48-53_Periods1_ww3bb3",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-17_13-32-40_Colortable_PeriodicBW_vars\VCL_Pipe_1_2025-05-17_16-37-45_Periods2_ww1bb1",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-17_13-32-40_Colortable_PeriodicBW_vars\VCL_Pipe_1_2025-05-17_17-28-33_Periods2_ww2bb2",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-17_13-32-40_Colortable_PeriodicBW_vars\VCL_Pipe_1_2025-05-17_18-35-21_Periods2_ww3bb3",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-17_13-32-40_Colortable_PeriodicBW_vars\VCL_Pipe_1_2025-05-17_19-24-21_Periods5_ww1bb1",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-17_13-32-40_Colortable_PeriodicBW_vars\VCL_Pipe_1_2025-05-17_20-11-04_Periods5_ww2bb2",
        # r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipe_vars_1_2025-05-17_13-32-40_Colortable_PeriodicBW_vars\VCL_Pipe_1_2025-05-17_21-00-49_Periods5_ww3bb3",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\MASTER~2\VCL_VA~1\VCL_Pipe_vars_1_2025-05-11_19-12-06_Colortable_Vmag\VCL_Pipeline_2025-05-12_00-11-19_PsCol_Accent_invC_default"
    ]

    if not base_dirs_list:
        print("The 'base_dirs_list' is empty. Please add directory paths to process.")
    else:
        for base_dir_item in base_dirs_list:
            print(f"\n\n--- Processing base directory: {base_dir_item} ---")
            if not os.path.exists(base_dir_item) or not os.path.isdir(base_dir_item):
                print(f"SKIPPING: The specified base directory '{base_dir_item}' does not exist or is not a directory.")
                continue

            move_nested_folders_to_base(base_dir_item)

            print(f"\n--- Cleaning up empty directories in: {base_dir_item} ---")
            #cleanup_empty_dirs(base_dir_item) # Pass True for initial_call

            print(f"\n--- Operation Complete for: {base_dir_item} ---")

        print("\n\n--- All Specified Directories Processed ---")