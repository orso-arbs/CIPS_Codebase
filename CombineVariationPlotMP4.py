import os
import shutil

def get_individual_run_comment(run_folder_name):
    """
    Extracts the comment part from an individual variation run folder name.
    Example: "VCL_Pipeline_2025-05-11_19-53-42_PsCol_orangehot_invC_1" -> "PsCol_orangehot_invC_1"
    Assumes a prefix like "VCL_Pipeline_DATE_TIME_" or similar structure with at least 4 underscore-separated parts before the comment.
    """
    try:
        parts = run_folder_name.split('_')
        # Assumes prefix: IDENTIFIER_SCRIPTNAME_DATE_TIME_COMMENT...
        # Example: VCL_Pipeline_2025-05-11_19-53-42_PsCol_orangehot_invC_1
        # parts[0]=VCL, parts[1]=Pipeline, parts[2]=Date, parts[3]=Time
        # Comment starts from parts[4]
        if len(parts) > 4: 
            return "_".join(parts[4:])
        else: 
            # Fallback: if the name doesn't fit the expected VCL_Pipeline_DATE_TIME_Comment structure,
            # return the full folder name as the comment. This is safer than trying to guess a different structure.
            return run_folder_name 
    except Exception as e:
        print(f"  Warning: Could not robustly parse comment for '{run_folder_name}': {e}. Using folder name as comment.")
        return run_folder_name

def combine_variation_mp4s(variation_set_dir):
    """
    Combines .mp4 files from multiple 'individual variation run folders' (which are
    subdirectories of variation_set_dir) into a common folder.

    The new filename format is:
    [comment_of_individual_run_folder]_[name_of_folder_containing_mp4]_[original_filename].mp4

    Args:
        variation_set_dir (str): 
            The path to the "Variation main folder" that CONTAINS the individual 
            variation run folders.
            Example: "C:\\...\\VCL_Pipeline_variations_1_2025-05-10_20-34-05"
            This directory would contain subfolders like:
            - "VCL_Pipeline_2025-05-11_19-53-42_PsCol_orangehot_invC_1"
            - "VCL_Pipeline_2025-05-11_20-00-00_AnotherVariation_2"
    """
    if not os.path.isdir(variation_set_dir):
        print(f"Error: The 'Variation main folder' directory '{variation_set_dir}' does not exist.")
        return

    # Define the name for the folder that will store combined plots
    combined_plots_folder_name = "variation_plots_combined"
    # This folder will be created directly inside variation_set_dir
    combined_plots_path = os.path.join(variation_set_dir, combined_plots_folder_name)

    # Create the combined_plots_folder if it doesn't exist
    try:
        os.makedirs(combined_plots_path, exist_ok=True)
        if os.path.isdir(combined_plots_path) and not os.listdir(combined_plots_path): 
             print(f"Successfully created folder: '{combined_plots_path}'")
        elif os.path.isdir(combined_plots_path):
             print(f"Folder '{combined_plots_path}' already exists. Files will be added/overwritten.")
    except OSError as e:
        print(f"Error creating directory '{combined_plots_path}': {e}")
        return

    print(f"\nSearching for .mp4 files within individual run folders inside '{variation_set_dir}'...")
    mp4_copied_count = 0
    individual_run_folders_processed_count = 0

    # Iterate through items (potential individual variation run folders) in the variation_set_dir
    for item_name in os.listdir(variation_set_dir):
        current_individual_run_folder_path = os.path.join(variation_set_dir, item_name)

        # Process only if it's a directory and NOT the combined_plots_folder itself
        if os.path.isdir(current_individual_run_folder_path) and item_name != combined_plots_folder_name:
            individual_run_folder_name = item_name
            individual_run_folders_processed_count += 1
            
            print(f"\nProcessing Individual Run Folder: '{individual_run_folder_name}'")
            
            # Extract the comment for this specific individual run folder
            run_comment = get_individual_run_comment(individual_run_folder_name)
            print(f"  Using comment for naming: '{run_comment}'")

            # Walk through the directory tree of the current_individual_run_folder_path
            for dirpath, dirnames, filenames in os.walk(current_individual_run_folder_path):
                for filename in filenames:
                    if filename.endswith(".mp4"):
                        original_mp4_path = os.path.join(dirpath, filename)
                        
                        # Get the name of the immediate parent folder of the .mp4 file
                        mp4_parent_folder_name = os.path.basename(dirpath)
                        
                        # Create the new filename:
                        # [run_comment]_[mp4_parent_folder_name]_[original_filename].mp4
                        new_filename = f"{run_comment}_{mp4_parent_folder_name}_{filename}"
                        destination_mp4_path = os.path.join(combined_plots_path, new_filename)
                        
                        try:
                            # Copy the file
                            shutil.copy2(original_mp4_path, destination_mp4_path)
                            print(f"  Copied: '{original_mp4_path}' \n    TO -> '{destination_mp4_path}'")
                            mp4_copied_count += 1
                        except Exception as e:
                            print(f"  Error copying file '{original_mp4_path}': {e}")
    
    print("\n--- Summary ---")
    if individual_run_folders_processed_count == 0:
        print(f"No 'individual variation run folders' found directly inside '{variation_set_dir}' (excluding '{combined_plots_folder_name}').")
        print("Please ensure this script is pointed to a 'Variation main folder' that CONTAINS your individual run folders.")
    elif mp4_copied_count == 0:
        print(f"Processed {individual_run_folders_processed_count} individual run folder(s), but no .mp4 files were found to copy.")
    else:
        print(f"Finished copying. Total .mp4 files copied: {mp4_copied_count} from {individual_run_folders_processed_count} individual run folder(s).")
    
    print(f"All combined .mp4 files are located in: '{combined_plots_path}'")


if __name__ == '__main__':
    # --- IMPORTANT: SET YOUR "VARIATION MAIN FOLDER" PATH HERE ---
    # This directory should be a specific experiment set folder, 
    # e.g., "VCL_Pipeline_variations_1_2025-05-10_20-34-05",
    # which in turn contains the individual variation run folders like
    # "VCL_Pipeline_2025-05-11_19-53-42_PsCol_orangehot_invC_1".

    # Example for Windows:
    # This is the path to a folder like "VCL_Pipeline_variations_1_2025-05-10_20-34-05"
    variation_set_directory = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipeline_variations_1_2025-05-11_19-12-06"
    
    # Example for macOS/Linux:
    # variation_set_directory = "/path/to/your/VCL_variations/VCL_Pipeline_variations_1_2025-05-10_20-34-05"
    
    # --- VERIFY THIS PATH ---
    # Replace this with the actual path to your "Variation main folder"
    # variation_set_directory = r"YOUR_ACTUAL_VARIATION_MAIN_FOLDER_PATH_HERE" 

    if variation_set_directory == r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations\VCL_Pipeline_variations_1_2025-05-10_20-34-05" or \
       variation_set_directory == "/path/to/your/VCL_variations/VCL_Pipeline_variations_1_2025-05-10_20-34-05" or \
       variation_set_directory == r"YOUR_ACTUAL_VARIATION_MAIN_FOLDER_PATH_HERE": # Added placeholder check
        print("Reminder: Please VERIFY the 'variation_set_directory' variable in the script.")
        print("It should point to a 'Variation main folder' (e.g., ...VCL_Pipeline_variations_1_2025-05-10_20-34-05),")
        print("which CONTAINS your individual variation run folders (e.g., ...PsCol_orangehot_invC_1).")
        print(f"Currently set to: {variation_set_directory}\n")

    # Call the function with the specified directory
    combine_variation_mp4s(variation_set_directory)

    # --- Example of how to use the function with a test directory structure ---
    # test_variation_set_dir_name = "MyExperimentSet_Alpha_MainRun_2025-01-15"
    # test_variation_set_dir_path = os.path.join(os.getcwd(), test_variation_set_dir_name)

    # if not os.path.exists(test_variation_set_dir_path):
    #     os.makedirs(test_variation_set_dir_path, exist_ok=True)
    #     print(f"\n--- Creating a test directory structure at: {test_variation_set_dir_path} ---")

    #     # Create dummy individual run folders inside the test_variation_set_dir_path
    #     run_folder1_name = "VCL_Pipeline_2025-01-15_10-00-00_ParamSet1_HighContrast"
    #     run_folder2_name = "VCL_Pipeline_2025-01-15_11-00-00_ParamSet2_LowContrast_Final"
    #     run_folder3_name = "Simple_Run_NoStandardPrefix" # Test fallback comment

    #     path_run1 = os.path.join(test_variation_set_dir_path, run_folder1_name)
    #     os.makedirs(os.path.join(path_run1, "outputs", "videos"), exist_ok=True)
    #     with open(os.path.join(path_run1, "outputs", "videos", "animation1.mp4"), "w") as f: f.write("dummy_content_r1_v1")
    #     with open(os.path.join(path_run1, "raw_data", "clip.mp4"), "w") as f: f.write("dummy_content_r1_v2")
        
    #     path_run2 = os.path.join(test_variation_set_dir_path, run_folder2_name)
    #     os.makedirs(os.path.join(path_run2, "results_final"), exist_ok=True) # Different mp4 parent folder name
    #     with open(os.path.join(path_run2, "results_final", "final_cut.mp4"), "w") as f: f.write("dummy_content_r2_v1")

    #     path_run3 = os.path.join(test_variation_set_dir_path, run_folder3_name)
    #     os.makedirs(os.path.join(path_run3, "media"), exist_ok=True)
    #     with open(os.path.join(path_run3, "media", "video.mp4"), "w") as f: f.write("dummy_content_r3_v1")
        
    #     print("\n--- Running with the test directory structure ---")
    #     combine_variation_mp4s(test_variation_set_dir_path)
        
    #     # To clean up the test directory:
    #     # print(f"\nTo clean up, manually delete the folder: {os.path.abspath(test_variation_set_dir_path)}")
    # else:
    #     print(f"\nSkipping test directory creation as '{os.path.abspath(test_variation_set_dir_path)}' already exists.")
    #     print(f"You can run: combine_variation_mp4s('{os.path.abspath(test_variation_set_dir_path)}') to test it.")
