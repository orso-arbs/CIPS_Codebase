import os

def get_folders_two_levels_deep(base_path):
    """
    Finds all folder paths exactly two levels deep from a base path.

    Args:
        base_path (str): The starting directory path.

    Returns:
        list: A list of absolute paths to folders two levels deep.
    """
    level2_folders = []
    if not os.path.isdir(base_path):
        print(f"Error: Base path '{base_path}' not found or is not a directory.")
        return level2_folders

    # Normalize base_path to remove trailing slash for consistent depth calculation
    # and count path separators to determine base depth.
    norm_base_path = base_path.rstrip(os.sep)
    base_path_sep_count = norm_base_path.count(os.sep)

    for root, dirs, _ in os.walk(base_path, topdown=True):
        # Calculate the depth of the current 'root' relative to 'base_path'
        current_root_sep_count = root.rstrip(os.sep).count(os.sep)
        current_root_relative_depth = current_root_sep_count - base_path_sep_count

        # We are interested in the children ('dirs') of folders that are 1 level deep.
        # These children will be 2 levels deep relative to base_path.
        if current_root_relative_depth == 1:
            for dir_name in dirs:
                level2_folders.append(os.path.join(root, dir_name))
            
            # Optimization: We've found the level 2 folders from this 'root' (a level 1 folder).
            # Tell os.walk not to go deeper into these level 2 folders from this point,
            # as we only care about folders *exactly* at level 2.
            dirs[:] = [] 
        elif current_root_relative_depth >= 2:
            # If 'root' itself is already at level 2 or deeper,
            # we don't need to explore its children for this task.
            dirs[:] = []
            
    return level2_folders

if __name__ == "__main__":
    # --- Set your base directory path here ---
    base_directory = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations"

    print(f"Searching for folders 2 levels deep in: '{base_directory}'\n")
    
    folders_found = get_folders_two_levels_deep(base_directory)

    if folders_found:
        print("Folders found at two levels deep:")
        for folder_path in folders_found:
            print(folder_path)
    else:
        print(f"No folders found two levels deep from '{base_directory}'.")
