import os
import argparse
import sys

def find_nested_directories(base_dir):
    """
    Scans a base directory to find and print specific subdirectories.

    This function walks through a directory structure with the following logic:
    1. It looks at the first-level subdirectories within `base_dir`.
    2. It processes only those directories whose names are in the format
       'part1_part2' (e.g., '20250607_2240236'). It specifically excludes
       directories with more complex names.
    3. For each matching first-level directory, it then lists its own
       subdirectories (the second-level directories) that ALSO match the
       'part1_part2' naming convention.
    4. The full path of each matching second-level directory is printed.

    Args:
        base_dir (str): The absolute or relative path to the directory to scan.
    """
    # --- 1. Validate the base directory ---
    # Check if the provided path exists and is actually a directory.
    if not os.path.isdir(base_dir):
        print(f"Error: The provided path '{base_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(1) # Exit the script with an error code

    print(f"Scanning base directory: {os.path.abspath(base_dir)}\n")

    # --- 2. Iterate through first-level directories ---
    try:
        # Use os.scandir for better performance than os.listdir.
        for entry_level1 in os.scandir(base_dir):
            if entry_level1.is_dir():
                # --- 3. Apply the name filtering logic to the first level ---
                name_parts_level1 = entry_level1.name.split('_')
                
                # We only proceed if the name has exactly two parts.
                if len(name_parts_level1) == 2:
                    
                    # --- 4. Iterate through second-level directories ---
                    path_level1 = entry_level1.path
                    
                    try:
                        for entry_level2 in os.scandir(path_level1):
                            # Check if the second-level entry is a directory.
                            if entry_level2.is_dir():
                                # --- 5. Apply the same filtering logic to the second level ---
                                name_parts_level2 = entry_level2.name.split('_')
                                if len(name_parts_level2) == 2:
                                    print(entry_level2.path)
                    except OSError as e:
                        # Handle potential permission errors for the second-level scan.
                        print(f"Warning: Could not scan directory '{path_level1}'. Reason: {e}", file=sys.stderr)

    except OSError as e:
        # Handle potential permission errors for the base directory scan.
        print(f"Error: Could not scan base directory '{base_dir}'. Reason: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':

    base_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars"
    find_nested_directories(base_dir)
