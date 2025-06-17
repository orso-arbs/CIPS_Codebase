import os
import shutil
import glob
from datetime import datetime


def move_mp4_files(image_dirs, moved_dir, create_subdirs=False, copy_instead=True):
    """
    Moves (or copies) MP4 files from a list of source directories to a destination directory.
    
    Parameters:
    -----------
    image_dirs : list of str
        List of source directories containing MP4 files to move
    moved_dir : str
        Destination directory where MP4 files will be moved to
    create_subdirs : bool, optional
        If True, creates a separate subdirectory for each source directory.
        If False, moves all MP4 files directly to moved_dir.
        Default is True.
    copy_instead : bool, optional
        If True, copies files instead of moving them.
        Default is False.
    
    Returns:
    --------
    dict
        Dictionary with stats about the operation:
        - 'total_files': Number of MP4 files found
        - 'moved_files': Number of MP4 files successfully moved
        - 'source_dirs_with_mp4': Number of source directories that had MP4 files
        - 'errors': List of errors encountered
        - 'moved_files_list': List of paths to successfully moved files
    """
    # Ensure the destination directory exists
    os.makedirs(moved_dir, exist_ok=True)
    
    stats = {
        'total_files': 0,
        'moved_files': 0,
        'source_dirs_with_mp4': 0,
        'errors': [],
        'moved_files_list': []
    }
    
    # Process each source directory
    for i, image_dir in enumerate(image_dirs):
        # Get all MP4 files in this directory
        mp4_files = glob.glob(os.path.join(image_dir, "*.mp4"))
        stats['total_files'] += len(mp4_files)
        
        if not mp4_files:
            print(f"No MP4 files found in: {image_dir}")
            continue
        
        stats['source_dirs_with_mp4'] += 1
        
        # Create a subdirectory if needed
        if create_subdirs:
            # Use the original folder name as subdirectory name
            subdir_name = os.path.basename(image_dir)
            
            # If there are multiple directories with the same name, add a timestamp
            target_dir = os.path.join(moved_dir, subdir_name)
            if os.path.exists(target_dir):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                subdir_name = f"{subdir_name}_{timestamp}"
                target_dir = os.path.join(moved_dir, subdir_name)
            
            os.makedirs(target_dir, exist_ok=True)
        else:
            target_dir = moved_dir
        
        # Move/Copy each MP4 file
        for mp4_file in mp4_files:
            try:
                filename = os.path.basename(mp4_file)
                destination = os.path.join(target_dir, filename)
                
                # Handle existing files at destination
                if os.path.exists(destination):
                    base, ext = os.path.splitext(filename)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_filename = f"{base}_{timestamp}{ext}"
                    destination = os.path.join(target_dir, new_filename)
                
                if copy_instead:
                    shutil.copy2(mp4_file, destination)
                    print(f"Copied: {mp4_file} → {destination}")
                else:
                    shutil.move(mp4_file, destination)
                    print(f"Moved: {mp4_file} → {destination}")
                
                stats['moved_files'] += 1
                stats['moved_files_list'].append(destination)
            
            except Exception as e:
                error_msg = f"Error processing {mp4_file}: {str(e)}"
                print(f"ERROR: {error_msg}")
                stats['errors'].append(error_msg)
    
    # Print summary
    print("\n--- Summary ---")
    print(f"Found {stats['total_files']} MP4 files in {stats['source_dirs_with_mp4']} directories")
    print(f"Successfully {'copied' if copy_instead else 'moved'} {stats['moved_files']} files")
    if stats['errors']:
        print(f"Encountered {len(stats['errors'])} errors")
    
    return stats


if __name__ == "__main__":
    # Example usage
    image_dirs = [
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250607_2240236\20250607_2240246\20250612_1429370\20250614_1949188\20250614_2132115",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250608_0303173\20250608_0303173\20250612_1638247\20250614_2137355\20250614_2250588",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250609_0028398\20250609_0028408\20250612_1843092\20250614_2257342\20250615_0044212",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250610_0004544\20250610_0004569\20250612_2023463\20250612_2228583\20250612_2318262",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250610_0529590\20250610_0529590\20250615_1239319\20250615_1440242\20250615_1602122",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250610_0646347\20250610_0646347\20250615_1609401\20250615_1727535\20250616_1233427",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250610_0803025\20250610_0803025\20250615_1734526\20250615_1952036\20250615_2108331",
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250610_0916439\20250610_0916439\20250615_2115060\20250615_2243023\20250616_0012372",
    ]
    
    moved_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\colortables_mp4"
    
    # Move the MP4 files
    stats = move_mp4_files(
        image_dirs=image_dirs,
        moved_dir=moved_dir,
        create_subdirs=False,  # Create a subfolder for each source directory
        copy_instead=True     # Copy instead of move (safer for testing)
    )
    
    print(f"\nFiles were copied to: {moved_dir}")
