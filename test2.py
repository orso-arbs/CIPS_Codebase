import os
import re

# Directory containing the images
source_dir = r"C:\Users\obs\Desktop\FrouzakisSF_Images"

# Pattern to match the current file names
pattern = r"T3\.0_(\d{4})\.png"

# Get all PNG files in the directory
files = [f for f in os.listdir(source_dir) if f.endswith('.png')]

# Counter for renamed files
renamed_count = 0

for filename in files:
    # Check if the filename matches our expected pattern
    match = re.match(pattern, filename)
    if match:
        # Extract the number part
        number = match.group(1)
        
        # Create new filename with 6-digit padding
        new_number = number.zfill(6)
        new_filename = f"visit_{new_number}.png"
        
        # Full paths for renaming
        old_path = os.path.join(source_dir, filename)
        new_path = os.path.join(source_dir, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)
        renamed_count += 1
        print(f"Renamed: {filename} → {new_filename}")

print(f"\nRenamed {renamed_count} files out of {len(files)} PNG files found.")
