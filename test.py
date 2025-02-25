import os
import time

# Example values (replace with actual ones in your script)
img_files = ["image1.png", "image2.png", "image3.png"]
all_images = img_files

for i in range(len(all_images)):
    # Clear the current line
    print("\r", end='', flush=True)  # Move cursor to beginning and erase previous content
    print(f"Plotting data for image name: {os.path.basename(img_files[i])} \t {i+1}/{len(all_images)}", end='', flush=True)
    time.sleep(3)  # Simulate some processing time

print("\nDone!")  # Move to a new line after completion
