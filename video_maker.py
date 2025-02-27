import cv2
import os
import time
import datetime
import glob
import re

### start inform
start_time = time.time()
current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
print(f"\n {os.path.basename(__file__)}: ", datetime.datetime.now(), "\n")

# Directory containing PNG images
plot_image_folder = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Cellpose\Cellpose1\CP_Results_2025-02-25_21-47\plots_2025-02-25_22-52"
output_video = os.path.join(plot_image_folder, f"FB_segmented_growth_statistics_{current_date}.mp4")

fps = 5  # Frames per second

# Get all PNG files in the directory
images = glob.glob(os.path.join(plot_image_folder, "*.png"))

if not images:
    print("No PNG images found in the directory.")
    exit()

# Rename images to have 6 digits in the filename
for image in images:
    dirname, basename = os.path.split(image)
    name, ext = os.path.splitext(basename)
    match = re.search(r'(\d+)', name)
    if match:
        number = int(match.group(1))
        new_name = f"{name[:match.start()]}{number:06d}{ext}"
        new_image_path = os.path.join(dirname, new_name)
        os.rename(image, new_image_path)

# Get all renamed PNG files in the directory
images = glob.glob(os.path.join(plot_image_folder, "*.png"))

# Sort images numerically based on the number in the filename
def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

images = sorted(images, key=extract_number)

# Read the first image to get dimensions
frame = cv2.imread(images[0])
height, width, layers = frame.shape

# Define the video codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use "XVID" for .avi
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Add images to the video
for image in images:
    frame = cv2.imread(image)
    video.write(frame)

# Release the video writer
video.release()
print(f"Video saved as {output_video}")

### end inform
end_time = time.time()
elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)
print(f"\n Code Completely Executed in {int(minutes)} min {seconds:.2f} sec \n")