import cv2
import os
import glob

# Directory containing PNG images
image_folder = "C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Cellpose\Cellpose1\CP_Results_2025-02-24_21-27\plots_2025-02-25_21-41"  # Change this to your directory path
output_video = f"FB_segmented_growth_statistics_{current_date}.mp4"
fps = 1  # Frames per second

# Get all PNG files in the directory and sort them by name
images = sorted(glob.glob(os.path.join(image_folder, "*.png")))

if not images:
    print("No PNG images found in the directory.")
    exit()

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