import cv2
import os
import time
import datetime
import glob
import re

import sys
import os
sys.path.append(os.path.abspath(r"C:/Users/obs/OneDrive/ETH/ETH_MSc/Masters Thesis/Python Code/Python_Orso_Utility_Scripts_MscThesis")) # dir containing Format_1 
import Format_1 as F_1




def create_video_from_images(plot_image_folder, video_output_dir, fps=5):

    output_video = os.path.join(video_output_dir, f"FB_segmented_growth_statistics_{current_date}.mp4")

    # Get all PNG files in the directory
    images = glob.glob(os.path.join(plot_image_folder, "*.png"))

    if not images:
        print("No PNG images found in the directory.")
        return

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