import cv2
import os
import time
import datetime
import glob
import re
import subprocess
import platform

import sys
import os
sys.path.append(os.path.abspath(r"C:/Users/obs/OneDrive/ETH/ETH_MSc/Masters Thesis/Python Code/Python_Orso_Utility_Scripts_MscThesis")) # dir containing Format_1 
import Format_1 as F_1




def create_video_from_images(plot_image_folders, video_output_dirs=None, fps=5, output_dir_comments=None, n_images=None):
    """
    Create videos from images in one or multiple folders.
    
    Args:
        plot_image_folders: str or list of str, paths to folders containing images
        video_output_dirs: str or list of str or None, paths for video output
        fps: int, frames per second
        output_dir_comments: str or list of str or None, comments for video filenames
        n_images: int or None, number of images to use (None for all images)
    """
    # Convert inputs to lists if they're not already
    if isinstance(plot_image_folders, str):
        plot_image_folders = [plot_image_folders]
    
    # Default to same directory as images if output_dirs not specified
    if video_output_dirs is None:
        video_output_dirs = plot_image_folders
    elif isinstance(video_output_dirs, str):
        video_output_dirs = [video_output_dirs] * len(plot_image_folders)
        
    # Handle comments
    if output_dir_comments is None:
        output_dir_comments = [""] * len(plot_image_folders)
    elif isinstance(output_dir_comments, str):
        output_dir_comments = [output_dir_comments] * len(plot_image_folders)
        
    # Ensure all lists have the same length
    if not (len(plot_image_folders) == len(video_output_dirs) == len(output_dir_comments)):
        raise ValueError("Number of input folders, output directories, and comments must match")
    
    # Process each folder
    for img_dir, vid_dir, comment in zip(plot_image_folders, video_output_dirs, output_dir_comments):
        print(f"Processing folder: {img_dir}")
        
        output_video = os.path.join(vid_dir, f"FB_segmented_growth_statistics_{comment}.mp4")
        
        # Get all PNG files in the directory
        images = glob.glob(os.path.join(img_dir, "*.png"))
        
        if not images:
            print(f"No PNG images found in directory: {img_dir}")
            continue
            
        # Sort images numerically
        images = sorted(images, key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else 0)
        
        # Limit number of images if specified
        if n_images is not None:
            print(f"Using first {n_images} images out of {len(images)} available")
            images = images[:n_images]
        
        # Read first image to get dimensions
        frame = cv2.imread(images[0])
        height, width, layers = frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        # Add images to video
        for image in images:
            frame = cv2.imread(image)
            video.write(frame)
        
        video.release()
        print(f"Video saved to: {output_video}")
        
    return None

# Example usage when script is run directly
if __name__ == "__main__":
    # Example directories containing PNG sequences
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
    
    comments = ["BW", "BBWW", "WBW", "WWBBWW", "BWB", "BBWWBB", "WB", "WWBB"]
    
    # Example 1: Test with just 3 images
    create_video_from_images(
        plot_image_folders=image_dirs,
        video_output_dirs=image_dirs,
        fps=5,
        output_dir_comments=comments,
        n_images=None  # Only use first n_images images for testing. None for all images
    )