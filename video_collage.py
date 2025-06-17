import os
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import re

def find_mp4_in_dir(directory):
    """Find the first MP4 file in a directory."""
    mp4_files = glob(os.path.join(directory, "*.mp4"))
    return mp4_files[0] if mp4_files else None

def create_label_overlay(frame, text, font_size=16):
    """Create a label overlay with matplotlib."""
    # Configure matplotlib for LaTeX rendering
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = font_size
    plt.rcParams['font.family'] = 'serif'
    
    # Create figure with transparent background
    dpi = 100
    height, width = frame.shape[:2]
    figsize = (width/dpi, height/dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=(0, 0, 0, 0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # Add text using LaTeX formatting
    ax.text(0.05, 0.95, f"${text}$", 
            fontsize=font_size,
            color='white',
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=3))
    
    # Convert to image array - using buffer_rgba() instead of deprecated tostring_rgb()
    fig.canvas.draw()
    buffer = fig.canvas.buffer_rgba()
    overlay = np.asarray(buffer)
    # Convert RGBA to RGB
    overlay = overlay[:, :, :3]
    plt.close(fig)
    
    # Resize and apply overlay
    overlay = cv2.resize(overlay, (width, height))
    mask = np.any(overlay > 0, axis=2)
    
    frame_with_label = frame.copy()
    frame_with_label[mask] = overlay[mask]
    
    return frame_with_label

def get_unique_output_path(base_dir, base_name):
    """Generate a unique output file path."""
    i = 1
    while True:
        out_path = os.path.join(base_dir, f"{base_name}{i}.mp4")
        if not os.path.exists(out_path):
            return out_path
        i += 1

def create_video_grid_collage(
    image_dirs,
    comments=None,
    output_base_dir=None,
    output_base_name="grid_collage",
    manual_grid=None,
    n_frames=None,
    log_level=1,
    font_size=16,
    fps=5
):
    """
    Create a grid collage video from multiple video directories.
    
    Args:
        image_dirs: List of directories containing MP4 videos
        comments: List of labels for each video
        output_base_dir: Directory to save output video
        output_base_name: Base name for output video file
        manual_grid: Manual grid size as [rows, cols]
        n_frames: Number of frames to process (None for all)
        log_level: Logging verbosity level
        font_size: Font size for labels
        fps: Frames per second for output video
    
    Returns:
        str: Path to the output video file
    """
    print("=== Video Grid Collage Creator ===")
    
    # Default output directory if not specified
    if not output_base_dir:
        output_base_dir = os.path.dirname(image_dirs[0])
    
    # Default comments if not specified
    if not comments:
        comments = [f"Seq{i+1}" for i in range(len(image_dirs))]
    elif len(comments) < len(image_dirs):
        comments.extend([f"Seq{i+1}" for i in range(len(comments), len(image_dirs))])
    
    # 1. Find MP4 files and set up video captures
    print("\nLoading videos...")
    video_paths = []
    video_caps = []
    
    for i, directory in enumerate(image_dirs):
        mp4_path = find_mp4_in_dir(directory)
        if not mp4_path:
            print(f"Warning: No MP4 file found in directory {i+1}: {directory}")
            continue
        
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video file: {mp4_path}")
            continue
            
        video_paths.append(mp4_path)
        video_caps.append(cap)
        print(f"Loaded video {i+1}: {os.path.basename(mp4_path)}")
    
    # Check if we have any videos
    n_videos = len(video_caps)
    if n_videos == 0:
        print("Error: No valid videos found.")
        return None
    
    # 2. Set up grid
    if manual_grid:
        rows, cols = manual_grid
        if rows * cols < n_videos:
            print(f"Warning: Manual grid size {rows}x{cols} is too small for {n_videos} videos.")
            # Calculate a suitable grid size
            cols = int(np.ceil(np.sqrt(n_videos)))
            rows = int(np.ceil(n_videos / cols))
    else:
        # Calculate a suitable grid size
        cols = int(np.ceil(np.sqrt(n_videos)))
        rows = int(np.ceil(n_videos / cols))
    
    print(f"Using {rows}x{cols} grid for {n_videos} videos")
    
    # 3. Get video properties
    first_cap = video_caps[0]
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Determine common frame count for all videos
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in video_caps]
    common_frame_count = min(frame_counts)
    
    if n_frames is not None and n_frames > 0:
        common_frame_count = min(common_frame_count, n_frames)
    
    print(f"Processing {common_frame_count} frames from each video")
    
    # 4. Setup output video
    collage_width = width * cols
    collage_height = height * rows
    
    output_path = get_unique_output_path(output_base_dir, output_base_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (collage_width, collage_height))
    
    if not out.isOpened():
        print("Error: Failed to create output video writer.")
        for cap in video_caps:
            cap.release()
        return None
    
    # 5. Process frames
    print("\nProcessing frames...")
    
    for frame_idx in range(common_frame_count):
        #if frame_idx % 10 == 0 or frame_idx == common_frame_count - 1:
        print(f"\rProcessing frame {frame_idx+1}/{common_frame_count}", end="", flush=True)
        
        # Create empty collage frame
        collage = np.zeros((collage_height, collage_width, 3), dtype=np.uint8)
        
        # Get frames from all videos
        for i, cap in enumerate(video_caps):
            ret, frame = cap.read()
            
            if not ret:
                print(f"\nWarning: Failed to read frame {frame_idx} from video {i}. Using black frame.")
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Apply label
            if i < len(comments):
                frame = create_label_overlay(frame, comments[i], font_size)
            
            # Calculate position in grid
            r = i // cols
            c = i % cols
            y_start = r * height
            y_end = (r + 1) * height
            x_start = c * width
            x_end = (c + 1) * width
            
            # Add to collage
            try:
                collage[y_start:y_end, x_start:x_end] = frame
            except Exception as e:
                print(f"\nError adding frame to collage: {e}")
        
        # Write collage frame to output video
        out.write(collage)
    
    # 6. Cleanup
    print("\n\nFinishing up...")
    for cap in video_caps:
        cap.release()
    out.release()
    
    print(f"\nVideo collage successfully created at: {output_path}")
    print(f"Grid: {rows}x{cols}, Total frames: {common_frame_count}")
    
    return output_path

def main():
    # Configuration
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
    output_base_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars"
    
    # Create video collage
    create_video_grid_collage(
        image_dirs=image_dirs,
        comments=comments,
        output_base_dir=output_base_dir,
        output_base_name="colortable",
        manual_grid=[4, 2],  # 4 rows, 2 columns
        n_frames=None,  # Process all frames
        log_level=1,
        font_size=16,
        fps=5
    )

if __name__ == "__main__":
    main()
