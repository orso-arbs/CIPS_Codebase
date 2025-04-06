import os
import cv2
import numpy as np
import glob
import re

# Base folder where all subfolders reside and where output videos will be saved.
base_folder = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop"

# List of folders that contain a single .mp4 video
video_dirs = [
    os.path.join(base_folder, r"CP_segment_1_2025-04-05_19-24-52_default\CP_extract_1_2025-04-05_19-37-12\CP_plotter_1_2025-04-05_19-38-02_default"),
    os.path.join(base_folder, r"CP_segment_1_2025-04-05_19-58-36_resample_False\CP_extract_1_2025-04-05_20-07-39\CP_plotter_1_2025-04-05_20-08-04_resample_False"),
    os.path.join(base_folder, r"CP_segment_1_2025-04-05_20-16-34_flow_threshold_0p2\CP_extract_1_2025-04-05_20-22-39\CP_plotter_1_2025-04-05_20-23-05_flow_threshold_0p2"),
    os.path.join(base_folder, r"CP_segment_1_2025-04-05_20-27-34_flow_threshold_0p6\CP_extract_1_2025-04-05_20-33-54\CP_plotter_1_2025-04-05_20-34-21_flow_threshold_0p6"),
    os.path.join(base_folder, r"CP_segment_1_2025-04-05_20-39-07_cellprob_threshold_0p5\CP_extract_1_2025-04-05_20-45-59\CP_plotter_1_2025-04-05_20-46-28_cellprob_threshold_0p5"),
    os.path.join(base_folder, r"CP_segment_1_2025-04-05_20-55-26_cellprob_threshold_0p8\CP_extract_1_2025-04-05_21-02-00\CP_plotter_1_2025-04-05_21-02-23_cellprob_threshold_0p8"),
    os.path.join(base_folder, r"CP_segment_1_2025-04-05_21-08-51_cellprob_threshold_1p0\CP_extract_1_2025-04-05_21-14-32\CP_plotter_1_2025-04-05_21-14-52_cellprob_threshold_1p0"),
    os.path.join(base_folder, r"CP_segment_1_2025-04-05_21-19-13_niter_1000"),
    os.path.join(base_folder, r"CP_segment_1_2025-04-05_21-29-40_niter_5000\CP_extract_1_2025-04-05_21-39-47\CP_plotter_1_2025-04-05_21-40-16_niter_5000")
]

def find_mp4_in_dir(directory):
    """Return the full path to the first .mp4 file found in the directory."""
    files = glob.glob(os.path.join(directory, "*.mp4"))
    return files[0] if files else None

# Gather video file paths from the directories
video_files = []
for d in video_dirs:
    mp4_file = find_mp4_in_dir(d)
    if mp4_file:
        video_files.append(mp4_file)
    else:
        print(f"No mp4 file found in {d}")

if not video_files:
    raise ValueError("No video files found.")

# Separate the default video (contains "default" in filename) from variant videos.
default_video_path = None
variant_video_paths = []
for vf in video_files:
    if "default" in os.path.basename(vf).lower():
        default_video_path = vf
    else:
        variant_video_paths.append(vf)

if default_video_path is None:
    raise ValueError("No default video (with 'default' in its name) found.")

print("Default video:", default_video_path)
print("Variant videos:", variant_video_paths)

def extract_comment_from_folder(folder_path):
    """
    Extract the comment part from the folder name using a regex.
    Assumes the folder name pattern is like:
    CP_segment_1_2025-04-05_19-24-52_default and we extract everything after the date.
    """
    folder_name = os.path.basename(folder_path)
    # Regex breakdown:
    # - \d{4}-\d{2}-\d{2} : date in YYYY-MM-DD
    # - _\d{2}-\d{2}-\d{2} : time in HH-MM-SS
    # - _(.+)$ : captures everything after the date-time as the comment.
    pattern = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(.*)"
    match = re.search(pattern, folder_name)
    if match:
        comment = match.group(1)
        print(f"Extracted comment from {folder_name}: {comment}")
        return comment
    else:
        print(f"Failed to extract comment from {folder_name}")
        return "unknown"

def get_unique_filename(base_path, base_name, ext=".mp4"):
    """
    Generate a unique filename in base_path with the given base_name.
    If base_name + ext exists, append a counter (e.g., base_name2, base_name3, …)
    """
    candidate = os.path.join(base_path, base_name + ext)
    counter = 2
    while os.path.exists(candidate):
        candidate = os.path.join(base_path, f"{base_name}{counter}{ext}")
        counter += 1
    return candidate

# Open the default video and read its properties.
default_cap = cv2.VideoCapture(default_video_path)
if not default_cap.isOpened():
    raise IOError("Could not open default video.")

fps = default_cap.get(cv2.CAP_PROP_FPS)
frame_count = int(default_cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(default_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(default_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load all default video frames into memory.
default_frames = []
for _ in range(frame_count):
    ret, frame = default_cap.read()
    if not ret:
        break
    default_frames.append(frame)
default_cap.release()

# Process each variant video: create a collage (default video on top, variant below).
for variant_path in variant_video_paths:
    variant_folder = os.path.dirname(variant_path)
    comment = extract_comment_from_folder(variant_folder)
    output_basename = f"CP_parameter_variation_{comment}"
    output_file = get_unique_filename(base_folder, output_basename)

    variant_cap = cv2.VideoCapture(variant_path)
    if not variant_cap.isOpened():
        print(f"Could not open variant video: {variant_path}")
        continue

    # Ensure the variant video has the same frame count.
    variant_frame_count = int(variant_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if variant_frame_count != frame_count:
        print(f"Frame count mismatch for {variant_path}. Skipping...")
        variant_cap.release()
        continue

    # Collage frame dimensions: same width, but height is the sum of both videos.
    collage_width = width
    collage_height = height * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (collage_width, collage_height))

    print(f"Processing collage for {variant_folder} --> {output_file}")

    frame_index = 0
    while True:
        ret_var, frame_var = variant_cap.read()
        if not ret_var or frame_index >= len(default_frames):
            break

        frame_def = default_frames[frame_index]
        collage_frame = np.vstack((frame_def, frame_var))
        out.write(collage_frame)
        frame_index += 1

    variant_cap.release()
    out.release()
    print(f"Saved collage video: {output_file}")

print("Processing complete.")
