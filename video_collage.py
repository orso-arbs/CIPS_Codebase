import os
import cv2
import numpy as np
from glob import glob

# --- Configuration ---
video_dirs = [
    r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_segment_1_2025-04-05_19-24-52_default\CP_extract_1_2025-04-05_19-37-12\CP_plotter_1_2025-04-05_19-38-02_default",
    r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_segment_1_2025-04-05_19-58-36_resample_False\CP_extract_1_2025-04-05_20-07-39\CP_plotter_1_2025-04-05_20-08-04_resample_False",
    r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_segment_1_2025-04-05_20-16-34_flow_threshold_0p2\CP_extract_1_2025-04-05_20-22-39\CP_plotter_1_2025-04-05_20-23-05_flow_threshold_0p2",
    r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_segment_1_2025-04-05_20-27-34_flow_threshold_0p6\CP_extract_1_2025-04-05_20-33-54\CP_plotter_1_2025-04-05_20-34-21_flow_threshold_0p6",
    r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_segment_1_2025-04-05_20-39-07_cellprob_threshold_0p5\CP_extract_1_2025-04-05_20-45-59\CP_plotter_1_2025-04-05_20-46-28_cellprob_threshold_0p5",
    r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_segment_1_2025-04-05_20-55-26_cellprob_threshold_0p8\CP_extract_1_2025-04-05_21-02-00\CP_plotter_1_2025-04-05_21-02-23_cellprob_threshold_0p8",
    r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_segment_1_2025-04-05_21-08-51_cellprob_threshold_1p0\CP_extract_1_2025-04-05_21-14-32\CP_plotter_1_2025-04-05_21-14-52_cellprob_threshold_1p0",
    r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_segment_1_2025-04-05_21-19-13_niter_1000\CP_extract_1_2025-04-05_21-25-04\CP_plotter_1_2025-04-05_21-25-22_niter_1000",
    r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_segment_1_2025-04-05_21-29-40_niter_5000\CP_extract_1_2025-04-05_21-39-47\CP_plotter_1_2025-04-05_21-40-16_niter_5000"
]

output_base_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop"
output_base_name = "CP_parameter_variation"

# --- Functions ---
def find_mp4_in_dir(directory):
    mp4_files = glob(os.path.join(directory, "*.mp4"))
    return mp4_files[0] if mp4_files else None

def extract_label(path):
    # Label is the last "_" section after the last timestamp
    components = path.split("_")
    for i in reversed(range(len(components))):
        if components[i].count("-") == 2:  # detects date format YYYY-MM-DD
            return "_".join(components[i+1:])
    return "unknown"

def get_unique_output_path(base_dir, base_name):
    i = 1
    while True:
        out_path = os.path.join(base_dir, f"{base_name}{i}.mp4")
        if not os.path.exists(out_path):
            return out_path
        i += 1

# --- Load Videos ---
videos = []
labels = []
for d in video_dirs:
    mp4_path = find_mp4_in_dir(d)
    if not mp4_path:
        print(f"No .mp4 file found in {d}")
        continue
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        print(f"Error opening video: {mp4_path}")
        continue
    videos.append(cap)
    labels.append(extract_label(d))

if len(videos) != 9:
    raise ValueError(f"Expected 9 videos, but got {len(videos)}.")

# --- Get Video Properties ---
frame_count = int(videos[0].get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(videos[0].get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(videos[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = videos[0].get(cv2.CAP_PROP_FPS)

# --- Setup Output Writer ---
collage_width = width * 3
collage_height = height * 3
output_path = get_unique_output_path(output_base_dir, output_base_name)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (collage_width, collage_height))

# --- Draw Text Helper ---
def draw_label(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = 30
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame

# --- Process and Write Frames ---
for _ in range(frame_count):
    frames = []
    for i, cap in enumerate(videos):
        ret, frame = cap.read()
        if not ret:
            print(f"Frame read failed for video {i}")
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame = draw_label(frame, labels[i])
        frames.append(frame)

    # Stack into 3x3 grid
    row1 = np.hstack(frames[0:3])
    row2 = np.hstack(frames[3:6])
    row3 = np.hstack(frames[6:9])
    collage = np.vstack([row1, row2, row3])
    out.write(collage)

# --- Cleanup ---
for cap in videos:
    cap.release()
out.release()
print(f"Video saved to: {output_path}")
