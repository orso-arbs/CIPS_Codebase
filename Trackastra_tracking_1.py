import os
import sys
import pandas as pd
import numpy as np
import glob
import pickle
import torch
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks
import Format_1 as F_1

def Trackastra_tracking_1(
    input_dir,
    output_dir_manual="",
    output_dir_comment="",
    max_distance=50,  # Maximum distance (in pixels) between adjacent frames to consider a cell match
    max_cell_division_distance=30,  # Maximum distance for potential mother-daughter links
    min_track_length=5,  # Minimum length of track to keep (in frames)
    track_type='tracklets',  # 'tracklets' or 'trajectories'
    Trackastra_tracking_log_level=2
):
    """
    Apply trackastra to track segmented cells from cellpose.
    
    Parameters
    ----------
    input_dir : str
        Directory containing the input data from CP_extract (with .pkl files)
    output_dir_manual : str, optional
        Manual specification of output directory, by default ""
    output_dir_comment : str, optional
        Comment to append to output directory name, by default ""
    max_distance : int, optional
        Maximum distance (in pixels) between adjacent frames to consider a cell match, by default 50
    max_cell_division_distance : int, optional
        Maximum distance for potential mother-daughter links, by default 30
    min_track_length : int, optional
        Minimum length of track to keep (in frames), by default 5
    track_type : str, optional
        Type of tracks to generate ('tracklets' or 'trajectories'), by default 'tracklets'
    Trackastra_tracking_log_level : int, optional
        Logging level for this function, by default 2
    
    Returns
    -------
    str
        Path to the output directory containing tracking results
    """
    # Create output directory for tracking results
    output_dir = F_1.F_out_dir(input_dir=input_dir, script_path=__file__, output_dir_comment=output_dir_comment, output_dir_manual=output_dir_manual)

    if Trackastra_tracking_log_level >= 1:
        print(f"Trackastra_tracking_1: Output directory: {output_dir}")
    
    # Find the DataFrame .pkl file in the input directory
    pkl_wildcard_str = os.path.join(input_dir, "*.pkl")
    pkl_files = glob.glob(pkl_wildcard_str)
    
    if not pkl_files:
        print(f"Error: No .pkl files found in {input_dir}")
        return output_dir
    
    # Load the DataFrame with segmentation masks
    df_path = pkl_files[0]  # Take the first .pkl file
    
    try:
        with open(df_path, 'rb') as f:
            df = pickle.load(f)
            
        if Trackastra_tracking_log_level >= 1:
            print(f"Loaded DataFrame from {df_path}")
            print(f"DataFrame contains {len(df)} rows")
    except Exception as e:
        print(f"Error loading DataFrame from {df_path}: {e}")
        return output_dir
    
    # We know the structure of the DataFrame from CP_extract_1 function
    # The key columns we need are: 'image_number'/'image_num' and 'masks'
    
    # Check which column name is used for image number
    image_num_col = 'image_number' if 'image_number' in df.columns else 'image_num'
    
    if image_num_col not in df.columns or 'masks' not in df.columns:
        print(f"Error: DataFrame is missing required columns. Expected 'image_number'/'image_num' and 'masks'.")
        print(f"Available columns: {df.columns.tolist()}")
        return output_dir
    
    # Create input data for trackastra
    tracking_data = []
    
    for idx, row in df.iterrows():
        image_num = row[image_num_col]
        masks = row['masks']
        
        if masks is None or not isinstance(masks, np.ndarray):
            if Trackastra_tracking_log_level >= 1:
                print(f"Warning: No valid mask data for image {image_num}, skipping")
            continue
        
        # Find unique cell IDs in the mask (excluding background 0)
        cell_ids = np.unique(masks)
        cell_ids = cell_ids[cell_ids > 0]
        
        for cell_id in cell_ids:
            # Create a binary mask for this specific cell
            cell_mask = masks == cell_id
            
            # Calculate centroid of the cell
            y_coords, x_coords = np.where(cell_mask)
            if len(y_coords) == 0:
                continue
                
            centroid_y = np.mean(y_coords)
            centroid_x = np.mean(x_coords)
            
            # Calculate area
            area = len(y_coords)
            
            # Store tracking data for this cell
            tracking_data.append({
                'frame': image_num,
                'x': centroid_x,
                'y': centroid_y,
                'area': area,
                'original_cell_id': cell_id
            })
    
    if not tracking_data:
        print("Error: No valid tracking data could be extracted from masks")
        return output_dir
    
    # Convert to DataFrame for trackastra
    tracking_df = pd.DataFrame(tracking_data)
    
    if Trackastra_tracking_log_level >= 2:
        print(f"Created tracking data with {len(tracking_df)} cell positions")
    
    # Save the raw tracking data
    tracking_df_path = os.path.join(output_dir, "raw_tracking_data.csv")
    tracking_df.to_csv(tracking_df_path, index=False)
    
    try:
        # Setup device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if Trackastra_tracking_log_level >= 1:
            print(f"Using device: {device}")

        # Initialize trackastra tracker with pre-trained model
        model = Trackastra.from_pretrained("general_2d", device=device)
        
        # Prepare masks array (time, y, x)
        masks_list = [row['masks'] for _, row in df.iterrows() if isinstance(row['masks'], np.ndarray)]
        masks_array = np.stack(masks_list)
        
        # Create dummy images array if needed for the model
        images_array = np.zeros_like(masks_array, dtype=np.float32)
        
        if Trackastra_tracking_log_level >= 1:
            print("Running trackastra tracking algorithm...")
        
        # Track the cells
        track_graph = model.track(images_array, masks_array, mode="greedy")
        
        # Generate CTC format tracking results
        ctc_tracks, masks_tracked = graph_to_ctc(
            track_graph,
            masks_array,
            outdir=os.path.join(output_dir, "ctc_tracks")
        )
        
        # Save tracking results
        np.save(os.path.join(output_dir, "masks_tracked.npy"), masks_tracked)
        
        # Get napari-compatible tracks for visualization
        napari_tracks, napari_graph, track_properties = graph_to_napari_tracks(track_graph)
        np.save(os.path.join(output_dir, "napari_tracks.npy"), napari_tracks)
        
        # Save the graph for later use
        with open(os.path.join(output_dir, "track_graph.pkl"), 'wb') as f:
            pickle.dump(track_graph, f)
            
        if Trackastra_tracking_log_level >= 1:
            print(f"Tracking complete. Results saved to {output_dir}")
            print(f"Number of tracks: {len(ctc_tracks)}")
        
        # Create a tracked DataFrame with the original data and tracking results
        tracked_df = df.copy()
        tracked_df['tracks'] = pd.Series(ctc_tracks)
        tracked_df['masks_tracked'] = pd.Series([m for m in masks_tracked])
        
        # Save the tracked DataFrame
        tracked_df_path = os.path.join(output_dir, "tracked_data.pkl")
        tracked_df.to_pickle(tracked_df_path)
        
        if Trackastra_tracking_log_level >= 1:
            print(f"Saved tracked DataFrame to {tracked_df_path}")
            
    except Exception as e:
        print(f"Error during tracking: {e}")
        if Trackastra_tracking_log_level >= 2:
            print(traceback.format_exc())
    
    return output_dir

if __name__ == "__main__":
    # Example usage when run as a script
    import argparse
    
    parser = argparse.ArgumentParser(description="Track cells using trackastra")
    parser.add_argument("--input_dir", required=True, help="Directory containing segmentation data")
    parser.add_argument("--output_dir", default="", help="Output directory for tracking results")
    parser.add_argument("--comment", default="", help="Comment for output directory name")
    parser.add_argument("--max_distance", type=int, default=50, help="Maximum linking distance")
    parser.add_argument("--min_track_length", type=int, default=5, help="Minimum track length to keep")
    parser.add_argument("--track_type", choices=['tracklets', 'trajectories'], default='tracklets',
                        help="Type of tracking to perform")
    parser.add_argument("--log_level", type=int, default=2, help="Logging verbosity")
    
    args = parser.parse_args()
    
    Trackastra_tracking_1(
        input_dir=args.input_dir,
        output_dir_manual=args.output_dir,
        output_dir_comment=args.comment,
        max_distance=args.max_distance,
        min_track_length=args.min_track_length,
        track_type=args.track_type,
        Trackastra_tracking_log_level=args.log_level
    )
