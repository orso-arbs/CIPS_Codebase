import os
import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import trackastra as ta

def main():
    # Hardcoded input path for testing
    input_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\SF_CP_analysis_pipeline_data\Visit_Projector_1_2025-05-10_14-46-34_A11_T-3_VM-hot\CP_segment_1_2025-05-10_15-40-15_cyto3\CP_extract_1_2025-05-10_15-46-46"
    
    # Load the DataFrame with segmentation masks
    pkl_files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
    if not pkl_files:
        print(f"Error: No .pkl files found in {input_dir}")
        return
    
    with open(os.path.join(input_dir, pkl_files[0]), 'rb') as f:
        df = pickle.load(f)
    
    print(f"Loaded DataFrame with {len(df)} frames")
    
    # Extract tracking data from masks
    tracking_data = []
    for idx, row in df.iterrows():
        image_num = row['image_number']
        masks = row['masks']
        
        if masks is None:
            continue
            
        # Process each cell in the mask
        cell_ids = np.unique(masks)
        cell_ids = cell_ids[cell_ids > 0]  # Exclude background
        
        for cell_id in cell_ids:
            cell_mask = masks == cell_id
            y_coords, x_coords = np.where(cell_mask)
            
            if len(y_coords) == 0:
                continue
                
            centroid_y = np.mean(y_coords)
            centroid_x = np.mean(x_coords)
            area = len(y_coords)
            
            tracking_data.append({
                'frame': image_num,
                'x': centroid_x,
                'y': centroid_y,
                'area': area,
                'cell_id': cell_id
            })
    
    # Convert to DataFrame for tracking
    track_df = pd.DataFrame(tracking_data)
    print(f"Created tracking data with {len(track_df)} cell positions")
    
    # Initialize and run tracker
    tracker = ta.Tracker()
    tracker.params['settings']['distance_threshold'] = 40  # Adjust if needed
    tracker.track_as_tracklets(track_df, ['frame', 'x', 'y'])
    
    # Get results
    tracklets = tracker.get_tracklets()
    print(f"Generated {len(np.unique(tracklets['tracklet_id']))} tracklets")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Cell trajectories
    plt.subplot(121)
    for track_id in np.unique(tracklets['tracklet_id']):
        track_data = tracklets[tracklets['tracklet_id'] == track_id]
        plt.plot(track_data['x'], track_data['y'], '-o', label=f'Track {track_id}' if track_id < 5 else "")
    
    plt.title('Cell Trajectories')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    if len(np.unique(tracklets['tracklet_id'])) < 10:
        plt.legend()
    
    # Plot 2: Timeline view
    plt.subplot(122)
    for track_id in np.unique(tracklets['tracklet_id']):
        track_data = tracklets[tracklets['tracklet_id'] == track_id]
        frames = track_data['frame']
        plt.plot([min(frames), max(frames)], [track_id, track_id], '-', linewidth=2)
        plt.plot(frames, [track_id] * len(frames), 'o')
    
    plt.title('Track Timeline')
    plt.xlabel('Frame Number')
    plt.ylabel('Track ID')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.join(input_dir, "tracking_test_results")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "tracking_results.png"))
    
    # Save tracking data
    tracklets.to_csv(os.path.join(output_dir, "tracklets.csv"), index=False)
    
    print(f"Results saved to {output_dir}")
    plt.show()

if __name__ == "__main__":
    main()
