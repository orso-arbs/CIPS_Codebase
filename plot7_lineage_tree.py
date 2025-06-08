import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import Format_1 as F_1

try:
    import napari
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False


def plotter_7_lineage_tree(
    input_dir,
    output_dir_manual="",
    output_dir_comment="",
    show_plot=0,
    interactive=True,  # Whether to show interactive napari view
    min_track_length=3,
    figsize=(12, 10),
    dpi=150,
    Plot_log_level=1
):
    """
    Create visualizations of cell tracks from Trackastra tracking results.
    """
    # Create output directory
    output_dir = F_1.F_out_dir(input_dir=input_dir, script_path=__file__, 
                              output_dir_comment=output_dir_comment, 
                              output_dir_manual=output_dir_manual)

    if Plot_log_level >= 1:
        print(f"plotter_7_lineage_tree: Output directory: {output_dir}")
        print("Napari available:", NAPARI_AVAILABLE)
    # Load tracking data
    track_graph_path = os.path.join(input_dir, "track_graph.pkl")
    napari_tracks_path = os.path.join(input_dir, "napari_tracks.npy")
    masks_path = os.path.join(input_dir, "masks_tracked.npy")
    
    if not all(os.path.exists(p) for p in [track_graph_path, napari_tracks_path, masks_path]):
        print("Error: Missing tracking data files")
        return output_dir
    
    try:
        # Load tracking data
        with open(track_graph_path, 'rb') as f:
            track_graph = pickle.load(f)
            
        napari_tracks = np.load(napari_tracks_path)
        masks_tracked = np.load(masks_path)
        
        if Plot_log_level >= 1:
            print("Loaded tracking data")
            print(f"Number of tracks: {len(np.unique(napari_tracks[:, 0]))}")
        
        # Create timeline visualization
        fig_timeline, ax_timeline = plt.subplots(figsize=figsize)
        
        # Group tracks by ID and plot their timepoints
        track_ids = np.unique(napari_tracks[:, 0])
        for y_pos, track_id in enumerate(track_ids):
            track_points = napari_tracks[napari_tracks[:, 0] == track_id]
            frames = track_points[:, 1]  # time points
            ax_timeline.plot([min(frames), max(frames)], [y_pos, y_pos], '-', linewidth=2)
            ax_timeline.plot(frames, [y_pos] * len(frames), 'o', markersize=4)
        
        ax_timeline.set_xlabel('Frame')
        ax_timeline.set_ylabel('Track ID')
        ax_timeline.set_title('Cell Track Timeline')
        ax_timeline.grid(True, alpha=0.3)
        
        # Save timeline plot
        plt.savefig(os.path.join(output_dir, "track_timeline.png"), dpi=dpi, bbox_inches='tight')
        if not show_plot:
            plt.close()
        
        # Create spatial trajectory plot
        fig_spatial, ax_spatial = plt.subplots(figsize=figsize)
        
        # Plot each track's spatial trajectory
        for track_id in track_ids:
            track_points = napari_tracks[napari_tracks[:, 0] == track_id]
            positions = track_points[:, 2:4]  # x,y coordinates
            ax_spatial.plot(positions[:, 0], positions[:, 1], '-o', linewidth=1, markersize=4, 
                          label=f'Track {track_id}' if track_id < 10 else "")
        
        ax_spatial.set_xlabel('X Position')
        ax_spatial.set_ylabel('Y Position')
        ax_spatial.set_title('Cell Trajectories')
        if len(track_ids) <= 10:
            ax_spatial.legend()
        ax_spatial.set_aspect('equal')
        
        # Save spatial plot
        plt.savefig(os.path.join(output_dir, "spatial_trajectories.png"), dpi=dpi, bbox_inches='tight')
        if not show_plot:
            plt.close()
        
        # Create interactive visualization if requested
        if interactive and NAPARI_AVAILABLE:
            viewer = napari.Viewer()
            viewer.add_labels(masks_tracked, name='Tracked Masks')
            viewer.add_tracks(napari_tracks, name='Cell Tracks')
            napari.run()
        
        if Plot_log_level >= 1:
            print(f"Saved visualizations to {output_dir}")
            
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        if Plot_log_level >= 2:
            print(traceback.format_exc())
    
    return output_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create tracking visualization plots")
    parser.add_argument("--input_dir", required=True, help="Directory containing tracking data")
    parser.add_argument("--output_dir", default="", help="Output directory for plots")
    parser.add_argument("--comment", default="", help="Comment for output directory name")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--interactive", action="store_true", help="Show interactive napari view")
    parser.add_argument("--min_length", type=int, default=3, help="Minimum track length to include")
    parser.add_argument("--log_level", type=int, default=1, help="Logging verbosity")
    
    args = parser.parse_args()
    
    plotter_7_lineage_tree(
        input_dir=args.input_dir,
        output_dir_manual=args.output_dir,
        output_dir_comment=args.comment,
        show_plot=1 if args.show else 0,
        interactive=args.interactive,
        min_track_length=args.min_length,
        Plot_log_level=args.log_level
    )
