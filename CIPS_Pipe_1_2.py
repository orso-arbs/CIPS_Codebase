import sys
import os
import Format_1 as F_1
import Trackastra_tracking_1 as tr1
import plot7_lineage_tree as p7

@F_1.ParameterLog(max_size = 1024 * 10)
def CIPS_pipeline_2(
    # General control
    input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\SF_CP_analysis_pipeline_data\Visit_Projector_1_2025-05-10_14-46-34_A11_T-3_VM-hot\CP_segment_1_2025-05-10_15-40-15_cyto3\CP_extract_1_2025-05-10_15-46-46",
    output_dir_manual=r"C:\Users\obs\Desktop\track test",
    output_dir_comment="Tracking_Analysis", 
    global_log_level=1,
    
    # Trackastra_tracking_1 args
    tr_output_dir_manual="",
    tr_output_dir_comment="",
    tr_max_distance=50,
    tr_max_cell_division_distance=30,
    tr_min_track_length=5,
    tr_track_type='tracklets',
    tr_Trackastra_tracking_log_level=2,
    
    # plot7_lineage_tree args
    p7_output_dir_manual="",
    p7_output_dir_comment="",
    p7_show_plot=0,
    p7_min_track_length=3,
    p7_Plot_log_level=1,

    # Control flags
    run_trackastra_tracking=True,
    run_plotter_7_lineage=True,
):
    """
    Runs a simplified version of the CIPS pipeline focusing on cell tracking and lineage visualization.
    """
    # Override log levels with global level if set
    if global_log_level is not None:
        tr_Trackastra_tracking_log_level = global_log_level
        p7_Plot_log_level = global_log_level

    # Create main output directory
    pipeline_output_dir = F_1.F_out_dir(
        input_dir=input_dir, 
        script_path=__file__, 
        output_dir_comment=output_dir_comment, 
        output_dir_manual=output_dir_manual
    )

    # Initialize tracking output directory
    tr1_output_dir = None

    # Apply Trackastra cell tracking
    if run_trackastra_tracking:
        print(f"--- Running Trackastra_tracking_1 ---")
        tr1_output_dir = tr1.Trackastra_tracking_1(
            input_dir=input_dir,
            output_dir_manual=tr_output_dir_manual,
            output_dir_comment=tr_output_dir_comment,
            max_distance=tr_max_distance,
            max_cell_division_distance=tr_max_cell_division_distance,
            min_track_length=tr_min_track_length,
            track_type=tr_track_type,
            Trackastra_tracking_log_level=tr_Trackastra_tracking_log_level
        )
    else:
        print("--- Skipping Trackastra_tracking_1 ---")
    
    # Plot lineage trees
    if run_plotter_7_lineage:
        if tr1_output_dir:
            print(f"--- Running plotter_7_lineage_tree ---")
            p7_output_dir = p7.plotter_7_lineage_tree(
                input_dir=tr1_output_dir,
                output_dir_manual=p7_output_dir_manual,
                output_dir_comment=p7_output_dir_comment,
                show_plot=p7_show_plot,
                min_track_length=p7_min_track_length,
                Plot_log_level=p7_Plot_log_level
            )
        else:
            print("--- Skipping plotter_7_lineage_tree (missing tracking output) ---")
    else:
        print("--- Skipping plotter_7_lineage_tree ---")

    F_1.ding()
    
    return {
        "pipeline_output_dir": pipeline_output_dir,
        "tr1_output_dir": tr1_output_dir
    }

if __name__ == "__main__":
    print("Running CIPS-Pipeline-2 (Tracking Analysis).")
    
    # Example usage with some custom parameters
    CIPS_pipeline_2(
        tr_max_distance=40,  # Adjust tracking distance
        tr_min_track_length=3,  # Shorter minimum track length
        p7_show_plot=1,  # Show plots interactively
    )
    print("CIPS-Pipeline-2 run finished.")
