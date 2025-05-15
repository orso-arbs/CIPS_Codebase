import sys
import os
import Format_1 as F_1

import Visit_Projector_1 as VP1
import CP_segment_1 as CPs1
import CP_extract_1 as CPe1
import dim1_manual_1 as d1
import dim2_VisIt_R_1 as d2
import plot1 as p1
import plot2_CPvsA11 as p2
import plot3_CPvsA11_Panel as p3_panel
import plot4_dimentions as p4


@F_1.ParameterLog(max_size = 1024 * 10) # 10KB 
def VCL_pipeline(
    # General control
    input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\SF_CP_analysis_pipeline_data", # leave empty to start pipeline with visit folder
    vcl_pipeline_output_dir_manual="", # leave empty to start pipeline with visit folder
    vcl_pipeline_output_dir_comment="", 

    # Visit_projector_1 args
    vp_input_dir="",
    vp_Database=r"euler.ethz.ch:/cluster/scratch/orsob/orsoMT_orsob/A11_states/A11_all_states.visit",
    vp_State_range_manual=[1],
    vp_Plots=["Pseudocolor - Isosurface"],
    vp_Pseudocolor_Variable="velocity_magnitude",
    vp_Pseudocolor_colortable="hot",
    vp_invertColorTable=0,
    vp_Isosurface_Variable="temperature",
    vp_Isosurface_ContourValue=3,
    vp_no_annotations=1,
    vp_viewNormal=[0, 0, -1],
    vp_viewUp=[1, 0, 0],
    vp_imageZoom=1,
    vp_parallelScale=80,
    vp_perspective=0,
    vp_Visit_projector_1_log_level=1,
    vp_Visit_projector_1_show_windows=0,
    vp_output_dir_manual="",
    vp_output_dir_comment="",

    # CP_segment_1 args
    cps_CP_model_type="cyto3",
    cps_gpu=True,
    cps_diameter_estimate_guess_px=None,
    cps_output_dir_comment="", # Base comment
    cps_CP_segment_log_level=1,

    # CP_extract_1 args
    cpe_CP_extract_log_level=0,
    # cpe_diameter_training_px=None, # Was commented out in original

    # dimentionalise_2_from_VisIt_R_Average args
    d2_CP_dimentionalise_log_level=0,
    d2_output_dir_comment="", # Base comment

    # plotter_1 args
    p1_output_dir_manual="",
    p1_output_dir_comment="",
    p1_video=1,

    # plotter_4_dimentionalisation args
    p4_output_dir_manual="",
    p4_output_dir_comment="",
    p4_show_plot=0,
    p4_Plot_log_level=0,
    p4_Panel_1_A11=0,
    p4_A11_manual_data_base_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction",
    p4_Panel_2_Dimentionalised_from_VisIt=1,

    # plotter_2_CPvsA11 args
    p2_output_dir_manual="",
    p2_output_dir_comment="",
    p2_video=1,

    # plotter_3_CPvsA11_Panel args
    p3_output_dir_manual="",
    p3_output_dir_comment="",
    p3_video=0,
    p3_show_plot=0,
    p3_Panel_1=0,
    p3_Panel_2=0,
    p3_Panel_3=0,
    p3_Panel_4=1,

    # Control flags for pipeline sections
    run_visit_projector=True,
    run_cp_segment=True,
    run_cp_extract=True,
    run_dimentionalise=True,
    run_plotter_1=True,
    run_plotter_4=True,
    run_plotter_2=True,
    run_plotter_3_panel=True,

    # run_cp_segment=False,
    # run_cp_extract=False,
    # run_dimentionalise=False,
    # run_plotter_1=False,
    # run_plotter_4=False,
    # run_plotter_2=False,
    # run_plotter_3_panel=False



):
    """
    Runs the VCL (VisIt-Cellpose-Lineage-Tree) pipeline with configurable parameters.
    Each major step can be toggled on/off using the 'run_*' boolean flags.
    Output directories are chained from one step to the next.
    The 'vp_input_dir' is the primary input for the first step (Visit_projector_1).
    The 'vp_output_dir_manual' can be used to specify a top-level directory for a run,
    into which Visit_Projector_1 will create its specific output folder.
    Subsequent steps will use the output of the previous step as their input.
    """

    #################################################### I/O
    start_time, current_date = F_1.start_inform(__file__)

    vcl_pipeline_output_dir = F_1.F_out_dir(input_dir = input_dir, script_path = __file__, output_dir_comment = vcl_pipeline_output_dir_comment, output_dir_manual = vcl_pipeline_output_dir_manual) # Format_1 required definition of output directory

    # Initialize output directory variables
    VP1_output_dir = None
    CPs1_output_dir = None
    CPe1_output_dir = None
    d2_output_dir = None

    #########################################        Visit

    if not vp_input_dir:
        vp_input_dir = vcl_pipeline_output_dir # to put the Visit output in the pipeline folder if the VisIt output folder is not specified

    if run_visit_projector:
        print(f"--- Running Visit_Projector_1 ---")
        # For variations, vp_output_dir_manual will be set by the calling script
        # to ensure outputs go into the correct variation folder.
        # vp_output_dir_comment will be specific to the variation.
        VP1_output_dir = VP1.Visit_projector_1(
            input_dir=vp_input_dir,
            Database=vp_Database, State_range_manual=vp_State_range_manual,
            Plots=vp_Plots,
            Pseudocolor_Variable=vp_Pseudocolor_Variable, Pseudocolor_colortable=vp_Pseudocolor_colortable, invertColorTable=vp_invertColorTable,
            Isosurface_Variable=vp_Isosurface_Variable, Isosurface_ContourValue=vp_Isosurface_ContourValue,
            no_annotations=vp_no_annotations, viewNormal=vp_viewNormal, viewUp=vp_viewUp, imageZoom=vp_imageZoom, parallelScale=vp_parallelScale, perspective=vp_perspective,
            Visit_projector_1_log_level=vp_Visit_projector_1_log_level, Visit_projector_1_show_windows=vp_Visit_projector_1_show_windows,
            output_dir_manual=vp_output_dir_manual, # This is key for variations
            output_dir_comment=vp_output_dir_comment,
        )
        # print(f"Visit_Projector_1 output: {VP1_output_dir}")
        print("Note: Visit window can now be closed. 'VisIt: Error - Can't delete the last window' is now inconsequential to the remaining code")
    else:
        print("--- Skipping Visit_Projector_1 ---")
        # If skipped, need a way to specify VP1_output_dir if subsequent steps are run
        # For a full pipeline run, this would typically not be skipped if later steps depend on it.
        # For parameter sweeps, this step will usually run.

    #########################################        Cellpose
    if run_cp_segment:
        if VP1_output_dir: # Proceed only if Visit output exists
            print(f"--- Running CP_segment_1 ---")
            CPs1_output_dir = CPs1.CP_segment_1(
                input_dir=VP1_output_dir,
                CP_model_type=cps_CP_model_type,
                gpu=cps_gpu,
                diameter_estimate_guess_px=cps_diameter_estimate_guess_px,
                output_dir_comment=cps_output_dir_comment,
                CP_segment_log_level=cps_CP_segment_log_level,
            )
            # print(f"CP_segment_1 output: {CPs1_output_dir}")
        else:
            print("--- Skipping CP_segment_1 (missing Visit_Projector_1 output) ---")
    else:
        print("--- Skipping CP_segment_1 ---")

    #########################################        Process Data
    if run_cp_extract:
        if CPs1_output_dir: # Proceed only if CP_segment output exists
            print(f"--- Running CP_extract_1 ---")
            CPe1_output_dir = CPe1.CP_extract_1(
                input_dir=CPs1_output_dir,
                CP_extract_log_level=cpe_CP_extract_log_level,
                # diameter_training_px=cpe_diameter_training_px, # if added as arg
            )
            # print(f"CP_extract_1 output: {CPe1_output_dir}")
        else:
            print("--- Skipping CP_extract_1 (missing CP_segment_1 output) ---")
    else:
        print("--- Skipping CP_extract_1 ---")

    if run_dimentionalise:
        if CPe1_output_dir: # Proceed only if CP_extract output exists
            print(f"--- Running dimentionalise_2_from_VisIt_R_Average ---")
            d2_output_dir = d2.dimentionalise_2_from_VisIt_R_Average(
                input_dir=CPe1_output_dir,
                CP_dimentionalise_log_level=d2_CP_dimentionalise_log_level,
                output_dir_comment=d2_output_dir_comment,
            )
            # print(f"dimentionalise_2_from_VisIt_R_Average output: {d2_output_dir}")
        else:
            print("--- Skipping dimentionalise_2_from_VisIt_R_Average (missing CP_extract_1 output) ---")
    else:
        print("--- Skipping dimentionalise_2_from_VisIt_R_Average ---")


    # At this point, d2_output_dir is equivalent to d1_output_dir in the original script for plotting
    plot_input_dir = d2_output_dir

    #########################################        Plot
    if run_plotter_1:
        if plot_input_dir:
            print(f"--- Running plotter_1 ---")
            p1_output_dir = p1.plotter_1(
                input_dir=plot_input_dir,
                output_dir_manual=p1_output_dir_manual, # Plotters usually create subfolders in their input_dir
                output_dir_comment=p1_output_dir_comment,
                video=p1_video
            )
            # print(f"plotter_1 output: {p1_output_dir}")
        else:
            print("--- Skipping plotter_1 (missing dimentionalisation output) ---")
    else:
        print("--- Skipping plotter_1 ---")

    if run_plotter_4:
        if plot_input_dir:
            print(f"--- Running plotter_4_dimentionalisation ---")
            p4_output_dir = p4.plotter_4_dimentionalisation(
                input_dir=plot_input_dir,
                output_dir_manual=p4_output_dir_manual,
                output_dir_comment=p4_output_dir_comment,
                show_plot=p4_show_plot, Plot_log_level=p4_Plot_log_level,
                Panel_1_A11=p4_Panel_1_A11, A11_manual_data_base_dir=p4_A11_manual_data_base_dir,
                Panel_2_Dimentionalised_from_VisIt=p4_Panel_2_Dimentionalised_from_VisIt,
            )
            # print(f"plotter_4_dimentionalisation output: {p4_output_dir}")
        else:
            print("--- Skipping plotter_4_dimentionalisation (missing dimentionalisation output) ---")
    else:
        print("--- Skipping plotter_4_dimentionalisation ---")

    if run_plotter_2:
        if plot_input_dir:
            print(f"--- Running plotter_2_CPvsA11 ---")
            p2_output_dir = p2.plotter_2_CPvsA11(
                input_dir=plot_input_dir,
                output_dir_manual=p2_output_dir_manual,
                output_dir_comment=p2_output_dir_comment,
                video=p2_video
            )
            # print(f"plotter_2_CPvsA11 output: {p2_output_dir}")
        else:
            print("--- Skipping plotter_2_CPvsA11 (missing dimentionalisation output) ---")
    else:
        print("--- Skipping plotter_2_CPvsA11 ---")

    if run_plotter_3_panel:
        if plot_input_dir:
            print(f"--- Running plotter_3_CPvsA11_Panel ---")
            p3_out_dir = p3_panel.plotter_3_CPvsA11_Panel(
                input_dir=plot_input_dir,
                output_dir_manual=p3_output_dir_manual,
                output_dir_comment=p3_output_dir_comment,
                video=p3_video, show_plot=p3_show_plot,
                Panel_1=p3_Panel_1,
                Panel_2=p3_Panel_2,
                Panel_3=p3_Panel_3,
                Panel_4=p3_Panel_4,
            )
            # print(f"plotter_3_CPvsA11_Panel output: {p3_out_dir}")
        else:
            print("--- Skipping plotter_3_CPvsA11_Panel (missing dimentionalisation output) ---")
    else:
        print("--- Skipping plotter_3_CPvsA11_Panel ---")

    ######################################################## end
    F_1.end_inform(__file__, start_time)
    F_1.ding()

    # Return the path of the final data directory (e.g., dimentionalisation output)
    # or a dictionary of key output paths if needed by the calling script.
    return {
        "VP1_output_dir": VP1_output_dir,
        "CPs1_output_dir": CPs1_output_dir,
        "CPe1_output_dir": CPe1_output_dir,
        "d2_output_dir": d2_output_dir, # Final data processing output before plotting
        "plot_input_dir": plot_input_dir
    }




# Example of how to run the pipeline with default settings
if __name__ == "__main__":
    print("Running VCL-Pipeline with default settings as a standalone script example.")
    
    VCL_pipeline(
        input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\SF_CP_analysis_pipeline_data", # directory to place outputs
        vcl_pipeline_output_dir_comment="Test00"
    )
    print("Default pipeline run finished.")