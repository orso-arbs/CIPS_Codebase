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
def CIPS_pipeline(
    # General control
    input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir",
    cips_pipeline_output_dir_manual="",
    cips_pipeline_output_dir_comment="Resolution 3000px2", 
    cips_pipeline_global_log_level=None,  # Added global log level parameter

    # Visit_projector_1 args
    vp_input_dir="",
    vp_Database=r"euler.ethz.ch:/cluster/scratch/orsob/orsoMT_orsob/A11_states/A11_all_states.visit",
    vp_State_range_manual=[100],
    vp_Plots=["Pseudocolor - Isosurface"],
    vp_Pseudocolor_Variable="velocity_magnitude", # "temperature", "density", "pressure", "velocity_magnitude"
                            # s1 :  H2      s10: HRR            s19: omega_x
                            # s2 :  O2      s11: stretch        s20: omega_y
                            # s3 :  H2O     s12: curvature      s21: omega_z
                            # s4 :  H       s13: atot
                            # s5 :  O       s14: an
                            # s6 :  OH      s15: at
                            # s7 :  HO2     s16: Sd
                            # s8 :  H2O2    s17: Sdd
                            # s9 :  N2      s18: Sa
    vp_Pseudocolor_colortable="hot", # Can be "hot", "CustomBW1", "CustomBW2", "PeriodicBW", "PointWise", etc.
    vp_invertColorTable=0,
    # Parameters for the periodic black and white color table
    Pseudocolor_periodic_num_periods = 3,   # periods of w-w-b-b points (4 points)
    distance_ww = 2.0,          # Relative length of solid white
    distance_wb = 1.0,          # Relative length of white-to-black gradient
    distance_bb = 2.0,          # Relative length of solid black
    distance_bw = 1.0,          # Relative length of black-to-white gradient
    # Parameters for the pointwise color table
    pointwise_color_points = None,  # List of [position, r, g, b, a] points for PointWise color table
    show_color_table_markers = True,  # Whether to show position markers and labels in color table preview
    
    vp_Isosurface_Variable="temperature",
    vp_Isosurface_ContourValue=3,
    vp_no_annotations=1,
    vp_viewNormal=[0, -1, 0],
    vp_viewUp=[0, 0, 1],
    vp_imageZoom=1,
    vp_parallelScale=80,
    vp_perspective=0,
    vp_WindowWidth = 3000, # Window size in px
    vp_WindowHeight = 3000, # Window size in px
    vp_Visit_projector_1_log_level=1,
    vp_Visit_projector_1_show_windows=0,
    vp_output_dir_manual="",
    vp_output_dir_comment="",

    # CP_segment_1 args
    cps_CP_model_type="cyto3",
    cps_gpu=True,
    cps_diameter_estimate_guess_px=None,
    cps_output_dir_comment="",
    cps_CP_segment_log_level=1,

    # CP_extract_1 args
    cpe_CP_extract_log_level=1,

    # dimentionalise_2_from_VisIt_R_Average args
    d2_CP_dimentionalise_log_level=1,
    d2_output_dir_comment="",    

    # plotter_1 args
    p1_output_dir_manual="",
    p1_output_dir_comment="",
    p1_video=1,

    # plotter_4_dimentionalisation args
    p4_output_dir_manual="",
    p4_output_dir_comment="",
    p4_show_plot=0,
    p4_Plot_log_level=1,  # Default to 1
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
    
    ):
    """
    Runs the CIPS pipeline with configurable parameters.
    
    Parameters added:
    ----------
    cips_pipeline_global_log_level : int, optional
        Global log level that will be used for all components unless specifically overridden.
        Default is 0 (minimal logging).
    pointwise_color_points : list of lists, optional
        Points defining the PointWise color table. Each point is a list [position, r, g, b, a].
        position should be a float between 0.0 and 1.0
        r, g, b, a are color values between 0-255
        Required if vp_Pseudocolor_colortable="PointWise". Default is None.
    show_color_table_markers : bool, optional
        Whether to show position markers and labels in color table preview images. Default is True.
    """

    # Override individual log levels with global level if set
    if cips_pipeline_global_log_level is not None:
        vp_Visit_projector_1_log_level = cips_pipeline_global_log_level
        cps_CP_segment_log_level = cips_pipeline_global_log_level
        cpe_CP_extract_log_level = cips_pipeline_global_log_level
        d2_CP_dimentionalise_log_level = cips_pipeline_global_log_level
        p4_Plot_log_level = cips_pipeline_global_log_level

    #################################################### I/O
    cips_pipeline_output_dir = F_1.F_out_dir(input_dir = input_dir, script_path = __file__, output_dir_comment = cips_pipeline_output_dir_comment, output_dir_manual = cips_pipeline_output_dir_manual) # Format_1 required definition of output directory

    # Initialize output directory variables
    VP1_output_dir = None
    CPs1_output_dir = None
    CPe1_output_dir = None
    d2_output_dir = None

    #########################################        Visit

    if not vp_input_dir:
        vp_input_dir = cips_pipeline_output_dir # to put the Visit output in the pipeline folder if the VisIt output folder is not specified

    if run_visit_projector:
        print(f"--- Running Visit_Projector_1 ---")
        VP1_output_dir = VP1.Visit_projector_1(
            input_dir=vp_input_dir,
            Database=vp_Database, State_range_manual=vp_State_range_manual,
            Plots=vp_Plots,
            Pseudocolor_Variable=vp_Pseudocolor_Variable, 
            Pseudocolor_colortable=vp_Pseudocolor_colortable, 
            invertColorTable=vp_invertColorTable, 
            Pseudocolor_periodic_num_periods=Pseudocolor_periodic_num_periods, 
            distance_ww=distance_ww, 
            distance_wb=distance_wb, 
            distance_bb=distance_bb, 
            distance_bw=distance_bw,
            pointwise_color_points=pointwise_color_points,
            show_color_table_markers=show_color_table_markers,
            Isosurface_Variable=vp_Isosurface_Variable, 
            Isosurface_ContourValue=vp_Isosurface_ContourValue,
            no_annotations=vp_no_annotations, viewNormal=vp_viewNormal, viewUp=vp_viewUp, imageZoom=vp_imageZoom, parallelScale=vp_parallelScale, perspective=vp_perspective,
            Visit_projector_1_log_level=vp_Visit_projector_1_log_level,  # Use processed log level
            Visit_projector_1_show_windows=vp_Visit_projector_1_show_windows,
            WindowWidth = vp_WindowWidth, WindowHeight = vp_WindowHeight,
            output_dir_manual=vp_output_dir_manual,
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
                CP_segment_log_level=cps_CP_segment_log_level,  # Use processed log level
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
                CP_extract_log_level=cpe_CP_extract_log_level,  # Use processed log level
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
                CP_dimentionalise_log_level=d2_CP_dimentionalise_log_level,  # Use processed log level
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
                show_plot=p4_show_plot, Plot_log_level=p4_Plot_log_level,  # Use processed log level
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
    F_1.ding()

    # Return the path of the final data directory (e.g., dimentionalisation output)
    # or a dictionary of key output paths if needed by the calling script.
    return {
        "cips_pipeline_output_dir": cips_pipeline_output_dir,
        "VP1_output_dir": VP1_output_dir,
        "CPs1_output_dir": CPs1_output_dir,
        "CPe1_output_dir": CPe1_output_dir,
        "d2_output_dir": d2_output_dir, # Final data processing output before plotting
        "plot_input_dir": plot_input_dir
    }




# Example of how to run the pipeline with default settings
if __name__ == "__main__":
    print("Running CIPS-Pipeline.")
    
    CIPS_pipeline(
    )
    print("CIPS-Pipeline run finished.")