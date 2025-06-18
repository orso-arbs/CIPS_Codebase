import sys
import os
import Format_1 as F_1
from Format_1 import Tee
import shutil
import tempfile
import traceback 

import Visit_Projector_1 as VP1
import CP_segment_1 as CPs1
import CP_extract_1 as CPe1
import Analysis_Altantzis2011 as A11
import plot1 as p1
import plot2_CPvsA11 as p2
import plot3_CPvsA11_Panel as p3_panel
import plot4_dimentions as p4
import plot6_colortables as p6c

@F_1.ParameterLog(max_size = 1024 * 10)
def CIPS_pipeline_2(
    # General control
    input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir",
    cips_pipeline_output_dir_manual="",
    cips_pipeline_output_dir_comment="T3_vmag_3000px_WWBBWW_cyto3_flowThres0p5", 
    cips_pipeline_global_log_level=None,

    # Stage output overrides (for resuming pipeline)
    cips_VP1_output_dir_override="",
    cips_CPs1_output_dir_override="",
    cips_CPe1_output_dir_override="",
    cips_A11_output_dir_override="",  # New override for Analysis_Altantzis2011 stage

    # Visit_projector_1 args
    vp_input_dir="",
    vp_Database=r"euler.ethz.ch:/cluster/scratch/orsob/orsoMT_orsob/A11_states/A11_all_states.visit",
    vp_State_range_manual=[],
    vp_Plots=["Pseudocolor - Isosurface"],
    vp_Pseudocolor_Variable="velocity_magnitude",
    vp_Pseudocolor_colortable="PointWise",
    vp_invertColorTable=0,
    # Parameters for the periodic black and white color table
    Pseudocolor_periodic_num_periods=3,
    distance_ww=2.0,
    distance_wb=1.0,
    distance_bb=2.0,
    distance_bw=1.0,
    # Parameters for the pointwise color table
    pointwise_color_points=[
        [0.0, 255, 255, 255, 255],  # White
        [0.3, 255, 255, 255, 255],  # White
        [0.45, 0, 0, 0, 255], # Black
        [0.55, 0, 0, 0, 255], # Black
        [0.7, 255, 255, 255, 255],  # White
        [1.0, 255, 255, 255, 255],  # White
    ],  
    show_color_table_markers=True,
    
    vp_Isosurface_Variable="temperature",
    vp_Isosurface_ContourValue=3,
    vp_no_annotations=1,
    vp_viewNormal=[0, -1, 0],
    vp_viewUp=[0, 0, 1],
    vp_imageZoom=1,
    vp_parallelScale=80,
    vp_perspective=0,
    vp_WindowWidth=3000,
    vp_WindowHeight=3000,
    vp_Visit_projector_1_log_level=2,
    vp_Visit_projector_1_show_windows=0,
    vp_output_dir_manual="",
    vp_output_dir_comment="",

    # CP_segment_1 args
    cps_CP_model_type_for_segmentation="cyto3", 
    cps_CP_model_type_for_diameter_estimation="cyto3",
    cps_gpu=True,
    cps_CP_empty_cache_onoff=True, 
    cps_diameter_estimate_guess_px=None,
    cps_channels=[0,0], 
    cps_flow_threshold=0.5, 
    cps_cellprob_threshold=0.0, 
    cps_resample=True, 
    cps_niter=0,
    cps_batch_size=2,
    cps_augment=True,
    cps_tile_overlap=0.1,
    cps_bsize=160,
    cps_max_images_per_batch=40,
    cps_CP_default_plot_onoff=0, 
    cps_CP_default_image_onoff=0, 
    cps_CP_default_seg_file_onoff=1,
    cps_output_dir_manual="",
    cps_output_dir_comment="",
    cps_CP_segment_log_level=1,

    # CP_extract_1 args
    cpe_output_dir_manual="",
    cpe_output_dir_comment="",
    cpe_CP_extract_log_level=1,
    
    # Analysis_Altantzis2011 args
    a11_output_dir_manual="",
    a11_output_dir_comment="",
    a11_Analysis_A11_log_level=2,
    a11_d_T=7.516e-3,
    a11_S_L=51.44,
    a11_T_b=1843.5,
    a11_show_plots=False,
    a11_plot_CST_detJ=True,
    a11_plot_CST_selection=True,
    a11_Convert_to_grayscale_image=True,

    # plotter_1 args
    p1_output_dir_manual="",
    p1_output_dir_comment="",
    p1_video=1,
    p1_Plot_log_level=1,

    # plotter_4_dimentionalisation args
    p4_output_dir_manual="",
    p4_output_dir_comment="",
    p4_show_plot=0,
    p4_Plot_log_level=1,
    p4_Panel_1_A11=0,
    p4_A11_manual_data_base_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction",
    p4_Panel_2_Dimentionalised_from_VisIt=1,

    # plotter_2_CPvsA11 args
    p2_output_dir_manual="",
    p2_output_dir_comment="",
    p2_video=1,
    p2_Plot_log_level=1,

    # plotter_3_CPvsA11_Panel args
    p3_output_dir_manual="",
    p3_output_dir_comment="",
    p3_video=0,
    p3_show_plot=0,
    p3_Plot_log_level=1,
    p3_Panel_1=0,
    p3_Panel_2=0,
    p3_Panel_3=0,
    p3_Panel_4=1,

    # plotter_6_colortables args
    p6c_output_dir_manual="",
    p6c_output_dir_comment="",
    p6c_show_plot=0,
    p6c_Plot_log_level=1,
    p6c_image_width_ratio=0.5,
    p6c_plot_width_ratio=0.5,
    p6c_plot_spacing=0.0,
    p6c_colorbar_width=0.1,
    p6c_colorbar_height=0.6,
    p6c_colorbar_x_pos=0.1,
    p6c_ScaleFactor=1.5,
    p6c_figsize=(18, 6),
    p6c_FontSizeFactor_Legends=1.4,
    p6c_FontSizeFactor_Axis=1.0,
    p6c_Legend_y_offset=1.3,
    p6c_dpi=100,
    p6c_save_fig=True,
    p6c_video=True,

    # Control flags for pipeline sections
    run_visit_projector=True,
    run_cp_segment=True,
    run_cp_extract=True,
    run_analysis_a11=True,
    run_plotter_6_colortables=True,
    run_plotter_1=True,
    run_plotter_4=True,
    run_plotter_2=True,
    run_plotter_3_panel=True,
):
    """
    Runs the enhanced CIPS pipeline with unified Analysis_Altantzis2011 module.
    
    This new version replaces the separate dimensionalization and spherical reconstruction
    steps with the integrated Analysis_Altantzis2011 pipeline.
    
    Parameters
    ----------
    run_analysis_a11 : bool, optional
        If True, runs the new Analysis_Altantzis2011 pipeline which combines
        dimensionalization, spherical reconstruction, and CST selection.
        Default is True.
    a11_output_dir_manual : str, optional
        Manual output directory for Analysis_Altantzis2011. Default is "".
    a11_output_dir_comment : str, optional
        Output directory comment for Analysis_Altantzis2011. Default is "".
    a11_Analysis_A11_log_level : int, optional
        Log level for Analysis_Altantzis2011. Default is 2.
    a11_d_T : float, optional
        Flame thickness in meters. Default is 7.516e-3.
    a11_S_L : float, optional
        Laminar flame speed in m/s. Default is 51.44.
    a11_T_b : float, optional
        Burned gas temperature in K. Default is 1843.5.
    a11_show_plots : bool, optional
        Whether to display plots during processing. Default is False.
    a11_plot_CST_detJ : bool, optional
        Whether to generate CST boundary with det(J) plot. Default is False.
    a11_plot_CST_selection : bool, optional
        Whether to generate CST selection plots. Default is True.
    a11_Convert_to_grayscale_image : bool, optional
        Whether to convert RGB images to grayscale for CST plots. Default is True.
    cips_A11_output_dir_override : str, optional
        If run_analysis_a11 is False, use this path as Analysis_Altantzis2011 output.
        Default is "".
    
    Returns
    -------
    list
        A list containing the output directories from each step in the pipeline.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    temp_log_file_handle = None
    temp_log_file_path = None 
    pipeline_output_directory_for_log = None
    
    stdout_tee = None
    stderr_tee = None
    
    results = {} # Initialize results to ensure it's always defined

    try:
        temp_log_file_handle = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_cips_pipe_log.txt', encoding='utf-8')
        temp_log_file_path = temp_log_file_handle.name
        
        stdout_tee = Tee(temp_log_file_handle, original_stdout)
        stderr_tee = Tee(temp_log_file_handle, original_stderr) 
        
        sys.stdout = stdout_tee
        sys.stderr = stderr_tee

        try:
            # Override individual log levels with global level if set
            if cips_pipeline_global_log_level is not None:
                vp_Visit_projector_1_log_level = cips_pipeline_global_log_level
                cps_CP_segment_log_level = cips_pipeline_global_log_level
                cpe_CP_extract_log_level = cips_pipeline_global_log_level
                a11_Analysis_A11_log_level = cips_pipeline_global_log_level
                p1_Plot_log_level = cips_pipeline_global_log_level
                p2_Plot_log_level = cips_pipeline_global_log_level
                p3_Plot_log_level = cips_pipeline_global_log_level
                p4_Plot_log_level = cips_pipeline_global_log_level
                p6c_Plot_log_level = cips_pipeline_global_log_level

            #################################################### I/O
            cips_pipeline_output_dir = F_1.F_out_dir(
                input_dir=input_dir,
                script_path=__file__,
                output_dir_comment=cips_pipeline_output_dir_comment,
                output_dir_manual=cips_pipeline_output_dir_manual
            ) 
            pipeline_output_directory_for_log = cips_pipeline_output_dir 

            # Initialize output directory variables
            VP1_output_dir = None
            CPs1_output_dir = None
            CPe1_output_dir = None
            A11_output_dir = None
            p6c_output_dir = None

            #########################################        Visit Projector
            if not vp_input_dir:
                vp_input_dir = cips_pipeline_output_dir 

            if run_visit_projector:
                print(f"--- Running Visit_Projector_1 ---")
                VP1_output_dir = VP1.Visit_projector_1(
                    input_dir=vp_input_dir,
                    Database=vp_Database,
                    State_range_manual=vp_State_range_manual,
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
                    no_annotations=vp_no_annotations,
                    viewNormal=vp_viewNormal,
                    viewUp=vp_viewUp,
                    imageZoom=vp_imageZoom,
                    parallelScale=vp_parallelScale,
                    perspective=vp_perspective,
                    Visit_projector_1_log_level=vp_Visit_projector_1_log_level,
                    Visit_projector_1_show_windows=vp_Visit_projector_1_show_windows,
                    WindowWidth=vp_WindowWidth,
                    WindowHeight=vp_WindowHeight,
                    output_dir_manual=vp_output_dir_manual,
                    output_dir_comment=vp_output_dir_comment,
                )
            else:
                if cips_VP1_output_dir_override and os.path.isdir(cips_VP1_output_dir_override):
                    VP1_output_dir = cips_VP1_output_dir_override
                    print(f"--- Skipping Visit_Projector_1 --- Using provided output directory: {VP1_output_dir}")
                else:
                    print(f"--- Skipping Visit_Projector_1 --- No valid override directory provided.")
            
            #########################################        Cellpose Segmentation
            if run_cp_segment:
                if VP1_output_dir: 
                    print(f"--- Running CP_segment_1 ---")
                    CPs1_output_dir = CPs1.CP_segment_1(
                        input_dir=VP1_output_dir,
                        CP_model_type_for_segmentation=cps_CP_model_type_for_segmentation, 
                        CP_model_type_for_diameter_estimation=cps_CP_model_type_for_diameter_estimation,
                        gpu=cps_gpu,
                        CP_empty_cache_onoff=cps_CP_empty_cache_onoff, 
                        diameter_estimate_guess_px=cps_diameter_estimate_guess_px,
                        channels=cps_channels,
                        flow_threshold=cps_flow_threshold,
                        cellprob_threshold=cps_cellprob_threshold,
                        resample=cps_resample,
                        niter=cps_niter,
                        batch_size=cps_batch_size,
                        augment=cps_augment,
                        tile_overlap=cps_tile_overlap,
                        bsize=cps_bsize,
                        max_images_per_batch=cps_max_images_per_batch,
                        CP_default_plot_onoff=cps_CP_default_plot_onoff,
                        CP_default_image_onoff=cps_CP_default_image_onoff,
                        CP_default_seg_file_onoff=cps_CP_default_seg_file_onoff,
                        output_dir_manual=cps_output_dir_manual,
                        output_dir_comment=cps_output_dir_comment,
                        CP_segment_log_level=cps_CP_segment_log_level,
                    )
                else:
                    print("--- Skipping CP_segment_1 (missing Visit_Projector_1 output) ---")
            else:
                if cips_CPs1_output_dir_override and os.path.isdir(cips_CPs1_output_dir_override):
                    CPs1_output_dir = cips_CPs1_output_dir_override
                    print(f"--- Skipping CP_segment_1 --- Using provided output directory: {CPs1_output_dir}")
                else:
                    print(f"--- Skipping CP_segment_1 --- No valid override directory provided.")

            #########################################        Cellpose Extract
            if run_cp_extract:
                if CPs1_output_dir: 
                    print(f"--- Running CP_extract_1 ---")
                    CPe1_output_dir = CPe1.CP_extract_1(
                        input_dir=CPs1_output_dir,
                        CP_extract_log_level=cpe_CP_extract_log_level,
                        output_dir_manual=cpe_output_dir_manual, 
                        output_dir_comment=cpe_output_dir_comment, 
                    )
                else:
                    print("--- Skipping CP_extract_1 (missing CP_segment_1 output) ---")
            else:
                if cips_CPe1_output_dir_override and os.path.isdir(cips_CPe1_output_dir_override):
                    CPe1_output_dir = cips_CPe1_output_dir_override
                    print(f"--- Skipping CP_extract_1 --- Using provided output directory: {CPe1_output_dir}")
                else:
                    print(f"--- Skipping CP_extract_1 --- No valid override directory provided.")

            #########################################        Analysis Altantzis 2011 (Unified Analysis Pipeline)
            if run_analysis_a11:
                if CPe1_output_dir: 
                    print(f"--- Running Analysis_Altantzis2011 ---")
                    A11_output_dir = A11.Analysis_Altantzis2011(
                        input_dir=CPe1_output_dir,
                        Analysis_A11_log_level=a11_Analysis_A11_log_level,
                        output_dir_manual=a11_output_dir_manual, 
                        output_dir_comment=a11_output_dir_comment,
                        d_T=a11_d_T,
                        S_L=a11_S_L,
                        T_b=a11_T_b,
                        show_plots=a11_show_plots,
                        plot_CST_detJ=a11_plot_CST_detJ,
                        plot_CST_selection=a11_plot_CST_selection,
                        Convert_to_grayscale_image=a11_Convert_to_grayscale_image
                    )
                else:
                    print("--- Skipping Analysis_Altantzis2011 (missing CP_extract_1 output) ---")
            else:
                if cips_A11_output_dir_override and os.path.isdir(cips_A11_output_dir_override):
                    A11_output_dir = cips_A11_output_dir_override
                    print(f"--- Skipping Analysis_Altantzis2011 --- Using provided output directory: {A11_output_dir}")
                else:
                    print(f"--- Skipping Analysis_Altantzis2011 --- No valid override directory provided.")

            # Use A11_output_dir for downstream processing
            plot_input_dir = A11_output_dir

            #########################################        Plotting
            if run_plotter_1:
                if plot_input_dir:
                    print(f"--- Running plotter_1 ---")
                    p1_output_dir = p1.plotter_1(
                        input_dir=plot_input_dir,
                        output_dir_manual=p1_output_dir_manual,
                        output_dir_comment=p1_output_dir_comment,
                        video=p1_video,
                        Plot_log_level=p1_Plot_log_level
                    )
                else:
                    print("--- Skipping plotter_1 (missing analysis output) ---")
            else:
                print("--- Skipping plotter_1 ---")

            if run_plotter_4:
                if plot_input_dir:
                    print(f"--- Running plotter_4_dimentionalisation ---")
                    p4_output_dir = p4.plotter_4_dimentionalisation(
                        input_dir=plot_input_dir,
                        output_dir_manual=p4_output_dir_manual,
                        output_dir_comment=p4_output_dir_comment,
                        show_plot=p4_show_plot,
                        Plot_log_level=p4_Plot_log_level,
                        Panel_1_A11=p4_Panel_1_A11,
                        A11_manual_data_base_dir=p4_A11_manual_data_base_dir,
                        Panel_2_Dimentionalised_from_VisIt=p4_Panel_2_Dimentionalised_from_VisIt,
                    )
                else:
                    print("--- Skipping plotter_4_dimentionalisation (missing analysis output) ---")
            else:
                print("--- Skipping plotter_4_dimentionalisation ---")

            if run_plotter_2:
                if plot_input_dir:
                    print(f"--- Running plotter_2_CPvsA11 ---")
                    p2_output_dir = p2.plotter_2_CPvsA11(
                        input_dir=plot_input_dir,
                        output_dir_manual=p2_output_dir_manual,
                        output_dir_comment=p2_output_dir_comment,
                        video=p2_video,
                        Plot_log_level=p2_Plot_log_level
                    )
                else:
                    print("--- Skipping plotter_2_CPvsA11 (missing analysis output) ---")
            else:
                print("--- Skipping plotter_2_CPvsA11 ---")

            if run_plotter_3_panel:
                if plot_input_dir:
                    print(f"--- Running plotter_3_CPvsA11_Panel ---")
                    p3_out_dir = p3_panel.plotter_3_CPvsA11_Panel(
                        input_dir=plot_input_dir,
                        output_dir_manual=p3_output_dir_manual,
                        output_dir_comment=p3_output_dir_comment,
                        video=p3_video,
                        show_plot=p3_show_plot,
                        Plot_log_level=p3_Plot_log_level,
                        Panel_1=p3_Panel_1,
                        Panel_2=p3_Panel_2,
                        Panel_3=p3_Panel_3,
                        Panel_4=p3_Panel_4,
                    )
                else:
                    print("--- Skipping plotter_3_CPvsA11_Panel (missing analysis output) ---")
            else:
                print("--- Skipping plotter_3_CPvsA11_Panel ---")

            #########################################        Color Table Plotting
            if run_plotter_6_colortables:
                if CPe1_output_dir:
                    print(f"--- Running plotter_6_colortables ---")
                    p6c_output_dir = p6c.plotter_6_colortables(
                        input_dir=CPe1_output_dir,
                        output_dir_manual=p6c_output_dir_manual,
                        output_dir_comment=p6c_output_dir_comment,
                        show_plot=p6c_show_plot,
                        Plot_log_level=p6c_Plot_log_level,
                        image_width_ratio=p6c_image_width_ratio,
                        plot_width_ratio=p6c_plot_width_ratio,
                        plot_spacing=p6c_plot_spacing,
                        colorbar_width=p6c_colorbar_width,
                        colorbar_height=p6c_colorbar_height,
                        colorbar_x_pos=p6c_colorbar_x_pos,
                        ScaleFactor=p6c_ScaleFactor,
                        figsize=p6c_figsize,
                        FontSizeFactor_Legends=p6c_FontSizeFactor_Legends,
                        FontSizeFactor_Axis=p6c_FontSizeFactor_Axis,
                        Legend_y_offset=p6c_Legend_y_offset,
                        dpi=p6c_dpi,
                        save_fig=p6c_save_fig,
                        video=p6c_video
                    )
                else:
                    print("--- Skipping plotter_6_colortables (missing CP_extract_1 output) ---")
            else:
                print("--- Skipping plotter_6_colortables ---")

            ######################################################## end
            F_1.ding()

            # Results dictionary to be returned
            results_dict = {
                "cips_pipeline_output_dir": cips_pipeline_output_dir,
                "VP1_output_dir": VP1_output_dir,
                "CPs1_output_dir": CPs1_output_dir,
                "CPe1_output_dir": CPe1_output_dir,
                "A11_output_dir": A11_output_dir,  # New unified analysis output
                "p6c_output_dir": p6c_output_dir,
                "plot_input_dir": plot_input_dir
            }
            results = list(results_dict.values())

        except Exception as e:
            print(f"\n!!! EXCEPTION OCCURRED IN CIPS_pipeline_2 !!!", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise

    finally:
        # Ensure streams are flushed before restoring
        if sys.stdout is stdout_tee and stdout_tee is not None: 
            sys.stdout.flush()
        if sys.stderr is stderr_tee and stderr_tee is not None: 
            sys.stderr.flush()

        # Restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        if temp_log_file_handle:
            temp_log_file_handle.close() 

            if pipeline_output_directory_for_log and os.path.isdir(pipeline_output_directory_for_log):
                dir_name_for_log_prefix = os.path.basename(pipeline_output_directory_for_log)
                log_file_name = f"{dir_name_for_log_prefix}_CIPS_Pipe_terminal_output.txt"
                final_log_path = os.path.join(pipeline_output_directory_for_log, log_file_name)
                try:
                    shutil.move(temp_log_file_path, final_log_path)
                    print(f"CIPS_pipeline_2 terminal output saved to: {final_log_path}")
                except Exception as e_move:
                    print(f"Error moving CIPS_pipeline_2 log: {e_move}")
                    print(f"Temporary log file remains at: {temp_log_file_path}")
            elif temp_log_file_path and os.path.exists(temp_log_file_path):
                print(f"CIPS_pipeline_2 output directory not determined. Log remains at: {temp_log_file_path}")
        elif temp_log_file_path and os.path.exists(temp_log_file_path): 
             print(f"Temporary log file may exist at: {temp_log_file_path} but was not properly closed or moved.")
        else:
            print("Failed to initialize or use temporary log file for CIPS_pipeline_2.")

    return results

# Example of how to run the pipeline with the new unified approach
if __name__ == "__main__":
    print("Running CIPS-Pipeline 2 with unified Analysis_Altantzis2011 approach")
    
    CIPS_pipeline_2(
        cips_pipeline_global_log_level=2, 
        cips_pipeline_output_dir_comment="T3_vmag_3000px_WWBBWW_cyto3_flowThres0p5",
        
        # Visit_Projector parameters
        
        # Cellpose segmentation parameters
        cps_max_images_per_batch=40,
        cps_batch_size=2,
        cps_augment=True,
        cps_tile_overlap=0.1,
        cps_bsize=160,
        cps_output_dir_comment="",
        
        # Cellpose extract parameters
        cpe_output_dir_comment="",
        cpe_CP_extract_log_level=2,	

        # Analysis Altantzis2011 parameters
        a11_output_dir_comment="",
        a11_Analysis_A11_log_level=2,
        a11_show_plots=False,
        a11_plot_CST_detJ=True,
        a11_plot_CST_selection=True,
        
        # Run all pipeline steps
        run_visit_projector=True, #cips_VP1_output_dir_override=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250610_0004544\20250610_0004569",
        run_cp_segment=True, #cips_CPs1_output_dir_override=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250618_1754539\20250618_1754549\20250618_1756011",
        run_cp_extract=True,
        run_analysis_a11=True, #cips_A11_output_dir_override=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250618_1754539\20250618_1754549\20250618_1756011\20250618_2359067\20250618_2359162",
        run_plotter_1=True,
        run_plotter_4=True,
        run_plotter_2=True,
        run_plotter_3_panel=True,
        run_plotter_6_colortables=True,
    )
    
    print("CIPS-Pipeline 2 run finished.")
