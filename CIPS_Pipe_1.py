import sys
import os
import Format_1 as F_1
from Format_1 import Tee # Import Tee class
import shutil
import tempfile
import traceback 

import Visit_Projector_1 as VP1
import CP_segment_1 as CPs1
import CP_extract_1 as CPe1
import dim1_manual_1 as d1
import dim2_VisIt_R_1 as d2
import plot1 as p1
import plot2_CPvsA11 as p2
import plot3_CPvsA11_Panel as p3_panel
import plot4_dimentions as p4
import plot6_colortables as p6c # Import for the new plotter_6_colortables
import Spherical_Reconstruction_1 as SR1  # Import the new spherical reconstruction module


@F_1.ParameterLog(max_size = 1024 * 10) # 10KB 
def CIPS_pipeline(
    # General control
    input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir",
    cips_pipeline_output_dir_manual="",
    cips_pipeline_output_dir_comment="", 
    cips_pipeline_global_log_level=None,  # Added global log level parameter

    # Stage output overrides (for resuming pipeline)
    cips_VP1_output_dir_override="",
    cips_CPs1_output_dir_override="",
    cips_CPe1_output_dir_override="",
    cips_d2_output_dir_override="",
    cips_SR1_output_dir_override="",  # New override for spherical reconstruction stage

    # Visit_projector_1 args
    vp_input_dir="",
    vp_Database=r"euler.ethz.ch:/cluster/scratch/orsob/orsoMT_orsob/A11_states/A11_all_states.visit",
    vp_State_range_manual=[],
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
    vp_Pseudocolor_colortable="PointWise", # Can be "hot", "CustomBW1", "CustomBW2", "PeriodicBW", "PointWise", etc.
    vp_invertColorTable=0,
    # Parameters for the periodic black and white color table
    Pseudocolor_periodic_num_periods = 3,   # periods of w-w-b-b points (4 points)
    distance_ww = 2.0,          # Relative length of solid white
    distance_wb = 1.0,          # Relative length of white-to-black gradient
    distance_bb = 2.0,          # Relative length of solid black
    distance_bw = 1.0,          # Relative length of black-to-white gradient
    # Parameters for the pointwise color table
    pointwise_color_points = [ # List of [position, r, g, b, a] points for PointWise color table
        [0.0, 0, 0, 0, 255], # Black
        [0.3, 0, 0, 0, 255], # Black
        [0.7, 255, 255, 255, 255],  # White
        [1.0, 255, 255, 255, 255],  # White
        ],  
    show_color_table_markers = True,  # Whether to show position markers and labels in color table preview
    
    vp_Isosurface_Variable="temperature",
    vp_Isosurface_ContourValue=3,
    vp_no_annotations=1,
    vp_viewNormal=[0, -1, 0],
    vp_viewUp=[0, 0, 1],
    vp_imageZoom=1,
    vp_parallelScale=80,
    vp_perspective=0,
    vp_WindowWidth = 2000, # Window size in px
    vp_WindowHeight = 2000, # Window size in px
    vp_Visit_projector_1_log_level=2,
    vp_Visit_projector_1_show_windows=0,
    vp_output_dir_manual="",
    vp_output_dir_comment="",

    # CP_segment_1 args
    cps_CP_model_type_for_segmentation="cyto3", 
    cps_CP_model_type_for_diameter_estimation="cyto3", # New pipeline parameter
    cps_gpu=True,
    cps_CP_empty_cache_onoff=True, 
    cps_diameter_estimate_guess_px=None,
    cps_channels=[0,0], 
    cps_flow_threshold=0.5, 
    cps_cellprob_threshold=0.0, 
    cps_resample=True, 
    cps_niter=0,
    cps_batch_size=4,
    cps_augment=True, # New
    cps_tile_overlap=0.1, # New
    cps_bsize=160, # Stick to multiples of 16. Cellpose uses 224 by default.
    cps_max_images_per_batch=40,  # New parameter for manual batch size
    cps_CP_default_plot_onoff=0, 
    cps_CP_default_image_onoff=0, 
    cps_CP_default_seg_file_onoff=1,
    cps_output_dir_manual="",
    cps_output_dir_comment="",
    cps_CP_segment_log_level=2,

    # CP_extract_1 args
    cpe_output_dir_manual="",
    cpe_output_dir_comment="",
    cpe_CP_extract_log_level=1,

    # dimentionalise_2_from_VisIt_R_Average args
    d2_output_dir_manual="",
    d2_output_dir_comment="",    
    d2_CP_dimentionalise_log_level=1,

    # Spherical_Reconstruction_1 args
    sr_output_dir_manual="",
    sr_output_dir_comment="",
    sr_Spherical_Reconstruction_log_level=2,
    sr_show_plots=False,
    sr_plot_CST_detJ=False,
    
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
    p6c_image_width_ratio=0.5,     # Width ratio for combined image subplot
    p6c_plot_width_ratio=0.5,      # Width ratio for property plot subplot
    p6c_plot_spacing=0.0,          # Horizontal spacing between plots
    p6c_colorbar_width=0.1,        # Width of colorbar relative to subplot width
    p6c_colorbar_height=0.6,       # Height of colorbar relative to subplot height
    p6c_colorbar_x_pos=0.1,        # X position of colorbar relative to subplot width
    p6c_ScaleFactor=1.5,           # Scale factor for zooming in on the spherical flame
    p6c_figsize=(18, 6),           # Figure size (width, height) in inches
    p6c_FontSizeFactor_Legends=1.4, # Factor to adjust font size for legends
    p6c_FontSizeFactor_Axis=1.0,   # Factor to adjust font size for axes labels
    p6c_Legend_y_offset=1.3,       # Y offset for legends
    p6c_dpi=100,
    p6c_save_fig=True,
    p6c_video=False,

    # Control flags for pipeline sections
    run_visit_projector=True,
    run_cp_segment=True,
    run_plotter_6_colortables=True,
    run_cp_extract=True,
    run_dimentionalise=True,
    run_spherical_reconstruction=True,  # New control flag
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
        Default is None (components use their individual log levels).
    pointwise_color_points : list of lists, optional
        Points defining the PointWise color table. Each point is a list [position, r, g, b, a].
        position should be a float between 0.0 and 1.0
        r, g, b, a are color values between 0-255
        Required if vp_Pseudocolor_colortable="PointWise". Default is None.
    show_color_table_markers : bool, optional
        Whether to show position markers and labels in color table preview images. Default is True.
    cps_CP_model_type_for_segmentation : str, optional # Existing parameter, ensure description is up-to-date
        Specifies the Cellpose model to use for the main segmentation task in CP_segment_1.
        Default is "cyto3".
    cps_CP_model_type_for_diameter_estimation : str, optional # New parameter docstring
        Specifies the Cellpose model to use for the initial diameter estimation step in CP_segment_1.
        Default is "cyto3".
    cps_channels : list, optional
        Channels for CP_segment_1. Default is [0,0].
    cps_flow_threshold : float, optional
        Flow threshold for CP_segment_1. Default is 0.7.
    cps_cellprob_threshold : float, optional
        Cell probability threshold for CP_segment_1. Default is 0.0.
    cps_resample : bool, optional
        Resample flag for CP_segment_1. Default is True.
    cps_niter : int, optional
        Number of iterations for CP_segment_1. Default is 0.
    cps_CP_empty_cache_onoff : bool, optional
        If True, clears CUDA GPU memory before starting Cellpose segmentation. Default is True.
    cps_tile : bool, optional
        Whether to run Cellpose on tiles of the image. Default is True.
    cps_tile_overlap : float, optional
        Fraction of overlap between tiles if cps_tile is True. Default is 0.1.
    cps_bsize : int, optional
        Size of tiles in pixels if cps_tile is True. Default is 224.
    cps_max_images_per_batch : int or None, optional
        Maximum number of images to process in a single batch for CP_segment_1.
        If None, all images are processed in one batch. If the number of images
        exceeds this value, they will be processed in multiple batches.
        This helps manage memory usage for large datasets. Default is None.
    cps_CP_default_plot_onoff : int, optional
        Default plot on/off for CP_segment_1. Default is 1.
    cps_CP_default_image_onoff : int, optional
        Default image on/off for CP_segment_1. Default is 1.
    cps_CP_default_seg_file_onoff : int, optional
        Default segmentation file on/off for CP_segment_1. Default is 1.
    cps_output_dir_manual : str, optional
        Manual output directory for CP_segment_1. Default is "".
    cpe_output_dir_manual : str, optional
        Manual output directory for CP_extract_1. Default is "".
    cpe_output_dir_comment : str, optional
        Output directory comment for CP_extract_1. Default is "".
    d2_output_dir_manual : str, optional
        Manual output directory for dimentionalise_2_from_VisIt_R_Average. Default is "".
    sr_output_dir_manual : str, optional
        Manual output directory for Spherical_Reconstruction_1. Default is "".
    sr_output_dir_comment : str, optional
        Output directory comment for Spherical_Reconstruction_1. Default is "".
    p1_Plot_log_level : int, optional
        Log level for plotter_1. Default is 1.
    p2_Plot_log_level : int, optional
        Log level for plotter_2_CPvsA11. Default is 1.
    p3_Plot_log_level : int, optional
        Log level for plotter_3_CPvsA11_Panel. Default is 1.
    # Parameters for plotter_6_colortables
    p6c_output_dir_manual : str, optional
        Manual output directory for plotter_6_colortables. Default is "".
    p6c_output_dir_comment : str, optional
        Output directory comment for plotter_6_colortables. Default is "".
    p6c_show_plot : int, optional
        Whether to display the plot (1) or not (0) for plotter_6_colortables. Default is 0.
    p6c_Plot_log_level : int, optional
        Logging level for plotter_6_colortables. Default is 1.
    p6c_image_width_ratio : float, optional
        Width ratio for the image subplot in plotter_6_colortables. Default is 0.45.
    p6c_plot_width_ratio : float, optional
        Width ratio for the property plot subplot in plotter_6_colortables. Default is 0.45.
    p6c_plot_spacing : float, optional
        Horizontal spacing between plots in plotter_6_colortables. Default is 0.0.
    p6c_colorbar_width : float, optional
        Width of colorbar relative to subplot width in plotter_6_colortables. Default is 0.1.
    p6c_colorbar_height : float, optional
        Height of colorbar relative to subplot height in plotter_6_colortables. Default is 0.6.
    p6c_colorbar_x_pos : float, optional
        X position of colorbar relative to subplot width in plotter_6_colortables. Default is 0.1.
    p6c_ScaleFactor : float, optional
        Scale factor for zooming in on the spherical flame in plotter_6_colortables. Default is 1.5.
    p6c_figsize : tuple, optional
        Figure size (width, height) in inches for plotter_6_colortables. Default is (18, 6).
    p6c_FontSizeFactor_Legends : float, optional
        Factor to adjust font size for legends in plotter_6_colortables. Default is 1.4.
    p6c_FontSizeFactor_Axis : float, optional
        Factor to adjust font size for axes labels in plotter_6_colortables. Default is 1.0.
    p6c_Legend_y_offset : float, optional
        Y offset for legends in plotter_6_colortables. Default is 1.3.
    p6c_dpi : int, optional
        DPI for the figure in plotter_6_colortables. Default is 100.
    p6c_save_fig : bool, optional
        Whether to save the figure in plotter_6_colortables. Default is True.
    p6c_video : bool, optional
        Whether to create a video of the plots in plotter_6_colortables. Default is False.
    # Stage output overrides
    cips_VP1_output_dir_override : str, optional
        If run_visit_projector is False, use this path as VP1.Visit_projector_1 output. Default is "".
    cips_CPs1_output_dir_override : str, optional
        If run_cp_segment is False, use this path as CPs1.CP_segment_1 output. Default is "".
    cips_CPe1_output_dir_override : str, optional
        If run_cp_extract is False, use this path as CPe1.CP_extract_1 output. Default is "".
    cips_d2_output_dir_override : str, optional
        If run_dimentionalise is False, use this path as d2.dimentionalise_2_from_VisIt_R_Average output. Default is "".
    cips_SR1_output_dir_override : str, optional
        If run_spherical_reconstruction is False, use this path as SR1.Spherical_Reconstruction_1 output. Default is "".
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

        # --- Start of original CIPS_pipeline logic, now within try...except block for logging ---
        try:
            # Override individual log levels with global level if set
            if cips_pipeline_global_log_level is not None:
                vp_Visit_projector_1_log_level = cips_pipeline_global_log_level
                cps_CP_segment_log_level = cips_pipeline_global_log_level
                cpe_CP_extract_log_level = cips_pipeline_global_log_level
                d2_CP_dimentionalise_log_level = cips_pipeline_global_log_level
                p1_Plot_log_level = cips_pipeline_global_log_level
                p2_Plot_log_level = cips_pipeline_global_log_level
                p3_Plot_log_level = cips_pipeline_global_log_level
                p4_Plot_log_level = cips_pipeline_global_log_level
                p6c_Plot_log_level = cips_pipeline_global_log_level # New

            #################################################### I/O
            cips_pipeline_output_dir = F_1.F_out_dir(input_dir = input_dir, script_path = __file__, output_dir_comment = cips_pipeline_output_dir_comment, output_dir_manual = cips_pipeline_output_dir_manual) 
            pipeline_output_directory_for_log = cips_pipeline_output_dir 

            # Initialize output directory variables
            VP1_output_dir = None
            CPs1_output_dir = None
            CPe1_output_dir = None
            d2_output_dir = None
            p6c_output_dir = None # New
            SR1_output_dir = None # New

            #########################################        Visit

            if not vp_input_dir:
                vp_input_dir = cips_pipeline_output_dir 

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
                    Visit_projector_1_log_level=vp_Visit_projector_1_log_level,
                    Visit_projector_1_show_windows=vp_Visit_projector_1_show_windows,
                    WindowWidth = vp_WindowWidth, WindowHeight = vp_WindowHeight,
                    output_dir_manual=vp_output_dir_manual,
                    output_dir_comment=vp_output_dir_comment,
                )
                print("Note: Visit window can now be closed. 'VisIt: Error - Can't delete the last window' is now inconsequential to the remaining code")
            else:
                if cips_VP1_output_dir_override and os.path.isdir(cips_VP1_output_dir_override):
                    VP1_output_dir = cips_VP1_output_dir_override
                    print(f"--- Skipping Visit_Projector_1 --- Using provided output directory: {VP1_output_dir}")
                else:
                    print(f"--- Skipping Visit_Projector_1 --- No valid override directory provided. Subsequent steps requiring its output may be skipped.")
            
            #########################################        Cellpose
            if run_cp_segment:
                if VP1_output_dir: 
                    print(f"--- Running CP_segment_1 ---")
                    CPs1_output_dir = CPs1.CP_segment_1(
                        input_dir=VP1_output_dir,
                        CP_model_type_for_segmentation=cps_CP_model_type_for_segmentation, 
                        CP_model_type_for_diameter_estimation=cps_CP_model_type_for_diameter_estimation, # Pass new parameter
                        gpu=cps_gpu,
                        CP_empty_cache_onoff=cps_CP_empty_cache_onoff, 
                        diameter_estimate_guess_px=cps_diameter_estimate_guess_px,
                        channels=cps_channels,
                        flow_threshold=cps_flow_threshold,
                        cellprob_threshold=cps_cellprob_threshold,
                        resample=cps_resample,
                        niter=cps_niter,
                        batch_size=cps_batch_size,
                        augment=cps_augment, # New
                        tile_overlap=cps_tile_overlap, # New
                        bsize=cps_bsize, # New
                        max_images_per_batch=cps_max_images_per_batch,  # New parameter
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
                    print(f"--- Skipping CP_segment_1 --- No valid override directory provided. Subsequent steps requiring its output may be skipped.")

            
            #########################################        Process Data
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
                    print(f"--- Skipping CP_extract_1 --- No valid override directory provided. Subsequent steps requiring its output may be skipped.")

            if run_dimentionalise:
                if CPe1_output_dir: 
                    print(f"--- Running dimentionalise_2_from_VisIt_R_Average ---")
                    d2_output_dir = d2.dimentionalise_2_from_VisIt_R_Average(
                        input_dir=CPe1_output_dir,
                        CP_dimentionalise_log_level=d2_CP_dimentionalise_log_level,
                        output_dir_manual=d2_output_dir_manual, 
                        output_dir_comment=d2_output_dir_comment,
                    )
                else:
                    print("--- Skipping dimentionalise_2_from_VisIt_R_Average (missing CP_extract_1 output) ---")
            else:
                if cips_d2_output_dir_override and os.path.isdir(cips_d2_output_dir_override):
                    d2_output_dir = cips_d2_output_dir_override
                    print(f"--- Skipping dimentionalise_2_from_VisIt_R_Average --- Using provided output directory: {d2_output_dir}")
                else:
                    print(f"--- Skipping dimentionalise_2_from_VisIt_R_Average --- No valid override directory provided. Subsequent steps requiring its output may be skipped.")

            #########################################        Spherical Reconstruction
            if run_spherical_reconstruction:
                if d2_output_dir:
                    print(f"--- Running Spherical_Reconstruction_1 ---")
                    SR1_output_dir = SR1.Spherical_Reconstruction_1(
                        input_dir=d2_output_dir,
                        Spherical_Reconstruction_log_level=sr_Spherical_Reconstruction_log_level,
                        output_dir_manual=sr_output_dir_manual,
                        output_dir_comment=sr_output_dir_comment,
                        show_plots=sr_show_plots,
                        plot_CST_detJ=sr_plot_CST_detJ,
                    )
                else:
                    print("--- Skipping Spherical_Reconstruction_1 (missing dimentionalisation output) ---")
            else:
                if cips_SR1_output_dir_override and os.path.isdir(cips_SR1_output_dir_override):
                    SR1_output_dir = cips_SR1_output_dir_override
                    print(f"--- Skipping Spherical_Reconstruction_1 --- Using provided output directory: {SR1_output_dir}")
                else:
                    print(f"--- Skipping Spherical_Reconstruction_1 --- No valid override directory provided.")
                    SR1_output_dir = None

            # Use SR1_output_dir for downstream processing if available, otherwise fall back to d2_output_dir
            plot_input_dir = SR1_output_dir if SR1_output_dir else d2_output_dir

            #########################################        Plot
            if run_plotter_1:
                if plot_input_dir:
                    print(f"--- Running plotter_1 ---")
                    p1_output_dir = p1.plotter_1(
                        input_dir=plot_input_dir,
                        output_dir_manual=p1_output_dir_manual, # Plotters usually create subfolders in their input_dir
                        output_dir_comment=p1_output_dir_comment,
                        video=p1_video,
                        Plot_log_level=p1_Plot_log_level # Added, assuming plotter_1 takes this
                    )
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
                        video=p2_video,
                        Plot_log_level=p2_Plot_log_level # Added, assuming plotter_2 takes this
                    )
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
                        Plot_log_level=p3_Plot_log_level, # Added, assuming plotter_3 takes this
                        Panel_1=p3_Panel_1,
                        Panel_2=p3_Panel_2,
                        Panel_3=p3_Panel_3,
                        Panel_4=p3_Panel_4,
                    )
                else:
                    print("--- Skipping plotter_3_CPvsA11_Panel (missing dimentionalisation output) ---")
            else:
                print("--- Skipping plotter_3_CPvsA11_Panel ---")

            #########################################        Plotter 6 Colortables
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
                    print("--- Skipping plotter_6_colortables (missing CP_segment_1 output) ---")
            else:
                print("--- Skipping plotter_6_colortables ---")

            ######################################################## end
            F_1.ding()

            # Results dictionary to be returned
            results_dict = {
                "cips_pipeline_output_dir": cips_pipeline_output_dir,
                "VP1_output_dir": VP1_output_dir,
                "CPs1_output_dir": CPs1_output_dir,
                "p6c_output_dir": p6c_output_dir,
                "CPe1_output_dir": CPe1_output_dir,
                "d2_output_dir": d2_output_dir,
                "SR1_output_dir": SR1_output_dir,  # Add SR1 output to results
                "plot_input_dir": plot_input_dir
            }
            results = list(results_dict.values())

        except Exception as e:
            print(f"\n!!! EXCEPTION OCCURRED IN CIPS_pipeline !!!", file=sys.stderr) # Will go to Tee (log and console)
            traceback.print_exc(file=sys.stderr) # Will go to Tee (log and console)
            raise # Re-raise the exception to be caught by the caller if any
        # --- End of original CIPS_pipeline logic ---

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
                    print(f"CIPS_pipeline terminal output saved to: {final_log_path}", file=original_stdout)
                except Exception as e_move:
                    print(f"Error moving CIPS_pipeline log from {temp_log_file_path} to {final_log_path}: {e_move}", file=original_stdout)
                    print(f"Temporary log file remains at: {temp_log_file_path}", file=original_stdout)
            elif temp_log_file_path and os.path.exists(temp_log_file_path):
                print(f"CIPS_pipeline output directory not determined or invalid. Terminal log remains at: {temp_log_file_path}", file=original_stdout)
        elif temp_log_file_path and os.path.exists(temp_log_file_path): 
             print(f"Temporary log file may exist at: {temp_log_file_path} but was not properly closed or moved.", file=original_stdout)
        else:
            print("Failed to initialize or use temporary log file for CIPS_pipeline.", file=original_stdout)

    return results # Return the collected results












# Example of how to run the pipeline with default settings
if __name__ == "__main__":
    print("Running CIPS-Pipeline.")
    
    # Example: Skip Visit_Projector, use its existing output, and run the rest
    CIPS_pipeline(
        cips_pipeline_global_log_level=None, # Example: Set global log level
        cips_pipeline_output_dir_comment="",
        run_visit_projector = False, 
        
        # 7 images
        #cips_VP1_output_dir_override = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_misc\VisIt_output_test_7_images",

        # WWBBWW                         
        cips_VP1_output_dir_override = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250610_0004544\20250610_0004569",
        
        # batch processing
        cps_max_images_per_batch=40,
        cps_batch_size=4,
        cps_augment=True, # New
        cps_tile_overlap=0.1, # New
        cps_bsize=160, # Stick to multiples of 16. Cellpose uses 224 by default.

        cps_output_dir_comment="2000px_manualbatch40_batchsize6_bsize160",
        run_cp_segment=True,
        run_cp_extract=True,
        run_dimentionalise=True,
        run_spherical_reconstruction=True,  # Run the new spherical reconstruction stage
        run_plotter_1=True,
        run_plotter_4=True,
        run_plotter_2=True,
        run_plotter_3_panel=True,
        run_plotter_6_colortables=True, # New
    )

    print("CIPS-Pipeline run finished.")


