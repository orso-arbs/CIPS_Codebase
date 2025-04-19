import sys
import os
import Format_1 as F_1

import Visit_Projector_1 as VP1
import CP_segment_1 as CPs1
import CP_extract_1 as CPe1
import CP_plotter_1 as CPp1
import CP_plotter_2_CPvsA11 as CPp2
import CP_plotter_3_CPvsA11_Panel as CPp3_panel


start_time, current_date = F_1.start_inform(__file__)



#########################################        Visit




# Visit
if 1==0:
    # A11 single timedumps #50 (if i remember correctly it's 50)
    #Database = r"euler.ethz.ch:/cluster/scratch/cfrouzak/spher_H2/postProc/fields/po_part2/po_s912k_post.nek5000"
    # A11 fist 20 timedumps
    Database = r"euler.ethz.ch:/cluster/scratch/orsob/MastersThesis/postProc/po_part1/po_s912k_post.nek5000"

    VP1_output_dir = VP1.Visit_projector_1(
        input_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\SF_CP_analysis_pipeline_data", # storage for this script
        Database = Database, State_range_manual = [],
        Plots = ["Pseudocolor-velocity_magnitude Isosurface-temperature colorTableName-CustomBW"],
        no_annotations = 1, viewNormal = [0,0,-1], viewUp = [1,0,0], imageZoom = 1, parallelScale = 20, perspective = 0,
        Visit_projector_1_log_level = 1, Visit_projector_1_show_windows = 0,
        output_dir_manual = "", output_dir_comment = "testing around",
    )

    print("Note: Visit window can now be closed. 'VisIt: Error - Can't delete the last window' is now inconsequentioal to the remaining code")




#########################################        Cellpose


# CP_segment_1
if 1==1:
    #visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop"
    #visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop small"
    #visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop small two only"
    #visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop Small First few"
    #visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\A11 FB poster selection"
    #visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\SF_CP_analysis_pipeline_data\Visit_Projector_1_2025-04-16_15-27-50_isoT3colVMag_orthogonal_standardView_CustomBW_0w-0p4w-0p6b-1b"



    #2 images for testing
    #visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\SF_CP_analysis_pipeline_data\Visit_Projector_1_2025-04-18_12-15-15_testing_around"

    # 20 images for testing
    visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\SF_CP_analysis_pipeline_data\Visit_Projector_1_2025-04-19_13-27-49_testing_around"
    
    #visit_images_dir = VP1_output_dir

    CP_model_type = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Python Code\msc python cellpose\CP Models"
    CP_model_type = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Python Code\msc python cellpose\CP Models\ZGX model_UoB"
    CP_model_type = "cyto3"

    CPs1_output_dir = CPs1.CP_segment_1(
        input_dir = visit_images_dir,
        CP_model_type = CP_model_type,
        gpu = True,
        diameter_estimate_guess = None, # must define for custom model. otherwise set to 0 or None for Cellpose diameter estimate from styles vector
        output_dir_comment = "cyto3",                     
        CP_segment_log_level = 1,
        )

# CP_extract_1
if 1==1:
    # BW 134 ball flame - Crop
    #CPs1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_segment_1_2025-03-12_13-42-11"
    
    # BW 134 ball flame - Crop Small First few
    #CPs1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop Small First few\CP_segment_1_2025-03-10_15-13-26"

    # A11 FB poster selection
    #CPs1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\A11 FB poster selection\CP_segment_1_2025-03-13_17-49-00"

    CPe1_output_dir = CPe1.CP_extract_1(
        input_dir = CPs1_output_dir,
        #masks = masks, flows = flows, styles = styles, diameter_estimate_used = diameter_estimate_used, CP_model_type = CP_model_type,
        CP_extract_log_level = -10,
        diameter_training_px = 30, # define for custom model
        )



#########################################        Plot




if 1==1: # video to evaluate CP segmentation settings and extracted results
    # \BW 134 ball flame - Crop
    #CPe1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_segment_1_2025-03-12_13-42-11\CP_extract_1_2025-03-12_13-48-48"
    
    #CPe1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop Small First few\CP_segment_1_2025-03-10_15-13-26\CP_extract_1_2025-03-10_15-21-44"

    # A11 FB poster selection
    #CPe1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\A11 FB poster selection\CP_segment_1_2025-03-13_15-47-30\CP_extract_1_2025-03-13_15-48-12"


    CPp1_output_dir = CPp1.CP_plotter_1(
        input_dir = CPe1_output_dir,
        output_dir_manual = "", output_dir_comment = "",
        video = 1
        )


if 1==0: # video comparing CP A11
    #CPe1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_segment_1_2025-03-12_13-42-11\CP_extract_1_2025-03-12_13-48-48"

    CPp2_output_dir = CPp2.CP_plotter_2_CPvsA11(
        input_dir = CPe1_output_dir,
        output_dir_manual = "", output_dir_comment = "",
        video = 1
        )


if 1==0: # panel comparing CP A11
    # \BW 134 ball flame - Crop
    #CPe1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_segment_1_2025-03-12_13-42-11\CP_extract_1_2025-03-12_13-48-48"

    # BW 134 ball flame - Crop Small First few
    #CPe1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop Small First few\CP_segment_1_2025-03-10_15-13-26\CP_extract_1_2025-03-11_12-30-00"

    CPp2_output_dir = CPp3_panel.CP_plotter_3_CPvsA11_Panel(
        input_dir = CPe1_output_dir,
        output_dir_manual = "", output_dir_comment = "",
        video = 0, show_plot = 0,
        Panel_1 = 0, # 
        Panel_2 = 0, # check non Dimentionalisation
        Panel_3 = 0, # 
        Panel_4 = 1, # panel comparing CP A11
        )








######################################################## end 
F_1.end_inform(__file__, start_time)