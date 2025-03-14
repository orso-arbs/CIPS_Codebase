import sys
import os
sys.path.append(os.path.abspath(r"C:/Users/obs/OneDrive/ETH/ETH_MSc/Masters Thesis/Python Code/Python_Orso_Utility_Scripts_MscThesis")) # dir containing Format_1 
import Format_1 as F_1

import CP_segment_1 as CPs1
import CP_extract_1 as CPe1
import CP_plotter_1 as CPp1
import CP_plotter_2_CPvsA11 as CPp2
import CP_plotter_3_CPvsA11_Panel as CPp3_panel


start_time, current_date = F_1.start_inform(__file__)

########################################################

if 1==1: 
    visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop"
    #visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop small"
    #visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop small two only"
    #visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop Small First few"
    #visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\A11 FB poster selection"

    CPs1_output_dir, masks, flows, styles, diameter_estimate, CP_model_type = CPs1.CP_segment_1(
        input_dir = visit_images_dir,
        CP_model_type = "cyto3",
        gpu = True,
        CP_segment_log_level = 1,
        )


if 1==1:
    # BW 134 ball flame - Crop
    #CPs1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_segment_1_2025-03-12_13-42-11"
    
    # BW 134 ball flame - Crop Small First few
    #CPs1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop Small First few\CP_segment_1_2025-03-10_15-13-26"

    # A11 FB poster selection
    #CPs1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\A11 FB poster selection\CP_segment_1_2025-03-13_17-49-00"

    CPe1_output_dir, CP_extract_df = CPe1.CP_extract_1(
        input_dir = CPs1_output_dir,
        #masks = masks, flows = flows, styles = styles, diameter_estimate = diameter_estimate, CP_model_type = CP_model_type,
        CP_extract_log_level = 0,
        )



##### plotting

if 1==1: # video to evaluate CP segmentation settings and extracted results
    # \BW 134 ball flame - Crop
    #CPe1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_segment_1_2025-03-12_13-42-11\CP_extract_1_2025-03-12_13-48-48"
    
    #CPe1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop Small First few\CP_segment_1_2025-03-10_15-13-26\CP_extract_1_2025-03-10_15-21-44"
    #CP_extract_df = None

    # A11 FB poster selection
    #CPe1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\A11 FB poster selection\CP_segment_1_2025-03-13_15-47-30\CP_extract_1_2025-03-13_15-48-12"


    CPp1_output_dir = CPp1.CP_plotter_1(
        input_dir = CPe1_output_dir,
        #CP_extract_df = CP_extract_df,
        output_dir_manual = "", output_dir_comment = "",
        video = 1
        )


if 1==0: # video comparing CP A11
    #CPe1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_segment_1_2025-03-12_13-42-11\CP_extract_1_2025-03-12_13-48-48"
    #CP_extract_df = None

    CPp2_output_dir = CPp2.CP_plotter_2_CPvsA11(
        input_dir = CPe1_output_dir,
        #CP_extract_df = CP_extract_df,
        output_dir_manual = "", output_dir_comment = "",
        video = 1
        )


if 1==0:
    # \BW 134 ball flame - Crop
    #CPe1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_segment_1_2025-03-12_13-42-11\CP_extract_1_2025-03-12_13-48-48"

    # BW 134 ball flame - Crop Small First few
    #CPe1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop Small First few\CP_segment_1_2025-03-10_15-13-26\CP_extract_1_2025-03-11_12-30-00"
    #CP_extract_df = None

    CPp2_output_dir = CPp3_panel.CP_plotter_3_CPvsA11_Panel(
        input_dir = CPe1_output_dir,
        #CP_extract_df = CP_extract_df,
        output_dir_manual = "", output_dir_comment = "",
        video = 0, show_plot = 1,
        Panel_1 = 0, # 
        Panel_2 = 0, # check non Dimentionalisation
        Panel_3 = 0, # 
        Panel_4 = 1, # panel comparing CP A11
        )








######################################################## end inform
F_1.end_inform(__file__, start_time)