import sys
import os
sys.path.append(os.path.abspath(r"C:/Users/obs/OneDrive/ETH/ETH_MSc/Masters Thesis/Python Code/Python_Orso_Utility_Scripts_MscThesis")) # dir containing Format_1 
import Format_1 as F_1

import CP_segment_1 as CPs1
import CP_extract_1 as CPe1
import CP_plotter_1 as CPp1


start_time, current_date = F_1.start_inform(__file__)

########################################################

if 1==0: 
    #visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop"
    #visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop small"
    #visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop small two only"
    visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop Small First few"

    CPs1_output_dir, masks, flows, styles, diameter_estimate = CPs1.CP_segment_1(
        input_dir = visit_images_dir,
        CP_model_type = "cyto3", gpu = True,
        CP_segment_log_level = 0,
        )


if 1==1:
    CPs1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop Small First few\CP_segment_1_2025-03-08_12-50-31"

    CPe1_output_dir, df = CPe1.CP_extract_1(
        input_dir = CPs1_output_dir,
        #masks, flows, styles, diameter_estimate
        CP_extract_log_level = 0,
        )


if 1==0:
    #CPe1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop Small First few\CP_segment_1_2025-03-08_12-46-24\CP_extract_1_2025-03-08_12-46-36\CP_plotter_1_2025-03-08_12-46-44"

    CPp1_output_dir = CPp1.CP_plotter_1(
        input_dir = CPe1_output_dir,
        df = df,
        output_dir_manual = "", output_dir_comment = ""
        )











######################################################## end inform
F_1.end_inform(__file__, start_time)