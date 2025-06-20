import sys
import os
import Format_1 as F_1

import CP_segment_1 as CPs1
import CP_extract_1 as CPe1
import plot1 as CPp1
import plotter_2_CPvsA11 as CPp2
import plot3_CPvsA11_Panel as CPp3_panel


start_time, current_date = F_1.start_inform(__file__)

########################################################
'''
find good cellpose parameters to optimise segmentation


    resample = True (default)

Whether to run dynamics on original image with resampled interpolated flows (True) or on the rescaled image (False) 

True -> Smoother large ROIs, slower
False -> More small ROIs, faster


findings with resample = False
CP efficiency:  minor differences in N_c
Segmentation:   no change. 


    flow_threshold = 0.4 (default)

Permitted error between flows from true ROI and network predicted flows.

Increase -> more #ROIs, might accept ill-shaped ROIs
Decrease -> less #ROIs, less ill-shaped ROIs#   flow_threshold = 0.4


findings with flow_threshold = 0.2
CP efficiency:  lower, especially in breakup transition (too little)
N_c:            less
Segmentation:   less esp in perifery.
D_mean:         bit bigger


findings with flow_threshold = 0.6                              ---> promising
CP efficiency:  higher, especially in breakup transition
N_c:            bit more
Segmentation:   more.
D_mean:         bit lower


findings with flow_threshold = 0.8                              ---> promising
CP efficiency:  higher, especially in breakup transition
N_c:            bit more
Segmentation:   more esp in perifery
D_mean:         bit lower


findings with flow_threshold = 1.0dd
CP efficiency:  bit less
N_c:            bit less
Segmentation:   bit less
D_mean:         bit lower



    cellprob_threshold = 0.0 (default)

Minimum probability that cell is in 

Increase -> less #ROIs, especially dim areas
Decrease -> more #ROIs


findings with cellprob_threshold = 1.0
CP efficiency:  lower
N_c:            little lower
Segmentation:   minor differences, no clear difference characteristic
D_mean:         bit lower


findings with cellprob_threshold = 0.8
CP efficiency:  bit lower
N_c:            similar, bit lower
Segmentation:   
D_mean:         similar, bit lower


findings with cellprob_threshold = 0.5
CP efficiency:  similar, slightly lower
N_c:            similar, slightly lower
Segmentation:   similar
D_mean:         similar, slightly lower


findings with cellprob_threshold = 0.1
CP efficiency:  similar
N_c:            similar, slightly higher
Segmentation:   similar
D_mean:         similar



    niter = None (default)

Set #iterations for dynamics simulation

None or (0)  sets #iterations proportional to ROI diameter
Bigger (i.e. 2000)  use for strongly elongated non-circular cells

Pixels converging to same points are same ROI

findings with niter = 100
CP efficiency:  same
N_c:            same
Segmentation:   same
D_mean:         same


findings with niter = 1000
CP efficiency:  same
N_c:            same
Segmentation:   same
D_mean:         same


findings with niter = 5000
CP efficiency:  same
N_c:            same
Segmentation:   same
D_mean:         same




    Estimated Diameter (diameter)

Used to resize images such that model trained diameter is similar to diameters in images.

= 0 or None makes automatic estimate
= D user estimate

Too big -> over-merge cells
Too small -> over-split cells








'''





def CP_main(
        # CPs1.CP_segment_1 arguments
        visit_images_dir,
        input_dir,
        CP_model_type = 'cyto3', gpu = True, # model = models.Cellpose() arguments
        diameter_estimate_guess = None, channels = [0,0], flow_threshold = 0.4, cellprob_threshold = 0.0, resample = True, niter = 0, # model.eval() arguments
        CP_segment_output_dir_manual = "", CP_segment_output_dir_comment = "",
        CP_segment_log_level = 1,

        # CP_extract_1 arguments
        #input_dir = CPs1_output_dir,
        #masks = masks, flows = flows, styles = styles, diameter_estimate_used = diameter_estimate_used, CP_model_type = CP_model_type,
        CP_extract_log_level = 0,

        # CPp1.CP_plotter_1 arguments
        #input_dir = CPe1_output_dir,
        #CP_extract_df = CP_extract_df,
        CP_plotter_1_output_dir_manual = "", CP_plotter_1_output_dir_comment = "",
        CP_plotter_1_video = 1
    ):



    # CP_segment_1
    if 1==1:

        CPs1_output_dir, masks, flows, styles, diameter_estimate_used, CP_model_type = CPs1.CP_segment_1(
            input_dir = visit_images_dir,
            CP_model_type = CP_model_type, gpu = gpu, # model = models.Cellpose() arguments
            diameter_estimate_guess = diameter_estimate_guess, channels = channels, flow_threshold = flow_threshold, cellprob_threshold = cellprob_threshold, resample = resample, niter = niter, # model.eval() arguments
            output_dir_manual = CP_segment_output_dir_manual, output_dir_comment = CP_segment_output_dir_comment,
            CP_segment_log_level = CP_segment_log_level,
            )


    # CP_extract_1
    if 1==1:
        # BW 134 ball flame - Crop
        #CPs1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_segment_1_2025-03-12_13-42-11"
        
        # BW 134 ball flame - Crop Small First few
        #CPs1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop Small First few\CP_segment_1_2025-03-10_15-13-26"

        # A11 FB poster selection
        #CPs1_output_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\A11 FB poster selection\CP_segment_1_2025-03-13_17-49-00"

        CPe1_output_dir, CP_extract_df = CPe1.CP_extract_1(
            input_dir = CPs1_output_dir,
            #masks = masks, flows = flows, styles = styles, diameter_estimate_used = diameter_estimate_used, CP_model_type = CP_model_type,
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
            output_dir_manual = CP_plotter_1_output_dir_manual, output_dir_comment = CP_plotter_1_output_dir_comment,
            video = CP_plotter_1_video,
            )




    # not: for functions below the arguments are not in CP_main yet
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



# CP_main(
#         # CPs1.CP_segment_1 arguments
#         visit_images_dir = visit_images_dir)


visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop"
#visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop small"
#visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop small two only"
#visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop Small First few"
#visit_images_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\A11 FB poster selection"


def run_variants():
    # Base arguments shared among all cases
    base_args = {
        "visit_images_dir": visit_images_dir,
        "input_dir": visit_images_dir,  # Same as visit_images_dir
        "CP_model_type": 'cyto3',
        "gpu": True,
        "diameter_estimate_guess": None,
        "channels": [0, 0],
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "resample": True,
        "niter": 0,
        "CP_segment_output_dir_manual": "",
        "CP_segment_output_dir_comment": "",  # Will be updated per variant
        "CP_segment_log_level": 1,
        "CP_extract_log_level": 0,
        "CP_plotter_1_output_dir_manual": "",
        "CP_plotter_1_output_dir_comment": "",  # Will be updated per variant
        "CP_plotter_1_video": 1
    }

    # Variants with specific modifications
    variants = [
        ("A", {
            "CP_segment_output_dir_comment": "flow_threshold_0p8",
            "CP_plotter_1_output_dir_comment": "flow_threshold_0p8",
            "flow_threshold": 0.8
        }),
        ("B", {
            "CP_segment_output_dir_comment": "flow_threshold_1p0",
            "CP_plotter_1_output_dir_comment": "flow_threshold_1p0",
            "flow_threshold": 1.0
        }),
        ("C", {
            "CP_segment_output_dir_comment": "cellprob_threshold_0p1",
            "CP_plotter_1_output_dir_comment": "cellprob_threshold_0p1",
            "cellprob_threshold": 0.1
        }),
        ("D", {
            "CP_segment_output_dir_comment": "niter_100",
            "CP_plotter_1_output_dir_comment": "niter_100",
            "niter": 100
        }),
    ]

    # Loop over each variant and call CP_main with updated arguments
    for label, modifications in variants:
        # Create a copy of the base arguments and update with modifications
        args = base_args.copy()
        args.update(modifications)

        print(f"\n\n{label}   {label}   {label}   {label}   {label}   {label}   {label}   {label}\nRunning variant {label} with the following arguments:\n")
        for key, value in args.items():
            print(f"  {key}: {value}")
        print("\n\n")

        # Execute the function for this variant
        CP_main(**args)

run_variants()