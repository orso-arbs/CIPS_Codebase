import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, plot, utils, io
import datetime
import glob
import os
import time
import pandas as pd
import csv
import pickle

import sys
import os
sys.path.append(os.path.abspath(r"C:/Users/obs/OneDrive/ETH/ETH_MSc/Masters Thesis/Python Code/Python_Orso_Utility_Scripts_MscThesis")) # dir containing Format_1 
import Format_1 as F_1






'''
Purpose:
	Take images and segment them with CellPose. 

Arguments:
	input_dir 				        - string
                                    - directory containing images
    output_dir_manual = ""	        - string
                                    - output_dir_manual allows setting output path. 
    output_dir_comment = ""         - string
                                    - sets comment. default output path created as nameScript_date+time_comment.

    CP_model_type = 'cyto3'         - string
                                    - Cellpose pretraining model.
                                    - 'cyto' or 'nuclei' or 'cyto2' or 'cyto3' or custon model (human in the loop trained for example)
                                    - 'cyto3' is the newest and is trained as a generalist making it versatile.
    gpu = False                     - Bool
                                    - use gpu to speed up. reccomended True. CUDA + torch required
    channels = [0,0]                - list (2)
                                    - define CHANNELS to run segementation on
                                    - grayscale=0, R=1, G=2, B=3
                                    - channels = [cytoplasm, nucleus]
                                    - if NUCLEUS channel does not exist, set the second channel to 0
                                    - IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
                                    - channels = [0,0] # IF YOU HAVE GRAYSCALE
                                    - channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
                                    - channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus
    diameter_estimate_guess = None - Int > 0
                                    - diameter estimate to resize images such that model trained diameter is similar to diameters in images.
                                    - if diameter is set to None, the size of the cells is estimated on a per image basis
                                    - you can set the average cell `diameter` in pixels yourself (recommended)
                                    - diameter can be a list or a single number for all images

    CP_default_seg_file_onoff = 1   - Bool
                                    - save the Cellpose default seg file
    CP_default_plot_onoff = 0       - Bool
                                    - plot and save the Cellpose default plot
    CP_default_image_onoff = 0      - Bool
                                    - plot and save the Cellpose default image (broken?)

    CP_segment_log_level = 0                - wether to output log of CP operaitions to console

Return:
	output_dir 				        - string
                                    - necessary. first return has to be the output directory as created in function by F_1.F_out_dir

    masks                           - list (N_images) of numpy arrays (Nx_image x Ny_image)
                                    - each pixel in the image is assigned to an ROI (0 = NO ROI; 1,2,… = ROI labels)
    flows                           - list (N_images) of lists of numpy arrays
                                        (
                                        flows[0] is XY flow in RGB,
                                        flows[1] is the cell probability in range 0-255,
                                        flows[2] is Z flow in range 0-255 (if it exists, otherwise zeros),
                                        flows[3] is [dY, dX, cellprob] (or [dZ, dY, dX, cellprob] for 3D), flows[4] is pixel destinations (for internal use)
                                        
                                        or (contradictory sources)
                                        
                                        - flows[k][0] = XY flow in HSV 0-255
                                        - flows[k][1] = XY flows at each pixel
                                        - flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics)
                                        - flows[k][3] = final pixel locations after Euler integration
                                        )
    styles                          - list (N_images) of arrays of length 256 or single 1D array
                                    - Style vector summarizing each image, also used to estimate size of objects in image.
    diameter_estimate_used               - list (N_images) or float
                                    - estimated diameter used to resize image to run model on



Other Output:
    Cellpose default seg file       - if CP_default_seg_file_onoff == 1
    Cellpose default plot           - if CP_default_plot_onoff = 0
    Cellpose default image (broken?)- if CP_default_image_onoff = 0
'''


@F_1.ParameterLog(max_size = 1024 * 10, log_level = 0) # 0.1KB per smallest unit in return (8 bits per ASCII character)
def CP_segment_1(input_dir, # Format_1 requires input_dir
    CP_model_type = 'cyto3', gpu = True, # model = models.Cellpose() arguments
    diameter_estimate_guess = None, channels = [0,0], flow_threshold = 0.4, cellprob_threshold = 0.0, resample = True, niter = 0, # model.eval() arguments
    CP_default_plot_onoff = 0, CP_default_image_onoff = 0, CP_default_seg_file_onoff = 1,
    output_dir_manual = "", output_dir_comment = "",
    CP_segment_log_level = 0,
    ):

    ### output 
    output_dir = F_1.F_out_dir(input_dir, __file__, output_dir_comment = output_dir_comment) # Format_1 required definition of output directory

    ### I/O 

    # list of files
    files = glob.glob(input_dir + r"\*.png")
    all_images = [io.imread(f) for f in files]
    N_images = len(all_images)
    print("\n loaded #images: ", N_images)



    ### CellPose

    # log 
    io.logger_setup() if CP_segment_log_level >= 1 else None

    print("\n CellPose Segmenting")
    if CP_model_type in ['cyto', 'nuclei', 'cyto2', 'cyto3']: # if its a cellpose pretraines base model
        model = models.Cellpose(model_type = CP_model_type, gpu = gpu)
        masks, flows, styles, diameter_estimate_used = model.eval(all_images, diameter=diameter_estimate_guess, channels=channels,
                                                        flow_threshold = flow_threshold, cellprob_threshold = cellprob_threshold,
                                                        resample = resample, niter = niter
                                                        )
    else: # if its a custom model
        model = models.CellposeModel(pretrained_model = CP_model_type, gpu = gpu) # CP_model_type here has to be fiull path of model file
        masks, flows, styles = model.eval(all_images, diameter=diameter_estimate_guess, channels=channels,
                                                            flow_threshold = flow_threshold, cellprob_threshold = cellprob_threshold,
                                                            resample = resample, niter = niter
                                                            )
    
    if isinstance(diameter_estimate_used, int):
        diameter_estimate_used = np.full(N_images, diameter_estimate_used)

    


    ### Save/Print Results

    print("\n Save Initial Results \n")


    # Write the parameters to the CSV file
    output_file = f"{output_dir}/CP_settings.pkl"
    params = {
        "gpu": gpu,
        "diameter_estimate_guess": diameter_estimate_guess,
        "CP_segment_output_dir_comment": output_dir_comment,
        "flow_threshold": flow_threshold,
        "cellprob_threshold": cellprob_threshold,
        "resample": resample,
        "niter": niter,
        "CP_model_type": CP_model_type
    }
    with open(output_file, "wb") as file:
        pickle.dump(params, file)

    for idx in range(N_images):
        maski = masks[idx]
        flowi = flows[idx]
        input_filename = os.path.basename(files[idx])

        if CP_default_seg_file_onoff == 1: # Save the seg file
            output_seg_filename = os.path.splitext(input_filename)[0] + "_CP_default_seg_file"
            output_seg_path = os.path.join(output_dir, output_seg_filename)
            io.masks_flows_to_seg(all_images, maski, flowi, output_seg_path, channels=channels, diams=diameter_estimate_used[idx])

        if CP_default_plot_onoff == 1: # Save the CP default plot
            fig_CP_default_plot = plt.figure(figsize=(12,5))
            plot.show_segmentation(fig_CP_default_plot, all_images[idx], maski, flowi[0], channels=channels)
            plt.tight_layout()

            output_filename = os.path.splitext(input_filename)[0] + "_CP_default_plot.png"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path)
            plt.close(fig_CP_default_plot)  # Close the figure to free up memory

        if CP_default_image_onoff == 1: # Save the image
            output_image_filename = os.path.splitext(input_filename)[0] + "_CP_default_image"
            output_image_path = os.path.join(output_dir, output_image_filename)
            io.save_masks(all_images, maski, flowi, output_image_path, png=True)

    # show return types and size
    if CP_segment_log_level >= 2:
        variables = {
            "output_dir": output_dir,
            "masks": masks,
            "flows": flows,
            "styles": styles,
            "diameter_estimate_used": diameter_estimate_used,
            "CP_model_type": CP_model_type,
        }

        # Print type and size
        for name, value in variables.items():
            print(f"{name}: type={type(value)}", end="")

            # Try to get size if possible
            try:
                size = sys.getsizeof(value)
                print(f", size={size} bytes")
            except TypeError:
                print(" (size not available)")

    return output_dir, masks, flows, styles, diameter_estimate_used, CP_model_type # Format_1 requires outpu_dir as first return