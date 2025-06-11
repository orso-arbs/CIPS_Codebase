import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, plot, io
import glob
import os
import pandas as pd
import torch
import gc

import sys
import os
import Format_1 as F_1








@F_1.ParameterLog(max_size = 1024 * 10, log_level = 0) # 0.1KB per smallest unit in return (8 bits per ASCII character)
def CP_segment_1(
    # input
    input_dir,

    # Cellpose parameters
    CP_model_type_for_segmentation = 'cyto3', 
    CP_model_type_for_diameter_estimation = 'cyto3', # New parameter for diameter estimation model
    gpu = True, CP_empty_cache_onoff = True, # model parameters
    diameter_estimate_guess_px = None, channels = [0,0], flow_threshold = 0.7, cellprob_threshold = 0.0, resample = True, niter = 0, # model.eval() parameters
    batch_size=8, augment=True, tile_overlap=0.1, bsize=8, # tiling/augment parameters for model.eval()
    CP_default_plot_onoff = 1, CP_default_image_onoff = 1, CP_default_seg_file_onoff = 1, # output default Cellpose files

    # output and logging 
    CP_segment_log_level = 0,
    output_dir_manual = "", output_dir_comment = "",
    ):


    """
    Segments images using the Cellpose library.

    This function takes a directory of images, applies Cellpose segmentation,
    and saves the results along with logging information.

    Parameters
    ----------
    input_dir : str
        Directory containing the input images (expects .png files).
    CP_model_type_for_segmentation : str, optional
        Specifies the Cellpose model to use for segmentation. Can be a pretrained model name
        ('cyto', 'nuclei', 'cyto2', 'cyto3', 'cpsam') or a path to a custom model file.
        Defaults to 'cyto3'.
    CP_model_type_for_diameter_estimation : str, optional # New docstring
        Specifies the Cellpose model to use for the initial diameter estimation step.
        This model is part of the Cellpose wrapper instance.
        Defaults to 'cyto3'.
    gpu : bool, optional
        Whether to use the GPU for computation (requires CUDA and PyTorch).
        Defaults to True.
    diameter_estimate_guess_px : float or None, optional
        Estimated diameter of the objects in pixels. If None or 0, Cellpose
        estimates the diameter automatically using a base model (cyto3 if a
        custom model is used). Providing an estimate is officiall recommended but the estimate
        usually works well. Defaults to None.
    channels : list of int, optional
        Specifies the channels to use for segmentation. For bilogical application that allows 
        distinction of cytoplasm and nucleus. The list then is: [cytoplasm, nucleus].
        Use 0 for grayscale, 1 for Red, 2 for Green, 3 for Blue.
        If the nucleus channel doesn't exist, set the second element to 0.
        Example: [0, 0] for grayscale, [2, 3] for Green cytoplasm and Blue nucleus.
        For the flame images, use i.e. [0, 0] for grayscale of the flame
        cells (cytoplasm and no "nucleus" structure), Defaults to [0, 0].
    flow_threshold : float, optional
        Flow error threshold. Pixels with flow errors above this threshold are
        not used in dynamics. Defaults to 0.4.
    cellprob_threshold : float, optional
        Cell probability threshold. Pixels with cell probability below this
        threshold are excluded. Defaults to 0.0.
    resample : bool, optional
        Whether to run dynamics on original image with resampled interpolated flows (True) or
        on the rescaled image (False). Defaults to True.
    niter : int, optional
        Number of iterations for dynamics simulation. If 0, Cellpose estimates
        the number of iterations. Defaults to 0.
    augment : bool, optional # Changed from tile to augment
        Whether to use augmented prediction by tiling image with overlapping tiles and flipping overlapped regions.
        Defaults to True.
    tile_overlap : float, optional
        Fraction of overlap between tiles if `augment` is True. Defaults to 0.1.
    bsize : int, optional
        Size of tiles in pixels if `augment` is True. Recommended to be 224 as in training.
        Defaults to 224.
    CP_default_plot_onoff : int, optional
        If 1, saves the default Cellpose segmentation plot for each image.
        Defaults to 0.
    CP_default_image_onoff : int, optional
        If 1, saves the default Cellpose image output (functionality might be broken).
        Defaults to 0.
    CP_default_seg_file_onoff : int, optional
        If 1, saves the Cellpose default segmentation file (_seg.npy) for each image.
        Defaults to 1.
    CP_segment_log_level : int, optional
        Controls the verbosity of logging to the console.
        0: Minimal logging.
        1: Basic Cellpose logs.
        2: Detailed logs including return types and sizes.
        Defaults to 0.
    output_dir_manual : str, optional
        If provided, specifies the exact output directory path. Overrides the
        default Format_1 naming convention. Defaults to "" (using the standard output_dir by using
        Format_1).
    output_dir_comment : str, optional
        A comment to append to the default output directory name.
        Defaults to "".

    Returns
    -------
    output_dir : str
        The path to the directory where results and logs are saved.
        This is always the first return value as required by Format_1.

    Notes
    -----
    - This function relies on `Format_1.py` for output directory creation (`F_out_dir`)
      and parameter logging (`@F_1.ParameterLog`).
    - Besides the returned variables, the function saves several files to the `output_dir`:
        - `_log.json`: Contains logged parameters and execution details.
        - `CP_settings.pkl`: A pickle file storing the Cellpose settings used.
        - Optional Cellpose default outputs (`_seg.npy`, `_plot.png`, `_img.png`)
        based on the `CP_default_*_onoff` flags.
        - `_seg.npy` file contains the extracted masks and flows used in the further data analysis.
    """


    #################################################### I/O 

    output_dir = F_1.F_out_dir(input_dir, __file__, output_dir_comment = output_dir_comment, output_dir_manual = output_dir_manual) # Format_1 required definition of output directory

    # list of files
    files = glob.glob(input_dir + r"\*.png")
    all_images = [io.imread(f) for f in files] # submitting all images to Cellpose like this runs cellpose in batch mode, which is faster than running it for each image separately
    N_images = len(all_images)
    print("\n loaded #images: ", N_images)


    #################################################### CellPose
    if CP_empty_cache_onoff == True: # Clear GPU memory before starting Cellpose segmentation
        gc.collect() # Collect garbage to free up memory
        torch.cuda.empty_cache()# Clear GPU memory before starting Cellpose segmentation
        print("\n Cleared CUDA GPU memory") if CP_segment_log_level >= 1 else None

    io.logger_setup() if CP_segment_log_level >= 2 else None

    # Segment the images with cellpose

    # --- The Simplified and Unified Way with Batch Processing ---
    print("\n CellPose Segmenting --- The Simplified and Unified Way with Batch Processing ---")

    # 1. Initialize a default Cellpose WRAPPER instance.
    #    This instance always contains a size model (by default from 'cyto3'),
    #    which is what we need for diameter estimation.
    print("\nInitializing Cellpose wrapper to enable diameter estimation...")
    CP_instance = models.Cellpose(model_type = CP_model_type_for_diameter_estimation, gpu=gpu)

    # 2. Load your desired segmentation model (custom OR built-in) into a core CellposeModel object.
    #    The `pretrained_model` parameter accepts both names like 'cyto3' and file paths.
    print(f"\nLoading segmentation model: {CP_model_type_for_segmentation}")
    CP_model_for_segmentation_obj = models.CellposeModel(pretrained_model=CP_model_type_for_segmentation, gpu=gpu)

    # 3. Overwrite the segmentation model within the wrapper.
    CP_instance.cp = CP_model_for_segmentation_obj

    # 4. Run evaluation.
    #    CP_instance.eval() will now:
    #      a) Use its original size model for diameter estimation (if diameter is None).
    #      b) Use the new, swapped-in model (your CP_model_type_for_segmentation) for the actual segmentation.
    print(f"\nRunning network with segmentation model: {CP_instance.cp.pretrained_model}")
    print(f"Estimating diameters with size model from: {CP_instance.sz.model_type}") # This is the actual model used by size estimator

    masks, flows, styles, diameter_estimate_used_px = CP_instance.eval(
        all_images,
        diameter=diameter_estimate_guess_px,  # Set to None or 0 for auto-detection
        channels=channels,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        batch_size = batch_size, # number of 224x224 patches to run simultaneously on the GPU. Reducing can reduce RAM memory usage
        augment=augment,  # Use augment parameter from function signature
        tile_overlap=tile_overlap, # Use tile_overlap parameter
        bsize=bsize, # Use bsize parameter
        resample=resample,
        niter=niter,
    )
    print("\n END of CellPose Segmenting --- END of The Simplified and Unified Way ---")

########################### new up / old below
    # Commented out old logic - ensure it's not active or update if needed.
    # For brevity, skipping detailed changes in the commented block, assuming new logic is primary.
    # If this old block were to be used, `CP_model_type` would become `CP_model_type_for_segmentation`
    # and `CP_model_for_diameter_estimate` (as a variable) would become `CP_model_type_for_diameter_estimation`
    # The `tile` parameter in `CP_instance.eval` would also need to be `augment`.

    # print("\n CellPose Segmenting")
    # if CP_model_type_for_segmentation in ['cyto', 'nuclei', 'cyto2', 'cyto3']: # if its a cellpose pretrained base model
    #     # Initialize Cellpose (with pretrained base model)
    #     print("\nInitialize Cellpose with pretrained base model: ", CP_model_type_for_segmentation) if CP_segment_log_level >= 2 else None
    #     CP_instance = models.Cellpose(model_type = CP_model_type_for_segmentation, gpu = gpu)

    #     # Run CP network (with pretrained base model)
    #     print("\n Running CP network with model: ", CP_model_type_for_segmentation) if CP_segment_log_level >= 1 else None
    #     # CP_model_for_diameter_estimate_val = CP_instance.sz.model_type # Old way of getting this
    #     print("estimating diameters with model: ", CP_instance.sz.model_type) if CP_segment_log_level >= 1 else None
    #     masks, flows, styles, diameter_estimate_used_px = CP_instance.eval(
    #         all_images, diameter=diameter_estimate_guess_px, channels=channels,
    #         flow_threshold = flow_threshold, cellprob_threshold = cellprob_threshold,
    #         augment=augment, tile_overlap=tile_overlap, bsize=bsize, # Tiling parameters
    #         resample = resample, niter = niter,
    #         )

    # else: # if its a custom model
    #     # Initialize Cellpose (with default pretrained base model cyto3 for size estimation)
    #     print("\n Initializing CellPose with default pretrained base Model cyto3 for size estimation") if CP_segment_log_level >= 1 else None
    #     CP_instance = models.Cellpose(gpu=gpu, model_type="cyto3") # Ensure size model is set
        
    #     # Load and assign the custom model using CellposeModel.
    #     print("\n Loading Custom Model for segmentation: ", CP_model_type_for_segmentation) if CP_segment_log_level >= 1 else None
    #     custom_model_obj = models.CellposeModel(pretrained_model=CP_model_type_for_segmentation, gpu=gpu)
    #     CP_instance.cp = custom_model_obj # Assign custom model for segmentation

    #     # Run CP network (with custom model for segmentation)
    #     print("\n Running CP network with segmentation model: ", CP_model_type_for_segmentation) if CP_segment_log_level >= 1 else None
    #     # CP_model_for_diameter_estimate_val = CP_instance.sz.model_type # Old way
    #     print("estimating diameters with size model: ", CP_instance.sz.model_type ) if CP_segment_log_level >= 1 else None
    #     masks, flows, styles, diameter_estimate_used_px = CP_instance.eval(
    #         all_images,
    #         diameter=diameter_estimate_guess_px, 
    #         channels=channels,
    #         flow_threshold=flow_threshold,
    #         cellprob_threshold=cellprob_threshold,
    #         augment=augment, tile_overlap=tile_overlap, bsize=bsize, # Tiling parameters
    #         resample=resample,
    #         niter=niter,
    #     )

####################################

    if isinstance(diameter_estimate_used_px, int):
        diameter_estimate_used_px = np.full(N_images, diameter_estimate_used_px)

    diameter_training_px = CP_instance.diam_mean # diameter used to train the model in pixels (diameter of the training set)
    if isinstance(diameter_training_px, int):
        diameter_training_px = np.full(N_images, diameter_training_px)

    print("\n Path to Cellpose Model used: ", CP_instance.cp.pretrained_model )



    #################################################### Save/Print Results

    print("\n Save Initial Results \n")

    # Write the parameters to the pkl file

    F_1.debug_info(output_dir_comment) if CP_segment_log_level >= 4 else None
    CP_settings = {
        "CP_model_type_for_segmentation": CP_model_type_for_segmentation,
        "CP_segmentation_model_actual_path": CP_instance.cp.pretrained_model, # Renamed from CP_model_path
        "CP_model_type_for_diameter_estimation_input": CP_model_type_for_diameter_estimation, # New: logs the input parameter
        "CP_size_model_type_in_wrapper": CP_instance.sz.model_type, # New: logs the actual size model in the wrapper
        "gpu": gpu,
        "CP_empty_cache_onoff": CP_empty_cache_onoff,
        "diameter_estimate_guess_px": diameter_estimate_guess_px,
        "diameter_training_px": diameter_training_px,
        "flow_threshold": flow_threshold,
        "cellprob_threshold": cellprob_threshold,
        "resample": resample,
        "niter": niter,
        "augment": augment, # Changed from "tile" to "augment", uses augment parameter
        "tile_overlap": tile_overlap,
        "bsize": bsize,
        "CP_segment_output_dir_comment": output_dir_comment,
    }
    # Convert to DataFrame (single row)
    CP_settings_df = pd.DataFrame([CP_settings])

    if CP_segment_log_level >= 4:
        F_1.debug_info(CP_settings_df["CP_model_type_for_segmentation"]) 
        F_1.debug_info(CP_settings_df["CP_segmentation_model_actual_path"]) # Updated key
        F_1.debug_info(CP_settings_df["CP_model_type_for_diameter_estimation_input"]) # Updated key
        F_1.debug_info(CP_settings_df["CP_size_model_type_in_wrapper"]) # Updated key
        F_1.debug_info(CP_settings_df["gpu"])
        F_1.debug_info(CP_settings_df["CP_empty_cache_onoff"])
        F_1.debug_info(CP_settings_df["diameter_estimate_guess_px"])
        F_1.debug_info(CP_settings_df["diameter_training_px"])
        F_1.debug_info(CP_settings_df["flow_threshold"])
        F_1.debug_info(CP_settings_df["cellprob_threshold"])
        F_1.debug_info(CP_settings_df["resample"])
        F_1.debug_info(CP_settings_df["niter"])
        F_1.debug_info(CP_settings_df["augment"]) # Updated key
        F_1.debug_info(CP_settings_df["tile_overlap"])
        F_1.debug_info(CP_settings_df["bsize"])
        F_1.debug_info(CP_settings_df["CP_segment_output_dir_comment"])

    # Save as pickle
    CP_settings_df.to_pickle(f"{output_dir}/CP_settings.pkl")

    # Save CP default outputs: seg file, plot and image
    for idx in range(N_images):
        maski = masks[idx]
        flowi = flows[idx]
        input_filename = os.path.basename(files[idx])

        if CP_default_seg_file_onoff == 1: # Save the seg file
            output_seg_filename = os.path.splitext(input_filename)[0] + "_CP_default_seg_file"
            output_seg_path = os.path.join(output_dir, output_seg_filename)
            io.masks_flows_to_seg(all_images, maski, flowi, output_seg_path, channels=channels, diams=diameter_estimate_used_px[idx])

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
    if CP_segment_log_level >= 4:
        variables = {
            "output_dir": output_dir,
            "masks": masks,
            "flows": flows,
            "styles": styles,
            "diameter_estimate_used_px": diameter_estimate_used_px,
            "CP_model_type_for_segmentation": CP_model_type_for_segmentation,
            "CP_model_type_for_diameter_estimation_input": CP_model_type_for_diameter_estimation, # Added for debug
        }

        # Print type and size
        for name, value in variables.items():
            print(f"{name}: type={type(value)}, value: {value}", end="")

            # Try to get size if possible
            try:
                size = sys.getsizeof(value)
                print(f", size={size} bytes")
            except TypeError:
                print(" (size not available)")


    #################################################### return

    return output_dir # Format_1 requires outpu_dir as first return