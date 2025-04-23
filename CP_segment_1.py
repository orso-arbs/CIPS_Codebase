import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, plot, io
import glob
import os
import pandas as pd

import sys
import os
import Format_1 as F_1








@F_1.ParameterLog(max_size = 1024 * 10, log_level = 0) # 0.1KB per smallest unit in return (8 bits per ASCII character)
def CP_segment_1(
    # input
    input_dir,

    # Cellpose parameters
    CP_model_type = 'cyto3', gpu = True, # models.Cellpose() parameters
    diameter_estimate_guess_px = None, channels = [0,0], flow_threshold = 0.4, cellprob_threshold = 0.0, resample = True, niter = 0, # model.eval() parameters
    CP_default_plot_onoff = 0, CP_default_image_onoff = 0, CP_default_seg_file_onoff = 1, # output default Cellpose files

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
    CP_model_type : str, optional
        Specifies the Cellpose model to use. Can be a pretrained model name
        ('cyto', 'nuclei', 'cyto2', 'cyto3') or a path to a custom model file.
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

    output_dir = F_1.F_out_dir(input_dir, __file__, output_dir_comment = output_dir_comment) # Format_1 required definition of output directory

    # list of files
    files = glob.glob(input_dir + r"\*.png")
    all_images = [io.imread(f) for f in files]
    N_images = len(all_images)
    print("\n loaded #images: ", N_images)



    #################################################### CellPose


    # Segment the imafges with cellpose
    print("\n CellPose Segmenting")
    if CP_model_type in ['cyto', 'nuclei', 'cyto2', 'cyto3']: # if its a cellpose pretrained base model
        # Initialize Cellpose (with pretrained base model)
        print("\nInitialize Cellpose with pretrained base model: ", CP_model_type) if CP_segment_log_level >= 2 else None
        CP_instance = models.Cellpose(model_type = CP_model_type, gpu = gpu)

        # Run CP network (with pretrained base model)
        print("\n Running CP network with model: ", CP_model_type) if CP_segment_log_level >= 1 else None
        CP_model_for_diameter_estimate = CP_instance.sz.model_type
        print("estimating diameters with model: ", CP_model_for_diameter_estimate) if CP_segment_log_level >= 1 else None
        masks, flows, styles, diameter_estimate_used_px = CP_instance.eval(
            all_images, diameter=diameter_estimate_guess_px, channels=channels,
            flow_threshold = flow_threshold, cellprob_threshold = cellprob_threshold,
            resample = resample, niter = niter,
            )

    else: # if its a custom model
        # Initialize Cellpose (with default pretrained base model cyto3)
        print("\n Initializing CellPose with default pretrained base Model cyto3") if CP_segment_log_level >= 1 else None
        CP_instance = models.Cellpose(gpu=gpu)
        
        # Load and assign the custom model using CellposeModel. With this separate loading for a custom model, you can still have cellpose estimate the diameters using a pretrained model which by default is cyto3
        print("\n Loading Custom Model") if CP_segment_log_level >= 1 else None
        CP_model = models.CellposeModel(pretrained_model=CP_model_type, gpu=gpu)
        CP_instance.cp = CP_model

        # Run CP network (with custom model)
        print("\n Running CP network with model: ", CP_model_type) if CP_segment_log_level >= 1 else None
        CP_model_for_diameter_estimate = CP_instance.sz.model_type
        print("estimating diameters with model: ", CP_model_for_diameter_estimate ) if CP_segment_log_level >= 1 else None
        masks, flows, styles, diameter_estimate_used_px = CP_instance.eval(
            all_images,
            diameter=diameter_estimate_guess_px,  # = 0 or None for Cellpose diameter estimation. Will use a pretrained base model or default cyto3 to estimate diameter
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            resample=resample,
            niter=niter,
        )

    if isinstance(diameter_estimate_used_px, int):
        diameter_estimate_used_px = np.full(N_images, diameter_estimate_used_px)

    diameter_training_px = CP_instance.diam_mean # diameter used to train the model in pixels (diameter of the training set)
    if isinstance(diameter_training_px, int):
        diameter_training_px = np.full(N_images, diameter_training_px)

    print("\n Path to Cellpose Model used: ", CP_instance.cp.pretrained_model )



    #################################################### Save/Print Results

    print("\n Save Initial Results \n")

    # Write the parameters to the pkl file

    F_1.debug_info(output_dir_comment)
    CP_settings = {
        "CP_model_type": CP_model_type,
        "CP_model_path": CP_instance.cp.pretrained_model,
        "CP_model_for_diameter_estimate": CP_model_for_diameter_estimate,
        "gpu": gpu,
        "diameter_estimate_guess_px": diameter_estimate_guess_px,
        "diameter_training_px": diameter_training_px,
        "flow_threshold": flow_threshold,
        "cellprob_threshold": cellprob_threshold,
        "resample": resample,
        "niter": niter,
        "CP_segment_output_dir_comment": output_dir_comment,
    }
    # Convert to DataFrame (single row)
    CP_settings_df = pd.DataFrame([CP_settings])

    F_1.debug_info(CP_settings_df["CP_model_type"])
    F_1.debug_info(CP_settings_df["CP_model_path"])
    F_1.debug_info(CP_settings_df["CP_model_for_diameter_estimate"])
    F_1.debug_info(CP_settings_df["gpu"])
    F_1.debug_info(CP_settings_df["diameter_estimate_guess_px"])
    F_1.debug_info(CP_settings_df["diameter_training_px"])
    F_1.debug_info(CP_settings_df["flow_threshold"])
    F_1.debug_info(CP_settings_df["cellprob_threshold"])
    F_1.debug_info(CP_settings_df["resample"])
    F_1.debug_info(CP_settings_df["niter"])
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
    if CP_segment_log_level >= 2:
        variables = {
            "output_dir": output_dir,
            "masks": masks,
            "flows": flows,
            "styles": styles,
            "diameter_estimate_used_px": diameter_estimate_used_px,
            "CP_model_type": CP_model_type,
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