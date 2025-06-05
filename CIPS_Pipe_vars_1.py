import os
import datetime
import traceback
import Format_1 as F_1 # Assuming Format_1.py is in the Python path or same directory

from CIPS_Pipe_1 import CIPS_pipeline
import CombineVariationPlotMP4 as cvp

@F_1.ParameterLog(max_size = 1024 * 10) # 10KB 
def CIPS_variation_1(
    # General control
    input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations",
    cips_variation_output_dir_manual="",
    cips_variation_output_dir_comment="Image_Resolution_State_0", 
):
    #################################################### I/O
    cips_variations_1_output_dir = F_1.F_out_dir(input_dir = input_dir, script_path = __file__, output_dir_comment = cips_variation_output_dir_comment, output_dir_manual = cips_variation_output_dir_manual) # Format_1 required definition of output directory
    print(f"CIPS_variations_1_output_dir: {cips_variations_1_output_dir}")

    ####################################################

    # create Error log file
    error_log_file = os.path.join(cips_variations_1_output_dir, "variation_errors.txt")

    # Define parameter variations
    # Each dict contains parameters to override in CIPS_pipeline
    variations = [
        # num_periods=1 variations
        {
            "cips_pipeline_output_dir_comment": "1024x1024",
            "vp_WindowWidth": 1024,
            "vp_WindowHeight": 1024,
            "vp_output_dir_comment": "1024x1024",
        },
        {
            "cips_pipeline_output_dir_comment": "3000x3000",
            "vp_WindowWidth": 3000,
            "vp_WindowHeight": 3000,
            "vp_output_dir_comment": "3000x3000",
        },
    ]


    # run variations
    for i, var_params in enumerate(variations):
        print(f"\n\n\n--- Starting Variation {i+1}/{len(variations)}: {var_params['vp_output_dir_comment']} ---")

        # Prepare arguments for CIPS_pipeline
        pipeline_args = {
            # General arguments
            'input_dir': cips_variations_1_output_dir, # set the directory for the pipeline
            "cips_pipeline_output_dir_comment": var_params["cips_pipeline_output_dir_comment"], # For overall pipeline logging

            # Visit_Projector_1 arguments
            "vp_output_dir_comment": var_params["vp_output_dir_comment"], # VP1 uses this for its own subfolder name
            "vp_WindowWidth": var_params["vp_WindowWidth"],
            "vp_WindowHeight": var_params["vp_WindowHeight"],

        }

        try:
            CIPS_pipeline(**pipeline_args)
            print(f"--- Variation {var_params['cips_pipeline_output_dir_comment']} completed successfully. ---")

        except Exception as e: # log an error if it occurs
            print(f"!!! ERROR during variation: {var_params['cips_pipeline_output_dir_comment']} !!!")
            error_timestamp = datetime.datetime.now().isoformat()
            error_details = (
                f"Timestamp: {error_timestamp}\n"
                f"Variation Index: {i}\n"
                f"Variation Name: {var_params['cips_pipeline_output_dir_comment']}\n"
                f"Parameters Used: {pipeline_args}\n"
                f"Error Type: {type(e).__name__}\n"
                f"Error Message: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}\n"
                f"{'-'*80}\n"
            )
            with open(error_log_file, "a") as f_err:
                f_err.write(error_details)
            print(f"Error details logged to {error_log_file}")
            print(f"--- Continuing to next variation. ---")

    print("\nAll variations attempted.\n")

    print("Attempting to combine MP4 files from all variations using CombineVariationPlotMP4.py")
    cvp.combine_variation_mp4s(cips_variations_1_output_dir) 

    print("\nEncountered Error: \n")
    F_1.ding() # Notify completion of all sweeps
    return cips_variations_1_output_dir

if __name__ == "__main__":
    # General script start info    
    CIPS_variation_1()