import os
import datetime
import traceback
import Format_1 as F_1 # Assuming Format_1.py is in the Python path or same directory

from VCL_Pipe_1 import VCL_pipeline
import CombineVariationPlotMP4 as cvp

@F_1.ParameterLog(max_size = 1024 * 10) # 10KB 
def VCL_variation_1(
    # General control
    input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations",
    vcl_variation_output_dir_manual="",
    vcl_variation_output_dir_comment="test", 
):
    #################################################### I/O
    vcl_variations_1_output_dir = F_1.F_out_dir(input_dir = input_dir, script_path = __file__, output_dir_comment = vcl_variation_output_dir_comment, output_dir_manual = vcl_variation_output_dir_manual) # Format_1 required definition of output directory
    print(f"VCL_variations_1_output_dir: {vcl_variations_1_output_dir}")

    ####################################################

    # create Error log file
    error_log_file = os.path.join(vcl_variations_1_output_dir, "variation_errors.txt")

    # Define parameter variations
    # Each dict contains parameters to override in VCL_pipeline
    # Specifically, these will affect Visit_Projector_1 via vp_* argumentss
    variations = [
        {
            "vcl_pipeline_output_dir_comment": "CustomBW",
            "vp_output_dir_comment": "CustomBW",
            "vp_Pseudocolor_colortable": "CustomBW",
            "vp_invertColorTable": 0, # Assuming default
        },
        {
            "vcl_pipeline_output_dir_comment": "orangehot_invC_1",
            "vp_output_dir_comment": "orangehot_invC_1",
            "vp_Pseudocolor_colortable": "orangehot",
            "vp_invertColorTable": 1,
        },
        {
            "vcl_pipeline_output_dir_comment": "Purples",
            "vp_output_dir_comment": "Purples",
            "vp_Pseudocolor_colortable": "Purples",
            "vp_invertColorTable": 0, # Assuming default
        },
        {
            "vcl_pipeline_output_dir_comment": "hot_and_cold",
            "vp_output_dir_comment": "hot_and_cold",
            "vp_Pseudocolor_colortable": "hot_and_cold",
            "vp_invertColorTable": 0, # Assuming default
        },
        {
            "vcl_pipeline_output_dir_comment": "BrBG",
            "vp_output_dir_comment": "BrBG",
            "vp_Pseudocolor_colortable": "BrBG",
            "vp_invertColorTable": 0, # Assuming default
        },
        {
            "vcl_pipeline_output_dir_comment": "difference",
            "vp_output_dir_comment": "difference",
            "vp_Pseudocolor_colortable": "difference",
            "vp_invertColorTable": 0, # Assuming default
        },
        {
            "vcl_pipeline_output_dir_comment": "plasma",
            "vp_output_dir_comment": "plasma",
            "vp_Pseudocolor_colortable": "plasma",
            "vp_invertColorTable": 0, # Assuming default
        },
        {
            "vcl_pipeline_output_dir_comment": "turbo",
            "vp_output_dir_comment": "turbo",
            "vp_Pseudocolor_colortable": "turbo",
            "vp_invertColorTable": 0, # Assuming default
        },
        {
            "vcl_pipeline_output_dir_comment": "hot",
            "vp_output_dir_comment": "hot",
            "vp_Pseudocolor_colortable": "hot",
            "vp_invertColorTable": 0, # Assuming default
        },
        {
            "vcl_pipeline_output_dir_comment": "Accent",
            "vp_output_dir_comment": "Accent",
            "vp_Pseudocolor_colortable": "Accent",
            "vp_invertColorTable": 0, # Assuming default
        },
    ]
    # run variations
    for i, var_params in enumerate(variations):
        print(f"\n--- Starting Variation {i+1}/{len(variations)}: {var_params['vp_output_dir_comment']} ---")

        # Prepare arguments for VCL_pipeline
        pipeline_args = {
            'input_dir': vcl_variations_1_output_dir, # set the directory for the pipeline
            "vcl_pipeline_output_dir_comment": var_params["vcl_pipeline_output_dir_comment"], # For overall pipeline logging

            # Visit_Projector_1 arguments
            #"vp_input_dir": vcl_variations_1_output_dir, # Directs VP1 output into this folder
            "vp_output_dir_comment": var_params["vp_output_dir_comment"], # VP1 uses this for its own subfolder name
            "vp_Pseudocolor_colortable": var_params["vp_Pseudocolor_colortable"],
            "vp_invertColorTable": var_params["vp_invertColorTable"],

        }

        try:
            VCL_pipeline(**pipeline_args)
            print(f"--- Variation {var_params['vcl_pipeline_output_dir_comment']} completed successfully. ---")

        except Exception as e: # log an error if it occurs
            print(f"!!! ERROR during variation: {var_params['vcl_pipeline_output_dir_comment']} !!!")
            error_timestamp = datetime.datetime.now().isoformat()
            error_details = (
                f"Timestamp: {error_timestamp}\n"
                f"Variation Index: {i}\n"
                f"Variation Name: {var_params['vcl_pipeline_output_dir_comment']}\n"
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
    cvp.combine_variation_mp4s(vcl_variations_1_output_dir) 

    print("\nEncountered Error: \n")
    F_1.ding() # Notify completion of all sweeps
    return vcl_variations_1_output_dir

if __name__ == "__main__":
    # General script start info    
    VCL_variation_1()