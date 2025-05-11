import os
import datetime
import traceback
import Format_1 as F_1 # Assuming Format_1.py is in the Python path or same directory
from VCL_Pipeline import VCL_pipeline # Assuming VCL-Pipeline.py is in the Python path or same directory

@F_1.ParameterLog(max_size = 1024 * 10) # 10KB 
def VCL_variation_1(
    # General control
    input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL_variations",
    vcl_variation_output_dir_manual="",
    vcl_variation_output_dir_comment="", 
):
    #################################################### I/O
    start_time, current_date = F_1.start_inform(__file__)

    vcl_variations_1_output_dir = F_1.F_out_dir(input_dir = input_dir, script_path = __file__, output_dir_comment = vcl_variation_output_dir_manual, output_dir_manual = vcl_variation_output_dir_comment) # Format_1 required definition of output directory


    ####################################################

    # create Error log file
    error_log_file = os.path.join(vcl_variations_1_output_dir, "variation_errors.txt")

    # Define parameter variations
    # Each dict contains parameters to override in VCL_pipeline
    # Specifically, these will affect Visit_Projector_1 via vp_* argumentss
    variations = [
        {
            "vcl_pipeline_output_dir_comment": "PsCol_CustomBW_invC_0",
            "vp_output_dir_comment": "PsCol_CustomBW_invC_0", # For folder and comment
            "vp_Pseudocolor_colortable": "CustomBW",
            "vp_invertColorTable": 0, # Default, but explicit for clarity in variation name
        },
        {
            "vcl_pipeline_output_dir_comment": "PsCol_orangehot_invC_1",
            "vp_output_dir_comment": "PsCol_orangehot_invC_1", # For folder and comment
            "vp_Pseudocolor_colortable": "orangehot",
            "vp_invertColorTable": 1,
        },
        # Add more variations here as needed
    ]
    # run variations
    for i, var_params in enumerate(variations):
        print(f"\n--- Starting Variation {i+1}/{len(variations)}: {var_params['vp_output_dir_comment']} ---")

        # Prepare arguments for VCL_pipeline
        pipeline_args = {
            "vcl_pipeline_output_dir_comment": var_params["vcl_pipeline_output_dir_comment"], # For overall pipeline logging

            # Visit_Projector_1 specific arguments
            "vp_input_dir": vcl_variations_1_output_dir, # Directs VP1 output into this folder
            "vp_output_dir_comment": var_params["vp_output_dir_comment"], # VP1 uses this for its own subfolder name
            "vp_Pseudocolor_colortable": var_params["vp_Pseudocolor_colortable"],
            "vp_invertColorTable": var_params["vp_invertColorTable"],
            
            # Add other vp_* parameters if they are part of the sweep
            # Otherwise, they will use defaults from VCL_pipeline

            # Potentially other parameters for other pipeline stages if varied
            # "cps_CP_model_type": "new_model", # Example
        }

        try:
            VCL_pipeline(**pipeline_args)
            print(f"--- Variation {var_params['vcl_pipeline_output_dir_comment']} completed successfully. ---")

        except Exception as e:
            print(f"!!! ERROR during variation: {var_params['vcl_pipeline_output_dir_comment']} !!!")
            error_timestamp = datetime.datetime.now().isoformat()
            error_details = (
                f"Timestamp: {error_timestamp}\n"
                f"Variation Index: {i}\n"
                f"Variation Name: {var_params['vcl_pipeline_output_dir_comment']}\n"
                f"Parameters Used: {pipeline_args}\n" # Log the actual args passed
                f"Error Type: {type(e).__name__}\n"
                f"Error Message: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}\n"
                f"{'-'*80}\n"
            )
            with open(error_log_file, "a") as f_err:
                f_err.write(error_details)
            print(f"Error details logged to {error_log_file}")
            print(f"--- Continuing to next variation. ---")

    print("\nAll variations attempted.")
    F_1.ding() # Notify completion of all sweeps

if __name__ == "__main__":
    # General script start info
    script_start_time, script_current_date = F_1.start_inform(__file__)
    
    VCL_variation_1()
    
    # General script end info
    F_1.end_inform(__file__, script_start_time)