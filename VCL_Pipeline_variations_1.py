import os
import datetime
import traceback
import Format_1 as F_1 # Assuming Format_1.py is in the Python path or same directory
from VCL_Pipeline import run_vcl_pipeline # Assuming VCL-Pipeline.py is in the Python path or same directory

def run_parameter_sweeps():
    # Base directory for all variation outputs
    variations_base_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VCL-Pipeline_Variaitons"
    os.makedirs(variations_base_dir, exist_ok=True)

    # Error log file
    error_log_file = os.path.join(variations_base_dir, "variation_errors.txt")

    # Define parameter variations
    # Each dict contains parameters to override in run_vcl_pipeline
    # Specifically, these will affect Visit_Projector_1 via vp_* arguments
    variations = [
        {
            "vp_Pseudocolor_colortable": "CustomBW",
            "vp_invertColorTable": 0, # Default, but explicit for clarity in variation name
            "variation_name": "PsCol_CustomBW_invC_0" # For folder and comment
        },
        {
            "vp_Pseudocolor_colortable": "orangehot",
            "vp_invertColorTable": 1,
            "variation_name": "PsCol_orangehot_invC_1" # For folder and comment
        },
        # Add more variations here as needed
    ]

    # Input directory for Visit_Projector_1 (where it reads data from)
    visit_projector_input_data_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\SF_CP_analysis_pipeline_data"

    for i, var_params in enumerate(variations):
        print(f"\n--- Starting Variation {i+1}/{len(variations)}: {var_params['variation_name']} ---")

        # Create a unique output directory for this specific variation run using Format_1 logic
        # This directory will be passed as vp_output_dir_manual to run_vcl_pipeline,
        # so Visit_Projector_1 (and subsequent steps if they also use such a pattern)
        # will create their outputs within this variation-specific folder.
        variation_run_base_path = F_1.create_output_dir(
            output_dir_manual="", # Let Format_1 create it in the default base
            output_dir_default_base=variations_base_dir,
            output_dir_comment=var_params["variation_name"], # Comment for the main variation folder
            script_name="VCL_Variation_Run" # Script name for the main variation folder
        )
        print(f"Variation output base directory: {variation_run_base_path}")

        # Prepare arguments for run_vcl_pipeline
        pipeline_args = {
            "pipeline_run_comment_suffix": var_params["variation_name"], # For overall pipeline logging

            # Visit_Projector_1 specific arguments
            "vp_input_dir": visit_projector_input_data_dir,
            "vp_output_dir_manual": variation_run_base_path, # Directs VP1 output into this folder
            "vp_output_dir_comment": var_params["variation_name"], # VP1 uses this for its own subfolder name
            "vp_Pseudocolor_colortable": var_params["vp_Pseudocolor_colortable"],
            "vp_invertColorTable": var_params["vp_invertColorTable"],
            
            # Add other vp_* parameters if they are part of the sweep
            # Otherwise, they will use defaults from run_vcl_pipeline

            # Potentially other parameters for other pipeline stages if varied
            # "cps_CP_model_type": "new_model", # Example
        }

        try:
            run_vcl_pipeline(**pipeline_args)
            print(f"--- Variation {var_params['variation_name']} completed successfully. ---")

        except Exception as e:
            print(f"!!! ERROR during variation: {var_params['variation_name']} !!!")
            error_timestamp = datetime.datetime.now().isoformat()
            error_details = (
                f"Timestamp: {error_timestamp}\n"
                f"Variation Index: {i}\n"
                f"Variation Name: {var_params['variation_name']}\n"
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
    
    run_parameter_sweeps()
    
    # General script end info
    F_1.end_inform(__file__, script_start_time)