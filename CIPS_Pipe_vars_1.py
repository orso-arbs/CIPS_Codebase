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
    cips_variation_output_dir_comment="Colortable_PeriodicBW_vars", 
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
            "Pseudocolor_periodic_num_periods": 1,
            "distance_ww": 1.0,
            "distance_bb": 1.0,
            "cips_pipeline_output_dir_comment": "Periods1_ww1bb1",
            "vp_output_dir_comment": "Periods1_ww1bb1",
            "vp_Pseudocolor_colortable": "PeriodicBW",
        },
        {
            "Pseudocolor_periodic_num_periods": 1,
            "distance_ww": 2.0,
            "distance_bb": 2.0,
            "cips_pipeline_output_dir_comment": "Periods1_ww2bb2",
            "vp_output_dir_comment": "Periods1_ww2bb2",
            "vp_Pseudocolor_colortable": "PeriodicBW",
        },
        {
            "Pseudocolor_periodic_num_periods": 1,
            "distance_ww": 3.0,
            "distance_bb": 3.0,
            "cips_pipeline_output_dir_comment": "Periods1_ww3bb3",
            "vp_output_dir_comment": "Periods1_ww3bb3",
            "vp_Pseudocolor_colortable": "PeriodicBW",
        },
        # num_periods=2 variations
        {
            "Pseudocolor_periodic_num_periods": 2,
            "distance_ww": 1.0,
            "distance_bb": 1.0,
            "cips_pipeline_output_dir_comment": "Periods2_ww1bb1",
            "vp_output_dir_comment": "Periods2_ww1bb1",
            "vp_Pseudocolor_colortable": "PeriodicBW",
        },
        {
            "Pseudocolor_periodic_num_periods": 2,
            "distance_ww": 2.0,
            "distance_bb": 2.0,
            "cips_pipeline_output_dir_comment": "Periods2_ww2bb2",
            "vp_output_dir_comment": "Periods2_ww2bb2",
            "vp_Pseudocolor_colortable": "PeriodicBW",
        },
        {
            "Pseudocolor_periodic_num_periods": 2,
            "distance_ww": 3.0,
            "distance_bb": 3.0,
            "cips_pipeline_output_dir_comment": "Periods2_ww3bb3",
            "vp_output_dir_comment": "Periods2_ww3bb3",
            "vp_Pseudocolor_colortable": "PeriodicBW",
        },
        # num_periods=5 variations
        {
            "Pseudocolor_periodic_num_periods": 5,
            "distance_ww": 1.0,
            "distance_bb": 1.0,
            "cips_pipeline_output_dir_comment": "Periods5_ww1bb1",
            "vp_output_dir_comment": "Periods5_ww1bb1",
            "vp_Pseudocolor_colortable": "PeriodicBW",
        },
        {
            "Pseudocolor_periodic_num_periods": 5,
            "distance_ww": 2.0,
            "distance_bb": 2.0,
            "cips_pipeline_output_dir_comment": "Periods5_ww2bb2",
            "vp_output_dir_comment": "Periods5_ww2bb2",
            "vp_Pseudocolor_colortable": "PeriodicBW",
        },
        {
            "Pseudocolor_periodic_num_periods": 5,
            "distance_ww": 3.0,
            "distance_bb": 3.0,
            "cips_pipeline_output_dir_comment": "Periods5_ww3bb3",
            "vp_output_dir_comment": "Periods5_ww3bb3",
            "vp_Pseudocolor_colortable": "PeriodicBW",
        },
    ]

    # run variations
    for i, var_params in enumerate(variations):
        print(f"\n--- Starting Variation {i+1}/{len(variations)}: {var_params['vp_output_dir_comment']} ---")

        # Prepare arguments for CIPS_pipeline
        pipeline_args = {
            'input_dir': cips_variations_1_output_dir, # set the directory for the pipeline
            "cips_pipeline_output_dir_comment": var_params["cips_pipeline_output_dir_comment"], # For overall pipeline logging

            # Visit_Projector_1 arguments
            "vp_output_dir_comment": var_params["vp_output_dir_comment"], # VP1 uses this for its own subfolder name
            "vp_Pseudocolor_colortable": var_params["vp_Pseudocolor_colortable"],
            "vp_invertColorTable": var_params.get("vp_invertColorTable", 0),

            # Add the new periodic colortable parameters
            "Pseudocolor_periodic_num_periods": var_params["Pseudocolor_periodic_num_periods"],
            "distance_ww": var_params["distance_ww"],
            "distance_bb": var_params["distance_bb"],
            # Keep default values for wb and bw
            "distance_wb": 1.0,
            "distance_bw": 1.0,
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