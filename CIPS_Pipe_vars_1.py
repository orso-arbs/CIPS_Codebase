# CIPS variations code BW segs manual

import os
import datetime
import traceback
import Format_1 as F_1

from CIPS_Pipe_1 import CIPS_pipeline
import CombineVariationPlotMP4 as cvp

@F_1.ParameterLog(max_size = 1024 * 10) # 10KB 
def CIPS_variation_1(
    # General control
    input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations",
    cips_variation_output_dir_manual="",
    cips_variation_output_dir_comment="makeItRun_BWvars_2000px_flowthresh0p5_manualbatch40_batchsize4_bsize160_second_attempt", 
):
    #################################################### I/O
    cips_variations_1_output_dir = F_1.F_out_dir(input_dir = input_dir, script_path = __file__, output_dir_comment = cips_variation_output_dir_comment, output_dir_manual = cips_variation_output_dir_manual)
    print(f"CIPS_variations_1_output_dir: {cips_variations_1_output_dir}")

    ####################################################

    # create Error log file
    error_log_file = os.path.join(cips_variations_1_output_dir, "variation_errors.txt")

    # Define parameter variations
    variations = [
        {
            "cips_pipeline_output_dir_comment": "BWB_2000px_manualbatch40_batchsize4_bsize160",
            "run_visit_projector": False, 
            "cips_VP1_output_dir_override": r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250610_0529590\20250610_0529590",
            "cps_output_dir_comment": "BWB_2000px_manualbatch40_batchsize4_bsize160",
            "cps_batch_size": 2,
            "cps_bsize": 160,
        },
        {
            "cips_pipeline_output_dir_comment": "BBWWBB_2000px_manualbatch40_batchsize4_bsize160",
            "run_visit_projector": False, 
            "cips_VP1_output_dir_override": r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250610_0646347\20250610_0646347",
            "cps_output_dir_comment": "BBWWBB_2000px_manualbatch40_batchsize4_bsize160",
            "cps_batch_size": 2,
            "cps_bsize": 160,
        },
        {
            "cips_pipeline_output_dir_comment": "WB_2000px_manualbatch40_batchsize4_bsize160",
            "run_visit_projector": False, 
            "cips_VP1_output_dir_override": r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250610_0803025\20250610_0803025",
            "cps_output_dir_comment": "WB_2000px_manualbatch40_batchsize4_bsize160",
            "cps_batch_size": 2,
            "cps_bsize": 160,
        },
        {
            "cips_pipeline_output_dir_comment": "WWBB_2000px_manualbatch40_batchsize4_bsize160",
            "run_visit_projector": False, 
            "cips_VP1_output_dir_override": r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250610_0916439\20250610_0916439",
            "cps_output_dir_comment": "WWBB_2000px_manualbatch40_batchsize4_bsize160",
            "cps_batch_size": 2,
            "cps_bsize": 160,
        },
    ]

    # run variations
    for i, var_params in enumerate(variations):
        print(f"\n\n\n--- Starting Variation {i+1}/{len(variations)}: {var_params['cips_pipeline_output_dir_comment']} ---")

        # Prepare arguments for CIPS_pipeline
        pipeline_args = {
            'input_dir': cips_variations_1_output_dir,
            "cips_pipeline_output_dir_comment": var_params["cips_pipeline_output_dir_comment"],

            # run segmentation only
            "run_visit_projector": var_params["run_visit_projector"],
            "cips_VP1_output_dir_override": var_params["cips_VP1_output_dir_override"],
            "cps_output_dir_comment": var_params["cps_output_dir_comment"],
            "cps_batch_size": var_params["cps_batch_size"],
            "cps_bsize": var_params["cps_bsize"],

        }

        try:
            CIPS_pipeline(**pipeline_args)
            print(f"--- Variation {var_params['cips_pipeline_output_dir_comment']} completed successfully. ---")

        except Exception as e:
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
    cvp.combine_variation_mp4s(cips_variations_1_output_dir, log_level=1) 

    print("\nVariation runs completed")
    F_1.ding()
    return cips_variations_1_output_dir

if __name__ == "__main__":
    CIPS_variation_1()


