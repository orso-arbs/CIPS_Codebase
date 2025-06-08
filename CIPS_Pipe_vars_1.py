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
    cips_variation_output_dir_comment="ColorTables_BW_variations_2000px", 
):
    #################################################### I/O
    cips_variations_1_output_dir = F_1.F_out_dir(input_dir = input_dir, script_path = __file__, output_dir_comment = cips_variation_output_dir_comment, output_dir_manual = cips_variation_output_dir_manual)
    print(f"CIPS_variations_1_output_dir: {cips_variations_1_output_dir}")

    ####################################################

    # create Error log file
    error_log_file = os.path.join(cips_variations_1_output_dir, "variation_errors.txt")

    # Define parameter variations
    variations = [
        # {
        #     "cips_pipeline_output_dir_comment": "BW",
        #     "vp_output_dir_comment": "PointWiseCustom",
        #     "vp_Pseudocolor_colortable": "PointWise",
        #     "vp_invertColorTable": 0,
        #     "pointwise_color_points": [
        #         [0.0, 0, 0, 0, 255], # Black
        #         [1.0, 255, 255, 255, 255],  # White
        #     ]
        # },
        # {
        #     "cips_pipeline_output_dir_comment": "BBWW",
        #     "vp_output_dir_comment": "PointWiseCustom",
        #     "vp_Pseudocolor_colortable": "PointWise",
        #     "vp_invertColorTable": 0,
        #     "pointwise_color_points": [
        #         [0.0, 0, 0, 0, 255], # Black
        #         [0.3, 0, 0, 0, 255], # Black
        #         [0.7, 255, 255, 255, 255],  # White
        #         [1.0, 255, 255, 255, 255],  # White
        #     ]
        # },
        # {
        #     "cips_pipeline_output_dir_comment": "WB",
        #     "vp_output_dir_comment": "PointWiseCustom",
        #     "vp_Pseudocolor_colortable": "PointWise",
        #     "vp_invertColorTable": 0,
        #     "pointwise_color_points": [
        #         [0.0, 255, 255, 255, 255],  # White
        #         [1.0, 0, 0, 0, 255], # Black
        #     ]
        # },
        # {
        #     "cips_pipeline_output_dir_comment": "WWBB",
        #     "vp_output_dir_comment": "PointWiseCustom",
        #     "vp_Pseudocolor_colortable": "PointWise",
        #     "vp_invertColorTable": 0,
        #     "pointwise_color_points": [
        #         [0.0, 255, 255, 255, 255],  # White
        #         [0.3, 255, 255, 255, 255],  # White
        #         [0.7, 0, 0, 0, 255], # Black
        #         [1.0, 0, 0, 0, 255], # Black
        #     ]
        # },
        {
            "cips_pipeline_output_dir_comment": "WBW",
            "vp_output_dir_comment": "PointWiseCustom",
            "vp_Pseudocolor_colortable": "PointWise",
            "vp_invertColorTable": 0,
            "pointwise_color_points": [
                [0.0, 255, 255, 255, 255],  # White
                [0.5, 0, 0, 0, 255], # Black
                [1.0, 255, 255, 255, 255],  # White
            ]
        },
        {
            "cips_pipeline_output_dir_comment": "WWBBWW",
            "vp_output_dir_comment": "PointWiseCustom",
            "vp_Pseudocolor_colortable": "PointWise",
            "vp_invertColorTable": 0,
            "pointwise_color_points": [
                [0.0, 255, 255, 255, 255],  # White
                [0.3, 255, 255, 255, 255],  # White
                [0.45, 0, 0, 0, 255], # Black
                [0.55, 0, 0, 0, 255], # Black
                [0.7, 255, 255, 255, 255],  # White
                [1.0, 255, 255, 255, 255],  # White
            ]
        },
        {
            "cips_pipeline_output_dir_comment": "BWB",
            "vp_output_dir_comment": "PointWiseCustom",
            "vp_Pseudocolor_colortable": "PointWise",
            "vp_invertColorTable": 0,
            "pointwise_color_points": [
                [0.0, 0, 0, 0, 255], # Black
                [0.5, 255, 255, 255, 255],  # White
                [1.0, 0, 0, 0, 255], # Black
            ]
        },
        {
            "cips_pipeline_output_dir_comment": "BBWWBB",
            "vp_output_dir_comment": "PointWiseCustom",
            "vp_Pseudocolor_colortable": "PointWise",
            "vp_invertColorTable": 0,
            "pointwise_color_points": [
                [0.0, 0, 0, 0, 255],  # Black
                [0.3, 0, 0, 0, 255],   # Black
                [0.45, 255, 255, 255, 255],  # White
                [0.55, 255, 255, 255, 255],  # White
                [0.7, 0, 0, 0, 255],     # Black
                [1.0, 0, 0, 0, 255]    # Black
            ]
        },
    ]

    # run variations
    for i, var_params in enumerate(variations):
        print(f"\n\n\n--- Starting Variation {i+1}/{len(variations)}: {var_params['vp_output_dir_comment']} ---")

        # Prepare arguments for CIPS_pipeline
        pipeline_args = {
            'input_dir': cips_variations_1_output_dir,
            "cips_pipeline_output_dir_comment": var_params["cips_pipeline_output_dir_comment"],

            # Visit_Projector_1 arguments
            "vp_output_dir_comment": var_params["vp_output_dir_comment"],
            "vp_Pseudocolor_colortable": var_params["vp_Pseudocolor_colortable"],
            "vp_invertColorTable": var_params.get("vp_invertColorTable", 0),
                        
            # Add pointwise_color_points if present
            "pointwise_color_points": var_params.get("pointwise_color_points", None),
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