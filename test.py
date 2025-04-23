import pandas as pd
import Format_1 as F_1

CP_model_type = "cyto3"
CP_model_path = "path/to/model"
gpu = True
diameter_estimate_guess = None
diameter_training_px = 30
flow_threshold = 0.4
cellprob_threshold = 0.0
resample = True
niter = 20
output_dir_comment = "cyto3"



F_1.debug_info(output_dir_comment)

CP_settings = {
    "CP_model_type": CP_model_type,
    "CP_model_path": CP_model_path,
    "gpu": gpu,
    "diameter_estimate_guess": diameter_estimate_guess,
    "diameter_training_px": diameter_training_px,
    "flow_threshold": flow_threshold,
    "cellprob_threshold": cellprob_threshold,
    "resample": resample,
    "niter": niter,
    "CP_segment_output_dir_comment": output_dir_comment,
}
# Convert to DataFrame (single row)
CP_settings_df = pd.DataFrame([CP_settings])
F_1.debug_info(CP_settings_df["CP_segment_output_dir_comment"])
F_1.debug_info(CP_settings_df["CP_model_type"])
print(CP_settings_df["CP_model_type"])