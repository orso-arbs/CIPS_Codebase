import numpy as np
import pandas as pd
import os
import pickle
import Format_1 as F_1
from cellpose import utils # Needed for utils.diameters

@F_1.ParameterLog(max_size = 1024 * 10, log_level = 0)
def Spherical_Reconstruction_1(
    # input
    input_dir, # Should be the output directory of CP_extract_1

    # output and logging
    Spherical_Reconstruction_log_level = 0,
    output_dir_manual = "", output_dir_comment = "",
    ):


    #################################################### I/O
    # Use the input_dir (output of CP_extract) as the base for the new output dir
    output_dir = F_1.F_out_dir(input_dir, __file__, output_dir_comment = output_dir_comment) # Format_1 required definition of output directory

    #################################################### Load Extracted Data

    dimentionalised_data_path = os.path.join(input_dir, 'dimentionalised_DataFrame.pkl')
    print(f"\n Loading extracted data from: {dimentionalised_data_path} \n") if Spherical_Reconstruction_log_level >= 1 else None
    try:
        dimentionalised_df = pd.read_pickle(dimentionalised_data_path)
    except FileNotFoundError:
        print(f"Error: Could not find extracted data file at {dimentionalised_data_path}")
        print("Ensure CP_dimentionalise ran successfully and produced 'dimentionalised_DataFrame.pkl' in the specified input directory.")
        return None # Or raise an error
    
    # Get number of images/rows from loaded data
    N_images = len(dimentionalised_df)
