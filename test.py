import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, plot, utils, io
import datetime
import glob
import os
import time
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(r"C:/Users/obs/OneDrive/ETH/ETH_MSc/Masters Thesis/Python Code/Python_Orso_Utility_Scripts_MscThesis")) # dir containing Format_1 
import Format_1 as F_1


input_dir

value = F_1.get_json_value(input_dir, json_data, *path_keys)














# Example usages
input_dir_value = get_json_value(data, "CP_segment_1", "arguments", 0, "value")
script_name_value = get_json_value(data, "CP_segment_1", "script_name")
returned_value2_value = get_json_value(data, "CP_segment_1", "returned_values", 1, "value")

print(f"Input Directory: {input_dir_value}")
print(f"Script Name: {script_name_value}")
print(f"Returned Value 2: {returned_value2_value}")
