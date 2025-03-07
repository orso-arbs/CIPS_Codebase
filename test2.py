import pandas as pd

# Define the column names from the 'data' dictionary
data_columns = [
    'image_file_name', 'image_file_path', 'image_Nx', 'image_Ny', 
    'seg_file_name', 'seg_file_path', 'ismanual', 'CP_model_type', 
    'channels', 'flows0', 'flows1', 'flows2', 'flows3', 'flows4', 
    'diameter_training', 'diameter_estimate', 'diameter_mean', 
    'diameter_median', 'diameter_distribution', 'outlines', 'masks', 
    'N_cells'
]

# Define the column names from 'df_columns'
df_columns = [
    'image_file_name', 'image_file_path', 'image_Nx', 'image_Ny',
    'seg_file_name', 'seg_file_path', 'ismanual', 'model', 'channels',
    'flows0', 'flows1', 'flows2', 'flows3', 'flows4',
    'diameter0', 'diameter1', 'diameter2', 'diameter_estimate', 'diameter_training',
    'diameter_mean', 'diameter_median', 'diameter_distribution', 'outlines', 'masks', 'N_cells'
]

# Find the difference in column names
missing_in_data = set(df_columns) - set(data_columns)
missing_in_df_columns = set(data_columns) - set(df_columns)

print(f"Missing in data: {missing_in_data}")
print(f"Missing in df_columns: {missing_in_df_columns}")
