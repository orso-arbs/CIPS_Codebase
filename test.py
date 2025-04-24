import pandas as pd

Plots = ["Pseudocolor-velocity_magnitude Isosurface-temperature3"], # Plots
Image_filenames_VisIt = ["0.png", "1.png", "2.png", "3.png",]
State_range = [0, 1, 2, 3] # State range for each image
Times_VisIt = [0.0, 0.1, 0.2, 0.3] # Times for each image
R_Average_VisIt = [0.5, 0.6, 0.7, 0.8] # R_Average for each image

VisIt_data_df = pd.DataFrame({
    'Plot': Plots[0] * len(State_range),
    'Image_filename_VisIt': Image_filenames_VisIt,
    'State_range_VisIt': State_range,
    'Time_VisIt': Times_VisIt,
    'R_Average_VisIt': R_Average_VisIt,
    })


extracted_df.at[i, 'Plot_VisIt'] = VisIt_data[i]
extracted_df.at[i, 'Image_filename_VisIt'] = VisIt_data[i]
extracted_df.at[i, 'State_range_VisIt'] = VisIt_data[i]
extracted_df.at[i, 'Time_VisIt'] = VisIt_data[i]
extracted_df.at[i, 'R_Average_VisIt'] = VisIt_data[i]



print(VisIt_data_df)
