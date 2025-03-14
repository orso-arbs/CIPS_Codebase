import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, plot, utils, io
import datetime
import glob
import os
import time
import pandas as pd
import math
from skimage import io as sk_io, color
import re
import csv
import pickle


import sys
import os
sys.path.append(os.path.abspath(r"C:/Users/obs/OneDrive/ETH/ETH_MSc/Masters Thesis/Python Code/Python_Orso_Utility_Scripts_MscThesis")) # dir containing Format_1 
import Format_1 as F_1

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)






@F_1.ParameterLog(max_size = 1024 * 10, log_level = 0) # 0.1KB per smallest unit in return (8 bits per ASCII character)
def CP_extract_1(
        input_dir,
        masks = None, flows = None, styles = None, diameter_estimate = None, # can be implemented as input
        CP_model_type = None, diameter_training_px = None,
        CP_extract_log_level = 0,
        ):
    
    ### output 
    output_dir = F_1.F_out_dir(input_dir, __file__, output_dir_comment = "") # Format_1 required definition of output directory



    ### Load data
    print(f"\n Loading data \n")

    # Load segmentation data
    seg_location = input_dir
    seg_files = glob.glob(os.path.join(seg_location, '*_seg.npy'))
    all_segs = []
    seg_filenames = []
    for seg_file in seg_files:
        seg = np.load(seg_file, allow_pickle=True).item()
        all_segs.append(seg)
        filename = os.path.basename(seg_file).replace('_seg.npy', '')
        seg_filenames.append(filename)
        #print(f"Adding file to all_segs: {seg_file}")  # Print the file being added
    N_seg = len(all_segs)
    print(f"Loaded {N_seg} seg files") if CP_extract_log_level == 1 else None

    # Load image data
    image_input_dir = os.path.dirname(input_dir) # one folder above
    image_files = glob.glob(os.path.join(image_input_dir, '*.png'))
    all_images = []
    all_grayscale_images = []
    image_number = []
    for image_file in image_files:
        # Load the image (RGBA)
        rgbA_image_px2 = sk_io.imread(image_file)
        
        # Convert to grayscale by ignoring the alpha channel
        grayscale_image = color.rgb2gray(rgbA_image_px2[..., :3])
        
        # Append both RGBA and grayscale images as a tuple
        all_images.append(rgbA_image_px2)
        all_grayscale_images.append(grayscale_image)

        # Extract the numeric part from the filename
        match = re.search(r'_(\d+)\.png$', os.path.basename(image_file))
        if match:
            image_number.append(int(match.group(1)))
        else:
            image_number.append(None)  # Handle cases where no number is found

    N_images = len(all_images)
    print(f"Loaded {N_images} images \n") if CP_extract_log_level == 1 else None

    if N_images != N_seg:
        raise ValueError("Number of images and segmentations do not match")

    # import Cellpose model settings
    CP_settings_file = f"{input_dir}/CP_settings.pkl"
    with open(CP_settings_file, "rb") as file:
        params = pickle.load(file)
        gpu = params["gpu"]
        diameter_estimate_manual = params["diameter_estimate_manual"]
        CP_segment_output_dir_comment = params["CP_segment_output_dir_comment"]
        flow_threshold = params["flow_threshold"]
        cellprob_threshold = params["cellprob_threshold"]
        resample = params["resample"]
        niter = params["niter"]
        CP_model_type = params["CP_model_type"]

    ### Tabularize data
    print(f"\n Extracting data \n")

    # Initialize DataFrame
    CP_extract_df_columns = [
        'image_file_name', 'image_file_path', 'image_number', 'image_Nx_px', 'image_Ny_px',
        'seg_file_name', 'seg_file_path', 'ismanual', 'CP_model_type', 'channels',
        'flows0', 'flows1', 'flows2', 'flows3', 'flows4',
        'diameter_estimate', 'diameter_training_px',
        'diameter_mean_px', 'diameter_median_px', 'diameter_distribution_px', 'outlines', 'masks', 'N_cells',
        'A_image_px2', 'A_empty_px2', 'A_FB_px2', 'Ar_px2_FBperimage', 'D_FB_px',
        'A_CP_mask_px', 'Ar_px2_CP_maskperImage', 'Ar_px2_CP_maskperFB',
    ]
    CP_extract_df = pd.DataFrame(columns=CP_extract_df_columns)

    # Preallocate columns to avoid growing the DataFrame dynamically
    for col in CP_extract_df_columns:
        CP_extract_df[col] = np.nan  # Initialize all columns with NaN

    # Explicitly set multiple columns to 'object' to allow lists or complex data types
    columns_with_lists = [ 
        'ismanual', 'flows0', 'flows1', 'flows2', 'flows3', 'flows4', 'channels',
        'outlines', 'masks', 'diameter_distribution_px',
    ]
    for column in columns_with_lists:
        CP_extract_df[column] = pd.Series(dtype="object")  # Explicitly set dtype to 'object'



    for i in range(N_images):
        image_i = all_images[i]
        grayscale_image_i = all_grayscale_images[i]
        seg_i = all_segs[i]

        # read seg
        masks_i = seg_i['masks']
        outlines_i = seg_i['outlines']
        flow_i = seg_i['flows']
        diameter_estimate_px_i = seg_i["diameter"]
        channels = seg_i["chan_choose"]
        ismanual = seg_i['ismanual']
        seg_image_filename = seg_i["filename"] # gives seg filename though it shoud give images file name. weird...

        # training diameter
        if diameter_training_px is None and CP_model_type is not None:
            if CP_model_type in ['cyto3', "cyto", "cyto2", "cyto3"]:
                diameter_training_px = 30
            if CP_model_type == "nuclei":
                diameter_training_px = 17
        elif diameter_training_px is None and CP_model_type is None:
            diameter_training_px = None
            print("NB: diameter_training_px can't be deduced. supply it or a standard CP_model_type as argument to CP_extract_1") if CP_extract_log_level == 1 else None

        # extract diameter tuple and from it mean and complete distribution
        diameters_tuple_i = utils.diameters(masks_i)
        median_diameter_px_i = diameters_tuple_i[0]
        diameter_array_px_i = diameters_tuple_i[1]
        mean_diameter_px_i = np.mean(diameter_array_px_i)

        # Calculate the relative frequency of each diameter in diameter_array_px_i
        unique_diameters, counts_diameters = np.unique(diameter_array_px_i, return_counts=True)
        total_diameters = diameter_array_px_i.size
        relative_diameter_frequencies = counts_diameters / total_diameters

        # CP effectiveness measures
        N_cells_i = np.max(masks_i)
        image_Nx_px = image_i.shape[0]
        image_Ny_px = image_i.shape[1]
        A_image_px2 = image_Nx_px * image_Ny_px
        A_empty_px2 = np.sum(grayscale_image_i == 1)
        A_FB_px2 = A_image_px2 - A_empty_px2
        Ar_px2_FBperimage = A_FB_px2 / A_image_px2
        D_FB_px = math.sqrt(A_FB_px2 * 4 / math.pi)
        R_FB_px = D_FB_px / 2

        A_CP_mask_px = np.count_nonzero(masks_i != 0)
        Ar_px2_CP_maskperImage = A_CP_mask_px / A_image_px2
        Ar_px2_CP_maskperFB = A_CP_mask_px / A_FB_px2
        

        # Create a new DataFrame row

        
        CP_extract_df.at[i, 'image_file_name'] = image_files[i]  # Specific image
        CP_extract_df.at[i, 'image_file_path'] = image_input_dir  # All images
        CP_extract_df.at[i, 'image_number'] = image_number[i]
        CP_extract_df.at[i, 'image_Nx_px'] = image_Nx_px
        CP_extract_df.at[i, 'image_Ny_px'] = image_Ny_px
        CP_extract_df.at[i, 'seg_file_name'] = seg_filenames[i]
        CP_extract_df.at[i, 'seg_file_path'] = seg_files[i]
        
        CP_extract_df.at[i, 'CP_model_type'] = params['CP_model_type']
        CP_extract_df.at[i, 'channels'] = channels
        CP_extract_df.at[i, 'gpu'] = params['gpu']
        CP_extract_df.at[i, 'diameter_estimate_manual'] = params['diameter_estimate_manual']
        CP_extract_df.at[i, 'CP_segment_output_dir_comment'] = params['CP_segment_output_dir_comment']
        CP_extract_df.at[i, 'flow_threshold'] = params['flow_threshold']
        CP_extract_df.at[i, 'cellprob_threshold'] = params['cellprob_threshold']
        CP_extract_df.at[i, 'resample'] = params['resample']
        CP_extract_df.at[i, 'niter'] = params['niter']

        CP_extract_df.at[i, 'ismanual'] = ismanual
        CP_extract_df.at[i, 'flows0'] = flow_i[0]
        CP_extract_df.at[i, 'flows1'] = flow_i[1]
        CP_extract_df.at[i, 'flows2'] = flow_i[2]
        CP_extract_df.at[i, 'flows3'] = flow_i[3]
        CP_extract_df.at[i, 'flows4'] = flow_i[4]
        CP_extract_df.at[i, 'outlines'] = outlines_i
        CP_extract_df.at[i, 'masks'] = masks_i

        CP_extract_df.at[i, 'diameter_training_px'] = diameter_training_px
        CP_extract_df.at[i, 'diameter_estimate_px'] = diameter_estimate_px_i
        CP_extract_df.at[i, 'diameter_mean_px'] = mean_diameter_px_i
        CP_extract_df.at[i, 'diameter_median_px'] = median_diameter_px_i
        CP_extract_df.at[i, 'diameter_distribution_px'] = diameter_array_px_i
        CP_extract_df.at[i, 'N_cells'] = N_cells_i
        CP_extract_df.at[i, 'A_image_px2'] = A_image_px2
        CP_extract_df.at[i, 'A_empty_px2'] = A_empty_px2
        CP_extract_df.at[i, 'A_FB_px2'] = A_FB_px2
        CP_extract_df.at[i, 'Ar_px2_FBperimage'] = Ar_px2_FBperimage
        CP_extract_df.at[i, 'D_FB_px'] = D_FB_px
        CP_extract_df.at[i, 'R_FB_px'] = R_FB_px
        CP_extract_df.at[i, 'A_CP_mask_px'] = A_CP_mask_px
        CP_extract_df.at[i, 'Ar_px2_CP_maskperImage'] = Ar_px2_CP_maskperImage
        CP_extract_df.at[i, 'Ar_px2_CP_maskperFB'] = Ar_px2_CP_maskperFB


        # clean diameter_distribution_px
        CP_extract_df['diameter_distribution_px'] = CP_extract_df['diameter_distribution_px'].apply(
            lambda x: np.array([x]) if isinstance(x, np.ndarray) and x.ndim == 0 else x
        )


    # Print columns that have None values
    none_columns = CP_extract_df.columns[CP_extract_df.isnull().any()].tolist()
    print(f"NB: Columns with None values: {none_columns}") if CP_extract_log_level == 1 else None











    ################# Match Images and A11 data^

    print("\n Non Dimentionalissing and matching CP and A11 data \n")  if CP_extract_log_level == 1 else None

    # Load A11 data
    A11_SF_K_mean = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_K_mean_as_mean_stretch_rate_vs_time_manual_extraction.txt")
    A11_SF_N_c = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_N_c_as_number_of_cells_vs_time_manual_extraction.txt")
    A11_SF_R_mean = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_R_mean_as_average_radius_of_the_wrinkled_flame_fron_vs_time_manual_extraction.txt")
    A11_SF_R_mean_dot = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_R_mean_dot_as_first_time_derivative_of_the_average_radius_of_the_wrinkled_flame_front_vs_time_manual_extraction.txt")
    A11_SF_s_a = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_s_a_as_average_normal_component_of_the_absolute_propagation_velocity_vs_time_manual_extraction.txt")
    A11_SF_s_d = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_s_d_as_average_density_weighted_displacement_speed_vs_time_manual_extraction.txt")
    A11_SF_A = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_A_as_flame_surface_area_of_the_wrinkled_spherical_front_vs_time_manual_extraction.txt")
    A11_SF_a_t = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_a_t_as_average_total_aerodynamic_strain_vs_time_manual_extraction.txt")
    A11_SF_iHRR = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_iHRR_as_integral_heat_release_rate_vs_time_manual_extraction.txt")
    A11_SF_K_geom = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_K_geom_as_geometric_stretch_rate_vs_time_manual_extraction.txt")

    # reference vales
    ''' A11 p.36
    "...reference values from detailed description of chemistry is based on the mech-
    anism of Li et al. [2004], consisting of 9 species and 19 non-duplicate
    elementary reactions. The planar premixed flame structure calculated us-
    ing PREMIX [Rupley et al., 1995]..."
    '''
    d_T = 7.516 * 1e-3 # flame thickness
    S_L = 51.44 # laminar flame speed
    T_b = 1843.5 # burned gas temperature

    t_ref = d_T/S_L # flame time scale
    
    t_max = 6.81 # max time estimated from plots
    R0 = 10 * d_T # initial Spherical flame radius

    # Calculate the time values for each DataFrame
    CP_extract_df['time'] = np.nan # initialize time collumn filled with nan
    min_image_num = CP_extract_df["image_number"].min()
    max_image_num = CP_extract_df["image_number"].max()
    if min_image_num == max_image_num:
        CP_extract_df["time"] = 0  # Assign 0 to all if there's no range
        print("Warning: min_image_num == max_image_num")
    else:
        # Linear mapping from [min_image_num, 134] to [0, t_max]
        CP_extract_df["time"] = (CP_extract_df["image_number"] - min_image_num) / (134 - min_image_num) * t_max
        print("CP_extract_df[time]", CP_extract_df["time"]) if CP_extract_log_level == 1 else None


    # Add non-dimensional columns to the DataFrame
    nonDim_columns = [
        'd_T_per_px', 'image_Nx_nonDim', 'image_Ny_nonDim', 'diameter_training_nonDim', 'diameter_estimate_nonDim',
        'diameter_mean_nonDim', 'diameter_median_nonDim', 'diameter_distribution_nonDim',
        'A_image_nonDim2', 'A_empty_nonDim2', 'A_FB_nonDim2', 'D_FB_nonDim', 'R_FB_nonDim', 'A_CP_mask_nonDim',
    ]
    # Add these columns to the existing DataFrame
    for col in nonDim_columns:
        CP_extract_df[col] = np.nan  # Initialize them with NaN
    # Add non-dimensional columns to the DataFrame
    columns_with_lists = [ 
        'diameter_distribution_nonDim',
    ]
    for column in columns_with_lists:
        CP_extract_df[column] = pd.Series(dtype="object")  # Explicitly set dtype to 'object'


    # extract pixel to nonDimentionalised length scaling.
    for i in range(N_images):
        print("i = ", i) if CP_extract_log_level == 1 else None


        # find d_T_per_px
        A11_SF_R_mean = A11_SF_R_mean.sort_values(by='time')
        CP_extract_df = CP_extract_df.sort_values(by='time')
        CP_extract_df['R_mean_interpolated'] = np.interp(CP_extract_df['time'], A11_SF_R_mean['time'], A11_SF_R_mean['R_mean']) # Interpolate 'R_mean' at the 'time' values in CP_extract_df
        CP_extract_df['d_T_per_px'] = CP_extract_df['R_mean_interpolated'] / CP_extract_df['R_FB_px']

        # Calculate non-dimensionalised values
        d_T_per_px_i = CP_extract_df.loc[i, 'd_T_per_px']

        image_Nx_nonDim = CP_extract_df.loc[i, "image_Nx_px"] * d_T_per_px_i
        image_Ny_nonDim = CP_extract_df.loc[i, "image_Ny_px"] * d_T_per_px_i

        diameter_training_px_i = CP_extract_df.loc[i, 'diameter_training_px']
        diameter_estimate_px_i = CP_extract_df.loc[i, 'diameter_estimate_px']
        diameter_median_px_i = CP_extract_df.loc[i, 'diameter_median_px']
        diameter_mean_px_i = CP_extract_df.loc[i, 'diameter_mean_px']

        masks_i = CP_extract_df.loc[i, 'masks']
        diameters_tuple_i_placeholder = utils.diameters(masks_i)
        diameter_array_px_i_placeholder = diameters_tuple_i_placeholder[1]
        diameter_distribution_px_i = diameter_array_px_i_placeholder

        diameter_training_nonDim = diameter_training_px_i * d_T_per_px_i if diameter_training_px_i is not None and d_T_per_px_i is not None else None
        diameter_estimate_nonDim_i = diameter_estimate_px_i * d_T_per_px_i if diameter_estimate_px_i is not None and d_T_per_px_i is not None else None
        diameter_median_nonDim_i = diameter_median_px_i * d_T_per_px_i if diameter_median_px_i is not None and d_T_per_px_i is not None else None
        diameter_mean_nonDim_i = diameter_mean_px_i * d_T_per_px_i
        diameter_distribution_nonDim_i = diameter_distribution_px_i * d_T_per_px_i

        A_image_nonDim = CP_extract_df.loc[i, "A_image_px2"] * d_T_per_px_i**2
        A_empty_nonDim = CP_extract_df.loc[i, "A_empty_px2"] * d_T_per_px_i**2
        A_FB_nonDim = CP_extract_df.loc[i, "A_FB_px2"] * d_T_per_px_i**2
        D_FB_nonDim = CP_extract_df.loc[i, "D_FB_px"] * d_T_per_px_i
        R_FB_nonDim = CP_extract_df.loc[i, "R_FB_px"] * d_T_per_px_i
        A_CP_mask_nonDim = CP_extract_df.loc[i, "A_CP_mask_px"] * d_T_per_px_i**2


        CP_extract_df.at[i, 'd_T_per_px_i'] = d_T_per_px_i
        CP_extract_df.at[i, 'image_Nx_nonDim'] = image_Nx_nonDim
        CP_extract_df.at[i, 'image_Ny_nonDim'] = image_Ny_nonDim
        CP_extract_df.at[i, 'diameter_training_nonDim'] = diameter_training_nonDim
        CP_extract_df.at[i, 'diameter_estimate_nonDim'] = diameter_estimate_nonDim_i
        CP_extract_df.at[i, 'diameter_mean_nonDim'] = diameter_mean_nonDim_i
        CP_extract_df.at[i, 'diameter_median_nonDim'] = diameter_median_nonDim_i
        CP_extract_df.at[i, 'diameter_distribution_nonDim'] = diameter_distribution_nonDim_i
        CP_extract_df.at[i, 'A_image_nonDim2'] = A_image_nonDim
        CP_extract_df.at[i, 'A_empty_nonDim2'] = A_empty_nonDim
        CP_extract_df.at[i, 'A_FB_nonDim2'] = A_FB_nonDim
        CP_extract_df.at[i, 'D_FB_nonDim'] = D_FB_nonDim
        CP_extract_df.at[i, 'R_FB_nonDim'] = R_FB_nonDim
        CP_extract_df.at[i, 'A_CP_mask_nonDim'] = A_CP_mask_nonDim

        # clean diameter_distribution_nonDim
        CP_extract_df['diameter_distribution_nonDim'] = CP_extract_df['diameter_distribution_nonDim'].apply(
            lambda x: np.array([x]) if isinstance(x, np.ndarray) and x.ndim == 0 else x
        )











    ### Save

    print(f"\n Saving data \n")

    ### Save DataFrame to CSV
    csv_filename = f'segmentation_DataFrame.csv'
    CP_extract_df.to_csv(os.path.join(output_dir, csv_filename), sep='\t', index=False)

    # Save DataFrame to Pickle
    pickle_filename = f'segmentation_DataFramee.pkl'
    CP_extract_df.to_pickle(os.path.join(output_dir, pickle_filename))
    #
    # # Load DataFrame from Pickle
    # CP_extract_df = pd.read_pickle(os.path.join(seg_location, pickle_filename))

    # Save DataFrame to Excel
    excel_filename = f'segmentation_DataFramee.xlsx'
    CP_extract_df.to_excel(os.path.join(output_dir, excel_filename), index=False)
    #


    ### return
    return output_dir, CP_extract_df # Format_1 requires outpu_dir as first return




