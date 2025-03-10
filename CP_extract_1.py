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
    for image_file in image_files:
        # Load the image (RGBA)
        rgbA_image_px2 = sk_io.imread(image_file)
        
        # Convert to grayscale by ignoring the alpha channel
        grayscale_image = color.rgb2gray(rgbA_image_px2[..., :3])
        
        # Append both RGBA and grayscale images as a tuple
        all_images.append(rgbA_image_px2)
        all_grayscale_images.append(grayscale_image)
    N_images = len(all_images)
    print(f"Loaded {N_images} images \n") if CP_extract_log_level == 1 else None

    if N_images != N_seg:
        raise ValueError("Number of images and segmentations do not match")



    ### Load data
    print(f"\n Extracting data \n")

    # Initialize DataFrame
    df_columns = [
        'image_file_name', 'image_file_path', 'image_Nx_px', 'image_Ny_px',
        'seg_file_name', 'seg_file_path', 'ismanual', 'CP_model_type', 'channels',
        'flows0', 'flows1', 'flows2', 'flows3', 'flows4',
        'diameter_estimate', 'diameter_training_px',
        'diameter_mean_px', 'diameter_median_px', 'diameter_distribution_px', 'outlines', 'masks', 'N_cells',
        'A_image_px2', 'A_empty_px2', 'A_FB_px2', 'Ar_px2_FBperimage', 'D_FB_px',
        'A_CP_mask_px', 'Ar_px2_CP_maskperImage', 'Ar_px2_CP_maskperFB',
        'time',
    ]
    df = pd.DataFrame(columns=df_columns)

    
    ''' A11 p.36
    reference values from detailed description of chemistry is based on the mech-
    anism of Li et al. [2004], consisting of 9 species and 19 non-duplicate
    elementary reactions. The planar premixed flame structure calculated us-
    ing PREMIX [Rupley et al., 1995]'
    '''
    d_T = 7.516 * 1e-3 # flame thickness
    S_L = 51.44 # laminar flame speed
    T_b = 1843.5 # burned gas temperature

    t_ref = d_T/S_L # flame time scale
    
    t_max = 6.81 # max time estimated from plots
    t = np.linspace(0, t_max, 134)
    R0 = 10 * d_T # initial Spherical flame radius

    time_values = np.linspace(0, t_max, N_images)  # Generate time values



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
            print("NB: diameter_training_px can't be deduced. supply it or a standard CP_model_type as argument to CP_extract_1")

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
        D_FB_px = math.sqrt(A_FB_px2) * 4 / math.pi

        A_CP_mask_px = np.count_nonzero(masks_i != 0)
        Ar_px2_CP_maskperImage = A_CP_mask_px / A_image_px2
        Ar_px2_CP_maskperFB = A_CP_mask_px / A_FB_px2

        # Create a new DataFrame row
        new_row = pd.DataFrame([{
            'image_file_name': image_files[i], # specific image
            'image_file_directory': image_input_dir, # all images
            'image_Nx_px': image_Nx_px,
            'image_Ny_px': image_Ny_px,
            'seg_file_name': seg_filenames[i],
            'seg_file_path': seg_files[i],
            'ismanual': ismanual,
            'CP_model_type': CP_model_type,
            'channels': channels,
            'flows0': flow_i[0],
            'flows1': flow_i[1],
            'flows2': flow_i[2],
            'flows3': flow_i[3],
            'flows4': flow_i[4],
            'outlines': outlines_i,
            'masks': masks_i,

            'diameter_training_px': diameter_training_px,
            'diameter_estimate_px': diameter_estimate_px_i,
            'diameter_mean_px': median_diameter_px_i,
            'diameter_median_px': mean_diameter_px_i,
            'diameter_distribution_px': diameter_array_px_i,

            'N_cells': N_cells_i,
            'A_image_px2': A_image_px2,
            'A_empty_px2': A_empty_px2,
            'A_FB_px2': A_FB_px2,
            'Ar_px2_FBperimage': Ar_px2_FBperimage,
            'D_FB_px': D_FB_px,
            'A_CP_mask_px': A_CP_mask_px,
            'Ar_px2_CP_maskperImage': Ar_px2_CP_maskperImage,
            'Ar_px2_CP_maskperFB': Ar_px2_CP_maskperFB,
            'time': time_values[i]  # Assign time based on index
        }])

        # Concatenate the new row to the DataFrame
        df = pd.concat([df, new_row], ignore_index=True)

    # Print columns that have None values
    none_columns = df.columns[df.isnull().any()].tolist()
    print(f"NB: Columns with None values: {none_columns}")


    ################# Dimentionalise

    












    ### Save

    print(f"\n Saving data \n")

    ### Save DataFrame to CSV
    csv_filename = f'segmentation_DataFrame.csv'
    df.to_csv(os.path.join(output_dir, csv_filename), sep='\t', index=False)

    # Save DataFrame to Pickle
    pickle_filename = f'segmentation_DataFramee.pkl'
    df.to_pickle(os.path.join(output_dir, pickle_filename))
    #
    # # Load DataFrame from Pickle
    # df = pd.read_pickle(os.path.join(seg_location, pickle_filename))

    # Save DataFrame to Excel
    excel_filename = f'segmentation_DataFramee.xlsx'
    df.to_excel(os.path.join(output_dir, excel_filename), index=False)
    #


    ### return
    return output_dir, df # Format_1 requires outpu_dir as first return




