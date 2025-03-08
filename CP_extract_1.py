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
        CP_model_type = None, diameter_training = None,
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
        rgba_image = sk_io.imread(image_file)
        
        # Convert to grayscale by ignoring the alpha channel
        grayscale_image = color.rgb2gray(rgba_image[..., :3])
        
        # Append both RGBA and grayscale images as a tuple
        all_images.append(rgba_image)
        all_grayscale_images.append(grayscale_image)
    N_images = len(all_images)
    print(f"Loaded {N_images} images \n") if CP_extract_log_level == 1 else None

    if N_images != N_seg:
        raise ValueError("Number of images and segmentations do not match")



    ### Load data
    print(f"\n Extracting data \n")

    # Initialize DataFrame
    df_columns = [
        'image_file_name', 'image_file_path', 'image_Nx', 'image_Ny',
        'seg_file_name', 'seg_file_path', 'ismanual', 'CP_model_type', 'channels',
        'flows0', 'flows1', 'flows2', 'flows3', 'flows4',
        'diameter_estimate', 'diameter_training',
        'diameter_mean', 'diameter_median', 'diameter_distribution', 'outlines', 'masks', 'N_cells',
        'A_image', 'A_empty', 'A_FB', 'Ar_FBperimage', 'D_FB',
        'A_CP_mask', 'Ar_CP_maskperImage', 'Ar_CP_maskperFB',
    ]
    df = pd.DataFrame(columns=df_columns)



    for i in range(N_images):
        image_i = all_images[i]
        grayscale_image_i = all_grayscale_images[i]
        seg_i = all_segs[i]

        # read seg
        masks_i = seg_i['masks']
        outlines_i = seg_i['outlines']
        flow_i = seg_i['flows']
        diameter_estimate_i = seg_i["diameter"]
        channels = seg_i["chan_choose"]
        ismanual = seg_i['ismanual']
        seg_image_filename = seg_i["filename"] # gives seg filename though it shoud give images file name. weird...

        # training diameter
        if diameter_training is None and CP_model_type is not None:
            if CP_model_type in ['cyto3', "cyto", "cyto2", "cyto3"]:
                diameter_training = 30
            if CP_model_type == "nuclei":
                diameter_training = 17
        elif diameter_training is None and CP_model_type is None:
            diameter_training = None
            print("NB: diameter_training can't be deduced. supply it or a standard CP_model_type as argument to CP_extract_1")

        # extract diameter tuple and from it mean and complete distribution
        diameters_tuple_i = utils.diameters(masks_i)
        median_diameter_i = diameters_tuple_i[0]
        diameter_array_i = diameters_tuple_i[1]
        mean_diameter_i = np.mean(diameter_array_i)

        # Calculate the relative frequency of each diameter in diameter_array_i
        unique_diameters, counts_diameters = np.unique(diameter_array_i, return_counts=True)
        total_diameters = diameter_array_i.size
        relative_diameter_frequencies = counts_diameters / total_diameters

        # CP effectiveness measures
        N_cells_i = np.max(masks_i)
        image_Nx = image_i.shape[0]
        image_Ny = image_i.shape[1]
        A_image = image_Nx * image_Ny
        A_empty = np.sum(grayscale_image_i == 1)
        A_FB = A_image - A_empty
        Ar_FBperimage = A_FB / A_image
        D_FB = math.sqrt(A_FB) * 4 / math.pi

        A_CP_mask = np.count_nonzero(masks_i != 0)
        Ar_CP_maskperImage = A_CP_mask / A_image
        Ar_CP_maskperFB = A_CP_mask / A_FB

        # Create a new DataFrame row
        new_row = pd.DataFrame([{
            'image_file_name': image_files[i], # specific image
            'image_file_directory': image_input_dir, # all images
            'image_Nx': image_Nx,
            'image_Ny': image_Ny,
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

            'diameter_training': diameter_training,
            'diameter_estimate': diameter_estimate_i,
            'diameter_mean': median_diameter_i,
            'diameter_median': mean_diameter_i,
            'diameter_distribution': diameter_array_i,

            'N_cells': N_cells_i,
            'A_image': A_image,
            'A_empty': A_empty,
            'A_FB': A_FB,
            'Ar_FBperimage': Ar_FBperimage,
            'D_FB': D_FB,
            'A_CP_mask': A_CP_mask,
            'Ar_CP_maskperImage': Ar_CP_maskperImage,
            'Ar_CP_maskperFB': Ar_CP_maskperFB
        }])

        # Concatenate the new row to the DataFrame
        df = pd.concat([df, new_row], ignore_index=True)

    # Print columns that have None values
    none_columns = df.columns[df.isnull().any()].tolist()
    print(f"NB: Columns with None values: {none_columns}")


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




