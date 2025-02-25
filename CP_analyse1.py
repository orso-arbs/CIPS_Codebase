import numpy as np
import matplotlib.pyplot as plt
from cellpose import plot, utils, io
from cellpose.io import imread
from cellpose import plot
import cellpose
import os
import datetime
import time
import glob

### start inform
start_time = time.time()
current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
print(f"\n {os.path.basename(__file__)}: ", datetime.datetime.now(), "\n")



### Settings
seg_location = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Cellpose\Cellpose1\CP_Results_2025-02-24_21-27"
img_location = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Cellpose\Cellpose1\BW 134 ball flame - Crop small"



### I/O

# Find all segmentation files
seg_files = glob.glob(os.path.join(seg_location, '*_seg.npy'))
#print(f"Found {len(seg_files)} files in seg_files \n")

# Load segmentation data and extract filenames
all_segs = []
seg_filenames = []
for seg_file in seg_files:
    seg = np.load(seg_file, allow_pickle=True).item()
    all_segs.append(seg)
    filename = os.path.basename(seg_file).replace('_seg.npy', '')
    seg_filenames.append(filename)
    #print(f"Adding file to all_segs: {seg_file}")  # Print the file being added
print(f"Loaded {len(all_segs)} seg files \n")

# Find all image files
img_files = glob.glob(os.path.join(img_location, '*.png'))
#print(f"Found {len(img_files)} files in img_files \n")

# Load image data
all_images = []
for img_file in img_files:
    img = io.imread(img_file)
    all_images.append(img)
    #print(f"Adding file to all_images: {img_file}")  # Print the file being added
print(f"Loaded {len(all_images)} images \n")

if 1==0: # Print all keys from each seg dictionary and their properties
    for i, seg in enumerate(all_segs):
        print(f"\n\n Keys for seg {i+1} ({seg_filenames[i]}):")
        print(f"filename key: {seg['filename']}")
        for key, value in seg.items():
            value_type = type(value)
            value_size = value.size if hasattr(value, 'size') else 'N/A'
            value_shape = value.shape if hasattr(value, 'shape') else 'N/A'
            value_length = len(value) if hasattr(value, '__len__') else 'N/A'
            example_value = value[0] if hasattr(value, '__getitem__') and hasattr(value, '__len__') and len(value) > 0 else value
            print(f"  Key: {key}")
            print(f"    Type: {value_type}")
            print(f"    Size: {value_size}")
            print(f"    Shape: {value_shape}")
            print(f"    Length: {value_length}")
            print(f"    Example value: {example_value}")

            # If the value is a list, print properties of each element in the list
            if isinstance(value, list):
                for j, item in enumerate(value):
                    item_type = type(item)
                    item_size = item.size if hasattr(item, 'size') else 'N/A'
                    item_shape = item.shape if hasattr(item, 'shape') else 'N/A'
                    item_length = len(item) if hasattr(item, '__len__') else 'N/A'
                    example_item_value = item[0] if hasattr(item, '__getitem__') and hasattr(item, '__len__') and len(item) > 0 else item
                    print(f"    List item {j}:")
                    print(f"      Type: {item_type}")
                    print(f"      Size: {item_size}")
                    print(f"      Shape: {item_shape}")
                    print(f"      Length: {item_length}")
                    print(f"      Example value: {example_item_value}")
        print("\n")



### Operate

all_diameter_tuple = []
all_diameter_distribution = []
all_median_diameters = []
all_mean_diameters = []
all_N_cells = []
all_relative_pixel_frequencies = []
all_relative_diameter_frequencies = []

if len(all_images) != len(all_segs):
    raise ValueError("Number of images and segmentations do not match")
else:
    for i in range(len(all_images)):
        img = all_images[i]
        seg = all_segs[i]

        # read masks and number of cells
        masks_i = seg['masks']
        N_cells_i = np.max(masks_i)
        all_N_cells.append(N_cells_i)
        print(f"Image {i+1} has {N_cells_i} cells")

        # extract diameter tuple and from it mean and complete distribution
        diameters_tuple_i = cellpose.utils.diameters(masks_i)
        all_diameter_tuple.append(diameters_tuple_i)

        median_diameter_i = diameters_tuple_i[0]
        all_median_diameters.append(median_diameter_i)

        diameter_array_i = diameters_tuple_i[1]
        all_diameter_distribution.append(diameter_array_i)

        # Calculate the mean diameter
        mean_diameter_i = np.mean(diameter_array_i)
        all_mean_diameters.append(mean_diameter_i)

        # Calculate the relative frequency of each diameter in diameter_array_i
        unique_diameters, counts_diameters = np.unique(diameter_array_i, return_counts=True)
        total_diameters = diameter_array_i.size
        relative_diameter_frequencies = counts_diameters / total_diameters
        all_relative_diameter_frequencies.append(relative_diameter_frequencies)

        # Calculate the relative frequency of each cell number in masks_i
        unique, counts = np.unique(masks_i, return_counts=True)
        total_pixels = masks_i.size
        relative_frequencies = counts / total_pixels
        all_relative_pixel_frequencies.append(relative_frequencies)


        if 1==0: # Plot the diameters
            plt.figure()
            plt.hist(diameters_i, bins=20, edgecolor='black')
            plt.title(f'Diameters for Image {i+1}')
            plt.xlabel('Diameter')
            plt.ylabel('Frequency')
            plt.show()

        if 1==0: # plot the image and outlines
            img_height, img_width = img.shape[:2]
            print(f"Original image size: {img_width}x{img_height}")

            fig = plt.figure(figsize=(10,10))
            outlines = utils.outlines_list(seg['masks'])
            plt.imshow(img, cmap='gray')
            for o in outlines:
                plt.plot(o[:,0], o[:,1], color='b')
            plt.show()





### Save/Print Results

# Plot all diameter distributions with the image number on the curve and the mean as a horizontal line
plt.figure(figsize=(10, 6))
colors = plt.cm.get_cmap('tab10', len(all_diameter_distribution))

if 1==0:   # Plot all diameter distributions with the image number on the curve and the mean as a horizontal line
    for i, diameter_array in enumerate(all_diameter_distribution):
        plt.plot(diameter_array, label=f'{seg_filenames[i]}', color=colors(i))
        plt.axhline(y=all_median_diameters[i], color=colors(i), linestyle='--', label=f'Median {seg_filenames[i]}')
        plt.axhline(y=all_mean_diameters[i], color=colors(i), linestyle='-', label=f'Mean {seg_filenames[i]}')
        plt.hist(diameter_array, bins=20, alpha=0.5, label=f'{seg_filenames[i]}', color=colors(i))

    plt.xlabel('Cell Index')
    plt.ylabel('Diameter')
    plt.title('Diameter Distributions for All Images')
    plt.legend()
    plt.show()

if 1==0:    # Plot all relative diameter frequencies with the image number on the curve and the mean as a horizontal line
    plt.figure(figsize=(10, 6))
    colors = plt.cm.get_cmap('tab10', len(all_diameter_distribution))   

    for i, (diameter_array, relative_diameter_frequencies) in enumerate(zip(all_diameter_distribution, all_relative_diameter_frequencies)):
        unique_diameters, counts_diameters = np.unique(diameter_array, return_counts=True)
        plt.plot(unique_diameters, relative_diameter_frequencies, label=f'{seg_filenames[i]}', color=colors(i))
        plt.axhline(y=all_median_diameters[i], color=colors(i), linestyle='--', label=f'Median {seg_filenames[i]}')
        plt.axhline(y=all_mean_diameters[i], color=colors(i), linestyle='-', label=f'Mean {seg_filenames[i]}')  

    plt.xlabel('Diameter')
    plt.ylabel('Frequency')
    plt.title('Diameter Frequency Distributions for All Images')
    plt.legend()
    plt.show()



if 1==0:    # Plot histograms for the diameter distributions
    plt.figure(figsize=(10, 6))
    for i, diameter_array in enumerate(all_diameter_distribution):
        plt.hist(diameter_array, bins=20, alpha=0.5, label=f'{seg_filenames[i]}', color=colors(i))
    # 
    plt.xlabel('Diameter')
    plt.ylabel('Frequency')
    plt.title('Histogram of Diameter Distributions for All Images')
    plt.legend()
    plt.show()





### end inform
end_time = time.time()
elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)
print(f"\n Code Completely Executed in {int(minutes)} min {seconds:.2f} sec \n")