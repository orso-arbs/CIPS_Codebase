import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cellpose import plot, utils, io
from cellpose.io import imread
from cellpose import plot
import cellpose
import os
import datetime
import time
import glob
import pandas as pd

### start inform
start_time = time.time()
current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
print(f"\n {os.path.basename(__file__)}: ", datetime.datetime.now(), "\n")

### Settings
seg_location = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Cellpose\Cellpose1\CP_Results_2025-02-24_21-27"
img_location = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Cellpose\Cellpose1\BW 134 ball flame - Crop small"
plot_folder_location = os.path.join(seg_location, f'plots_{current_date}')


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
print(f"Loaded {len(all_segs)} seg files")

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



### Extract

print(f"\n Extracting data \n")

# Initialize DataFrame
df_columns = [
    'image_file_name', 'image_file_path', 'image_Nx', 'image_Ny',
    'seg_file_name', 'seg_file_path', 'ismanual', 'model', 'chan_choose',
    'flows0', 'flows1', 'flows2', 'flows3', 'flows4',
    'diameter0', 'diameter1', 'diameter2', 'diameter_estimate', 'diameter_training',
    'diameter_mean', 'diameter_median', 'diameter_distribution', 'outlines', 'masks', 'N_cells'
]
df = pd.DataFrame(columns=df_columns)

if len(all_images) != len(all_segs):
    raise ValueError("Number of images and segmentations do not match")
else:
    for i in range(len(all_images)):
        img = all_images[i]
        seg = all_segs[i]

        # read masks and number of cells
        masks_i = seg['masks']
        N_cells_i = np.max(masks_i)

        # extract diameter tuple and from it mean and complete distribution
        diameters_tuple_i = cellpose.utils.diameters(masks_i)
        median_diameter_i = diameters_tuple_i[0]
        diameter_array_i = diameters_tuple_i[1]

        # Calculate the mean diameter
        mean_diameter_i = np.mean(diameter_array_i)

        # Calculate the relative frequency of each diameter in diameter_array_i
        unique_diameters, counts_diameters = np.unique(diameter_array_i, return_counts=True)
        total_diameters = diameter_array_i.size
        relative_diameter_frequencies = counts_diameters / total_diameters

        # Extract other relevant data from seg
        ismanual = seg.get('ismanual', None)
        model = seg.get('model', None)
        chan_choose = seg.get('chan_choose', None)
        flows = seg.get('flows', [None]*5)
        diameter = seg.get('diameter', [None]*3)
        diameter_estimate = seg.get('diameter_estimate', None)
        diameter_training = seg.get('diameter_training', None)
        outlines = utils.outlines_list(masks_i)

        # Create a new DataFrame row
        new_row = pd.DataFrame([{
            'image_file_name': os.path.splitext(os.path.basename(img_files[i]))[0],  # Remove file extension
            'image_file_path': img_files[i],
            'image_Nx': img.shape[0],
            'image_Ny': img.shape[1],
            'seg_file_name': seg_filenames[i],
            'seg_file_path': seg_files[i],
            'ismanual': ismanual,
            'model': model,
            'chan_choose': chan_choose,
            'flows0': flows[0],
            'flows1': flows[1],
            'flows2': flows[2],
            'flows3': flows[3],
            'flows4': flows[4],
            'diameter0': diameter[0],
            'diameter1': diameter[1],
            'diameter2': diameter[2],
            'diameter_estimate': diameter_estimate,
            'diameter_training': diameter_training,
            'diameter_mean': median_diameter_i,
            'diameter_median': mean_diameter_i,
            'diameter_distribution': diameter_array_i,
            'outlines': outlines,
            'masks': masks_i,
            'N_cells': len(diameter_array_i)
        }])

        # Concatenate the new row to the DataFrame
        df = pd.concat([df, new_row], ignore_index=True)

# Print columns that have None values
none_columns = df.columns[df.isnull().any()].tolist()
print(f"NB: Columns with None values: {none_columns}")


### Save

print(f"\n Saving data \n")

### Save DataFrame to CSV
csv_filename = f'segmentation_data_{current_date}.csv'
df.to_csv(os.path.join(seg_location, csv_filename), sep='\t', index=False)

# Save DataFrame to Pickle
pickle_filename = f'segmentation_data_{current_date}.pkl'
df.to_pickle(os.path.join(seg_location, pickle_filename))
#
# # Load DataFrame from Pickle
# df = pd.read_pickle(os.path.join(seg_location, pickle_filename))

# Save DataFrame to Excel
excel_filename = f'segmentation_data_{current_date}.xlsx'
df.to_excel(os.path.join(seg_location, excel_filename), index=False)
#
# # Load DataFrame from Excel
# df = pd.read_excel(os.path.join(seg_location, excel_filename))



### Plot
print(f"\n Plotting \n")

# Create a folder to save the plots
if not os.path.exists(plot_folder_location):
    os.makedirs(plot_folder_location)

# auxillary function to plot the data

# Number of rows in the DataFrame
num_rows = len(df)

# Find the maximum frequency for all histograms
bin_size = 2
max_frequency = 0
for i in range(num_rows):
    unique_diameters, counts_diameters = np.unique(df.loc[i, 'diameter_distribution'], return_counts=True)
    bins = np.arange(0, max(unique_diameters) + bin_size, bin_size)
    hist, _ = np.histogram(df.loc[i, 'diameter_distribution'], bins=bins)
    max_frequency = max(max_frequency, hist.max())

for i in range(num_rows): # Plot the data for each row
    # Create a new figure for each row
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Get the image, outlines, and masks for the current row
    img = io.imread(df.loc[i, 'image_file_path'])
    outlines = df.loc[i, 'outlines']
    masks = df.loc[i, 'masks']

    # Plot: original image
    axes[0,0].imshow(img, cmap='gray')
    axes[0,0].set_title(f"Original Image {i+1}")
    axes[0,0].axis('off')

    # Plot: image with outlines
    axes[0,1].imshow(img, cmap='gray')
    for outline in outlines:
        axes[0,1].plot(outline[:, 0], outline[:, 1], color='red')
    axes[0,1].set_title(f"Image {i+1} with Outlines")
    axes[0,1].axis('off')

    # Plot: image with masks
    axes[0,2].imshow(img, cmap='gray')
    axes[0,2].imshow(plot.mask_overlay(img, masks), alpha=0.5, cmap='gist_rainbow')
    axes[0,2].set_title(f"Image {i+1} with Masks")
    axes[0,2].axis('off')

    # Plot: Diameter distribution vs. diameter frequency (bin count histogram)
    unique_diameters, counts_diameters = np.unique(df.loc[i, 'diameter_distribution'], return_counts=True)
    bins = np.arange(0, max(unique_diameters) + bin_size, bin_size)
    axes[1, 0].hist(df.loc[i, 'diameter_distribution'], bins=bins)
    axes[1, 0].set_title("Diameter Distribution")
    axes[1, 0].set_xlabel("Diameter")
    axes[1, 0].set_ylabel("Frequency")

    mean_diameter = df.loc[i, 'diameter_mean']
    median_diameter = df.loc[i, 'diameter_median']
    axes[1, 0].axvline(mean_diameter, color='blue', linestyle='dashed', linewidth=1)
    axes[1, 0].text(mean_diameter, axes[1, 0].get_ylim()[1] * 0.9, f'Mean: {mean_diameter:.2f}', color='blue')
    axes[1, 0].axvline(median_diameter, color='green', linestyle='dashed', linewidth=1)
    axes[1, 0].text(median_diameter, axes[1, 0].get_ylim()[1] * 0.8, f'Median: {median_diameter:.2f}', color='green')

    axes[1, 0].set_xlim(0, df['diameter_distribution'].apply(lambda x: np.max(x)).max()*1.05) # df['diameter_distribution'].min()
    axes[1, 0].set_ylim(0, max_frequency*1.05)

    # Plot 1: Image number vs. median diameter, mean diameter, and amount of cells (up to current image)
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    ax1.plot(range(num_rows), df['diameter_mean'], label='Mean Diameter', color='blue')
    ax1.plot(range(num_rows), df['diameter_median'], label='Median Diameter', color='green')
    ax2.plot(range(num_rows), df['N_cells'], label='Number of Cells', color='red')
    axes[1, 1].axvline(i, color='blue', label=f'shown image: {i:.2f}', linestyle='dashed', linewidth=3)
    #axes[1, 1].text(i, axes[1, 1].get_ylim()[1] * 0.9, f'shown image: {i:.2f}', color='blue')

    ax1.set_xlim(0, num_rows - 1)
    ax1.set_ylim(min(df['diameter_mean'].min(), df['diameter_median'].min()), max(df['diameter_mean'].max(), df['diameter_median'].max())*1.05)
    ax2.set_ylim(df['N_cells'].min(), df['N_cells'].max()*1.05)

    ax1.set_title("Diameter and Cell Count")
    ax1.set_xlabel("Image Number")
    ax1.set_ylabel("Diameter")
    ax2.set_ylabel("Number of Cells")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Plot: Leave empty
    axes[1, 2].axis('off')


    # Adjust layout and save the figure as a PNG file
    plt.tight_layout()
    plot_filename = os.path.join(plot_folder_location, f'plot_{i+1}.png')
    plt.savefig(plot_filename)
    plt.close(fig)


#ToDO

# make colormap on masks be nice and add numbers







### end inform
end_time = time.time()
elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)
print(f"\n Code Completely Executed in {int(minutes)} min {seconds:.2f} sec \n")