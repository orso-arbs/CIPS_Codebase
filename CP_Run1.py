import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io
from cellpose.io import imread
from cellpose import plot
import datetime
import glob
import os
import time

### start inform
start_time = time.time()
current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
print(f"\n {os.path.basename(__file__)}: ", datetime.datetime.now(), "\n")
print("Current time:", datetime.datetime.now())  # Add this line to print current time


### Settings

main_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Cellpose\Cellpose1"
input_files_dir = main_dir + r"\BW 134 ball flame - Crop small"

output_dir = main_dir + r"\CP_Results_" + current_date
os.makedirs(output_dir, exist_ok=True)



### I/O 

# list of files
# PUT PATH TO YOUR FILES HERE!
files = glob.glob(input_files_dir + r"\*.png")
#files = glob.glob(input_files_dir + r"\bw_0134.png")
imgs = [imread(f) for f in files]
nimg = len(imgs)
print("\n loaded #images: ", nimg, "\n")



### CellPose

# log 
io.logger_setup()

# model_type='cyto' or 'nuclei' or 'cyto2' or 'cyto3'
model = models.Cellpose(model_type='cyto3', gpu=True)


# define CHANNELS to run segementation on
# grayscale=0, R=1, G=2, B=3
# channels = [cytoplasm, nucleus]
# if NUCLEUS channel does not exist, set the second channel to 0
channels = [[0,0]]
# IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
# channels = [0,0] # IF YOU HAVE GRAYSCALE
# channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
# channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

# if diameter is set to None, the size of the cells is estimated on a per image basis
# you can set the average cell `diameter` in pixels yourself (recommended)
# diameter can be a list or a single number for all images

print("\n CellPose Segmenting \n")

masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=channels)


### or to run one of the other models, or a custom model, specify a CellposeModel
#model = models.CellposeModel(model_type='livecell_cp3')

#masks, flows, styles = model.eval(imgs, diameter=30, channels=[0,0])




### Save/Print Results

print("\n Save/Plot Results \n")

nimg = len(imgs)
for idx in range(nimg):
    maski = masks[idx]
    flowi = flows[idx][0]
    input_filename = os.path.basename(files[idx])

    fig = plt.figure(figsize=(12,5))
    plot.show_segmentation(fig, imgs[idx], maski, flowi, channels=channels)
    plt.tight_layout()
        

    # Save the plot
    output_filename = os.path.splitext(input_filename)[0] + "_segmented_plot.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    plt.close(fig)  # Close the figure to free up memory

    # Save the image
    output_image_filename = os.path.splitext(input_filename)[0] + "_segmented_image"
    output_image_path = os.path.join(output_dir, output_image_filename)
    io.save_masks(imgs, maski, flowi, output_image_path, png=True)

    # Save the seg file
    output_seg_filename = os.path.splitext(input_filename)[0] + "_segmented"
    output_seg_path = os.path.join(output_dir, output_seg_filename)
    io.masks_flows_to_seg(imgs, maski, flows[idx], output_seg_path, channels=channels, diams=diams)




### end inform
end_time = time.time()
elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)
print(f"\n Code Completely Executed in {int(minutes)} min {seconds:.2f} sec \n")