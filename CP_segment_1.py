import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io
from cellpose.io import imread
from cellpose import plot
import datetime
import glob
import os
import time
import Misc_functions_1 as misc_f

def CP_segment():

    start_time, current_date = misc_f.start_inform(script_location=__file__)


    ### Settings

    main_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Cellpose\Cellpose1"
    #input_files_dir = main_dir + r"\BW 134 ball flame - Crop small"
    #input_files_dir = main_dir + r"\BW 134 ball flame - Crop"
    input_files_dir = main_dir + r"\BW 134 ball flame - Crop Small First few"






    misc_f.end_inform(script_location=__file__, start_time = start_time)







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
    flowi = flows[idx]
    input_filename = os.path.basename(files[idx])

    fig = plt.figure(figsize=(12,5))
    plot.show_segmentation(fig, imgs[idx], maski, flowi[0], channels=channels)
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
    io.masks_flows_to_seg(imgs, maski, flowi, output_seg_path, channels=channels, diams=diams)




### end inform
end_time = time.time()
elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)
print(f"\n Code Completely Executed in {int(minutes)} min {seconds:.2f} sec \n")