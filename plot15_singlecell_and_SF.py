import os
import pickle
from pathlib import Path
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from matplotlib.patches import ConnectionPatch, Rectangle
from skimage import measure
import Format_1 as F_1  # Direct import of Format_1 module

# --- Matplotlib Configuration ---
# Set up LaTeX font for all plot text
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)

def plot_singlecell_and_SF(
    input_dir,
    output_dir_manual="",
    output_dir_comment="single_cell_and_SF",
    image_numbers=[],
    cell_id=95,
    # --- Left Plot (Segmented Full View) Parameters ---
    left_plot_params={
        "ScaleFactor": 1.5,
        "show_masks": True,
        "show_outlines": True,
        "alpha": 0.5,
        "contour_color": 'w',
        "contour_linewidth": 0.8,
        "cells_to_color": [], # Empty list colors all cells
    },
    # --- Right Plot (Zoomed Single Cell) Parameters ---
    right_plot_params={
        "zoom_factor": 3.0,
        "show_mask": True,
        "show_outline": True,
        "show_diameter_circle": True,
        "show_centroid_cross": True,
        "mask_alpha": 0.5,
        "outline_color": 'w',
        "outline_linewidth": 1.5,
        "diameter_circle_color": 'r',
        "diameter_circle_linestyle": '-',
        "diameter_circle_linewidth": 1.5,
        "cross_color": 'black',
        "cross_size": 10,
        "cross_linewidth": 1.5,
    },
    # --- Frame and Connection Line Parameters ---
    frame_params={
        "color": 'yellow',
        "linewidth": 2,
        "linestyle": '--',
    },
    connection_line_params={
        "color": 'yellow',
        "linewidth": 2,
        "linestyle": '--',
    },
    # --- Label and Font Parameters ---
    labels={
        "left_x": {"text": "", "fontsize": 12, "position": "top"},
        "left_y": {"text": "", "fontsize": 12},
        "right_x": {"text": "x [px]", "fontsize": 20, "position": "top"},
        "right_y": {"text": "y [px]", "fontsize": 20},
    },
    fontsizes={
        "right_tick_labels": 16,
        "left_tick_labels": 16,
    },
    # Add new parameter for label padding
    label_padding={
        "left_x": 10,
        "left_y": 10,
        "right_x": 10,
        "right_y": 10,
    },
    # --- General Figure Parameters ---
    fig_size=(16, 8),
    dpi=300,
    show_plot=0
):
    """
    Generates a composite plot with a full segmented view on the left and a zoomed-in
    single-cell view on the right. A frame on the left plot indicates the zoomed area,
    with lines connecting it to the right plot.

    Parameters are organized into dictionaries for clarity.
    """
    start_time = time.time()
    print(f"Starting plot generation at {time.strftime('%H:%M:%S')}")
    
    # --- Setup Directories and Load Data ---
    output_dir = F_1.F_out_dir(
        input_dir=input_dir, 
        script_path=__file__, 
        output_dir_comment=output_dir_comment,
        output_dir_manual=output_dir_manual
    )
    print(f"Output directory: {output_dir}")

    input_dir = Path(input_dir)
    df_path = input_dir / "Analysis_A11_final_df.pkl"
    if not df_path.exists():
        print(f"Error: Data file not found at {df_path}")
        return

    print(f"Loading dataframe from {df_path}...")
    t0 = time.time()
    with open(df_path, 'rb') as f:
        df = pickle.load(f)
    print(f"Dataframe loaded in {time.time() - t0:.2f} seconds")

    # --- Image Processing Loop ---
    all_image_numbers = df['image_number'].unique()
    target_image_numbers = image_numbers if image_numbers else all_image_numbers

    print(f"Processing {len(target_image_numbers)} images for cell ID {cell_id}...")

    for image_num in target_image_numbers:
        img_start_time = time.time()
        print(f"\nProcessing image number {image_num} at {time.strftime('%H:%M:%S')}...")
        
        # Find the row for the current image number
        row_data = df[df['image_number'] == image_num]
        if row_data.empty:
            print(f"Warning: Image number {image_num} not found in DataFrame. Skipping.")
            continue
        row = row_data.iloc[0]

        try:
            # --- Load Image and Mask Data ---
            print(f"  Loading image and mask data...")
            t0 = time.time()
            image_file_path = row['image_file_path']
            original_img = plt.imread(image_file_path)
            full_mask = row['masks']
            print(f"  Data loaded in {time.time() - t0:.2f} seconds")

            if original_img is None:
                print(f"Error reading image file: {image_file_path}")
                continue
            if full_mask is None or cell_id not in np.unique(full_mask):
                print(f"Cell ID {cell_id} not found in mask for image {image_num}. Skipping.")
                continue

            # --- Create Figure and Axes ---
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)

            # ==========================================================
            # Plot 1: Full Segmented Image (Left)
            # ==========================================================
            print(f"  Processing left plot (full segmented image)...")
            t0 = time.time()
            
            img_height, img_width = original_img.shape[:2]
            center_x, center_y = img_width // 2, img_height // 2
            
            # Define the zoom for the left plot
            sf_scale = left_plot_params.get("ScaleFactor", 1.5)
            zoom_half_size = int(row['D_SF_px'] * sf_scale / 2)
            left_x_min = max(0, center_x - zoom_half_size)
            left_x_max = min(img_width, center_x + zoom_half_size)
            left_y_min = max(0, center_y - zoom_half_size)
            left_y_max = min(img_height, center_y + zoom_half_size)

            # Crop the image and mask for the left plot
            left_img_cropped = original_img[left_y_min:left_y_max, left_x_min:left_x_max]
            left_mask_cropped = full_mask[left_y_min:left_y_max, left_x_min:left_x_max]
            
            ax1.imshow(left_img_cropped, extent=[left_x_min, left_x_max, left_y_max, left_y_min])
            ax1.set_xlabel(labels["left_x"]["text"], fontsize=labels["left_x"]["fontsize"])
            ax1.set_ylabel(labels["left_y"]["text"], fontsize=labels["left_y"]["fontsize"])
            ax1.set_axis_off() # Typically we don't want axes on the image

            # Color mapping for masks
            distinct_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 165, 0), (128, 0, 128), (165, 42, 42), (255, 192, 203)]

            # Overlay masks and outlines on the left plot
            unique_cells = np.unique(left_mask_cropped)
            unique_cells = unique_cells[unique_cells > 0]
            print(f"  Found {len(unique_cells)} cells in the left plot")

            # Configure left plot axes - replace set_axis_off() with proper tick config
            ax1.tick_params(axis='both', direction='in', labelsize=fontsizes.get("left_tick_labels", 16))
            
            # Set x-axis position for left plot
            x_axis_pos_left = labels["left_x"].get("position", "top")
            if x_axis_pos_left == "top":
                ax1.xaxis.set_ticks_position('top')
                ax1.xaxis.set_label_position('top')
                ax1.spines['bottom'].set_visible(False)
                ax1.spines['top'].set_visible(True)
            
            # Set axis labels with padding
            ax1.set_xlabel(labels["left_x"]["text"], 
                          fontsize=labels["left_x"]["fontsize"],
                          labelpad=label_padding.get("left_x", 10))
            ax1.set_ylabel(labels["left_y"]["text"], 
                          fontsize=labels["left_y"]["fontsize"],
                          labelpad=label_padding.get("left_y", 10))

            if left_plot_params["show_masks"]:
                print(f"  Applying masks to {len(unique_cells)} cells...")
                mask_start = time.time()
                
            for c_id in unique_cells:
                if left_plot_params["cells_to_color"] and c_id not in left_plot_params["cells_to_color"]:
                    continue

                cell_mask_layer = (left_mask_cropped == c_id)
                if left_plot_params["show_masks"]:
                    color_rgb = distinct_colors[int(c_id) % len(distinct_colors)]
                    normalized_color = (color_rgb[0]/255, color_rgb[1]/255, color_rgb[2]/255, left_plot_params["alpha"])
                    mask_overlay = np.zeros((*cell_mask_layer.shape, 4))
                    mask_overlay[cell_mask_layer] = normalized_color
                    ax1.imshow(mask_overlay, extent=[left_x_min, left_x_max, left_y_max, left_y_min])
                
                if left_plot_params["show_outlines"]:
                    contours = measure.find_contours(cell_mask_layer, 0.5)
                    for contour in contours:
                        ax1.plot(contour[:, 1] + left_x_min, contour[:, 0] + left_y_min,
                                 color=left_plot_params["contour_color"],
                                 linewidth=left_plot_params["contour_linewidth"])
            
            if left_plot_params["show_masks"]:
                print(f"  Masks applied in {time.time() - mask_start:.2f} seconds")
            print(f"  Left plot completed in {time.time() - t0:.2f} seconds")

            # ==========================================================
            # Plot 2: Single Cell Zoom (Right)
            # ==========================================================
            print(f"  Processing right plot (single cell zoom)...")
            t0 = time.time()
            
            props = measure.regionprops((full_mask == cell_id).astype(int))
            centroid_y, centroid_x = props[0].centroid
            cell_area = props[0].area
            approx_diameter = 2 * np.sqrt(cell_area / np.pi)

            # Define zoom window for the right plot
            zoom_size = int(approx_diameter * right_plot_params["zoom_factor"])
            right_x_min = max(0, int(centroid_x - zoom_size / 2))
            right_x_max = min(img_width, int(centroid_x + zoom_size / 2))
            right_y_min = max(0, int(centroid_y - zoom_size / 2))
            right_y_max = min(img_height, int(centroid_y + zoom_size / 2))

            # ==========================================================
            # Frame and Connection Lines - MOVED HERE to go behind the right plot
            # ==========================================================
            print(f"  Drawing frame and connection lines...")
            
            # 1. Draw the zoom rectangle on the left plot
            rect_width = right_x_max - right_x_min
            rect_height = right_y_max - right_y_min
            rect = Rectangle((right_x_min, right_y_min), rect_width, rect_height,
                             linewidth=frame_params["linewidth"],
                             edgecolor=frame_params["color"],
                             linestyle=frame_params["linestyle"],
                             facecolor='none')
            ax1.add_patch(rect)
            
            # 2. Draw connecting lines with low zorder to make them appear behind the right plot
            connection_params = connection_line_params.copy()
            connection_params['zorder'] = 0  # Low zorder to go behind right plot
            
            # Top-left corner
            con_tl = ConnectionPatch(xyA=(right_x_min, right_y_min), xyB=(0, 1),
                                     coordsA='data', coordsB='axes fraction',
                                     axesA=ax1, axesB=ax2, **connection_params)
            # Top-right corner
            con_tr = ConnectionPatch(xyA=(right_x_max, right_y_min), xyB=(1, 1),
                                     coordsA='data', coordsB='axes fraction',
                                     axesA=ax1, axesB=ax2, **connection_params)
            # Bottom-left corner
            con_bl = ConnectionPatch(xyA=(right_x_min, right_y_max), xyB=(0, 0),
                                     coordsA='data', coordsB='axes fraction',
                                     axesA=ax1, axesB=ax2, **connection_params)
            # Bottom-right corner
            con_br = ConnectionPatch(xyA=(right_x_max, right_y_max), xyB=(1, 0),
                                     coordsA='data', coordsB='axes fraction',
                                     axesA=ax1, axesB=ax2, **connection_params)
            
            fig.add_artist(con_tl)
            fig.add_artist(con_tr)
            fig.add_artist(con_bl)
            fig.add_artist(con_br)
            
            # Add a solid background to the right plot to hide the lines
            ax2.patch.set_facecolor('white')
            ax2.patch.set_zorder(1)  # Higher zorder than connection lines
            
            print(f"  Frame and connection lines drawn in {time.time() - t0:.2f} seconds")
            
            # Now continue with right plot elements (with higher zorder)
            print(f"  Rendering right plot elements...")
            t0 = time.time()
            
            # Crop for the right plot
            right_img_cropped = original_img[right_y_min:right_y_max, right_x_min:right_x_max]
            right_mask_cropped = full_mask[right_y_min:right_y_max, right_x_min:right_x_max]

            # Display image with higher zorder than connections
            ax2.imshow(right_img_cropped, extent=[right_x_min, right_x_max, right_y_max, right_y_min], 
                      zorder=2)
            
            # Overlay mask, outline, and features for the right plot (all with higher zorder)
            cell_mask_layer_right = (right_mask_cropped == cell_id)
            if right_plot_params["show_mask"]:
                color_rgb = distinct_colors[int(cell_id) % len(distinct_colors)]
                normalized_color = (color_rgb[0]/255, color_rgb[1]/255, color_rgb[2]/255, right_plot_params["mask_alpha"])
                mask_overlay_right = np.zeros((*cell_mask_layer_right.shape, 4))
                mask_overlay_right[cell_mask_layer_right] = normalized_color
                ax2.imshow(mask_overlay_right, extent=[right_x_min, right_x_max, right_y_max, right_y_min], 
                          zorder=3)
            
            if right_plot_params["show_outline"]:
                contours = measure.find_contours(cell_mask_layer_right, 0.5)
                for contour in contours:
                    ax2.plot(contour[:, 1] + right_x_min, contour[:, 0] + right_y_min,
                             color=right_plot_params["outline_color"],
                             linewidth=right_plot_params["outline_linewidth"], zorder=4)
            
            if right_plot_params["show_centroid_cross"]:
                ax2.plot([centroid_x - right_plot_params["cross_size"]/2, centroid_x + right_plot_params["cross_size"]/2],
                         [centroid_y, centroid_y],
                         color=right_plot_params["cross_color"], linewidth=right_plot_params["cross_linewidth"], zorder=5)
                ax2.plot([centroid_x, centroid_x],
                         [centroid_y - right_plot_params["cross_size"]/2, centroid_y + right_plot_params["cross_size"]/2],
                         color=right_plot_params["cross_color"], linewidth=right_plot_params["cross_linewidth"], zorder=5)
            
            if right_plot_params["show_diameter_circle"]:
                circle = plt.Circle((centroid_x, centroid_y), approx_diameter / 2,
                                    fill=False, edgecolor=right_plot_params["diameter_circle_color"],
                                    linestyle=right_plot_params["diameter_circle_linestyle"],
                                    linewidth=right_plot_params["diameter_circle_linewidth"], zorder=6)
                ax2.add_patch(circle)

            # --- Configure Right Plot Axes ---
            ax2.set_xlim(right_x_min, right_x_max)
            ax2.set_ylim(right_y_max, right_y_min) # Inverted y-axis for images
            ax2.tick_params(axis='both', direction='in', labelsize=fontsizes["right_tick_labels"])

            x_axis_pos = labels["right_x"].get("position", "top")
            if x_axis_pos == "top":
                ax2.xaxis.set_ticks_position('top')
                ax2.xaxis.set_label_position('top')
                ax2.spines['bottom'].set_visible(False)
                ax2.spines['top'].set_visible(True)
            
            # Add label padding to right plot as well
            ax2.set_xlabel(labels["right_x"]["text"], 
                          fontsize=labels["right_x"]["fontsize"],
                          labelpad=label_padding.get("right_x", 10))
            ax2.set_ylabel(labels["right_y"]["text"], 
                          fontsize=labels["right_y"]["fontsize"],
                          labelpad=label_padding.get("right_y", 10))


            # --- Finalize and Save Plot ---
            print(f"  Finalizing and saving plot...")
            t0 = time.time()
            plt.tight_layout()
            
            output_filename = f"plot15_SF_and_singlecell_{int(image_num):04d}_cell_{cell_id}"
            
            # Fix: Use os.path.join instead of / operator for path construction
            png_path = os.path.join(output_dir, f"{output_filename}.png")
            svg_path = os.path.join(output_dir, f"{output_filename}.svg")

            print(f"  Saving PNG...")
            png_save_time = time.time()
            fig.savefig(png_path, format="png", dpi=dpi, bbox_inches='tight')
            print(f"  PNG saved in {time.time() - png_save_time:.2f} seconds")
            
            print(f"  Saving SVG...")
            svg_save_time = time.time()
            fig.savefig(svg_path, format="svg", dpi=dpi, bbox_inches='tight')
            print(f"  SVG saved in {time.time() - svg_save_time:.2f} seconds")
            
            print(f"  > Saved plot for image {image_num} to {output_dir}")
            print(f"  > Plot finalized and saved in {time.time() - t0:.2f} seconds")
            print(f"  > Total processing time for image {image_num}: {time.time() - img_start_time:.2f} seconds")

            if show_plot:
                plt.show()
            
            plt.close(fig)

        except Exception as e:
            print(f"--- An error occurred while processing image {image_num}: {e} ---")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Finished at {time.strftime('%H:%M:%S')}")

if __name__ == "__main__":
    # This block controls the execution of the script.
    # All parameters can be modified here directly.
    
    # --- REQUIRED: Set the input directory containing the .pkl file ---
    # Please update this path to your actual data directory
    INPUT_DIRECTORY = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187"
    
    plot_singlecell_and_SF(
        input_dir=INPUT_DIRECTORY,
        
        # --- Image Selection ---
        # Provide a list of image numbers, e.g., [79, 80].
        # An empty list [] will process all images in the dataframe.
        image_numbers=[79], 
        cell_id=95,  # The specific cell ID to zoom in on
        
        # --- Left Plot Parameters (Full View) ---
        left_plot_params={
            "ScaleFactor": 1.5,          # Zoom level for the overall view, based on flame diameter
            "show_masks": True,          # Show colored masks for all cells
            "show_outlines": False,       # Show outlines for all cells
            "alpha": 0.4,                # Transparency of the masks
            "contour_color": 'white',    # Color of cell outlines
            "contour_linewidth": 0.5,
            "cells_to_color": [],        # List of specific cells to color, or [] for all
        },
        
        # --- Right Plot Parameters (Single Cell Zoom) ---
        right_plot_params={
            "zoom_factor": 3.0,                  # Zoom level relative to the target cell's diameter
            "show_mask": True,                   # Show the colored mask for the target cell
            "show_outline": True,                # Show the outline for the target cell
            "show_diameter_circle": True,        # Show a circle representing the cell's approximate diameter
            "show_centroid_cross": True,         # Show a cross at the cell's centroid
            "mask_alpha": 0.5,                   # Transparency of the single cell mask
            "outline_color": 'white',
            "outline_linewidth": 3.0,
            "diameter_circle_color": 'red',
            "diameter_circle_linestyle": '-',
            "diameter_circle_linewidth": 4.0,
            "cross_color": 'black',
            "cross_size": 20,
            "cross_linewidth": 3.0,
        },

        # --- Frame and Line Styling ---
        frame_params={
            "color": 'black',
            "linewidth": 2,
            "linestyle": '-',
        },
        connection_line_params={
            "color": 'black',
            "linewidth": 2,
            "linestyle": '',
        },
        
        # --- Labels and Fonts ---
        labels={
            "left_x": {"text": r"x-position [px]", "fontsize": 20, "position": "top"},
            "left_y": {"text": r"y-position [px]", "fontsize": 20},
            "right_x": {"text": r"x-position [px]", "fontsize": 20, "position": "top"},
            "right_y": {"text": r"y-position [px]", "fontsize": 20},
        },
        fontsizes={
            "left_tick_labels": 16,
            "right_tick_labels": 16,
        },
        # New parameter for controlling label padding
        label_padding={
            "left_x": 15,
            "left_y": 15,
            "right_x": 15,
            "right_y": 15,
        },
        
        # --- General Figure Settings ---
        fig_size=(18, 9),  # Width, Height in inches
        show_plot=0        # Set to 1 to display plots interactively, 0 to just save them
    )