import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from skimage import measure
import pickle
from pathlib import Path
import Format_1 as F_1
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Set up LaTeX font
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)

def plot_singlecell_and_sf(
    input_dir,
    output_dir_manual="",
    output_dir_comment="single_cell_and_SF_plots",
    image_numbers=[],
    cell_id=95,
    # Left plot parameters (from plot13)
    left_zoom_factor=1.0,
    left_ScaleFactor=None,
    left_show_masks=True,
    left_show_outlines=True,
    left_cells_to_color=[],
    left_alpha=0.5,
    left_contour_color='w',
    left_contour_linestyle='-',
    left_contour_linewidth=0.8,
    # Right plot parameters (from plot15)
    right_zoom_factor=5.0,
    right_show_mask=True,
    right_show_outline=True,
    right_show_diameter_circle=True,
    right_show_centroid_cross=True,
    right_mask_alpha=0.5,
    right_outline_color='w',
    right_outline_linestyle='-',
    right_outline_linewidth=0.8,
    right_diameter_circle_color='r',
    right_diameter_circle_linestyle='-',
    right_diameter_circle_linewidth=1.5,
    right_cross_color='black',
    right_cross_size=10,
    right_cross_linewidth=2,
    # Frame and connecting lines parameters
    frame_color='black',
    frame_linewidth=1,
    frame_linestyle='-',
    connection_color='black',
    connection_linewidth=1,
    connection_linestyle='--',
    connection_zorder=5,
    # Label and Font arguments
    left_x_label_content="x [px]",
    left_y_label_content="y [px]",
    right_x_label_content="x [px]",
    right_y_label_content="y [px]",
    left_x_label_size=12,
    left_y_label_size=12,
    right_x_label_size=12,
    right_y_label_size=12,
    left_x_label_pos=(0.5, 1.05),
    left_y_label_pos=(-0.05, 0.5),
    right_x_label_pos=(0.5, 1.05),
    right_y_label_pos=(-0.05, 0.5),
    left_tick_fontsize=10,
    right_tick_fontsize=10,
    left_tick_padding=5,  # Parameter for left tick padding
    right_tick_padding=5, # Parameter for right tick padding
    # Subplot labels and titles
    show_subplot_labels=True,
    left_subplot_label="(a)",
    right_subplot_label="(b)",
    subplot_label_fontsize=20,
    subplot_label_weight='bold',
    subplot_label_pos=(0.05, 0.95),  # Position relative to axes (x, y)
    subplot_label_va='top',          # Vertical alignment: 'top', 'center', 'bottom'
    subplot_label_ha='left',         # Horizontal alignment: 'left', 'center', 'right'
    left_subplot_title="",           # Optional title for left subplot
    right_subplot_title="",          # Optional title for right subplot
    subplot_title_fontsize=16,       # Font size for subplot titles
    subplot_title_weight='normal',   # Weight for subplot titles: 'normal', 'bold'
    subplot_title_pos=(0.5, 0.02),   # Position relative to axes (x, y)
    subplot_title_va='bottom',       # Vertical alignment: 'top', 'center', 'bottom'
    subplot_title_ha='center',       # Horizontal alignment: 'left', 'center', 'right'
    # General plot arguments
    show_plot=0,
    dpi=300
):
    """
    Creates a two-panel plot with a segmented image on the left and a zoomed-in single cell on the right.

    Args:
        input_dir (str): Path to the directory containing Analysis_A11_final_df.pkl.
        output_dir_manual (str, optional): Manual output directory. Defaults to "".
        output_dir_comment (str, optional): Comment for the output directory. Defaults to "single_cell_and_SF_plots".
        image_numbers (list, optional): List of image numbers to process. If empty, all images are processed. Defaults to [].
        cell_id (int, optional): The ID of the cell to be featured in the right plot. Defaults to 95.
        left_zoom_factor (float, optional): Zoom factor for the left plot. Defaults to 1.5.
        left_ScaleFactor (float, optional): Scale factor for zooming based on flame radius. Defaults to None.
        left_show_masks (bool, optional): Whether to show masks on the left plot. Defaults to True.
        left_show_outlines (bool, optional): Whether to show outlines on the left plot. Defaults to True.
        left_cells_to_color (list, optional): List of cell IDs to color on the left plot. Defaults to [].
        left_alpha (float, optional): Transparency of the masks on the left plot. Defaults to 0.5.
        left_contour_color (str, optional): Color of cell outlines on the left plot. Defaults to 'w'.
        left_contour_linestyle (str, optional): Linestyle of cell outlines on the left plot. Defaults to '-'.
        left_contour_linewidth (float, optional): Linewidth of cell outlines on the left plot. Defaults to 0.8.
        right_zoom_factor (float, optional): Zoom factor for the right plot. Defaults to 5.0.
        right_show_mask (bool, optional): Whether to show the mask on the right plot. Defaults to True.
        right_show_outline (bool, optional): Whether to show the outline on the right plot. Defaults to True.
        right_show_diameter_circle (bool, optional): Whether to show the diameter circle on the right plot. Defaults to True.
        right_show_centroid_cross (bool, optional): Whether to show the centroid cross on the right plot. Defaults to True.
        right_mask_alpha (float, optional): Transparency of the mask on the right plot. Defaults to 0.5.
        right_outline_color (str, optional): Color of the cell outline on the right plot. Defaults to 'w'.
        right_outline_linestyle (str, optional): Linestyle of the cell outline on the right plot. Defaults to '-'.
        right_outline_linewidth (float, optional): Linewidth of the cell outline on the right plot. Defaults to 0.8.
        right_diameter_circle_color (str, optional): Color of the diameter circle on the right plot. Defaults to 'r'.
        right_diameter_circle_linestyle (str, optional): Linestyle of the diameter circle on the right plot. Defaults to '-'.
        right_diameter_circle_linewidth (float, optional): Linewidth of the diameter circle on the right plot. Defaults to 1.5.
        right_cross_color (str, optional): Color of the centroid cross on the right plot. Defaults to 'black'.
        right_cross_size (int, optional): Size of the centroid cross on the right plot. Defaults to 10.
        right_cross_linewidth (int, optional): Linewidth of the centroid cross on the right plot. Defaults to 2.
        frame_color (str, optional): Color of the frame highlighting the zoomed region. Defaults to 'r'.
        frame_linewidth (float, optional): Linewidth of the frame. Defaults to 1.
        frame_linestyle (str, optional): Line style of the frame. Defaults to '-'.

        connection_color (str, optional): Color of the connecting lines between plots. Defaults to 'r'.
        connection_linewidth (float, optional): Linewidth of the connecting lines. Defaults to 1.
        connection_linestyle (str, optional): Line style of the connecting lines. Defaults to '-'.

        connection_zorder (int, optional): Z-order of the connecting lines. Defaults to 5.
        left_x_label_content (str, optional): X-axis label for the left plot. Defaults to "x [px]".
        left_y_label_content (str, optional): Y-axis label for the left plot. Defaults to "y [px]".
        right_x_label_content (str, optional): X-axis label for the right plot. Defaults to "x [px]".
        right_y_label_content (str, optional): Y-axis label for the right plot. Defaults to "y [px]".
        left_x_label_size (int, optional): Font size for the x-axis label of the left plot. Defaults to 12.
        left_y_label_size (int, optional): Font size for the y-axis label of the left plot. Defaults to 12.
        right_x_label_size (int, optional): Font size for the x-axis label of the right plot. Defaults to 12.
        right_y_label_size (int, optional): Font size for the y-axis label of the right plot. Defaults to 12.
        left_x_label_pos (tuple, optional): Position of the x-axis label of the left plot. Defaults to (0.5, 1.05).
        left_y_label_pos (tuple, optional): Position of the y-axis label of the left plot. Defaults to (-0.05, 0.5).
        right_x_label_pos (tuple, optional): Position of the x-axis label of the right plot. Defaults to (0.5, 1.05).
        right_y_label_pos (tuple, optional): Position of the y-axis label of the right plot. Defaults to (-0.05, 0.5).
        left_tick_fontsize (int, optional): Font size for the tick labels of the left plot. Defaults to 10.
        right_tick_fontsize (int, optional): Font size for the tick labels of the right plot. Defaults to 10.
        left_tick_padding (int, optional): Padding between tick labels and axes for the left plot. Defaults to 5.
        right_tick_padding (int, optional): Padding between tick labels and axes for the right plot. Defaults to 5.
        show_subplot_labels (bool, optional): Whether to show subplot labels. Defaults to True.
        left_subplot_label (str, optional): Label for the left subplot. Defaults to "(a)".
        right_subplot_label (str, optional): Label for the right subplot. Defaults to "(b)".
        subplot_label_fontsize (int, optional): Font size for subplot labels. Defaults to 20.
        subplot_label_weight (str, optional): Font weight for subplot labels. Defaults to 'bold'.
        subplot_label_pos (tuple, optional): Position for subplot labels relative to axes. Defaults to (0.05, 0.95).
        subplot_label_va (str, optional): Vertical alignment for subplot labels. Defaults to 'top'.
        subplot_label_ha (str, optional): Horizontal alignment for subplot labels. Defaults to 'left'.
        left_subplot_title (str, optional): Optional title for the left subplot. Defaults to "".
        right_subplot_title (str, optional): Optional title for the right subplot. Defaults to "".
        subplot_title_fontsize (int, optional): Font size for subplot titles. Defaults to 16.
        subplot_title_weight (str, optional): Font weight for subplot titles. Defaults to 'normal'.
        subplot_title_pos (tuple, optional): Position for subplot titles relative to axes. Defaults to (0.5, 0.02).
        subplot_title_va (str, optional): Vertical alignment for subplot titles. Defaults to 'bottom'.
        subplot_title_ha (str, optional): Horizontal alignment for subplot titles. Defaults to 'center'.
        show_plot (int, optional): Whether to display the plot. Defaults to 0.
        dpi (int, optional): DPI for the saved plot. Defaults to 300.
    """
    output_dir = F_1.F_out_dir(input_dir=input_dir, script_path=__file__, output_dir_comment=output_dir_comment, output_dir_manual=output_dir_manual)
    
    # Convert output_dir to Path object if it's not already
    output_dir = Path(output_dir)

    df_path = Path(input_dir) / "Analysis_A11_final_df.pkl"
    with open(df_path, 'rb') as f:
        df = pickle.load(f)

    if not image_numbers:
        image_numbers = df['image_number'].unique()

    for image_num in image_numbers:
        image_data = df[df['image_number'] == image_num]
        if image_data.empty:
            print(f"Image number {image_num} not found in DataFrame.")
            continue

        row = image_data.iloc[0]
        image_file_path = row['image_file_path']
        masks = row['masks']

        if cell_id not in np.unique(masks):
            print(f"Cell ID {cell_id} not in image {image_num}. Skipping.")
            continue

        original_img = plt.imread(image_file_path)
        height, width = original_img.shape[:2]

        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 8))

        # Left plot (from plot13_segmented_image.py)
        center_x, center_y = width // 2, height // 2
        if left_ScaleFactor:
            D_SF_px = row['D_SF_px']
            zoom_half_size = int(D_SF_px * left_ScaleFactor / 2)
            x_min_left = max(0, center_x - zoom_half_size)
            x_max_left = min(width, center_x + zoom_half_size)
            y_min_left = max(0, center_y - zoom_half_size)
            y_max_left = min(height, center_y + zoom_half_size)
        else:
            new_width = int(width / left_zoom_factor)
            new_height = int(height / left_zoom_factor)
            x_min_left = max(0, center_x - new_width // 2)
            x_max_left = min(width, center_x + new_width // 2)
            y_min_left = max(0, center_y - new_height // 2)
            y_max_left = min(height, center_y + new_height // 2)
        
        zoomed_image_left = original_img[y_min_left:y_max_left, x_min_left:x_max_left]
        zoomed_mask_left = masks[y_min_left:y_max_left, x_min_left:x_max_left]
        
        ax_left.imshow(zoomed_image_left, extent=[x_min_left, x_max_left, y_max_left, y_min_left])

        if left_show_masks:
            distinct_colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (0, 255, 255), (255, 0, 255), (255, 165, 0), (128, 0, 128),
                (165, 42, 42), (255, 192, 203)
            ]
            mask_overlay = np.zeros((*zoomed_mask_left.shape, 4))
            unique_cells = np.unique(zoomed_mask_left)
            unique_cells = unique_cells[unique_cells > 0]
            
            for cid in unique_cells:
                if not left_cells_to_color or cid in left_cells_to_color:
                    cell_mask = zoomed_mask_left == cid
                    color_index = int(cid) % 10
                    color_rgb = distinct_colors[color_index]
                    normalized_color = (color_rgb[0]/255, color_rgb[1]/255, color_rgb[2]/255, left_alpha)
                    mask_overlay[cell_mask] = normalized_color
            ax_left.imshow(mask_overlay, extent=[x_min_left, x_max_left, y_max_left, y_min_left])

        if left_show_outlines:
            unique_cells = np.unique(zoomed_mask_left)[1:]
            for cid in unique_cells:
                if not left_cells_to_color or cid in left_cells_to_color:
                    cell_mask = zoomed_mask_left == cid
                    contours = measure.find_contours(cell_mask, 0.5)
                    for contour in contours:
                        ax_left.plot(contour[:, 1] + x_min_left, contour[:, 0] + y_min_left,
                                     color=left_contour_color,
                                     linestyle=left_contour_linestyle,
                                     linewidth=left_contour_linewidth)

        ax_left.tick_params(axis='both', direction='in', labelsize=left_tick_fontsize, pad=left_tick_padding)
        ax_left.xaxis.set_ticks_position('top')
        ax_left.xaxis.set_label_position('top')
        ax_left.set_xlabel(left_x_label_content, fontsize=left_x_label_size)
        ax_left.set_ylabel(left_y_label_content, fontsize=left_y_label_size)
        ax_left.xaxis.set_label_coords(left_x_label_pos[0], left_x_label_pos[1])
        ax_left.yaxis.set_label_coords(left_y_label_pos[0], left_y_label_pos[1])
        
        # Right plot (from plot15_singlecell.py)
        props = measure.regionprops((masks == cell_id).astype(int))
        centroid_y, centroid_x = props[0].centroid
        cell_area = props[0].area
        cell_diameter = 2 * np.sqrt(cell_area / np.pi)
        
        zoom_window_size = int(cell_diameter * right_zoom_factor)
        x_min_right = max(0, int(centroid_x - zoom_window_size // 2))
        x_max_right = min(width, int(centroid_x + zoom_window_size // 2))
        y_min_right = max(0, int(centroid_y - zoom_window_size // 2))
        y_max_right = min(height, int(centroid_y + zoom_window_size // 2))
        
        zoomed_image_right = original_img[y_min_right:y_max_right, x_min_right:x_max_right]
        zoomed_mask_right = masks[y_min_right:y_max_right, x_min_right:x_max_right]
        
        ax_right.imshow(zoomed_image_right, extent=[x_min_right, x_max_right, y_max_right, y_min_right])
        
        if right_show_mask:
            mask_overlay_right = np.zeros((*zoomed_mask_right.shape, 4))
            cell_mask_zoomed = zoomed_mask_right == cell_id
            if np.any(cell_mask_zoomed):
                color_index = cell_id % 10
                color_rgb = distinct_colors[color_index]
                normalized_color = (color_rgb[0]/255, color_rgb[1]/255, color_rgb[2]/255, right_mask_alpha)
                mask_overlay_right[cell_mask_zoomed] = normalized_color
                ax_right.imshow(mask_overlay_right, extent=[x_min_right, x_max_right, y_max_right, y_min_right])

        if right_show_outline:
            cell_mask_zoomed = zoomed_mask_right == cell_id
            contours = measure.find_contours(cell_mask_zoomed, 0.5)
            for contour in contours:
                ax_right.plot(contour[:, 1] + x_min_right, contour[:, 0] + y_min_right,
                              color=right_outline_color,
                              linestyle=right_outline_linestyle,
                              linewidth=right_outline_linewidth)

        if right_show_centroid_cross:
            ax_right.plot([centroid_x - right_cross_size/2, centroid_x + right_cross_size/2],
                          [centroid_y, centroid_y],
                          color=right_cross_color, linewidth=right_cross_linewidth)
            ax_right.plot([centroid_x, centroid_x],
                          [centroid_y - right_cross_size/2, centroid_y + right_cross_size/2],
                          color=right_cross_color, linewidth=right_cross_linewidth)

        if right_show_diameter_circle:
            circle = plt.Circle((centroid_x, centroid_y), cell_diameter / 2,
                                fill=False, edgecolor=right_diameter_circle_color,
                                linestyle=right_diameter_circle_linestyle,
                                linewidth=right_diameter_circle_linewidth)
            ax_right.add_patch(circle)

        ax_right.tick_params(axis='both', direction='in', labelsize=right_tick_fontsize, pad=right_tick_padding)
        ax_right.xaxis.set_ticks_position('top')
        ax_right.xaxis.set_label_position('top')
        ax_right.set_xlabel(right_x_label_content, fontsize=right_x_label_size)
        ax_right.set_ylabel(right_y_label_content, fontsize=right_y_label_size)
        ax_right.xaxis.set_label_coords(right_x_label_pos[0], right_x_label_pos[1])
        ax_right.yaxis.set_label_coords(right_y_label_pos[0], right_y_label_pos[1])
        
        # Frame on the left plot
        rect = Rectangle((x_min_right, y_min_right), x_max_right - x_min_right, y_max_right - y_min_right,
                         linewidth=frame_linewidth, edgecolor=frame_color, linestyle=frame_linestyle,
                         facecolor='none', zorder=10)
        ax_left.add_patch(rect)

        # Connecting lines
        # Top-left
        con1 = mpl.patches.ConnectionPatch(xyA=(x_min_right, y_min_right), xyB=(x_min_right, y_min_right),
                                          coordsA=ax_left.transData, coordsB=ax_right.transData,
                                          color=connection_color, linewidth=connection_linewidth,
                                          linestyle=connection_linestyle, zorder=connection_zorder)
        # Top-right
        con2 = mpl.patches.ConnectionPatch(xyA=(x_max_right, y_min_right), xyB=(x_max_right, y_min_right),
                                          coordsA=ax_left.transData, coordsB=ax_right.transData,
                                          color=connection_color, linewidth=connection_linewidth,
                                          linestyle=connection_linestyle, zorder=connection_zorder)
        # Bottom-left
        con3 = mpl.patches.ConnectionPatch(xyA=(x_min_right, y_max_right), xyB=(x_min_right, y_max_right),
                                          coordsA=ax_left.transData, coordsB=ax_right.transData,
                                          color=connection_color, linewidth=connection_linewidth,
                                          linestyle=connection_linestyle, zorder=connection_zorder)
        # Bottom-right
        con4 = mpl.patches.ConnectionPatch(xyA=(x_max_right, y_max_right), xyB=(x_max_right, y_max_right),
                                          coordsA=ax_left.transData, coordsB=ax_right.transData,
                                          color=connection_color, linewidth=connection_linewidth,
                                          linestyle=connection_linestyle, zorder=connection_zorder)
        
        ax_right.set_zorder(ax_left.get_zorder()+1)
        
        fig.add_artist(con1)
        fig.add_artist(con2)
        fig.add_artist(con3)
        fig.add_artist(con4)

        # Add subplot labels if requested
        if show_subplot_labels:
            # Left subplot label
            ax_left.text(
                x_min_left + (x_max_left - x_min_left) * subplot_label_pos[0],
                y_min_left + (y_max_left - y_min_left) * (1 - subplot_label_pos[1]),
                left_subplot_label,
                fontsize=subplot_label_fontsize,
                fontweight=subplot_label_weight,
                ha=subplot_label_ha,
                va=subplot_label_va,
                transform=ax_left.transData  # Use data coordinates
            )
            
            # Right subplot label
            ax_right.text(
                x_min_right + (x_max_right - x_min_right) * subplot_label_pos[0],
                y_min_right + (y_max_right - y_min_right) * (1 - subplot_label_pos[1]),
                right_subplot_label,
                fontsize=subplot_label_fontsize,
                fontweight=subplot_label_weight,
                ha=subplot_label_ha,
                va=subplot_label_va,
                transform=ax_right.transData  # Use data coordinates
            )
        
        # Add subplot titles if provided
        if left_subplot_title:
            ax_left.text(
                x_min_left + (x_max_left - x_min_left) * subplot_title_pos[0],
                y_max_left - (y_max_left - y_min_left) * subplot_title_pos[1],
                left_subplot_title,
                fontsize=subplot_title_fontsize,
                fontweight=subplot_title_weight,
                ha=subplot_title_ha,
                va=subplot_title_va,
                transform=ax_left.transData  # Use data coordinates
            )
            
        if right_subplot_title:
            ax_right.text(
                x_min_right + (x_max_right - x_min_right) * subplot_title_pos[0],
                y_max_right - (y_max_right - y_min_right) * subplot_title_pos[1],
                right_subplot_title,
                fontsize=subplot_title_fontsize,
                fontweight=subplot_title_weight,
                ha=subplot_title_ha,
                va=subplot_title_va,
                transform=ax_right.transData  # Use data coordinates
            )

        output_filename = f"plot15_singlecell_and_SF_{int(image_num):04d}_cell{cell_id}"
        fig.savefig(output_dir / f"{output_filename}.png", dpi=dpi, bbox_inches='tight')
        fig.savefig(output_dir / f"{output_filename}.svg", dpi=dpi, bbox_inches='tight')

        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        print(f"Saved plot for image {image_num} and cell {cell_id}")

if __name__ == "__main__":
    # --- V S C O D E   R U N N A B L E   S E C T I O N ---
    # Define all arguments here
    
    # --- Input and Output ---
    input_directory = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187"
    manual_output_directory = ""  # Optional: Or a specific path
    output_comment = "single_cell_and_SF_plots"
    
    # --- Image and Cell Selection ---
    images_to_process = [79]  # Empty list [] for all images
    target_cell_id = 95

    # --- Left Plot (Segmented Image) Parameters ---
    left_plot_zoom = 1.0
    left_plot_scale_factor = None # Overrides zoom if not None
    left_plot_show_masks = True
    left_plot_show_outlines = True
    left_plot_cells_to_color = [] # Empty for all cells
    left_plot_mask_alpha = 0.5
    left_plot_contour_color = 'white'
    left_plot_contour_linestyle = '-'
    left_plot_contour_linewidth = 1
    
    # --- Right Plot (Single Cell) Parameters ---
    right_plot_zoom = 2.5
    right_plot_show_mask = True
    right_plot_show_outline = True
    right_plot_show_diameter_circle = True
    right_plot_show_centroid_cross = True
    right_plot_mask_alpha = 0.5
    right_plot_outline_color = 'white'
    right_plot_outline_linestyle = '-'
    right_plot_outline_linewidth = 3
    right_plot_diameter_circle_color = 'red'
    right_plot_diameter_circle_linestyle = '-'
    right_plot_diameter_circle_linewidth = 4
    right_plot_cross_color = 'black'
    right_plot_cross_size = 20
    right_plot_cross_linewidth = 3

    # --- Frame and Connecting Line Parameters ---
    frame_color = 'black'
    frame_linewidth = 2
    frame_linestyle = '-'
    connection_color = 'black'
    connection_linewidth = 1
    connection_linestyle = '--'
    connection_zorder = 5
    
    # --- Label and Font Arguments ---
    left_x_label = r"x [px]"
    left_y_label = r"y [px]"
    right_x_label = r"x [px]"
    right_y_label = r"y [px]"
    
    left_xlabel_fontsize = 20
    left_ylabel_fontsize = 20
    right_xlabel_fontsize = 20
    right_ylabel_fontsize = 20

    left_xlabel_position = (0.5, 1.08)
    left_ylabel_position = (-0.12, 0.5)  # Moved further left
    right_xlabel_position = (0.5, 1.08)
    right_ylabel_position = (-0.12, 0.5)  # Moved further left
    
    left_ticks_fontsize = 20
    right_ticks_fontsize = 20
    left_tick_padding = 5  # Added padding for tick labels
    right_tick_padding = 5  # Added padding for tick labels
    
    # --- Subplot Labels and Titles ---
    show_labels = True
    left_label = r"\textbf{(a)}"  # Bold in LaTeX
    right_label = r"\textbf{(b)}"  # Bold in LaTeX
    label_fontsize = 20
    label_weight = 'bold'
    label_position = (0.03, 0.03)  # Bottom left
    label_vertical_align = 'bottom'
    label_horizontal_align = 'left'
    
    left_title = "Full image with segmented cells"
    right_title = "Single cell detail"
    title_fontsize = 20
    title_weight = 'normal'
    title_position = (0.5, -0.01)
    title_vertical_align = 'top'
    title_horizontal_align = 'center'
    
    # --- General Execution Arguments ---
    display_plot = 0 # 1 to show, 0 to save and close
    figure_dpi = 300
    
    # --- Function Call ---
    plot_singlecell_and_sf(
        input_dir=input_directory,
        output_dir_manual=manual_output_directory,
        output_dir_comment=output_comment,
        image_numbers=images_to_process,
        cell_id=target_cell_id,
        # Left plot args
        left_zoom_factor=left_plot_zoom,
        left_ScaleFactor=left_plot_scale_factor,
        left_show_masks=left_plot_show_masks,
        left_show_outlines=left_plot_show_outlines,
        left_cells_to_color=left_plot_cells_to_color,
        left_alpha=left_plot_mask_alpha,
        left_contour_color=left_plot_contour_color,
        left_contour_linestyle=left_plot_contour_linestyle,
        left_contour_linewidth=left_plot_contour_linewidth,
        # Right plot args
        right_zoom_factor=right_plot_zoom,
        right_show_mask=right_plot_show_mask,
        right_show_outline=right_plot_show_outline,
        right_show_diameter_circle=right_plot_show_diameter_circle,
        right_show_centroid_cross=right_plot_show_centroid_cross,
        right_mask_alpha=right_plot_mask_alpha,
        right_outline_color=right_plot_outline_color,
        right_outline_linestyle=right_plot_outline_linestyle,
        right_outline_linewidth=right_plot_outline_linewidth,
        right_diameter_circle_color=right_plot_diameter_circle_color,
        right_diameter_circle_linestyle=right_plot_diameter_circle_linestyle,
        right_diameter_circle_linewidth=right_plot_diameter_circle_linewidth,
        right_cross_color=right_plot_cross_color,
        right_cross_size=right_plot_cross_size,
        right_cross_linewidth=right_plot_cross_linewidth,
        # Frame and connecting line args
        frame_color=frame_color,
        frame_linewidth=frame_linewidth,
        frame_linestyle=frame_linestyle,
        connection_color=connection_color,
        connection_linewidth=connection_linewidth,
        connection_linestyle=connection_linestyle,
        connection_zorder=connection_zorder,
        # Label and Font args
        left_x_label_content=left_x_label,
        left_y_label_content=left_y_label,
        right_x_label_content=right_x_label,
        right_y_label_content=right_y_label,
        left_x_label_size=left_xlabel_fontsize,
        left_y_label_size=left_ylabel_fontsize,
        right_x_label_size=right_xlabel_fontsize,
        right_y_label_size=right_ylabel_fontsize,
        left_x_label_pos=left_xlabel_position,
        left_y_label_pos=left_ylabel_position,
        right_x_label_pos=right_xlabel_position,
        right_y_label_pos=right_ylabel_position,
        left_tick_fontsize=left_ticks_fontsize,
        right_tick_fontsize=right_ticks_fontsize,
        left_tick_padding=left_tick_padding,
        right_tick_padding=right_tick_padding,
        # Subplot labels and titles args
        show_subplot_labels=show_labels,
        left_subplot_label=left_label,
        right_subplot_label=right_label,
        subplot_label_fontsize=label_fontsize,
        subplot_label_weight=label_weight,
        subplot_label_pos=label_position,
        subplot_label_va=label_vertical_align,
        subplot_label_ha=label_horizontal_align,
        left_subplot_title=left_title,
        right_subplot_title=right_title,
        subplot_title_fontsize=title_fontsize,
        subplot_title_weight=title_weight,
        subplot_title_pos=title_position,
        subplot_title_va=title_vertical_align,
        subplot_title_ha=title_horizontal_align,
        # General args
        show_plot=display_plot,
        dpi=figure_dpi
    )