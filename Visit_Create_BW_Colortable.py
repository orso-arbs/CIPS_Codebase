import os
import sys
import pandas as pd
import time
import json
from PIL import Image, ImageDraw, ImageFont

import Format_1 as F_1 # Assuming Format_1.py contains F_out_dir and ParameterLog

# Definition of the color table creation function
# (Copied from the provided immersive artifact: periodic_bw_color_table_generator)
def create_periodic_bw_color_table(num_periods, distance_ww, distance_wb, distance_bb, distance_bw, visit_module_ref, colortable_output_dir, show_markers=True):
    """
    Creates and adds a VisIt color table with periodic black and white segments.
    This version builds a list of control point data first for conciseness.

    Args:
        num_periods (int): The number of times the white-black-white pattern should repeat.
        distance_ww (float): Relative length of the solid white segment.
        distance_wb (float): Relative length of the white-to-black gradient segment.
        distance_bb (float): Relative length of the solid black segment.
        distance_bw (float): Relative length of the black-to-white gradient segment.
        visit_module_ref: Reference to the VisIt Python module (e.g., the 'visit' module).
        colortable_output_dir (str): Directory to save the color table JSON and preview image.
        show_markers (bool): Whether to show position markers and labels in the preview image.
    """
    if num_periods <= 0:
        print(f"Error: Number of periods must be positive. Cannot create color table PeriodicBW.")
        return

    total_relative_distance_one_period = float(distance_ww + distance_wb + distance_bb + distance_bw)

    if total_relative_distance_one_period <= 0:
        print(f"Error: Sum of segment lengths (distance_ww, distance_wb, distance_bb, distance_bw) must be positive. Cannot create color table PeriodicBW.")
        return
    

    # Calculate the normalized deltas for each segment based on the total length
    # of the colormap (which is total_relative_distance mapped to [0,1])
    # norm_factor is the scaling factor to map one unit of relative distance to the [0,1] colormap range
    distance_ww_wb_bb = float(distance_ww + distance_wb + distance_bb)
    total_relative_distance = float(distance_ww_wb_bb * num_periods + distance_bw * (num_periods - 1))

    norm_factor = 1.0 / (total_relative_distance)
    
    delta_ww = distance_ww * norm_factor
    delta_wb = distance_wb * norm_factor
    delta_bb = distance_bb * norm_factor
    delta_bw = distance_bw * norm_factor

    white_color = (255, 255, 255, 255)  # RGBA for white
    black_color = (0, 0, 0, 255)      # RGBA for black

    current_pos = 0.0
    point_definitions = [] # List to store (color_tuple, position_float)

    # Add the very first control point (start of the first white segment)
    point_definitions.append((white_color, current_pos))

    # Loop through each period
    for i in range(num_periods):
        # White segment
        current_pos += delta_ww
        current_pos = min(current_pos, 1.0)
        point_definitions.append((white_color, current_pos))

        # White-to-Black gradient endpoint
        current_pos += delta_wb
        current_pos = min(current_pos, 1.0)
        point_definitions.append((black_color, current_pos))

        # Black segment endpoint
        current_pos += delta_bb
        current_pos = min(current_pos, 1.0)
        
        # For the last period, make sure we end at exactly 1.0 with black
        if i == num_periods - 1:
            point_definitions.append((black_color, 1.0))
        else:
            # For non-final periods, add the black point at current position
            point_definitions.append((black_color, current_pos))
            # And add the transition back to white for the next period
            current_pos += delta_bw
            current_pos = min(current_pos, 1.0)
            point_definitions.append((white_color, current_pos))

    # Ensure the very last point is exactly at 1.0 and black
    if point_definitions[-1][1] < 1.0 or point_definitions[-1][0] != black_color:
        point_definitions.append((black_color, 1.0))

    # Initialize ColorControlPointList
    ccpl = visit_module_ref.ColorControlPointList()
    # By default, continuous color tables use linear smoothing and do not have equal spacing,
    # which is what we want for this kind of gradient definition.
    # ccpl.smoothing = ccpl.Linear (this is default)
    # ccpl.equalSpacingFlag = 0 (this is default)

    # Add control points from the generated list
    for color_tuple, pos_float in point_definitions:
        p = visit_module_ref.ColorControlPoint()
        p.colors = color_tuple
        p.position = pos_float
        ccpl.AddControlPoints(p)

    # Save control points to JSON
    os.makedirs(colortable_output_dir, exist_ok=True)
    json_path = os.path.join(colortable_output_dir, f"PeriodicBW_control_points.json")
    json_data = {
        "control_points": [
            {"color": list(color), "position": pos} 
            for color, pos in point_definitions
        ],
        "parameters": {
            "num_periods": num_periods,
            "distance_ww": distance_ww,
            "distance_wb": distance_wb,
            "distance_bb": distance_bb,
            "distance_bw": distance_bw
        }
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    # Generate preview image
    width, height = 512, 50
    preview = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()  # For labels
    
    # Draw the color gradient first
    for i in range(width):
        x = i / (width - 1)  # Normalize to [0,1]
        # Find surrounding control points
        for j in range(len(point_definitions)-1):
            p1 = point_definitions[j]
            p2 = point_definitions[j+1]
            if p1[1] <= x <= p2[1]:
                # Linear interpolation
                t = (x - p1[1]) / (p2[1] - p1[1]) if p2[1] != p1[1] else 0
                color = [int(p1[0][k] * (1-t) + p2[0][k] * t) for k in range(3)]
                draw.line([(i,0), (i,height)], fill=tuple(color))
                break

    # Add markers and labels if enabled
    if show_markers:
        label_height = height + 15  # Height for text labels
        preview_with_labels = Image.new('RGB', (width, label_height), 'white')
        preview_with_labels.paste(preview, (0, 0))
        draw = ImageDraw.Draw(preview_with_labels)

        # Draw period markers and labels
        for i in range(num_periods):
            # Calculate start position for this period
            period_start_pos = i * (distance_ww_wb_bb) / total_relative_distance
            if i > 0:
                period_start_pos += (i * distance_bw) / total_relative_distance

            # Draw full red vertical line for period start
            period_x = int(period_start_pos * width)
            draw.line([(period_x, 0), (period_x, height)], fill=(255,0,0), width=2)

            # Get exact positions from control points for this period
            points_in_period = point_definitions[i*4:(i+1)*4] if i < num_periods-1 else point_definitions[i*4:]
            
            # Find white and black points from the control points
            for j, (color, pos) in enumerate(points_in_period):
                x = int(pos * width)
                if color == white_color:
                    # Red dashed line for white points
                    for y in range(0, height, 4):
                        draw.line([(x, y), (x, min(y+2, height))], fill=(255,0,0), width=1)
                    draw.text((x-4, height+2), "w", fill=(255,0,0), font=font)
                elif color == black_color:
                    # Red dashed line for black points
                    for y in range(0, height, 4):
                        draw.line([(x, y), (x, min(y+2, height))], fill=(255,0,0), width=1)
                    draw.text((x-4, height+2), "b", fill=(255,0,0), font=font)

        preview_path = os.path.join(colortable_output_dir, f"PeriodicBW_preview.png")
        preview_with_labels.save(preview_path)
    else:
        # Save without labels
        preview_path = os.path.join(colortable_output_dir, f"PeriodicBW_preview.png")
        preview.save(preview_path)
    
    # Add the color table to VisIt
    visit_module_ref.AddColorTable("PeriodicBW", ccpl)
    print(f"Color table 'PeriodicBW' created and saved to {colortable_output_dir}")

def create_pointwise_bw_color_table(points_list, visit_module_ref, colortable_output_dir, 
                                   name="PointWise", width=100, height=400, 
                                   show_markers=True, show_wb_curve=False,
                                   curve_width=60, label_fontsize=12, curve_color='red'):
    """
    Creates and adds a custom VisIt color table from a list of point definitions.
    Also creates matplotlib-based color bar visualizations.
    
    Args:
        points_list (list): A list of lists where each inner list contains:
                          [position, r, g, b, a] values.
                          position is a float between 0.0 and 1.0
                          r, g, b, a are color values between 0-255
        visit_module_ref: Reference to the VisIt Python module.
        colortable_output_dir (str): Directory to save the color table JSON and preview image.
        name (str): Name for the color table in VisIt (default: "PointWise")
        width (int): Width of the preview image in pixels (default: 100)
        height (int): Height of the preview image in pixels (default: 400)
        show_markers (bool): Whether to show position markers and labels (default: True)
        show_wb_curve (bool): Whether to show white-black curve (default: False)
        curve_width (int): Width of the white-black curve area (default: 60)
        label_fontsize (int): Font size for the W/B labels (default: 12)
        curve_color (str): Color for the white/black curve and labels (default: 'red')
    """
    if not points_list or len(points_list) < 2:
        print(f"Error: At least two points are needed to define a color table. Cannot create {name}.")
        return
    
    # Validate points format and sort by position
    validated_points = []
    for point in points_list:
        if len(point) != 5:
            print(f"Warning: Skipping malformed point {point}. Expected format: [position, r, g, b, a]")
            continue
            
        position, r, g, b, a = point
        if not (0 <= position <= 1 and 
                0 <= r <= 255 and 0 <= g <= 255 and 
                0 <= b <= 255 and 0 <= a <= 255):
            print(f"Warning: Skipping point with out-of-range values: {point}")
            continue
            
        validated_points.append((position, (r, g, b, a)))
    
    if len(validated_points) < 2:
        print(f"Error: Not enough valid points after validation. Cannot create {name}.")
        return
    
    # Sort points by position
    validated_points.sort(key=lambda x: x[0])
    
    # Ensure the first point starts at 0.0 and the last ends at 1.0
    if validated_points[0][0] != 0.0:
        first_color = validated_points[0][1]
        validated_points.insert(0, (0.0, first_color))
        print(f"Added point at position 0.0 with color {first_color}")
        
    if validated_points[-1][0] != 1.0:
        last_color = validated_points[-1][1]
        validated_points.append((1.0, last_color))
        print(f"Added point at position 1.0 with color {last_color}")
    
    # Initialize ColorControlPointList for VisIt
    ccpl = visit_module_ref.ColorControlPointList()
    
    # Add control points from the validated list
    for pos_float, color_tuple in validated_points:
        p = visit_module_ref.ColorControlPoint()
        p.colors = color_tuple
        p.position = pos_float
        ccpl.AddControlPoints(p)
    
    # Save control points to JSON
    os.makedirs(colortable_output_dir, exist_ok=True)
    json_path = os.path.join(colortable_output_dir, f"{name}_control_points.json")
    json_data = {
        "control_points": [
            {"position": pos, "color": list(color)} 
            for pos, color in validated_points
        ],
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    # Add the color table to VisIt
    visit_module_ref.AddColorTable(name, ccpl)
    print(f"Color table '{name}' created and added to VisIt")
    
    # Generate matplotlib-based color bar visualizations
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    
    # Convert validated_points to colormap format for matplotlib
    cmap_points = []
    cmap_colors = []
    
    for pos, color in validated_points:
        # Convert RGBA 0-255 to RGB 0-1
        r, g, b, a = color
        cmap_points.append(pos)
        cmap_colors.append((r/255, g/255, b/255))
    
    # Create a custom colormap from the points
    cmap = LinearSegmentedColormap.from_list(name, list(zip(cmap_points, cmap_colors)))
    
    # Calculate grayscale values for each position in the colormap
    positions = np.linspace(0, 1, 256)
    colors = cmap(positions)
    grayscale = np.dot(colors[:, :3], [0.299, 0.587, 0.114])  # Convert RGB to grayscale using luminance formula
    
    # UNLABELED COLORBAR
    fig_unlabeled, ax_unlabeled = plt.subplots(figsize=(width/100, height/100))
    fig_unlabeled.patch.set_alpha(0)  # Make figure background transparent
    
    # Create a vertical colorbar
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax_unlabeled,
        orientation='vertical'
    )
    
    # Remove ticks and set transparent background
    cb.set_ticks([])
    cb.outline.set_visible(False)  # Remove colorbar outline
    ax_unlabeled.patch.set_alpha(0)  # Make axes background transparent
    
    # Add a black box around the colorbar
    ax_unlabeled.set_frame_on(True)
    for spine in ax_unlabeled.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.5)
    
    # Save unlabeled color bar with transparent background
    unlabeled_path = os.path.join(colortable_output_dir, f"{name}_colorbar_unlabeled.png")
    fig_unlabeled.savefig(unlabeled_path, bbox_inches='tight', dpi=100, transparent=True)
    plt.close(fig_unlabeled)
    
    # LABELED COLORBAR WITH WHITENESS/BLACKNESS CURVE
    if show_wb_curve:
        # Create figure for labeled colorbar
        fig_labeled, ax_labeled = plt.subplots(figsize=(width/100 * 1.0, height/100))
        fig_labeled.patch.set_alpha(0)  # Make figure background transparent
        
        # Create the colorbar
        cb_labeled = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax_labeled,
            orientation='vertical'
        )
        
        # Remove ticks and set transparent background
        cb_labeled.set_ticks([])
        cb_labeled.outline.set_visible(False)  # Remove colorbar outline
        ax_labeled.patch.set_alpha(0)  # Make axes background transparent
        
        # Plot the whiteness/blackness curve
        # Position the curve on top of the colorbar
        pos_scatter = np.linspace(0, 1, 50)  # Positions for smoother curve
        whiteness_curve = []
        
        for pos in pos_scatter:
            # Get color at this position
            color_at_pos = cmap(pos)[:3]  # RGB tuple
            
            # Calculate grayscale using luminance formula
            gray_val = 0.299*color_at_pos[0] + 0.587*color_at_pos[1] + 0.114*color_at_pos[2]
            
            # Map to x position (1=white=left, 0=black=right)
            # Scale to fit within the colorbar width
            whiteness_curve.append(gray_val)
        
        # Get the position of the colorbar for overlay
        bbox = ax_labeled.get_position()
        bar_width = bbox.width
        
        # Create a twin axis for the curve
        ax_curve = ax_labeled.twiny()
        ax_curve.set_position(bbox)  # Match the colorbar position
        
        # Plot the curve in the specified color
        ax_curve.plot(whiteness_curve, pos_scatter, color=curve_color, linewidth=2)
        
        # Remove ticks and labels from the curve axis
        ax_curve.set_xticks([])
        ax_curve.set_yticks([])
        
        # Set axis limits to match the colorbar
        ax_curve.set_xlim(0, 1)
        ax_curve.set_ylim(0, 1)
        
        # Add 'W' and 'B' labels at bottom in LaTeX format
        ax_curve.text(0.9, 0.02, r'$\mathbf{W}$', color=curve_color, 
                     ha='center', va='bottom', fontsize=label_fontsize)
        ax_curve.text(0.1, 0.02, r'$\mathbf{B}$', color=curve_color, 
                     ha='center', va='bottom', fontsize=label_fontsize)
        
        # Save labeled color bar with transparent background
        labeled_path = os.path.join(colortable_output_dir, f"{name}_colorbar_labeled.png")
        fig_labeled.savefig(labeled_path, bbox_inches='tight', dpi=100, transparent=True)
        plt.close(fig_labeled)
    
    print(f"Color table visualizations saved to {colortable_output_dir}")
    return name  # Return the name of the created color table

# Example usage (commented out)
# if __name__ == "__main__":
#     # This is just a demonstration - in practice, you would pass the visit module
#     # Example points list format: [position, r, g, b, a]
#     sample_points = [
#         [0.0, 255, 255, 255, 255],  # White at 0.0
#         [0.3, 200, 200, 200, 255],  # Light gray at 0.3
#         [0.6, 100, 100, 100, 255],  # Dark gray at 0.6
#         [1.0, 0, 0, 0, 255]         # Black at 1.0
#     ]
#     
#     # You would need the actual VisIt module and a directory
#     # create_pointwise_bw_color_table(sample_points, visit, "output/dir")
