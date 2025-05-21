import os
import sys
import pandas as pd
import time
import json
from PIL import Image, ImageDraw, ImageFont

import Format_1 as F_1 # Assuming Format_1.py contains F_out_dir and ParameterLog

# Definition of the color table creation function
# (Copied from the provided immersive artifact: periodic_bw_color_table_generator)
def create_periodic_bw_color_table(num_periods, distance_ww, distance_wb, distance_bb, distance_bw, visit_module_ref, colortable_output_dir):
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

    # Add markers and labels
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
    
    # Add the color table to VisIt
    visit_module_ref.AddColorTable("PeriodicBW", ccpl)
    print(f"Color table 'PeriodicBW' created and saved to {colortable_output_dir}")
