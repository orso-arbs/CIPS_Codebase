import os
import sys
import pandas as pd
import time

import Format_1 as F_1 # Assuming Format_1.py contains F_out_dir and ParameterLog

# Definition of the color table creation function
# (Copied from the provided immersive artifact: periodic_bw_color_table_generator)
def create_periodic_bw_color_table(color_table_name, num_periods, ww, wb, bb, bw, visit_module_ref):
    """
    Creates and adds a VisIt color table with periodic black and white segments.
    This version builds a list of control point data first for conciseness.

    Args:
        color_table_name (str): The name for the new color table.
        num_periods (int): The number of times the white-black-white pattern should repeat.
        ww (float): Relative length of the solid white segment.
        wb (float): Relative length of the white-to-black gradient segment.
        bb (float): Relative length of the solid black segment.
        bw (float): Relative length of the black-to-white gradient segment.
        visit_module_ref: Reference to the VisIt Python module (e.g., the 'visit' module).
    """
    if num_periods <= 0:
        print(f"Error: Number of periods must be positive. Cannot create color table '{color_table_name}'.")
        return

    total_relative_distance_one_period = float(ww + wb + bb + bw)

    if total_relative_distance_one_period <= 0:
        print(f"Error: Sum of segment lengths (ww, wb, bb, bw) must be positive. Cannot create color table '{color_table_name}'.")
        return

    white_color = (255, 255, 255, 255)  # RGBA for white
    black_color = (0, 0, 0, 255)      # RGBA for black

    current_pos = 0.0
    point_definitions = [] # List to store (color_tuple, position_float)

    # Calculate the normalized deltas for each segment based on the total length
    # of the colormap (which is num_periods * total_relative_distance_one_period mapped to [0,1])
    # norm_factor is the scaling factor to map one unit of relative distance to the [0,1] colormap range
    norm_factor = 1.0 / (total_relative_distance_one_period * num_periods)
    
    delta_ww = ww * norm_factor
    delta_wb = wb * norm_factor
    delta_bb = bb * norm_factor
    delta_bw = bw * norm_factor

    # Add the very first control point (start of the first white segment)
    point_definitions.append((white_color, current_pos))

    # Loop through each period to define control points
    for i in range(num_periods):
        # End of White segment / Start of White-to-Black gradient
        current_pos += delta_ww
        current_pos = min(current_pos, 1.0) 
        point_definitions.append((white_color, current_pos))

        # End of White-to-Black gradient / Start of Black segment
        current_pos += delta_wb
        current_pos = min(current_pos, 1.0)
        point_definitions.append((black_color, current_pos))

        # End of Black segment / Start of Black-to-White gradient
        current_pos += delta_bb
        current_pos = min(current_pos, 1.0)
        point_definitions.append((black_color, current_pos))

        # End of Black-to-White gradient / End of period (also start of next white segment)
        current_pos += delta_bw
        
        pos_to_set = current_pos
        if i == num_periods - 1: # Ensure the very last point is exactly at 1.0
            pos_to_set = 1.0
        else: # For intermediate periods, clamp to avoid overshooting
            pos_to_set = min(pos_to_set, 1.0)
        point_definitions.append((white_color, pos_to_set))

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

    # Add the color table to VisIt
    visit_module_ref.AddColorTable(color_table_name, ccpl)
    print(f"Color table '{color_table_name}' created and added to VisIt.")
