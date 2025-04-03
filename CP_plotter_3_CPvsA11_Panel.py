import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, plot, utils, io
import datetime
import glob
import os
import time
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
plt.rcParams['text.usetex'] = False  # Keep False unless you have a full LaTeX installation
import numpy as np
from skimage import io as sk_io, color, measure

import sys
import os
sys.path.append(os.path.abspath(r"C:/Users/obs/OneDrive/ETH/ETH_MSc/Masters Thesis/Python Code/Python_Orso_Utility_Scripts_MscThesis")) # dir containing Format_1 
import Format_1 as F_1

import video_maker_1 as vm1


@F_1.ParameterLog(max_size = 1024 * 10) # 10KB 
def CP_plotter_3_CPvsA11_Panel(input_dir, # Format_1 requires input_dir
    CP_extract_df = None, # if None a .pkl file has to be in the input_dir. otherwise no CP_extract data is provided.
    output_dir_manual = "", output_dir_comment = "",
    video = 1, show_plot = 1,
    Panel_1 = 0, Panel_2 = 0, Panel_3 = 0, Panel_4 = 0,
    ):

    ### output 
    output_dir = F_1.F_out_dir(input_dir, __file__, output_dir_comment = "") # Format_1 required definition of output directory

    pkl_files = glob.glob(os.path.join(input_dir, "*.pkl"))

    ### Load CP extraxct data
    if pkl_files:
        CP_extract_df_pkl = pkl_files[0] # If a .pkl file exists, use it as the pickle file path. If multiple .pkl files exist the first is used.
    else:
        CP_extract_df_pkl = None

    if CP_extract_df is None and CP_extract_df_pkl is None:
        raise ValueError("No CP_extract data provided. Provide Data.")
    elif CP_extract_df is None and CP_extract_df_pkl is not None:
        print(f"Loading CP_extract data from pickle file {os.path.basename(CP_extract_df_pkl)}")
        CP_extract_df = pd.read_pickle(CP_extract_df_pkl)
    elif CP_extract_df is not None and CP_extract_df_pkl is None:
        print("Loading CP_extract data from passed DataFrame in function argument")
        # No action needed since CP_extract_df is already passed
    elif CP_extract_df is not None and CP_extract_df_pkl is not None:
        print("Both CP_extract_df and CP_extract_df_pkl provided. Using data from passed DataFrame in function argument")
        # No action needed since CP_extract_df is already passed
    else:
        raise ValueError("Loading CP_extract data disambiguation failed. Check CP_extract_df and CP_extract_df_pkl")



    # Load A11 data
    A11_SF_K_mean = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction\A11_SF_K_mean_as_mean_stretch_rate_vs_time_manual_extraction.txt")
    A11_SF_N_c = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction\A11_SF_N_c_as_number_of_cells_vs_time_manual_extraction.txt")
    A11_SF_R_mean = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction\A11_SF_R_mean_as_average_radius_of_the_wrinkled_flame_fron_vs_time_manual_extraction.txt")
    A11_SF_R_mean_dot = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction\A11_SF_R_mean_dot_as_first_time_derivative_of_the_average_radius_of_the_wrinkled_flame_front_vs_time_manual_extraction.txt")
    A11_SF_s_a = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction\A11_SF_s_a_as_average_normal_component_of_the_absolute_propagation_velocity_vs_time_manual_extraction.txt")
    A11_SF_s_d = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction\A11_SF_s_d_as_average_density_weighted_displacement_speed_vs_time_manual_extraction.txt")
    A11_SF_A = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction\A11_SF_A_as_flame_surface_area_of_the_wrinkled_spherical_front_vs_time_manual_extraction.txt")
    A11_SF_a_t = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction\A11_SF_a_t_as_average_total_aerodynamic_strain_vs_time_manual_extraction.txt")
    A11_SF_iHRR = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction\A11_SF_iHRR_as_integral_heat_release_rate_vs_time_manual_extraction.txt")
    A11_SF_K_geom = pd.read_csv(r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction\A11_SF_K_geom_as_geometric_stretch_rate_vs_time_manual_extraction.txt")


    # auxillary function to plot the data

    # Number of rows in the DataFrame
    N_images = len(CP_extract_df)

    # Find the maximum frequency for all histograms
    max_diameter = max([diameter for sublist in CP_extract_df['diameter_distribution_nonDim'] for diameter in sublist])
    
    print(f"\nPlotting data")


    if Panel_1 == 1:
        # Create the figure with a custom GridSpec layout
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)

        # plot panel
        ax_0_0 = fig.add_subplot(gs[0, 0])
        ax_0_1 = fig.add_subplot(gs[0, 1])
        ax_0_2 = fig.add_subplot(gs[0, 2])
        ax_1_0 = fig.add_subplot(gs[1, 0])
        ax_1_12 = fig.add_subplot(gs[1, 1:3]) # spanning across two columns


        # Plot: Image number vs. median diameter, mean diameter, and amount of cells (up to current image)

        ax_1_12.plot(CP_extract_df['time'], CP_extract_df['diameter_mean_nonDim'], label="Cell Mean Diameter", color='green')
        ax_1_12.plot(CP_extract_df['time'], CP_extract_df['diameter_training_nonDim'], label=f"Cellpose Training Diameter", color='aquamarine')
        
        #S = max(CP_extract_df['diameter_mean_nonDim'].max(), CP_extract_df['diameter_median_nonDim'].max()) / CP_extract_df['D_FB_nonDim'].max()
        #ax_1_12.plot(range(N_images), CP_extract_df['D_FB_nonDim'] * S, label=f"{(CP_extract_df.iloc[i]['D_FB_nonDim']*S):.2f} = Spherical Flame Diameter * {S:.3f}", color='orange')
            
        
        S2 = 1e-1
        ax_1_12.plot(CP_extract_df['time'], CP_extract_df['R_FB_nonDim'] * S2, label=f"Image deduced Spherical Flame Radius * {S2:.3f}", color='olive')
        ax_1_12.plot(A11_SF_R_mean['time'], A11_SF_R_mean['R_mean'] * S2, label=f"A11 Spherical Flame Radius * {S2:.3f}", color='olive', linestyle='dashed')
        
        S3 = 1
        ax_1_12_L = ax_1_12.twinx() 
        ax_1_12_L.plot(A11_SF_iHRR['time'], A11_SF_iHRR['iHRR'] * S3, label=f"A11 integral heat release rate * {S2:.3f}", color='orange', linestyle='dashed')

        ax_1_12_R = ax_1_12.twinx()
        ax_1_12_R.plot(CP_extract_df['time'], CP_extract_df['N_cells'], label=f"Number of cells", color='red')
        
        # Create a third y-axis for the dotted line plots
        ax_1_12_RR = ax_1_12.twinx()  # Second twin axis
        ax_1_12_RR.plot(CP_extract_df['time'], CP_extract_df['Ar_px2_CP_maskperFB'], label="$A_{Cell masks}/A_{Spherical Flame}$", color='gray')



        # Set the limits and labels for the axes

        ax_1_12.set_xlim(0, 7)
        ax_1_12.set_ylim(0, max_diameter*1.05)
        ax_1_12_L.set_ylim(0, A11_SF_iHRR['iHRR'].max()*1.05)
        ax_1_12_R.set_ylim(CP_extract_df['N_cells'].min(), CP_extract_df['N_cells'].max()*1.05)
        ax_1_12_RR.set_ylim(0, 1)


        solid_line = mlines.Line2D([], [], color='black', linestyle='-', label="Cellpose (Solid)")
        dashed_line = mlines.Line2D([], [], color='black', linestyle='--', label="A11 (Dashed)")
        ax_1_12.set_title("Diameter and Cell Count")
        ax_1_12.legend(handles=[solid_line, dashed_line], loc='upper center', fontsize=10, frameon=False)
        
        ax_1_12.set_xlabel("time")
        ax_1_12.set_ylabel("Diameter", color='green')
        ax_1_12_L.set_ylabel("Heat Release Rate", color='orange')
        ax_1_12_R.set_ylabel("Number of Cells", color='red')
        ax_1_12_RR.set_ylabel("$A_{Cell masks}/A_{Spherical Flame}$", color='gray')

        ax_1_12.legend(loc='upper left')
        ax_1_12_L.legend(loc='lower left')
        ax_1_12_R.legend(loc='upper right')
        ax_1_12_RR.legend(loc='lower right')

        ax_1_12_L.spines["right"].set_position(("outward", 0))  # Slightly to the right
        ax_1_12_L.yaxis.set_label_position("right")
        ax_1_12_L.yaxis.set_ticks_position("right")

        ax_1_12_R.tick_params(axis='y', labelcolor='red')
        ax_1_12_R.spines["right"].set_color('red')
        ax_1_12_R.spines["right"].set_position(("outward", 45))  # Move further right
        ax_1_12_R.yaxis.set_label_position("right")
        ax_1_12_R.yaxis.set_ticks_position("right")

        ax_1_12_RR.tick_params(axis='y', labelcolor='gray')
        ax_1_12_RR.spines["right"].set_color('gray')
        ax_1_12_RR.spines["right"].set_position(("outward", 90))  # Move even further right
        ax_1_12_RR.yaxis.set_label_position("right")
        ax_1_12_RR.yaxis.set_ticks_position("right")

        ax_1_12_L.tick_params(axis='y', labelcolor='orange')
        ax_1_12_L.spines["right"].set_color('orange')
        ax_1_12_L.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax_1_12_L.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        offset_text = ax_1_12_L.yaxis.get_offset_text()
        offset_text.set_position((1.05, 1))  # Move to the right and above the axis

        # Adjust layout and save the figure as a PNG file
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f'plot_panel.png')
        plt.savefig(plot_filename)
        plt.show() if show_plot == 1 else None
        plt.close(fig)

        print("\n") # new line





    if Panel_2 == 1:
        fig, (ax_1, ax_2, ax_3, ax_4) = plt.subplots(4, 1, figsize=(8, 10))  # 2 rows, 1 column

        # First plot: Plotting data from CP_extract_df
        
        # Plotting the first set of data
        ax_1.plot(CP_extract_df['time'], CP_extract_df['R_FB_nonDim'], label="R_FB_nonDim", color='olive', linestyle='solid')
        ax_1.set_xlabel('Time')
        ax_1.set_ylabel('Radius')
        ax_1.set_title('Spherical Flame Radius Comparison')

        # Twin axes 1
        ax_1_twin = ax_1.twinx()  # Create a twin axes sharing the same x-axis
        ax_1_twin.plot(A11_SF_R_mean['time'], A11_SF_R_mean['R_mean'], label="A11 R_mean", color='green', linestyle='dashed')
        ax_1_twin.spines["right"].set_position(("outward", 0))  # Slightly to the right

        # Twin axes 2
        ax_1_twin2 = ax_1.twinx()  # Create a twin axes sharing the same x-axis
        ax_1_twin2.plot(CP_extract_df['time'], CP_extract_df['D_FB_px'], label="D_FB_px", color='blue', linestyle='dashed')
        ax_1_twin2.spines["right"].set_position(("outward", 40))  # Slightly to the right

        # Calculate R_mean_interpolated_i
        ax_1_twin3 = ax_1.twinx()  # Create a twin axes sharing the same x-axis
        CP_extract_df['R_mean_interpolated_i'] = (CP_extract_df['d_T_per_px'] * CP_extract_df['D_FB_px']) / 2
        ax_1_twin3.plot(CP_extract_df['time'], CP_extract_df['R_mean_interpolated_i'], label="R_mean_interpolated_i", color='black', linestyle='solid')
        ax_1_twin3.set_ylabel('R_mean_interpolated_i', color='black')
        ax_1_twin3.tick_params(axis='y', labelcolor='black')
        ax_1_twin3.spines["right"].set_position(("outward", 80))  # Slightly to the right

        # Second axis for d_T_per_px
        ax_2.plot(CP_extract_df['time'], CP_extract_df['d_T_per_px'], label="d_T_per_px", color='blue', linestyle='solid')
        ax_2.set_ylabel('d_T_per_px', color='blue')
        ax_2.tick_params(axis='y', labelcolor='blue')

        # Third axis for diameter_mean_nonDim
        ax_3.plot(CP_extract_df['time'], CP_extract_df['diameter_mean_nonDim'], label="diameter_mean_nonDim", color='red', linestyle='solid')
        ax_3.set_ylabel('diameter_mean_nonDim', color='red')
        ax_3.tick_params(axis='y', labelcolor='red')

        ax_3_twin = ax_3.twinx()  # Create a twin axes sharing the same x-axis
        ax_3_twin.plot(CP_extract_df['time'], CP_extract_df['diameter_mean_px'], label="diameter_mean_px", color='green', linestyle='dashed')
        ax_3_twin.spines["right"].set_position(("outward", 0))  # Slightly to the right


        # Collect handles and labels from all axes
        handles, labels = [], []
        for ax in [ax_1, ax_1_twin, ax_1_twin2, ax_1_twin3, ax_2, ax_3, ax_3_twin]:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

        # Add a unified legend for the entire figure
        ax_4.legend(handles, labels, loc='center', ncol=1)  # Place legend at the center of ax_4

        plt.subplots_adjust(top=0.85)  # Adjust this value to add more space on top
        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plot
        plt.show()



    if Panel_3 == 1:
        # Create the figure with a custom GridSpec layout
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 1, figure=fig)

        # plot panel
        ax_0 = fig.add_subplot(gs[0, 0])
        ax_1 = fig.add_subplot(gs[1, 0])
        ax_2 = fig.add_subplot(gs[2, 0])


        # Plot: Image number vs. median diameter, mean diameter, and amount of cells (up to current image)

        ax_0.plot(CP_extract_df['time'], CP_extract_df['diameter_mean_nonDim'], label="Cell Mean Diameter", color='green')
        ax_0.plot(CP_extract_df['time'], CP_extract_df['diameter_training_nonDim'], label=f"Cellpose Training Diameter", color='aquamarine')
        
        #S = max(CP_extract_df['diameter_mean_nonDim'].max(), CP_extract_df['diameter_median_nonDim'].max()) / CP_extract_df['D_FB_nonDim'].max()
        #ax_0.plot(range(N_images), CP_extract_df['D_FB_nonDim'] * S, label=f"{(CP_extract_df.iloc[i]['D_FB_nonDim']*S):.2f} = Spherical Flame Diameter * {S:.3f}", color='orange')
            
        
        S2 = 1e-1
        ax_0.plot(CP_extract_df['time'], CP_extract_df['R_FB_nonDim'] * S2, label=f"Image deduced Spherical Flame Radius * {S2:.3f}", color='olive')
        ax_0.plot(A11_SF_R_mean['time'], A11_SF_R_mean['R_mean'] * S2, label=f"A11 Spherical Flame Radius * {S2:.3f}", color='olive', linestyle='dashed')
        
        S3 = 1
        ax_0_L = ax_0.twinx() 
        ax_0_L.plot(A11_SF_iHRR['time'], A11_SF_iHRR['iHRR'] * S3, label=f"A11 integral heat release rate * {S2:.3f}", color='orange', linestyle='dashed')

        ax_0_R = ax_0.twinx()
        ax_0_R.plot(CP_extract_df['time'], CP_extract_df['N_cells'], label=f"Number of cells", color='red')
        
        # Create a third y-axis for the dotted line plots
        ax_0_RR = ax_0.twinx()  # Second twin axis
        ax_0_RR.plot(CP_extract_df['time'], CP_extract_df['Ar_px2_CP_maskperFB'], label="$A_{Cell masks}/A_{Spherical Flame}$", color='gray')



        # Set the limits and labels for the axes

        ax_0.set_xlim(0, 7)
        ax_0.set_ylim(0, max_diameter*1.05)
        ax_0_L.set_ylim(0, A11_SF_iHRR['iHRR'].max()*1.05)
        ax_0_R.set_ylim(CP_extract_df['N_cells'].min(), CP_extract_df['N_cells'].max()*1.05)
        ax_0_RR.set_ylim(0, 1)


        solid_line = mlines.Line2D([], [], color='black', linestyle='-', label="Cellpose (Solid)")
        dashed_line = mlines.Line2D([], [], color='black', linestyle='--', label="A11 (Dashed)")
        ax_0.set_title("Diameter and Cell Count")
        ax_0.legend(handles=[solid_line, dashed_line], loc='upper center', fontsize=10, frameon=False)
        
        ax_0.set_xlabel("time")
        ax_0.set_ylabel("Diameter", color='green')
        ax_0_L.set_ylabel("Heat Release Rate", color='orange')
        ax_0_R.set_ylabel("Number of Cells", color='red')
        ax_0_RR.set_ylabel("$A_{Cell masks}/A_{Spherical Flame}$", color='gray')

        ax_0.legend(loc='upper left')
        ax_0_L.legend(loc='lower left')
        ax_0_R.legend(loc='upper right')
        ax_0_RR.legend(loc='lower right')

        ax_0_L.spines["right"].set_position(("outward", 0))  # Slightly to the right
        ax_0_L.yaxis.set_label_position("right")
        ax_0_L.yaxis.set_ticks_position("right")

        ax_0_R.tick_params(axis='y', labelcolor='red')
        ax_0_R.spines["right"].set_color('red')
        ax_0_R.spines["right"].set_position(("outward", 45))  # Move further right
        ax_0_R.yaxis.set_label_position("right")
        ax_0_R.yaxis.set_ticks_position("right")

        ax_0_RR.tick_params(axis='y', labelcolor='gray')
        ax_0_RR.spines["right"].set_color('gray')
        ax_0_RR.spines["right"].set_position(("outward", 90))  # Move even further right
        ax_0_RR.yaxis.set_label_position("right")
        ax_0_RR.yaxis.set_ticks_position("right")

        ax_0_L.tick_params(axis='y', labelcolor='orange')
        ax_0_L.spines["right"].set_color('orange')
        ax_0_L.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax_0_L.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        offset_text = ax_0_L.yaxis.get_offset_text()
        offset_text.set_position((1.05, 1))  # Move to the right and above the axis

        # Adjust layout and save the figure as a PNG file
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f'plot_panel.png')
        plt.savefig(plot_filename)
        plt.show() if show_plot == 1 else None
        plt.close(fig)

        print("\n") # new line


    ''' A11 dataframes:
    A11_SF_iHRR
    A11_SF_A
    A11_SF_R_mean
    A11_SF_R_mean_dot
    A11_SF_N_c
    A11_SF_K_geom
    A11_SF_K_mean
    A11_SF_a_t
    A11_SF_s_a
    A11_SF_s_d
    '''


    if Panel_4 == 1:
        # Create the figure with a custom GridSpec layout
        fig = plt.figure(figsize=(15, 15))
        gs = gridspec.GridSpec(5, 2, figure=fig, height_ratios=[1, 1, 1, 1, 1])
        gs.update(hspace=0)  # Remove gaps

        solid_line = mlines.Line2D([], [], color='black', linestyle='-', label="Cellpose")
        dashed_line = mlines.Line2D([], [], color='black', linestyle='--', label="Altantzis 2011")

        fig.suptitle("Your Figure Title Here", fontsize=20, fontweight='bold', y=1.02)  # Adjust y for spacing
        fig.legend(handles=[dashed_line, solid_line], loc='upper center', fontsize=12, frameon=False, ncol=2)


        # Create left column subplots with shared x-axis
        ax_0_0 = fig.add_subplot(gs[0, 0])
        ax_1_0 = fig.add_subplot(gs[1, 0], sharex=ax_0_0)
        ax_2_0 = fig.add_subplot(gs[2, 0], sharex=ax_0_0)
        ax_3_0 = fig.add_subplot(gs[3, 0], sharex=ax_0_0)
        ax_4_0 = fig.add_subplot(gs[4, 0], sharex=ax_0_0)
        # Create right column subplots with shared x-axis
        ax_0_1 = fig.add_subplot(gs[0, 1])
        ax_1_1 = fig.add_subplot(gs[1, 1], sharex=ax_0_1)
        ax_2_1 = fig.add_subplot(gs[2, 1], sharex=ax_0_1)
        ax_3_1 = fig.add_subplot(gs[3, 1], sharex=ax_0_1)
        ax_4_1 = fig.add_subplot(gs[4, 1], sharex=ax_0_1)


        # Plot 0 0
        # Image number vs. median diameter, mean diameter, and amount of cells (up to current image)

        # A11 first plot column
        ax_0_0.plot(A11_SF_A['time'], A11_SF_A['A'] ,
                    label="A11 Spherical Flame Area $A_{SF}$", color='black', linestyle='dashed')
        ax_0_0.set_ylim(0, A11_SF_A['A'].max()*1.05)
        ax_0_0.tick_params(axis='y', labelcolor='black')
        ax_0_0.spines["left"].set_color('black')
        ax_0_0.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax_0_0.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        offset_text = ax_0_0.yaxis.get_offset_text()
        offset_text.set_position((0, 1))  # Move to the right and above the axis
        ax_0_0.set_ylabel("A11 Spherical Flame Area $A_{SF}$", color='black')


        ax_1_0.plot(A11_SF_iHRR['time'], A11_SF_iHRR['iHRR'],
                    label="A11 Integral heat release rate $iHRR$", color='black', linestyle='dashed')
        ax_1_0.set_ylim(0, A11_SF_iHRR['iHRR'].max()*1.05)
        ax_1_0.tick_params(axis='y', labelcolor='black')
        ax_1_0.spines["left"].set_color('black')
        ax_1_0.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax_1_0.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        offset_text = ax_1_0.yaxis.get_offset_text()
        offset_text.set_position((-0.5, -0.5))  # Move offset text to the left side (x=-0.1)
        ax_1_0.set_ylabel("A11 Integral Heat Release Rate $iHRR$", color='black')


        ax_2_0.plot(A11_SF_R_mean['time'], A11_SF_R_mean['R_mean'] ,
                    label="A11 Spherical Flame Radius $R_{mean}$", color='black', linestyle='dashed')
        ax_2_0.set_ylabel("A11 Spherical Flame Radius $R_{mean}$", color='black')
        ax_2_0.tick_params(axis='y', labelcolor='black')



        ax_3_0.plot(A11_SF_R_mean_dot['time'], A11_SF_R_mean_dot['R_mean_dot'] ,
                    label="A11 Spherical Flame Radius first \ntime derivative $\dot{R}_{\text{mean}}$", color='black', linestyle='dashed')
        ax_3_0.set_ylabel("A11 Spherical Flame Radius first \ntime derivative $\dot{R}_{\text{mean}}$", color='black')
        ax_3_0.tick_params(axis='y', labelcolor='black')


        ax_4_0.plot(A11_SF_N_c['time'], A11_SF_N_c['N_c'] ,
                    label="A11 Number of cells $N_c$", color='black', linestyle='dashed')
        ax_4_0.set_ylabel("A11 Number of cells $N_c$", color='black')
        ax_4_0.tick_params(axis='y', labelcolor='black')


        # A11 second plot column
        ax_0_1.plot(A11_SF_K_geom['time'], A11_SF_K_geom['K_geom'] ,
                    label="A11 geometric stretch rate $K_{geom}$", color='black', linestyle='dashed')
        ax_0_1.plot(A11_SF_K_mean['time'], A11_SF_K_mean['K_mean'] ,
                    label="A11 mean stretch rate $K_{mean}$", color='blue', linestyle='dashed')

        ax_1_1.plot(A11_SF_a_t['time'], A11_SF_a_t['a_t'] ,
                    label="A11 average total areodynamic strain $a_t$", color='black', linestyle='dashed')

        ax_2_1.plot(A11_SF_s_a['time'], A11_SF_s_a['s_a'] ,
                    label="A11 average normal absolute propagation velocity $s_a$", color='black', linestyle='dashed')

        ax_3_1.plot(A11_SF_s_d['time'], A11_SF_s_d['s_d'] ,
                    label="A11 average density weighed displacement speed $s_d$", color='black', linestyle='dashed')

        ax_4_1.plot(CP_extract_df['time'], CP_extract_df['d_T_per_px'] ,
                    label="Dimentionalisation $d_T/px$", color='black', linestyle='dotted')



        axes = [
            ax_0_0, ax_1_0, ax_2_0, ax_3_0, ax_4_0,
            ax_0_1, ax_1_1, ax_2_1, ax_3_1, ax_4_1
        ]
        axes_L = [
            ax_0_0, ax_1_0, ax_2_0, ax_3_0, ax_4_0
        ]
        axes_R = [
            ax_0_1, ax_1_1, ax_2_1, ax_3_1, ax_4_1
        ]


        # Loop through each subplot and apply the plots
        for ax in axes_R:
            # Add Twin Axes
            ax_R1 = ax.twinx()
            ax_R1.plot(CP_extract_df['time'], CP_extract_df['diameter_mean_nonDim'],
                        label="Cell Mean Diameter $D_{c,mean}$", color='green')
            ax_R1.set_ylabel("Cell Mean Diameter $ D_{c,mean}$", color='green')

            ax_R2 = ax.twinx()
            ax_R2.plot(CP_extract_df['time'], CP_extract_df['N_cells'],
                        label="CP Number of cells $N_c$", color='red')
            ax_R2.set_ylabel("CP Number of cells $N_c$", color='red')

            ax_R3 = ax.twinx()
            ax_R3.plot(CP_extract_df['time'], CP_extract_df['Ar_px2_CP_maskperFB'],
                        label="CP efficiency $\mu_{CP} = A_{CP}/A_{SF}$", color='gray')
            ax_R3.set_ylabel("CP efficiency $\mu_{CP} = A_{CP}/A_{SF}$", color='gray')

            # Set Limits
            ax_R1.set_ylim(0, max_diameter * 1.05)
            ax_R2.set_ylim(CP_extract_df['N_cells'].min(), CP_extract_df['N_cells'].max() * 1.05)
            ax_R3.set_ylim(0, 1)

            # Adjust Twin Axes Positions
            ax_R1.spines["right"].set_position(("outward", 0))
            ax_R2.spines["right"].set_position(("outward", 40))
            ax_R3.spines["right"].set_position(("outward", 80))

            ax_R1.spines["right"].set_color('green')
            ax_R2.spines["right"].set_color('red')
            ax_R3.spines["right"].set_color('gray')

            ax_R1.tick_params(axis='y', labelcolor='green')
            ax_R2.tick_params(axis='y', labelcolor='red')
            ax_R3.tick_params(axis='y', labelcolor='gray')

            ax.set_xlim(0, 7)

        for ax in axes_L:
            # Add Twin Axes
            ax_R1 = ax.twinx()
            ax_R1.plot(CP_extract_df['time'], CP_extract_df['diameter_mean_nonDim'],
                        label="Cell Mean Diameter $D_{c,mean}$", color='green')

            ax_R2 = ax.twinx()
            ax_R2.plot(CP_extract_df['time'], CP_extract_df['N_cells'],
                        label="CP Number of cells $N_c$", color='red')

            ax_R3 = ax.twinx()
            ax_R3.plot(CP_extract_df['time'], CP_extract_df['Ar_px2_CP_maskperFB'],
                        label="$\mu_{CP} = A_{CP}/A_{SF}$", color='gray')

            # Set Limits
            ax_R1.set_ylim(0, max_diameter * 1.05)
            ax_R2.set_ylim(CP_extract_df['N_cells'].min(), CP_extract_df['N_cells'].max() * 1.05)
            ax_R3.set_ylim(0, 1)

            # Turn off all ticks and tick labels for the twin y-axes
            ax_R1.tick_params(axis='both', which='both', length=0)  # Hide ticks
            ax_R1.tick_params(axis='y', labelleft=False, labelright=False)  # Hide y-axis labels
            ax_R1.set_ylabel('')  # Hide y-axis label

            ax_R2.tick_params(axis='both', which='both', length=0)  # Hide ticks
            ax_R2.tick_params(axis='y', labelleft=False, labelright=False)  # Hide y-axis labels
            ax_R2.set_ylabel('')  # Hide y-axis label

            ax_R3.tick_params(axis='both', which='both', length=0)  # Hide ticks
            ax_R3.tick_params(axis='y', labelleft=False, labelright=False)  # Hide y-axis labels
            ax_R3.set_ylabel('')  # Hide y-axis label

            ax.set_xlim(0, 7)

        for ax in axes:
            # Set x tick labels for the top two and bottom two plots
            if ax in [ax_0_0, ax_0_1]:  # Top two plots
                ax.tick_params(axis='x', labeltop=True, labelbottom=False)  # Show x-tick labels at the top, hide at the bottom
            elif ax in [ax_4_0, ax_4_1]:  # Bottom two plots
                ax.tick_params(axis='x', labeltop=False, labelbottom=True)  # Show x-tick labels at the bottom, hide at the top
            else:  # Middle plots
                ax.tick_params(axis='x', labeltop=False, labelbottom=False)  # No x-tick labels (optional)
            # Set x tick lines inside for all plots
            ax.tick_params(axis='x', direction='in', which='both', top=True, bottom=True)  # Make tick lines face inside


        # Explicitly set x-axis labels for the first and last row
        ax_0_0.set_xlabel("Time")  # Top-left subplot
        ax_0_1.set_xlabel("Time")  # Top-right subplot
        ax_4_0.set_xlabel("Time")  # Bottom-left subplot
        ax_4_1.set_xlabel("Time")  # Bottom-right subplot

        # Move top x-axis labels to the top row
        ax_0_0.xaxis.set_label_position("top")
        ax_0_1.xaxis.set_label_position("top")


        # Set y-axis labels for the right column at end beacuse othewise twinx() will overwrite the labels
        #ax_0_1.set_ylabel("A11 geometric and \textcolor{blue}{mean} stretch rate\n$K_{geom}$ and \textcolor{blue}{$K_{mean}$}", color='black')
        ax_0_1.set_ylabel(r"A11 stretch rate $K$", color='black')
        ax_0_1.tick_params(axis='y', labelcolor='black')
        black_label = mlines.Line2D([], [], color='black', label=r"$K_{geom}$", linestyle='dashed')
        blue_label = mlines.Line2D([], [], color='blue', label=r"$K_{mean}$", linestyle='dashed')
        ax_0_1.legend(handles=[black_label, blue_label], loc='upper left', fontsize=10, frameon=False)

        ax_1_1.set_ylabel("A11 average total\nareodynamic strain $a_t$", color='black')
        ax_1_1.tick_params(axis='y', labelcolor='black')

        ax_2_1.set_ylabel("A11 average normal absolute\npropagation velocity $s_a$", color='black')
        ax_2_1.tick_params(axis='y', labelcolor='black')

        ax_3_1.set_ylabel("A11 average density weighed\ndisplacement speed $s_d$", color='black')
        ax_3_1.tick_params(axis='y', labelcolor='black')

        ax_4_1.set_ylabel("Dimentionalisation $d_T/px$", color='black')
        ax_4_1.tick_params(axis='y', labelcolor='black')


        # move scientific notation from A and iHRR 
        offset_text = ax_0_0.yaxis.get_offset_text()
        offset_text.set_position((-0.08, 0))  # Move offset text to the left side (x=-0.1)

        offset_text = ax_1_0.yaxis.get_offset_text()
        offset_text.set_position((-0.08, 0))  # Move offset text to the left side (x=-0.1)




        plt.subplots_adjust(hspace=0)  # This removes the vertical spacing

        # Adjust layout and save the figure as a PNG file
        #plt.tight_layout()
        plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust rect to prevent overlap

        plot_filename = os.path.join(output_dir, f'plot_panel.png')
        plt.savefig(plot_filename)
        plt.show() if show_plot == 1 else None
        plt.close(fig)


        print("\n") # new line


    ### return
    return output_dir # Format_1 requires output_dir as first return
