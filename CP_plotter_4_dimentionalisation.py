import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
plt.rcParams['text.usetex'] = False  # Keep False unless you have a full LaTeX installation

import sys
import os
import Format_1 as F_1

import video_maker_1 as vm1



@F_1.ParameterLog(max_size = 1024 * 10) # 10KB 
def CP_plotter_4_dimentionalisation(input_dir, # Format_1 requires input_dir
    CP_extract_df = None, # if None a .pkl file has to be in the input_dir. otherwise no CP_extract data is provided.
    output_dir_manual = "", output_dir_comment = "",
    video = 1, show_plot = 1, Plot_log_level = 0,
    Panel_1_A11 = 1, A11_manual_data_base_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction",
    ):

    #################################################### I/O
    #
    output_dir = F_1.F_out_dir(input_dir, __file__, output_dir_comment = output_dir_comment) # Format_1 required definition of output directory

    #################################################### Load data
    pkl_files = glob.glob(os.path.join(input_dir, "*.pkl"))

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



    # Plotting Panel 1: dimentionalisation with A11 manually extracted data from the thesis 
    if Panel_1_A11 == 1:

        # Load A11 data from specified base directory
        A11_SF_K_mean = pd.read_csv(os.path.join(A11_manual_data_base_dir, "A11_SF_K_mean_as_mean_stretch_rate_vs_time_manual_extraction.txt"))
        A11_SF_N_c = pd.read_csv(os.path.join(A11_manual_data_base_dir, "A11_SF_N_c_as_number_of_cells_vs_time_manual_extraction.txt"))
        A11_SF_R_mean = pd.read_csv(os.path.join(A11_manual_data_base_dir, "A11_SF_R_mean_as_average_radius_of_the_wrinkled_flame_fron_vs_time_manual_extraction.txt"))
        A11_SF_R_mean_dot = pd.read_csv(os.path.join(A11_manual_data_base_dir, "A11_SF_R_mean_dot_as_first_time_derivative_of_the_average_radius_of_the_wrinkled_flame_front_vs_time_manual_extraction.txt"))
        A11_SF_s_a = pd.read_csv(os.path.join(A11_manual_data_base_dir, "A11_SF_s_a_as_average_normal_component_of_the_absolute_propagation_velocity_vs_time_manual_extraction.txt"))
        A11_SF_s_d = pd.read_csv(os.path.join(A11_manual_data_base_dir, "A11_SF_s_d_as_average_density_weighted_displacement_speed_vs_time_manual_extraction.txt"))
        A11_SF_A = pd.read_csv(os.path.join(A11_manual_data_base_dir, "A11_SF_A_as_flame_surface_area_of_the_wrinkled_spherical_front_vs_time_manual_extraction.txt"))
        A11_SF_a_t = pd.read_csv(os.path.join(A11_manual_data_base_dir, "A11_SF_a_t_as_average_total_aerodynamic_strain_vs_time_manual_extraction.txt"))
        A11_SF_iHRR = pd.read_csv(os.path.join(A11_manual_data_base_dir, "A11_SF_iHRR_as_integral_heat_release_rate_vs_time_manual_extraction.txt"))
        A11_SF_K_geom = pd.read_csv(os.path.join(A11_manual_data_base_dir, "A11_SF_K_geom_as_geometric_stretch_rate_vs_time_manual_extraction.txt"))

        #################################################### Plotting

        # auxillary function to plot the data

        # Number of rows in the DataFrame
        N_images = len(CP_extract_df)

        # Find the maximum frequency for all histograms
        max_diameter = max([diameter for sublist in CP_extract_df['diameter_distribution_nonDim'] for diameter in sublist])
    
        print("\nPlotting Panel 1: dimentionalisation with A11 manually extracted data from the thesis\n") if Plot_log_level >= 1 else None
        fig, (ax_1, ax_2, ax_3, ax_4) = plt.subplots(4, 1, figsize=(8, 10))  # 2 rows, 1 column
        
        # Subplot 1:
        ax_1.plot(CP_extract_df['time'], CP_extract_df['R_SF_nonDim'], label="R_SF_nonDim", color='olive', linestyle='solid')
        ax_1.set_xlabel('Time')
        ax_1.set_ylabel('Radius')
        ax_1.set_title('Spherical Flame Radius Comparison')
        ax_1.legend(loc='upper left')

        # Twin axes 1
        ax_1_twin = ax_1.twinx()  # Create a twin axes sharing the same x-axis
        ax_1_twin.plot(A11_SF_R_mean['time'], A11_SF_R_mean['R_mean'], label="A11 R_mean", color='green', linestyle='dashed')
        ax_1_twin.spines["right"].set_position(("outward", 0))  # Slightly to the right

        # Twin axes 2
        ax_1_twin2 = ax_1.twinx()  # Create a twin axes sharing the same x-axis
        ax_1_twin2.plot(CP_extract_df['time'], CP_extract_df['D_SF_px'], label="D_SF_px", color='blue', linestyle='dashed')
        ax_1_twin2.spines["right"].set_position(("outward", 40))  # Slightly to the right

        # Calculate R_mean_interpolated_i
        ax_1_twin3 = ax_1.twinx()  # Create a twin axes sharing the same x-axis
        CP_extract_df['R_mean_interpolated_i'] = (CP_extract_df['d_T_per_px'] * CP_extract_df['D_SF_px']) / 2
        ax_1_twin3.plot(CP_extract_df['time'], CP_extract_df['R_mean_interpolated_i'], label="R_mean_interpolated_i", color='black', linestyle='solid')
        ax_1_twin3.set_ylabel('R_mean_interpolated_i', color='black')
        ax_1_twin3.tick_params(axis='y', labelcolor='black')
        ax_1_twin3.spines["right"].set_position(("outward", 80))  # Slightly to the right

        # Second axis for d_T_per_px
        ax_2.plot(CP_extract_df['time'], CP_extract_df['d_T_per_px'], label="d_T_per_px", color='blue', linestyle='solid')
        ax_2.set_ylabel('d_T_per_px', color='blue')
        ax_2.tick_params(axis='y', labelcolor='blue')
        ax_2.legend(loc='upper left')


        # Third axis for diameter_mean_nonDim
        ax_3.plot(CP_extract_df['time'], CP_extract_df['diameter_mean_nonDim'], label="diameter_mean_nonDim", color='red', linestyle='solid')
        ax_3.set_ylabel('diameter_mean_nonDim', color='red')
        ax_3.tick_params(axis='y', labelcolor='red')
        ax_3.legend(loc='upper left')


        ax_3_twin = ax_3.twinx()  # Create a twin axes sharing the same x-axis
        ax_3_twin.plot(CP_extract_df['time'], CP_extract_df['diameter_mean_px'], label="diameter_mean_px", color='green', linestyle='dashed')
        ax_3_twin.spines["right"].set_position(("outward", 0))  # Slightly to the right

        # # Collect handles and labels from all axes
        # handles, labels = [], []
        # for ax in [ax_1, ax_1_twin, ax_1_twin2, ax_1_twin3, ax_2, ax_3, ax_3_twin]:
        #     h, l = ax.get_legend_handles_labels()
        #     handles.extend(h)
        #     labels.extend(l)

        # # Add a unified legend for the entire figure
        # ax_4.legend(handles, labels, loc='center', ncol=1)  # Place legend at the center of ax_4

        # Adjust layout and save the figure as a PNG file
        plt.legend()
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f'plot_panel.png')
        plt.savefig(plot_filename)
        plt.show() if show_plot == 1 else None
        plt.close(fig)

        print("\n") # new line

    
    return output_dir