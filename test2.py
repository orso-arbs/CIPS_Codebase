import pandas as pd

# List of file paths
file_paths = [
    r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_K_mean_as_mean_stretch_rate_vs_time_manual_extraction.txt",
    r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_N_c_as_number_of_cells_vs_time_manual_extraction.txt",
    r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_R_mean_as_average_radius_of_the_wrinkled_flame_fron_vs_time_manual_extraction.txt",
    r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_R_mean_dot_as_first_time_derivative_of_the_average_radius_of_the_wrinkled_flame_front_vs_time_manual_extraction.txt",
    r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_s_a_as_average_normal_component_of_the_absolute_propagation_velocity_vs_time_manual_extraction.txt",
    r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_s_d_as_average_density_weighted_displacement_speed_vs_time_manual_extraction.txt",
    r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_A_as_flame_surface_area_of_the_wrinkled_spherical_front_vs_time_manual_extraction.txt",
    r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_a_t_as_average_total_aerodynamic_strain_vs_time_manual_extraction.txt",
    r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_iHRR_as_integral_heat_release_rate_vs_time_manual_extraction.txt",
    r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_SF_K_geom_as_geometric_stretch_rate_vs_time_manual_extraction.txt"
]

# Variable to track the maximum 'time' value
max_time = float('-inf')

# Load all files and find the maximum 'time'
for file_path in file_paths:
    # Extract file name without path and extension
    file_name = file_path.split('\\')[-1].replace('.txt', '')
    
    # Simplify the variable name by removing 'A11_SF_' and '_as' parts
    variable_name = file_name.split('_as')[0]
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Assuming the 'time' column exists in each CSV, find the max time in this file
    if 'time' in df.columns:
        max_time = max(max_time, df['time'].max())

print(f"The maximum 'time' across all files is: {max_time}")
