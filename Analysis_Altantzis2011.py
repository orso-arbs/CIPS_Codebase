import numpy as np
import pandas as pd
import os
import pickle
import Format_1 as F_1
import warnings

# Import the component modules
from dim3_A11 import dim3_A11
from Spherical_Reconstruction_2 import Spherical_Reconstruction_2
from CST_Selection_1 import CST_Selection_1
import plot4_dimentions as p4  # Import plot4_dimentions module

@F_1.ParameterLog(max_size=1024 * 10, log_level=0)
def Analysis_Altantzis2011(
    # input
    input_dir,  # Should be the output directory of CP_extract_1

    # output and logging
    Analysis_A11_log_level=2,
    output_dir_manual="", 
    output_dir_comment="",
    
    # Dimensionalization parameters
    d_T=7.516e-3,  # flame thickness [m] from A11
    S_L=51.44,     # laminar flame speed [m/s] from A11
    T_b=1843.5,    # burned gas temperature [K] from A11
    
    # Spherical Reconstruction parameters
    show_plots=False,
    plot_CST_detJ=False,
    
    # CST Selection parameters
    plot_CST_selection=True,
    Convert_to_grayscale_image=True,
    
    # Plotting parameters for plot4
    run_plotter_4=True,
    plot4_show_plot=False,
    plot4_Panel_1_A11=0,
    plot4_Panel_2_Dimentionalised_from_VisIt=1,
    plot4_A11_manual_data_base_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Data\A11_manual_extraction"
):
    """
    Comprehensive analysis pipeline for segmentation data based on the Altantzis2011 approach.
    
    This function orchestrates a series of analysis steps:
    1. Dimensionalization of pixel data (dim3_A11)
    2. Spherical reconstruction of cell properties (Spherical_Reconstruction_2)
    3. Selection of cells within the Cubed Sphere Tile (CST_Selection_1)
    4. Visualization of dimensionalization results (plot4_dimentions)
    
    Parameters
    ----------
    input_dir : str
        Path to the directory containing the extracted_DataFrame.pkl from CP_extract_1
    Analysis_A11_log_level : int, optional
        Controls the verbosity of logging. Default is 2.
    output_dir_manual : str, optional
        If provided, specifies the output directory. Default is "".
    output_dir_comment : str, optional
        Comment to append to the output directory name. Default is "".
    d_T : float, optional
        Flame thickness in meters. Default is 7.516e-3 from A11 paper.
    S_L : float, optional
        Laminar flame speed in m/s. Default is 51.44 from A11 paper.
    T_b : float, optional
        Burned gas temperature in K. Default is 1843.5 from A11 paper.
    show_plots : bool, optional
        Whether to display plots during processing. Default is False.
    plot_CST_detJ : bool, optional
        Whether to generate CST boundary with det(J) plot. Default is False.
    plot_CST_selection : bool, optional
        Whether to generate CST selection plots. Default is True.
    Convert_to_grayscale_image : bool, optional
        Whether to convert RGB images to grayscale for CST plots. Default is True.
    run_plotter_4 : bool, optional
        Whether to run the dimensionalization plotting function. Default is True.
    plot4_show_plot : bool, optional
        Whether to display plots during processing for plot4. Default is False.
    plot4_Panel_1_A11 : int, optional
        Whether to generate Panel 1 (A11 manual data) in plot4. Default is 0 (off).
    plot4_Panel_2_Dimentionalised_from_VisIt : int, optional
        Whether to generate Panel 2 (VisIt dimensionalization) in plot4. Default is 1 (on).
    plot4_A11_manual_data_base_dir : str, optional
        Path to directory containing manual A11 data. Default is predefined path.
        
    Returns
    -------
    output_dir : str
        Path to the output directory containing all results.
    """
    #################################################### I/O
    # Create output directory
    output_dir = F_1.F_out_dir(
        input_dir=input_dir, 
        script_path=__file__, 
        output_dir_comment=output_dir_comment, 
        output_dir_manual=output_dir_manual
    )
    
    #################################################### Step 1: Dimensionalization
    print("\n[1/3] Running dimensionalization (dim3_A11)...") if Analysis_A11_log_level >= 1 else None
    
    # Initialize reference values dictionary
    ref_values = {
        'd_T': d_T,    # flame thickness [m]
        'S_L': S_L,    # laminar flame speed [m/s]
        'T_b': T_b,    # burned gas temperature [K]
        't_ref': d_T/S_L  # flame time scale [s]
    }
    
    # Run dimensionalization - each function creates its own subdirectory
    dim3_output_dir = dim3_A11(
        input_dir=output_dir,  # Use main output directory as Format_1 input_dir
        output_dir_manual="",  # Let Format_1 create subdirectory
        output_dir_comment="dim3_A11",  # Add a descriptive comment
        Analysis_A11_df=None,  # First function, so passing None
        dim3_A11_log_level=Analysis_A11_log_level,
        ref_values=ref_values
    )
    
    # Load the dimensionalized DataFrame from dim3_A11's output directory
    dim3_df_path = os.path.join(dim3_output_dir, 'Analysis_A11_df.pkl')
    print(f"Looking for dimensionalized DataFrame at: {dim3_df_path}") if Analysis_A11_log_level >= 1 else None
    
    #################################################### Step 2: Spherical Reconstruction
    print("\n[2/3] Running spherical reconstruction (Spherical_Reconstruction_2)...") if Analysis_A11_log_level >= 1 else None
    
    # Run spherical reconstruction - create its own subdirectory
    sr2_output_dir = Spherical_Reconstruction_2(
        input_dir=output_dir,  # Use main output directory as Format_1 input_dir
        output_dir_manual="",  # Let Format_1 create subdirectory
        output_dir_comment="SR2",  # Add a descriptive comment
        Analysis_A11_df=pd.read_pickle(dim3_df_path),  # Pass DataFrame from previous step
        SR2_log_level=Analysis_A11_log_level,
        show_plots=show_plots,
        plot_CST_detJ=plot_CST_detJ
    )
    
    # Load the updated DataFrame from SR2's output directory
    sr2_df_path = os.path.join(sr2_output_dir, 'Analysis_A11_df.pkl')
    print(f"Looking for spherically reconstructed DataFrame at: {sr2_df_path}") if Analysis_A11_log_level >= 1 else None
    
    #################################################### Step 3: CST Selection
    print("\n[3/3] Running CST selection (CST_Selection_1)...") if Analysis_A11_log_level >= 1 else None
    
    # Run CST selection - create its own subdirectory
    cst_output_dir = CST_Selection_1(
        input_dir=output_dir,  # Use main output directory as Format_1 input_dir
        output_dir_manual="",  # Let Format_1 create subdirectory
        output_dir_comment="CST",  # Add a descriptive comment
        Analysis_A11_df=pd.read_pickle(sr2_df_path),  # Pass DataFrame from previous step
        CST_log_level=Analysis_A11_log_level,
        show_plots=show_plots,
        plot_CST_selection=plot_CST_selection,
        Convert_to_grayscale_image=Convert_to_grayscale_image
    )
    
    # Load the final DataFrame from CST's output directory
    final_df_path = os.path.join(cst_output_dir, 'Analysis_A11_df.pkl')
    final_df = pd.read_pickle(final_df_path)
    print(f"Loaded CST filtered DataFrame from: {final_df_path}") if Analysis_A11_log_level >= 1 else None
    
    #################################################### Step 4: Plot Dimensionalization Results
    if run_plotter_4:
        print("\n[4/4] Running dimensionalization plotting (plot4_dimentions)...") if Analysis_A11_log_level >= 1 else None
        
        p4_output_dir = p4.plotter_4_dimentionalisation(
            input_dir=dim3_output_dir,  # Use dim3_A11 output as input
            output_dir_manual="",  # Let Format_1 create subdirectory
            output_dir_comment="plot4_dims",  # Add a descriptive comment
            CP_data=pd.read_pickle(dim3_df_path),  # Pass the DataFrame directly
            show_plot=plot4_show_plot,
            Plot_log_level=Analysis_A11_log_level,
            Panel_1_A11=plot4_Panel_1_A11,
            Panel_2_Dimentionalised_from_VisIt=plot4_Panel_2_Dimentionalised_from_VisIt,
            A11_manual_data_base_dir=plot4_A11_manual_data_base_dir
        )
        print(f"Dimensionalization plots saved to: {p4_output_dir}") if Analysis_A11_log_level >= 1 else None
    else:
        print("\n[4/4] Skipping dimensionalization plotting (plot4_dimentions)") if Analysis_A11_log_level >= 1 else None
    
    #################################################### Save Final Results
    # Copy the final DataFrame to the main output directory as well
    final_main_pkl_path = os.path.join(output_dir, 'Analysis_A11_final_df.pkl')
    final_main_csv_path = os.path.join(output_dir, 'Analysis_A11_final_df.csv')
    
    final_df.to_pickle(final_main_pkl_path)
    final_df.to_csv(final_main_csv_path, sep='\t', index=False)
    
    print(f"\nAnalysis pipeline complete!") if Analysis_A11_log_level >= 1 else None
    print(f"Final results saved to:") if Analysis_A11_log_level >= 1 else None
    print(f"  - {final_main_pkl_path}") if Analysis_A11_log_level >= 1 else None
    print(f"  - {final_main_csv_path}") if Analysis_A11_log_level >= 1 else None
    
    return output_dir

# Example usage when script is run directly
if __name__ == "__main__":
    print("Running Altantzis2011 Analysis Pipeline...")
    
    # CP_extract_1 output_dir of hot 3000px States 79 and 100
    #input_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\20250604_1311111\20250604_1312276\20250604_1312276\20250604_1313140\20250615_1635229"
    
    # CP_extract_1 output_dir of WWBBWW 2000px all 136 States
    input_dir = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250610_0004544\20250610_0004569\20250612_2023463\20250615_1831369"

    output_dir = Analysis_Altantzis2011(
        input_dir=input_dir,
        Analysis_A11_log_level=2,
        output_dir_comment="test_A11_pipeline",
        show_plots=False,
        plot_CST_detJ=True,
        plot_CST_selection=True,
        run_plotter_4=True,
        plot4_show_plot=False,
        plot4_Panel_2_Dimentionalised_from_VisIt=1
    )
    
    print(f"Analysis pipeline completed. Results in: {output_dir}")
    
    # The final directory structure would be something like:
    # output_dir/ (main Analysis_Altantzis2011 output)
    # ├── Analysis_A11_final_df.pkl
    # ├── Analysis_A11_final_df.csv
    # ├── _log.json
    # ├── 20XXXXXX_XXXXXX_dim3_A11/ (dim3_A11 output)
    # │   ├── Analysis_A11_df.pkl
    # │   ├── Analysis_A11_df.csv
    # │   └── _log.json
    # ├── 20XXXXXX_XXXXXX_SR2/ (Spherical_Reconstruction_2 output)
    # │   ├── Analysis_A11_df.pkl
    # │   ├── Analysis_A11_df.csv
    # │   ├── image_1_CST_boundary.png
    # │   └── _log.json
    # └── 20XXXXXX_XXXXXX_CST/ (CST_Selection_1 output)
    #     ├── Analysis_A11_df.pkl
    #     ├── Analysis_A11_df.csv
    #     ├── CST_classification_plots/
    #     │   └── image_X_CST_classification.png
    #     └── _log.json
