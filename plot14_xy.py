import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import Format_1 as F_1

# LaTeX settings
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

def plotter_14_xy(
    input_dir,
    y_column='N_cells_CSTx6',  # Default y_column
    x_column='R_SF_nonDim',  # Added x_column parameter with default value
    output_dir_manual="",
    output_dir_comment="",
    image_list=[],
    omit_image_list=[106],  # New parameter to omit specific images
    connect_with_lines=True,
    marker_style='',
    marker_size=6,
    marker_color='blue',
    line_style='-',
    line_width=1.5,
    line_color='blue',
    show_grid=True,     # New parameter to control grid display
    grid_style='--',     # New parameter for grid line style
    grid_width=0.5,      # New parameter for grid line width
    grid_color='gray',   # New parameter for grid color
    grid_alpha=0.5,      # New parameter for grid transparency
    x_label=None,  # Changed to None to use x_column as default
    y_label=None,
    legend_label=None,  # New parameter for custom legend label
    x_label_fontsize=20,
    y_label_fontsize=20,
    tick_label_fontsize=20,
    legend_fontsize=12,
    legend_loc='upper left',
    figsize=(10, 6),
    dpi=100,
    show_plot=0,
    Plot_log_level=1
):
    """
    Creates an x-y plot with configurable x and y axis columns.
    
    Parameters
    ----------
    input_dir : str
        Directory containing the Analysis_A11_final_df.pkl file
    y_column : str
        Column name to plot on y-axis
    x_column : str, optional
        Column name to plot on x-axis, by default 'R_SF_Average_VisIt'
    output_dir_manual : str, optional
        Manual output directory, by default ""
    output_dir_comment : str, optional
        Comment to append to the output directory name, by default ""
    image_list : list, optional
        List of image numbers to include in plot, if empty all images are used, by default []
    omit_image_list : list, optional
        List of image numbers to exclude from plot (applied after image_list filter), by default []
    connect_with_lines : bool, optional
        Whether to connect points with lines, by default True
    marker_style : str, optional
        Style of markers, by default 'o'
    marker_size : int, optional
        Size of markers, by default 6
    line_style : str, optional
        Style of lines, by default '-'
    line_width : float, optional
        Width of lines, by default 1.5
    line_color : str, optional
        Color of lines, by default 'blue'
    marker_color : str, optional
        Color of markers, by default 'blue'
    show_grid : bool, optional
        Whether to show grid lines, by default False
    grid_style : str, optional
        Style of grid lines, by default '--'
    grid_width : float, optional
        Width of grid lines, by default 0.5
    grid_color : str, optional
        Color of grid lines, by default 'gray'
    grid_alpha : float, optional
        Transparency of grid lines (0-1), by default 0.5
    x_label : str, optional
        Label for x-axis, by default None (will use x_column)
    y_label : str, optional
        Label for y-axis, by default None (will use y_column)
    legend_label : str, optional
        Custom text for the legend label. If None, no legend is shown.
    x_label_fontsize : int, optional
        Font size for x-axis label, by default 16
    y_label_fontsize : int, optional
        Font size for y-axis label, by default 16
    tick_label_fontsize : int, optional
        Font size for tick labels, by default 12
    legend_fontsize : int, optional
        Font size for legend, by default 12
    legend_loc : str, optional
        Location of legend, by default 'upper left'
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (10, 6)
    dpi : int, optional
        DPI for the figure, by default 100
    show_plot : int, optional
        Whether to display the plot (1) or not (0), by default 0
    Plot_log_level : int, optional
        Logging level, by default 1
    
    Returns
    -------
    str
        Path to the output directory
    """
    # Create output directory
    output_dir = F_1.F_out_dir(input_dir=input_dir, script_path=__file__, 
                             output_dir_comment=output_dir_comment, 
                             output_dir_manual=output_dir_manual)
    
    if Plot_log_level >= 1:
        print(f"plotter_14_xy: Output directory: {output_dir}")

    # Create directories for PNG and SVG outputs
    png_dir = os.path.join(output_dir, "png")
    svg_dir = os.path.join(output_dir, "svg")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    # Find the PKL file
    pandas_wildcard_str = os.path.join(input_dir, "Analysis_A11_final_df.pkl")
    pkl_files = glob.glob(pandas_wildcard_str)
    
    if not pkl_files:
        print(f"No Analysis_A11_final_df.pkl file found in {input_dir}")
        return output_dir
    
    # Load the DataFrame
    df_path = pkl_files[0]
    df = pd.read_pickle(df_path)
    
    if Plot_log_level >= 1:
        print(f"Loaded DataFrame from {df_path}")
    
    # Check if the required columns exist
    if x_column not in df.columns:
        print(f"Column '{x_column}' not found in DataFrame. Available columns: {df.columns.tolist()}")
        return output_dir
    
    if y_column not in df.columns:
        print(f"Column '{y_column}' not found in DataFrame. Available columns: {df.columns.tolist()}")
        return output_dir
    
    # Filter DataFrame based on image_list if provided
    if image_list:
        df = df[df['image_number'].isin(image_list)]
        if df.empty:
            print(f"No matching images found for the provided image_list: {image_list}")
            return output_dir
    
    # Exclude images in omit_image_list if provided
    if omit_image_list:
        df = df[~df['image_number'].isin(omit_image_list)]
        if df.empty:
            print(f"No images remaining after applying omit_image_list: {omit_image_list}")
            return output_dir
        if Plot_log_level >= 1:
            print(f"Excluded {len(omit_image_list)} images: {omit_image_list}")
    
    # Create figure
    plt.figure(figsize=figsize, dpi=dpi)
    
    # Set font size for all elements
    plt.rcParams['font.size'] = tick_label_fontsize
    
    # Create plot
    if connect_with_lines:
        plt.plot(df[x_column], df[y_column], 
                marker=marker_style, markersize=marker_size, 
                linestyle=line_style, linewidth=line_width,
                color=line_color, markerfacecolor=marker_color, 
                markeredgecolor='black', label=legend_label)
    else:
        plt.scatter(df[x_column], df[y_column], 
                  s=marker_size**2, marker=marker_style,
                  color=marker_color, edgecolors='black', 
                  label=legend_label)
    
    # Set labels
    plt.xlabel(x_label if x_label else x_column, fontsize=x_label_fontsize)
    plt.ylabel(y_label if y_label else y_column, fontsize=y_label_fontsize)
    
    # Set tick parameters for inward facing ticks
    plt.tick_params(axis='both', direction='in', which='both', labelsize=tick_label_fontsize)
    
    # Add grid if requested
    if show_grid:
        plt.grid(True, linestyle=grid_style, linewidth=grid_width, 
                alpha=grid_alpha, color=grid_color)
    
    # Add legend only if a legend_label is provided
    if legend_label is not None:
        plt.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figures - include both column names in filename
    base_filename = f"{x_column.replace(' ', '_')}_vs_{y_column.replace(' ', '_')}"
    png_path = os.path.join(png_dir, f"{base_filename}.png")
    svg_path = os.path.join(svg_dir, f"{base_filename}.svg")
    
    plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    
    if Plot_log_level >= 1:
        print(f"Saved figures to:\n  {png_path}\n  {svg_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # i = 100
    # print(
    #     f"Values for row {i}:\n"
    #     f"A_cell_SRec_mean_CST_nonDim2 {df['A_cell_SRec_mean_CST_nonDim2'].iloc[i]}\n"
    #     f"A_cell_SRec_mean_CST_px2 {df['A_cell_SRec_mean_CST_px2'].iloc[i]}\n"
    #     f"d_cell_SRec_mean_CST_nonDim {df['d_cell_SRec_mean_CST_nonDim'].iloc[i]}\n"
    #     f"d_cell_SRec_mean_CST_px {df['d_cell_SRec_mean_CST_px'].iloc[i]}\n"
    #     f"N_cells_CST {df['N_cells_CST'].iloc[i]}\n"
    #     f"\n"
    #     f"A_cell_SRec_mean_CSTx6_nonDim2 {df['A_cell_SRec_mean_CSTx6_nonDim2'].iloc[i]}\n"
    #     f"A_cell_SRec_mean_CSTx6_px2 {df['A_cell_SRec_mean_CSTx6_px2'].iloc[i]}\n"
    #     f"d_cell_SRec_mean_CSTx6_nonDim {df['d_cell_SRec_mean_CSTx6_nonDim'].iloc[i]}\n"
    #     f"d_cell_SRec_mean_CSTx6_px {df['d_cell_SRec_mean_CSTx6_px'].iloc[i]}\n"
    #     f"N_cells_CSTx6 {df['N_cells_CSTx6'].iloc[i]}\n"
    #     f"\n"
    #     f"length(d_cell_SRec_distribution_CST_nonDim) at row i {i}: {len(df['d_cell_SRec_distribution_CST_nonDim'].iloc[i])}\n"
    #     f"length(d_cell_SRec_distribution_CSTx6_nonDim) at row i {i}: {len(df['d_cell_SRec_distribution_CSTx6_nonDim'].iloc[i])}\n" 
    #     f"mean(d_cell_SRec_distribution_CST_nonDim) at row i {i}: {np.mean(df['d_cell_SRec_distribution_CST_nonDim'].iloc[i])}\n"
    #     f"mean(d_cell_SRec_distribution_CSTx6_nonDim) at row i {i}: {np.mean(df['d_cell_SRec_distribution_CSTx6_nonDim'].iloc[i])}\n"
    # )
    return output_dir



if __name__ == "__main__":
    #    Example usage
    plotter_14_xy(
        input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187",
        x_column="R_SF_nonDim",
        y_column="d_cell_SRec_mean_CSTx6_nonDim",
        output_dir_comment="d_cell_mean_nonDim vs R_SF_nonDim",
        image_list=[],
        line_color='green',
        x_label=r'$R_{SF}/\delta_T$',
        y_label=r'$\overline{d}_c/\delta_T$',
    )

    plotter_14_xy(
        input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187",
        x_column="R_SF_nonDim",
        y_column="A_cell_SRec_mean_CSTx6_nonDim2",
        output_dir_comment="A_cell_SRec_mean_CSTx6_nonDim2 vs R_SF_nonDim",
        image_list=[],
        line_color='darkgreen',
        x_label=r'$R_{SF}/\delta_T$',
        y_label=r'$\overline{A}_c/\delta_T$',
    )

    plotter_14_xy(
        input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187",
        x_column="R_SF_nonDim",
        y_column="N_cells_CSTx6",
        output_dir_comment="N_cells_CSTx6 vs R_SF_nonDim",
        image_list=[],
        line_color='red',
        x_label=r'$R_{SF}/\delta_T$',
        y_label=r'$N_{cells}$',
    )


    plotter_14_xy(
        input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187",
        x_column="R_SF_nonDim",
        y_column="contour_length_total_CSTx6_nonDim",
        output_dir_comment="contour_length_total_CSTx6_nonDim vs R_SF_nonDim",
        image_list=[],
        line_color='blue',
        x_label=r'$R_{SF}/\delta_T$',
        y_label=r'$L$/$\delta_T$',
    )

    plotter_14_xy(
        input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250628_2007187",
        x_column="Time_VisIt",
        y_column="R_SF_nonDim",
        output_dir_comment="R_SF_nonDim vs Time_VisIt",
        image_list=[],
        line_color='orange',
        x_label=r'$\tau$',
        y_label=r'$R_{SF}/\delta_T$',
    )

