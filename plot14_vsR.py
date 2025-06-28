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

def plotter_14_vsR(
    input_dir,
    y_column='N',
    x_column='R_SF_Average_VisIt',  # Added x_column parameter with default value
    output_dir_manual="",
    output_dir_comment="",
    image_list=[],
    connect_with_lines=True,
    marker_style='o',
    marker_size=6,
    line_style='-',
    line_width=1.5,
    line_color='blue',
    marker_color='blue',
    x_label=None,  # Changed to None to use x_column as default
    y_label=None,
    x_label_fontsize=16,
    y_label_fontsize=16,
    tick_label_fontsize=12,
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
    x_label : str, optional
        Label for x-axis, by default None (will use x_column)
    y_label : str, optional
        Label for y-axis, by default None (will use y_column)
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
        print(f"plotter_14_vsR: Output directory: {output_dir}")

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
                markeredgecolor='black', label=y_column)
    else:
        plt.scatter(df[x_column], df[y_column], 
                  s=marker_size**2, marker=marker_style,
                  color=marker_color, edgecolors='black', 
                  label=y_column)
    
    # Set labels
    plt.xlabel(x_label if x_label else x_column, fontsize=x_label_fontsize)
    plt.ylabel(y_label if y_label else y_column, fontsize=y_label_fontsize)
    
    # Set tick parameters for inward facing ticks
    plt.tick_params(axis='both', direction='in', which='both', labelsize=tick_label_fontsize)
    
    # Add legend
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
    
    return output_dir

if __name__ == "__main__":
    # Example usage
    plotter_14_vsR(
        input_dir=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554\20250625_1626096\20250626_1700136\20250626_1706361",
        x_column="R_SF_nonDim",
        y_column="d_cell_mean_nonDim",
        output_dir_comment="example_plot",
        image_list=[],
        connect_with_lines=True,
        marker_style='o',
        marker_size=8,
        line_color='blue',
        marker_color='red',
        x_label=r'R_SF_nonDim',
        y_label=r'Mean Cell Diameter (px)',
        show_plot=1,
        Plot_log_level=1
    )
