import Format_1 as F_1

input_dir = ""
output_dir_comment = ""

output_dir = F_1.F_out_dir(input_dir, __file__, output_dir_comment = output_dir_comment,
    output_dir_manual=r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\SF_CP_analysis_pipeline_data\test") # Format_1 required definition of output directory

print(output_dir)