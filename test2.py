import Format_1 as F_1
import time

path = r"C:\Users\obs\OneDrive\misc\Literature"

a = 1

@F_1.ParameterLog(max_size = 1024 * 10) # 10KB 
def test(input_dir):
    output_dir = F_1.F_out_dir(input_dir = input_dir, script_path = __file__) # Format_1 required definition of output directory

    # To wait for 1 second:
    time.sleep(1)

    print(a)

    return output_dir

test(path)