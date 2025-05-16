#####################################################################################################
#####                                                                                           #####
#####                                     Formater functions                                    #####
#####                                                                                           #####
#####################################################################################################

"""
These functions are used to format the output directory and save the arguments of a function to a JSON file.

--->  Rigorous data tracking  <---

"""

#####################################################################################################


"""
Prints the start time and script name, and records the start time.

Parameters
----------
script_location : str, optional
    The path to the script file. Defaults to the location of this script (__file__).

Returns
-------
start_time : float
    The time the function was called, obtained from time.time().
current_date : str
    The current date and time formatted as "YYYY-MM-DD_HH-MM-SS".
"""

import os
import datetime
import time

def start_inform(script_location: str = __file__, print_inform: bool = True):
    start_time = time.time()
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if print_inform:
        print(f"{os.path.basename(script_location)}:", datetime.datetime.now())
    return start_time, current_date

"""
Prints the script name and the execution duration.

Parameters
----------
script_location : str, optional
    The path to the script file. Defaults to the location of this script (__file__).
start_time : float, optional
    The start time recorded by `start_inform`. If 0 or not provided,
    the execution duration is not calculated or printed. Defaults to 0.

Returns
-------
end_time : float
    The time the function was called, obtained from time.time().
elapsed_time : float, optional
    The time elapsed since `start_time`. Only returned if `start_time` is provided and non-zero.
"""

import os
import datetime
import time

def end_inform(script_location: str = __file__, start_time: float = 0, print_inform: bool = True):
    end_time = time.time()
    if start_time == 0:
        if print_inform:
            print(f"{os.path.basename(script_location)}: Completely Executed")
            print(f"NB: No Execution duration since no start_time was provided")
        return end_time, elapsed_time
    else:
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        if print_inform:
            print(f"{os.path.basename(script_location)}: Completely Executed in {int(minutes)} min {seconds:.2f} sec")
        return end_time, elapsed_time



'''
purpose:
    set the output directory as formated
    if output_dir_manual is not provided, the output directory will be created in the input directory
    the output directory will have the name of the script, the current date and the optional comment

Input:
    input_dir - string, input directory
    output_dir_manual - string, optional manual output directory
    output_dir_comment - string, optional comment for the output directory

Output:
    output_dir - string, output directory formated or manual
    directory will be created if it does not exist
'''

import os
import datetime

def F_out_dir(input_dir, script_path,
                current_date=None, 
                output_dir_manual=None, output_dir_comment=None,
                create_output_dir = 1,
                ):
    if output_dir_manual:
        Comment = f"_{output_dir_comment.replace(' ', '_')}" if output_dir_comment else ""
        output_dir = output_dir_manual + Comment
        os.makedirs(output_dir, exist_ok=True) if create_output_dir==1 else None
    else:
        current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = os.path.splitext(os.path.basename(script_path))[0]
        folder_name = f"{script_name}_{current_date}" + (f"_{output_dir_comment.replace(' ', '_')}" if output_dir_comment else "")
        output_dir = os.path.join(input_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True) if create_output_dir==1 else None
    return output_dir



def get_json_value(input_dir, json_data, *path_keys):
    """
    Retrieve a value from JSON-like structure by navigating through keys/paths.
    
    :param json_data: The JSON data (as a Python dictionary) to search in.
    :param path_keys: A variable number of keys that represent the path to the desired value.
    :return: The value found at the specified path or None if not found.
    """
    log_file = os.path.join(input_dir, "_log.json")

    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    with open(log_file, 'r') as file:
        data = json.load(file)

    for key in path_keys:
        if isinstance(json_data, dict) and key in json_data:
            json_data = json_data[key]
        elif isinstance(json_data, list) and isinstance(key, int) and 0 <= key < len(json_data):
            json_data = json_data[key]
        else:
            return None  # Return None if the key does not exist or path is incorrect
    return json_data



'''
debugger
'''
def debug_info(x):
    print("Type:", type(x))

    # Check for shape/size
    if hasattr(x, 'shape'):
        print("Shape:", x.shape)
    elif hasattr(x, '__len__') and not isinstance(x, (str, bytes)):
        try:
            # Try to get sizes of nested structures like lists of lists
            def get_shape(obj):
                if hasattr(obj, '__len__') and not isinstance(obj, (str, bytes)):
                    return [len(obj)] + get_shape(obj[0])
                else:
                    return []
            shape = get_shape(x)
            print("Shape:", tuple(shape))
        except Exception:
            print("Length:", len(x))
    else:
        print("No shape or length available")

    print("Value:", x)
    print()



import winsound
import time

def ding():
    winsound.PlaySound(r"C:\Windows\Media\Windows Ding.wav", winsound.SND_FILENAME)
















'''
purpose:
    Saves the given dictionary of arguments to a JSON file in the specified directory,
    including name, value, type, size, and length.

Input:
    output_dir - string, output directory
    args_dict - dictionary, arguments to save

Output:
    JSON file created in output directory with formatted output

'''

import json
import os

def save_args_to_json(output_dir, args_dict, input_dir, function_name, log_level):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the file paths
    json_file_path = os.path.join(output_dir, "_log.json")
    input_json_path = os.path.join(input_dir, "_log.json")  # Read input_dir from function argument
    
    # Check if a JSON file exists in the input directory
    if os.path.exists(input_json_path):
        try:
            with open(input_json_path, 'r') as input_file:
                existing_data = json.load(input_file)  # Load existing JSON content
        except json.JSONDecodeError:
            existing_data = {}  # Handle case where file is not valid JSON
    else:
        existing_data = {}
    
    
    # Insert new log at the same level with the function name as a key, ensuring new logs come first
    updated_data = {function_name: args_dict}  # New log entry
    updated_data.update(existing_data)  # Append existing data after new entry
    
    new_log_size = sys.getsizeof(json.dumps(updated_data))
    print(f"{function_name} Log: new log entry size {new_log_size/1024:.2f} KB") if log_level == 1 else None

    # Write the merged dictionary to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(updated_data, json_file, indent=4)

import functools
import inspect



'''
purpose:
    Extracts the execution time and comment from the folder name.

    Example: folder_name = "Operator1_2025-02-28_13-02-57_comment_text"
    Extracted operation = "Operator1"
    Extracted execution_time = "2025-02-28_13-02-57"
    Extracted comment = "comment_text"

Input:
    folder_name - string, name of the folder

Output:
    execution_time - string, formatted as example: "Operator1_2025-02-28_13-02-57_comment_text"
    comment - string, comment text
'''

import re

def extract_execution_details(folder_name):
    # Extract the operation (everything before the first '_')
    operation_match = re.match(r'([^_]+)', folder_name)
    if operation_match:
        operation = operation_match.group(1)
    else:
        operation = None
    
    # First, try to extract the execution time (YYYY-MM-DD_HH-MM-SS)
    match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', folder_name)
    if match:
        execution_time = match.group(1)
    else:
        execution_time = None
    
    # Now, extract the comment (everything after the execution time and optional date)
    comment_match = re.search(r'_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(.+)', folder_name)
    if comment_match:
        comment = comment_match.group(1)
    else:
        comment = None

    return operation, execution_time, comment



"""
purpose:
    Truncates the value if it exceeds max_size. Returns a truncated string representation.

Input:
    value - any, value to truncate
    max_size - int, maximum size to truncate

Output:
    truncated value - string, truncated value

"""

import numpy as np

# Define a threshold for large data (e.g., 1 MB)

MAX_SIZE = 10240  # 10 KB
#MAX_SIZE = 102400  # 100 KB
#MAX_SIZE = 1024 * 1024  # 1 MB
#MAX_SIZE = 1024 * 1024 * 10  # 10 MB
#MAX_SIZE = 1024 * 1024 * 100  # 100 MB

def is_dataframe_like(obj):
    """ Checks if the object behaves like a DataFrame (supports Pandas, Polars, Modin, Dask, etc.) """
    return hasattr(obj, 'shape') and hasattr(obj, 'columns') and hasattr(obj, 'head')

def truncate_value(value, var_name, function_name, log_level, max_size=MAX_SIZE):
    """ Truncates large values and handles complex data types properly. """
    print("running trunacted_value on", var_name, "type", type(value)) if log_level == 1 else None
    try:
        if is_dataframe_like(value):  # Handle DataFrame-like objects
            #print(f"value {value} is_dataframe_like")
            size = deep_sizeof(value)
            #print(f"var_name {var_name}")
            #print(f"size of value: {size}")
            #print(f"type(size.item()) {type(size.item())}")
            #print(f"type(value.shape) {type(value.shape)}")
            #print(f"type(list(value.columns)) {type(list(value.columns))}")
            if size > max_size:
                print(f"{function_name} Log warning: truncating value of {var_name} (size {size/1024:.2f} KB exceeds {max_size/1024:.2f} KB)") if log_level == 1 else None
                # Calculate the number of rows to keep
                num_rows = max_size // (value.memory_usage(deep=True).sum() // len(value))
                truncated_value = value.head(num_rows)
                return {
                    "type": "DataFrame",
                    "size": size.type(),
                    "shape": value.shape,
                    "columns": list(value.columns),
                    "sample_data": truncated_value.to_dict(orient='split')
                }
            else:
                #print("else: returning value.to_dict(orient='split')")
                #print(f"size of value.to_dict(orient='split'): {size}")
                #print(f"type(value.to_dict(orient='split')) {type(value.to_dict(orient='split'))}")

                return value.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x).to_dict(orient='split')

        # Handle NumPy types by converting them to native Python types
        if isinstance(value, np.generic):  # Handle any NumPy scalar types (int64, float64, etc.)
            value = value.item()  # Convert NumPy scalar to native Python type

        if isinstance(value, np.ndarray):  # Handle NumPy arrays
            size = value.nbytes
            if size > max_size:
                print(f"{function_name} Log warning: truncating value of {var_name} (size {size/1024:.2f} KB exceeds {max_size/1024:.2f} KB)") if log_level == 1 else None
                # Calculate the number of elements to keep
                num_elements = max_size // value.itemsize
                if value.ndim > 1:
                    num_elements = (num_elements // value.shape[-1]) * value.shape[-1]  # Ensure divisibility
                truncated_value = value.flat[:num_elements].reshape(-1, value.shape[-1] if value.ndim > 1 else 1)
                return {
                    "type": "NumPy Array",
                    "size": size,
                    "shape": value.shape,
                    "sample_data": truncated_value.tolist()
                }
            else:
                return value.tolist()

        elif isinstance(value, list):  # Handle lists
            size = sum(sys.getsizeof(item) for item in value)
            if size > max_size:
                print(f"{function_name} Log warning: truncating value of {var_name} (size {size/1024:.2f} KB exceeds {max_size/1024:.2f} KB)") if log_level == 1 else None
                return [truncate_value(item, var_name, function_name, log_level, max_size) for item in value[:10]]  # First 10 elements only
            else:
                return [truncate_value(item, var_name, function_name, log_level, max_size) for item in value]

        elif isinstance(value, dict):  # Handle dictionaries
            size = sys.getsizeof(value)
            if size > max_size:
                print(f"{function_name} Log warning: truncating value of {var_name} (size {size/1024:.2f} KB exceeds {max_size/1024:.2f} KB)") if log_level == 1 else None
                return {k: truncate_value(v, k, function_name, log_level, max_size) for k, v in list(value.items())[:10]}  # First 10 items only
            else:
                return {k: truncate_value(v, k, function_name, log_level, max_size) for k, v in value.items()}
        
        elif isinstance(value, (tuple, set)):  # Handle tuples, sets
            size = sys.getsizeof(value)
            if size > max_size:
                print(f"{function_name} Log warning: truncating value of {var_name} (size {size/1024:.2f} KB exceeds {max_size/1024:.2f} KB)") if log_level == 1 else None
                return [truncate_value(item, var_name, function_name, log_level, max_size) for item in list(value)[:10]]  # First 10 elements only
            else:
                return [truncate_value(item, var_name, function_name, log_level, max_size) for item in value]

        elif isinstance(value, str):  # Handle large strings
            return value[:max_size]

        elif isinstance(value, bytes):  # Handle large byte sequences
            return value[:max_size]

        else:  # Default case
            return str(value)[:max_size]

    except Exception as e:
        return f"{function_name} Log warning: Error processing {var_name}: {str(e)}\n"


import pandas as pd
def deep_sizeof(obj, seen=None):
    """Recursively compute the total memory size of a Python object, 
       including nested lists, dicts, NumPy arrays, and DataFrames."""
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0  # Prevent infinite recursion for circular references
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(deep_sizeof(k, seen) + deep_sizeof(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(deep_sizeof(i, seen) for i in obj)
    elif isinstance(obj, np.ndarray):
        size += obj.nbytes  # Directly get size of NumPy array data
    elif isinstance(obj, pd.DataFrame):
        size += obj.memory_usage(deep=True).sum()  # Use Pandas' built-in method
        for col in obj.columns:
            size += deep_sizeof(obj[col].tolist(), seen)  # Recursively check elements
    elif isinstance(obj, pd.Series):
        size += obj.memory_usage(deep=True)  # Use Pandas' method for Series
        size += deep_sizeof(obj.tolist(), seen)  # Recursively check elements

    return size


'''
purpose:
    Decorator to log the arguments of any function to a JSON file in the same output
    directory where the function saves its output.
    Captures name, value, type, size, and length of each argument.

Input:
    output_dir - string, output directory
    Assumes:
    - The function returns multiple values.
    - The 0'th value is the output directory.
    - Execution time must be extracted from the folder name.

Output:
    JSON file created in output directory with formatted output
'''

import functools
import sys
import inspect

def ParameterLog(max_size=MAX_SIZE, log_level = 0):  # Allow user to specify max_size
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args_details = []

            print("")

            # Get function signature
            signature = inspect.signature(func)
            param_names = list(signature.parameters.keys())
            function_name = func.__name__
            script_location = inspect.getfile(func)
            script_name = os.path.splitext(os.path.basename(script_location))[0]


            # Process arguments
            for i, arg in enumerate(args):
                var_name = param_names[i] if i < len(param_names) else f"arg{i+1}"
                truncated_value = truncate_value(arg, var_name, function_name, log_level, max_size)  # Pass max_size
                args_details.append({
                    "name": var_name,
                    "value": truncated_value,
                    "type": type(arg).__name__,
                    "size": sys.getsizeof(arg),
                    "length": len(arg) if hasattr(arg, '__len__') else None
                })

            for key, arg in kwargs.items():
                truncated_value = truncate_value(arg, key, function_name, log_level, max_size)  # Pass max_size
                args_details.append({
                    "name": key,
                    "value": truncated_value,
                    "type": type(arg).__name__,
                    "size": sys.getsizeof(arg),
                    "length": len(arg) if hasattr(arg, '__len__') else None
                })

            # Extract input_dir if present in arguments
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            input_dir = bound_args.arguments.get("input_dir", None)
            if input_dir is None:
                raise ValueError("ParameterLog error: 'input_dir' argument is required but not found.")



            # start inform
            start_time, current_date_log_run = start_inform(script_location)

            ### Call function
            result = func(*args, **kwargs)

            # end inform
            end_time, elapsed_time = end_inform(script_location, start_time)




            # Handle multiple return values
            if isinstance(result, (tuple, list)) and len(result) > 0:
                output_dir = result[0]  
                returned_values = result[:]  
            else:
                output_dir = result  
                returned_values = [result] if isinstance(result, str) else result  

            # Validate output directory
            if not isinstance(output_dir, str) or not os.path.isdir(output_dir):
                raise ValueError(f"{function_name} Log error: Must return a valid output directory path.")

            # Extract execution details
            operation, execution_time, comment = extract_execution_details(os.path.basename(output_dir))
            if execution_time is None:
                raise ValueError(f"{function_name} Log error: Execution time not found in output directory: {output_dir}")

            # Log return values with proper variable names
            return_details = []
            closure_vars = inspect.getclosurevars(func).nonlocals  # Get variable names
            for i, returned_value in enumerate(returned_values):
                var_name = closure_vars.get(i, f"returned_value{i+1}")  # Try to get original name
                truncated_value = truncate_value(returned_value, var_name, function_name, log_level, max_size)  # Pass max_size
                return_details.append({
                    "name": var_name,
                    "value": truncated_value,
                    "type": type(returned_value).__name__,
                    "size": sys.getsizeof(returned_value),
                    "length": len(returned_value) if hasattr(returned_value, '__len__') else None
                })

            # Create log dictionary
            args_dict = {
                "function_name": func.__name__,
                "execution_time": execution_time,
                "comment": comment,
                "script_location": script_location,
                "script_name": script_name,
                "output_dir": output_dir,
                "elapsed_time_in_seconds": elapsed_time,
                "arguments": args_details,
                "returned_values": return_details
            }

            # Save log
            save_args_to_json(output_dir, args_dict, input_dir, function_name, log_level)

            print("")


            return result  

        return wrapper
    return decorator