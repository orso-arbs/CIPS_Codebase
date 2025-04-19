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

# --- Standard Library Imports ---
import os
import datetime
import time
import json
import sys
import re
import logging
import functools
import inspect

# --- Third-Party Imports ---
import numpy as np
import pandas as pd

#####################################################################################################
#                                        Timing Functions                                           #
#####################################################################################################

'''
purpose:
  Print the current time and name of script
  save the start time

Input:
  script_location - string, name of script to be printed
                  - default is the Misc_funcition script name

Output:
  Start time      - float, time.time() object
  Current date    - string, current date
'''
def start_inform(script_location=__file__):  # Pass filename as an argument
    start_time = time.time()
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"{os.path.basename(script_location)}:", datetime.datetime.now())
    return start_time, current_date

'''
purpose:
  Print the name of the script
  Print the time taken to execute the script

Input:
  script_location - string, script name
                  - default is the Misc_funcition script name

Output:
  End time        - float, time.time() object
  Elapsed time    - float, time taken to execute the script
'''
def end_inform(script_location=__file__, start_time=0):  # Pass filename as an argument
    end_time = time.time()
    if start_time == 0:
        print(f"{os.path.basename(script_location)}: Completely Executed")
        print(f"NB: No Execution duration since no start_time was provided")
        return end_time
    else:
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        print(f"{os.path.basename(script_location)}: Completely Executed in {int(minutes)} min {seconds:.2f} sec")
        return end_time, elapsed_time

#####################################################################################################
#                                     Directory Formatting                                          #
#####################################################################################################

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
def F_out_dir(input_dir, script_path,
                current_date=None,
                output_dir_manual=None, output_dir_comment=None,
                create_output_dir = 1,
                ):
    if output_dir_manual:
        output_dir = output_dir_manual
    else:
        # Use provided current_date if available, otherwise generate new one
        date_str = current_date if current_date else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = os.path.splitext(os.path.basename(script_path))[0]
        folder_name = f"{script_name}_{date_str}" + (f"_{output_dir_comment.replace(' ', '_')}" if output_dir_comment else "")
        output_dir = os.path.join(input_dir, folder_name)

    if create_output_dir == 1:
         os.makedirs(output_dir, exist_ok=True)

    return output_dir

#####################################################################################################
#                                     Utility Functions                                             #
#####################################################################################################

def get_json_value(input_dir, json_data, *path_keys):
    """
    Retrieve a value from JSON-like structure by navigating through keys/paths.

    :param json_data: The JSON data (as a Python dictionary) to search in.
    :param path_keys: A variable number of keys that represent the path to the desired value.
    :return: The value found at the specified path or None if not found.
    """
    # This function seems unrelated to the core logging task, keeping it as is.
    # However, it reads _log.json itself, which might conflict or be redundant
    # with the new logging approach if used elsewhere. Review usage if needed.
    log_file = os.path.join(input_dir, "_log.json")

    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")

    with open(log_file, 'r') as file:
        data = json.load(file) # Assumes data is the root to search

    current_data = data # Start search from root
    for key in path_keys:
        if isinstance(current_data, dict) and key in current_data:
            current_data = current_data[key]
        elif isinstance(current_data, list) and isinstance(key, int) and 0 <= key < len(current_data):
            current_data = current_data[key]
        else:
            return None  # Return None if the key does not exist or path is incorrect
    return current_data


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


# Define a threshold for large data (e.g., 1 MB)
MAX_SIZE = 10240  # 10 KB
#MAX_SIZE = 102400  # 100 KB
#MAX_SIZE = 1024 * 1024  # 1 MB
#MAX_SIZE = 1024 * 1024 * 10  # 10 MB
#MAX_SIZE = 1024 * 1024 * 100  # 100 MB

def is_dataframe_like(obj):
    """ Checks if the object behaves like a DataFrame (supports Pandas, Polars, Modin, Dask, etc.) """
    return hasattr(obj, 'shape') and hasattr(obj, 'columns') and hasattr(obj, 'head')

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
        # Use nbytes for the data buffer size, sys.getsizeof for the object overhead
        size = sys.getsizeof(obj) + obj.nbytes
    elif isinstance(obj, pd.DataFrame):
        # Use memory_usage(deep=True) for a more accurate DataFrame size
        try:
            size = obj.memory_usage(deep=True).sum()
        except Exception: # Fallback if memory_usage fails
             size = sys.getsizeof(obj) # Basic object size
    elif isinstance(obj, pd.Series):
         try:
            size = obj.memory_usage(deep=True)
         except Exception:
             size = sys.getsizeof(obj)

    # Add other types as needed

    return size


def truncate_value(value, var_name, function_name, log_level, max_size=MAX_SIZE):
    """
    Truncates large values and handles complex data types properly for logging.
    Returns a representation suitable for JSON serialization.
    """
    # Keep the print for debugging if log_level is high enough
    # print(f"Truncating {var_name} (type: {type(value).__name__}) in {function_name}") if log_level >= 1 else None

    try:
        # Estimate size first
        current_size = deep_sizeof(value)
        truncated = current_size > max_size

        # --- Handle specific types ---
        if is_dataframe_like(value):
            if truncated:
                print(f"{function_name} Log warning: truncating DataFrame '{var_name}' (size {current_size/1024:.2f} KB > {max_size/1024:.2f} KB)") if log_level >= 0 else None
                # Calculate rows to keep (ensure positive divisor)
                if len(value) > 0:
                    avg_row_size = current_size / len(value)
                    num_rows = max(1, int(max_size // avg_row_size)) if avg_row_size > 0 else len(value)
                else:
                    num_rows = 0
                sample_df = value.head(num_rows)
                sample_data = sample_df.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x).to_dict(orient='split')
                return {
                    "_log_type": "DataFrame",
                    "truncated": True,
                    "original_size_bytes": current_size,
                    "original_shape": value.shape,
                    "original_columns": list(value.columns),
                    "sample_data": sample_data
                }
            else:
                # Convert full DataFrame if small enough
                return value.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x).to_dict(orient='split')

        elif isinstance(value, np.generic):
            return value.item() # Convert NumPy scalar to native Python type

        elif isinstance(value, np.ndarray):
            if truncated:
                print(f"{function_name} Log warning: truncating ndarray '{var_name}' (size {current_size/1024:.2f} KB > {max_size/1024:.2f} KB)") if log_level >= 0 else None
                # Simple truncation: take first N elements
                num_elements = max_size // value.itemsize if value.itemsize > 0 else value.size
                sample_array = value.flat[:num_elements]
                return {
                    "_log_type": "ndarray",
                    "truncated": True,
                    "original_size_bytes": current_size,
                    "original_shape": value.shape,
                    "original_dtype": str(value.dtype),
                    "sample_data": sample_array.tolist() # Store flat sample as list
                }
            else:
                return value.tolist() # Convert full array to list

        elif isinstance(value, (list, tuple, set)):
            is_set = isinstance(value, set)
            value_list = list(value) if is_set else value # Work with list representation
            if truncated:
                print(f"{function_name} Log warning: truncating {type(value).__name__} '{var_name}' (size {current_size/1024:.2f} KB > {max_size/1024:.2f} KB)") if log_level >= 0 else None
                # Simple truncation: take first N items
                sample_list = value_list[:10] # Arbitrary sample size
                return {
                    "_log_type": type(value).__name__,
                    "truncated": True,
                    "original_size_bytes": current_size,
                    "original_length": len(value_list),
                    "sample_data": [truncate_value(item, f"{var_name}[{i}]", function_name, log_level, max_size) for i, item in enumerate(sample_list)] # Truncate items in sample
                }
            else:
                # Process all items recursively if within size limit
                processed_list = [truncate_value(item, f"{var_name}[{i}]", function_name, log_level, max_size) for i, item in enumerate(value_list)]
                return tuple(processed_list) if isinstance(value, tuple) else processed_list # Keep original type if tuple

        elif isinstance(value, dict):
            if truncated:
                print(f"{function_name} Log warning: truncating dict '{var_name}' (size {current_size/1024:.2f} KB > {max_size/1024:.2f} KB)") if log_level >= 0 else None
                # Simple truncation: take first N items
                sample_items = list(value.items())[:10] # Arbitrary sample size
                return {
                    "_log_type": "dict",
                    "truncated": True,
                    "original_size_bytes": current_size,
                    "original_length": len(value),
                    "sample_data": {k: truncate_value(v, f"{var_name}['{k}']", function_name, log_level, max_size) for k, v in sample_items} # Truncate values in sample
                }
            else:
                 # Process all items recursively
                return {k: truncate_value(v, f"{var_name}['{k}']", function_name, log_level, max_size) for k, v in value.items()}

        elif isinstance(value, str):
            if truncated:
                 print(f"{function_name} Log warning: truncating str '{var_name}' (size {current_size/1024:.2f} KB > {max_size/1024:.2f} KB)") if log_level >= 0 else None
                 return value[:max_size] + "... (truncated)"
            else:
                 return value

        elif isinstance(value, bytes):
             if truncated:
                 print(f"{function_name} Log warning: truncating bytes '{var_name}' (size {current_size/1024:.2f} KB > {max_size/1024:.2f} KB)") if log_level >= 0 else None
                 # Represent truncated bytes safely
                 return f"bytes(truncated, original_len={len(value)}, sample_hex={value[:min(max_size, 50)].hex()}...)"
             else:
                 # Represent small bytes as hex string for JSON compatibility
                 return f"bytes(len={len(value)}, data_hex={value.hex()})"

        # --- Default Case ---
        else:
            # Attempt string conversion, truncate if needed
            try:
                str_repr = str(value)
                # Estimate size of string representation
                str_size = sys.getsizeof(str_repr)
                if str_size > max_size:
                     print(f"{function_name} Log warning: truncating string representation of {type(value).__name__} '{var_name}' (size {str_size/1024:.2f} KB > {max_size/1024:.2f} KB)") if log_level >= 0 else None
                     return str_repr[:max_size] + "... (truncated string representation)"
                else:
                     return str_repr # Return string representation if small enough
            except Exception as e_str:
                 # Fallback if str() fails
                 logging.warning(f"Could not convert type {type(value).__name__} to string for logging: {e_str}", exc_info=True)
                 return f"<{type(value).__name__} object (unloggable)>"

    except Exception as e:
        # Log the exception during truncation itself
        logging.error(f"Error during truncate_value for {var_name} (type: {type(value).__name__}): {e}", exc_info=True)
        return f"Error processing value: {str(e)}"


#####################################################################################################
#                                     New Logging Components                                        #
#####################################################################################################

class JsonChainHandler(logging.Handler):
    """
    A logging handler that reads an existing JSON log from an input directory,
    prepends the new log record (formatted as {function_name: args_list}),
    and writes the updated JSON to an output directory.
    """
    def __init__(self, input_dir, output_dir, function_name):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.function_name = function_name # Function name is used as the key in JSON
        self.output_log_path = os.path.join(self.output_dir, "_log.json") if self.output_dir else None
        self.input_log_path = os.path.join(self.input_dir, "_log.json") if self.input_dir else None

    def emit(self, record):
        # Check if we have a valid output path
        if not self.output_log_path:
            logging.error(f"JsonChainHandler: Output directory not set for function {self.function_name}. Cannot write log.")
            return # Cannot proceed without output path

        try:
            # The formatter should have placed the processed list of args dicts into record.msg
            processed_args_list = record.msg

            existing_data = {}
            # Only read input if input_log_path is valid
            if self.input_log_path and os.path.exists(self.input_log_path):
                try:
                    with open(self.input_log_path, 'r', encoding='utf-8') as infile: # Specify encoding
                        existing_data = json.load(infile)
                        # Ensure existing_data is a dictionary
                        if not isinstance(existing_data, dict):
                             logging.warning(f"Existing log data in {self.input_log_path} was not a dictionary. Overwriting.")
                             existing_data = {}
                except json.JSONDecodeError:
                    logging.warning(f"Could not decode JSON from {self.input_log_path}. Starting fresh log.", exc_info=True)
                    existing_data = {}
                except FileNotFoundError:
                     existing_data = {} # Start fresh if file disappears
                except Exception:
                    logging.error(f"Error reading input log file {self.input_log_path}", exc_info=True)
                    existing_data = {} # Start fresh on other errors
            elif self.input_dir is None:
                 # Don't warn if input_dir was intentionally None (first step)
                 pass
            elif self.input_log_path: # Warn if input_dir was given but file doesn't exist
                 # This might be the first step, so not necessarily an error
                 # logging.info(f"Input log file {self.input_log_path} not found. Starting new log chain.")
                 pass


            # Prepend new log entry using the function name from the record attribute
            new_log_entry = {self.function_name: processed_args_list}
            updated_data = new_log_entry
            updated_data.update(existing_data) # Add old data after the new entry

            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)

            # Write the updated data
            try:
                with open(self.output_log_path, 'w', encoding='utf-8') as outfile: # Specify encoding
                    json.dump(updated_data, outfile, indent=4, ensure_ascii=False) # ensure_ascii=False for wider char support
            except Exception:
                 logging.error(f"Error writing output log file {self.output_log_path}", exc_info=True)

        except Exception:
            self.handleError(record) # Use standard error handling


class ArgTruncatingFormatter(logging.Formatter):
    """
    A logging formatter that takes a list of raw argument dictionaries from record.msg,
    truncates their values using truncate_value, and updates record.msg
    with the processed list of dictionaries.
    """
    def __init__(self, max_size=MAX_SIZE, log_level=0, fmt=None, datefmt=None, style='%', validate=True):
        # Pass style and validate for compatibility with standard Formatter
        # Default format string includes standard info but avoids '%(message)s'
        # as the message is modified in place and used directly by the handler.
        fmt = fmt or '%(asctime)s - %(name)s - %(levelname)s - Logged Args'
        super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate)
        self.max_size = max_size
        self.log_level = log_level

    def format(self, record):
        # record.msg = list of raw {'name': ..., 'value': ...} dicts
        # record.funcName = name of the function being logged
        args_list_raw = record.msg
        function_name = record.funcName

        processed_args_list = []
        if isinstance(args_list_raw, list):
            for arg_detail in args_list_raw:
                if isinstance(arg_detail, dict) and "name" in arg_detail and "value" in arg_detail:
                    var_name = arg_detail["name"]
                    raw_value = arg_detail["value"]
                    # Apply truncation
                    truncated_value = truncate_value(raw_value, var_name, function_name, self.log_level, self.max_size)
                    processed_args_list.append({
                        "name": var_name,
                        "value": truncated_value, # The (potentially) truncated value representation
                        "type": type(raw_value).__name__,
                        "original_size_bytes": deep_sizeof(raw_value), # Estimate original size
                        "original_length": len(raw_value) if hasattr(raw_value, '__len__') else None
                    })
                else:
                    logging.warning(f"ArgTruncatingFormatter encountered invalid item format in args list: {arg_detail}")
                    processed_args_list.append({"name": "invalid_arg_structure", "value": str(arg_detail)})
        else:
             logging.warning(f"ArgTruncatingFormatter received unexpected message type: {type(args_list_raw)}. Expected list.")
             processed_args_list = [{"name": "error_unexpected_log_message_type", "value": str(args_list_raw)}]

        # --- IMPORTANT ---
        # Update record.msg in place so the handler gets the processed list
        record.msg = processed_args_list

        # Return the standard formatted string (using the standard formatter logic)
        # This populates record.message based on the format string
        return super().format(record)


def StandardLog(max_size=MAX_SIZE, log_level=0):
    """
    Decorator to log function arguments using Python's logging framework,
    maintaining the chained JSON log structure. Logs a list of argument
    dictionaries under the function's name in the JSON log.

    Args:
        max_size (int): Maximum size in bytes for argument values before truncation.
        log_level (int): Log level to control internal printing (e.g., in truncate_value).
                         Note: This is separate from standard logging levels (INFO, DEBUG).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signature = inspect.signature(func)
            function_name = func.__name__
            try:
                script_location = inspect.getfile(func)
                module_name = func.__module__
            except TypeError:
                script_location = "unknown"
                module_name = "unknown"


            # Prepare list of raw arguments details {'name': ..., 'value': ...}
            raw_args_details = []
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults() # Include default values

            for name, value in bound_args.arguments.items():
                 raw_args_details.append({"name": name, "value": value})

            # Extract input_dir (must be present)
            input_dir = bound_args.arguments.get("input_dir", None)

            # --- Execute the original function ---
            start_time, current_date_for_dir = start_inform(script_location) # Get date for potential use in F_out_dir
            result = func(*args, **kwargs)
            end_inform(script_location, start_time)
            # --- Function execution finished ---

            # Extract output_dir from result
            output_dir = None
            original_result = result # Store original result to return later
            if isinstance(result, (tuple, list)) and len(result) > 0:
                if isinstance(result[0], str):
                     output_dir = result[0]
            elif isinstance(result, str):
                 output_dir = result

            # --- Validate directories ---
            # Input dir can be None for the first step
            valid_input_dir = input_dir is None or (isinstance(input_dir, str) and os.path.isdir(input_dir))
            # Output dir MUST be valid after the function call
            valid_output_dir = output_dir and isinstance(output_dir, str) and os.path.isdir(output_dir)

            if not valid_input_dir and input_dir is not None:
                 logging.warning(f"StandardLog in {function_name}: Provided 'input_dir' ('{input_dir}') is not a valid directory. Log chaining may be broken.")

            if not valid_output_dir:
                 raise ValueError(f"StandardLog error in {function_name}: Function did not return a valid output directory path as its first element (or only element), or the directory does not exist. Got: '{output_dir}'")

            # --- Set up logger and handler ---
            logger = logging.getLogger(f"{module_name}.{function_name}")
            logger.setLevel(logging.INFO) # Ensure logger processes INFO level

            # Create handler and formatter
            handler = JsonChainHandler(input_dir=input_dir, output_dir=output_dir, function_name=function_name)
            formatter = ArgTruncatingFormatter(max_size=max_size, log_level=log_level)
            handler.setFormatter(formatter)

            # Add handler, log, then remove handler
            # Check if handlers already exist to prevent duplicates if logger is reused somehow
            # (though getLogger should return the same instance)
            if not any(isinstance(h, JsonChainHandler) and h.output_dir == output_dir for h in logger.handlers):
                logger.addHandler(handler)
                try:
                    # Create LogRecord manually to pass raw args in msg and correct funcName
                    record = logging.LogRecord(
                        name=logger.name,
                        level=logging.INFO,
                        pathname=script_location,
                        lineno=inspect.getsourcelines(func)[1] if script_location != "unknown" else 0,
                        msg=raw_args_details, # Pass the raw list here
                        args=[],
                        exc_info=None,
                        func=function_name
                    )
                    logger.handle(record) # Process this record
                finally:
                    logger.removeHandler(handler) # Crucial cleanup
            else:
                 logging.debug(f"Skipping addHandler for {function_name} as a handler for this output dir might already exist.")


            return original_result # Return the original result

        return wrapper
    return decorator

# --- End of New Logging Components ---