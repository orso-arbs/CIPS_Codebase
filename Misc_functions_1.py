import os
import datetime
import time

### start inform
#
# purpose:
#   Print the current time and name of script
#   save the start time
#
# Input: 
#   script_location - string, name of script to be printed
#                   - default is the Misc_funcition script name
#
# Output:
#   Start time      - float, time.time() object	
#   Current date    - string, current date

def start_inform(script_location=__file__):  # Pass filename as an argument
    start_time = time.time()
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(f"\n{os.path.basename(script_location)}: ", datetime.datetime.now(), "\n")
    return start_time, current_date

### end inform
# 
# purpose:
#   Print the name of the script
#   Print the time taken to execute the script
#
# Input: 
#   script_location - string, script name
#                   - default is the Misc_funcition script name
#
# Output:
#   End time        - float, time.time() object
#   Elapsed time    - float, time taken to execute the script

def end_inform(script_location=__file__, start_time=0):  # Pass filename as an argument
    end_time = time.time()
    if start_time == 0:
        print(f"\n{os.path.basename(script_location)}: Completely Executed")
        print(f"\tNB: No Execution duration since no start_time was provided \n")
        return end_time
    else:
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        print(f"\n{os.path.basename(script_location)}: Completely Executed in {int(minutes)} min {seconds:.2f} sec \n")
        return end_time, elapsed_time


### Data Historian
# 
# purpose:
#   Save the history of the data
#
# Input:
