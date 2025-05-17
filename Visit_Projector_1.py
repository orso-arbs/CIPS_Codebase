import os
import sys
import pandas as pd
import time

import Format_1 as F_1

import Visit_Create_BW_Colortable as VPBWCT




@F_1.ParameterLog(max_size = 1024 * 10) # 10KB 
def Visit_projector_1(
    # input
    input_dir,

    # VisIt parameters
    Database, State_range_manual = [], # Data
    Plots = ["Pseudocolor-velocity_magnitude Isosurface-temperature3"], # Plots
    Pseudocolor_Variable = "velocity_magnitude", 
    Pseudocolor_colortable = "hot", # Can be "hot", "CustomBW1", "CustomBW2", "PeriodicBW", etc.
    invertColorTable = 0,
    # Parameters for the periodic black and white color table
    Pseudocolor_periodic_num_periods = 2, # periods of w-w-b-b points (4 points)
    distance_ww = 2.0,  # Relative length of solid white
    distance_wb = 1.0,  # Relative length of white-to-black gradient
    distance_bb = 2.0,  # Relative length of solid black
    distance_bw = 1.0,  # Relative length of black-to-white gradient

    Isosurface_Variable = "temperature", Isosurface_ContourValue = 3,
    no_annotations = 1, viewNormal = [0,0,-1], viewUp = [1,0,0], imageZoom = 1, parallelScale = 100, perspective = 1, # View
    Visit_projector_1_show_windows = 0, # Window

    # output and logging
    Visit_projector_1_log_level = 0,
    output_dir_manual = "", output_dir_comment = "",
):
    """
    Uses VisIt to output .png images from a .nek5000 database.

    Parameters
    ----------
    input_dir : str
        Used to set output directory.
    Database : str
        Path to the Nek5000 database.
    State_range_manual : list of int, optional
        States (time dumps in my case) to be processed. If empty ("[]"), all states are processed.
        Note that states usually start at 0.
    Plots : list of str, optional
        List of plot names to be created. Plots are defined inside the Visit_projector_1 function. Feel free to add ones!
        Default is ["Pseudocolor-velocity_magnitude Isosurface-temperature3"].
    Pseudocolor_Variable : str, optional
        Variable to use for the Pseudocolor plot. Default is "velocity_magnitude".
    Pseudocolor_colortable : str, optional
        Name of the color table for the Pseudocolor plot. Can be a built-in name or "CustomBW1", "CustomBW2",
        or the value of `Pseudocolor_periodic_name` to trigger custom generation. Default is "hot".
    invertColorTable : int, optional
        0 or 1 to invert the color table. Default is 0.
    Isosurface_Variable : str, optional
        Variable for the Isosurface operator. Default is "temperature".
    Isosurface_ContourValue : float or tuple, optional
        Contour value(s) for the Isosurface operator. Default is 3.
    no_annotations : int, optional
        If 1, no annotations are added to the plot. Default is 1.
    viewNormal : list of float, optional
        VisIt View normal vector. Default is [0,0,-1].
    viewUp : list of float, optional
        VisIt View up vector. Default is [1,0,0].
    imageZoom : float, optional
        VisIt Zoom level for the image. Default is 1.
    parallelScale : float, optional
        VisIt Parallel scale for the view. Default is 100.
    perspective : int, optional
        If 1, VisIt perspective view is used. Default is 1.
    Visit_projector_1_show_windows : int, optional
        If 1, VisIt windows are shown. Default is 0.
    Pseudocolor_periodic_name : str, optional
        The name to register for the custom periodic black and white color table.
        If `Pseudocolor_colortable` is set to this value, this table will be generated. Default is "PeriodicBW".
    Pseudocolor_periodic_num_periods : int, optional
        Number of periods for the periodic black and white color table. Default is 2.
    distance_ww : float, optional
        Relative length of the solid white segment in the periodic color table. Default is 2.0.
    distance_wb : float, optional
        Relative length of the white-to-black gradient in the periodic color table. Default is 1.0.
    distance_bb : float, optional
        Relative length of the solid black segment in the periodic color table. Default is 2.0.
    distance_bw : float, optional
        Relative length of the black-to-white gradient in the periodic color table. Default is 1.0.
    Visit_projector_1_log_level : int, optional
        Log level for the function. Default is 0.
    output_dir_manual : str, optional
        Manual output directory. If empty, a default directory is used.
    output_dir_comment : str, optional
        Comment for the output directory. Default is "".

    Returns
    -------
    output_dir : str
        Output directory where the images are saved.

    Notes
    -----
    - Pseudocolor_colortable: 
        - CustomBW: threshold between white and black. Low values are almost but not quite perfect white to have a contrast with the perfect white background to estimate the SF radius in pixels
        - PeriodicBW: triggers generation of the custom periodic black and white table.
    """

    # _VISIT_INITIALIZED used to launch visit only once in the case that multiple runs of the pipeline with VisIt are needed. This avoids errors.
    global _VISIT_INITIALIZED
    if '_VISIT_INITIALIZED' not in globals():
        _VISIT_INITIALIZED = False
    if not _VISIT_INITIALIZED:
        _VISIT_INITIALIZED = True

        # visit import and launch on the first run
        sys.path.append(r"C:\Users\obs\LLNL\VisIt3.4.2\lib\site-packages") # Consider making this path more flexible or an argument
        import visit as vi
        print("imported visit \n") if Visit_projector_1_log_level >= 2 else None
        vi.AddArgument("-nowin") if Visit_projector_1_show_windows == 0 else None
        vi.AddArgument("-quiet -nopty") if Visit_projector_1_log_level == 0 else None # -nopty might not be needed with -quiet
        vi.Launch() # loads rest of visit functions
        print("launched visit") if Visit_projector_1_log_level >= 1 else None
        if Visit_projector_1_log_level >= 3:
            print("Setting VisIt client debug level to 5")
            vi.SetDebugLevel("5") # SetDebugLevel expects a string like "1" through "5"
 
    import visit as vi # Ensure vi is defined for subsequent calls if not the first run
    print("imported visit \n") if Visit_projector_1_log_level >= 2 else None


    #################################################### I/O
    output_dir = F_1.F_out_dir(input_dir, __file__, output_dir_comment = output_dir_comment, output_dir_manual = output_dir_manual) # Format_1 required definition of output directory


    #################################################### VisIt

    # launch compute engine
    print(
    "\nif prompted, please confirm visit window: 'Select options for 'euler.ethz.ch' with 'ok'\n" \
    "If prompted, please provide Euler password\n" \
    "Then wait for Euler to allocate resources for the job\n" \
    "DO NOT close the appearing terminal and visit window\n"
    ) if Visit_projector_1_log_level >= 2 else None

    p = vi.GetMachineProfile("euler.ethz.ch")
    #print(p) # uncomment to see the machine profile
    p.userName="orsob"
    p.activeProfile = 1 # Assuming this is the correct parallel profile
    launch_profile = p.GetLaunchProfiles(p.activeProfile) # Get the active profile directly
    launch_profile.numProcessors = 4
    launch_profile.numNodes = 1
    launch_profile.timeLimit = "04:00:00"
    additional_args = f"--mem-per-cpu=4G --tmp=4G --output=/cluster/scratch/orsob/orsoMT_orsob/VisIt_logs_and_error_output/%j_visit.out --error=/cluster/scratch/orsob/orsoMT_orsob/VisIt_logs_and_error_output/%j_visit.err"
    launch_profile.launchArgs = additional_args # Set launchArgs on the retrieved profile object
    print(f"Updated launchArgs: {launch_profile.launchArgs}") if Visit_projector_1_log_level >= 2 else None
    
    vi.OpenComputeEngine(p)
    print("launched compute engine \n") if Visit_projector_1_log_level >= 2 else None

    # open database
    OpenSuccess = vi.OpenDatabase(Database)
    print(f"Opened database") if OpenSuccess == 1 and Visit_projector_1_log_level >= 1 else None
    if OpenSuccess == 0: # Check for failure
        print("Failed to open database")
        return output_dir # Exit if database open fails

    # define Expressions
    vi.DefineScalarExpression("X", "coord(mesh)[0]")
    vi.DefineScalarExpression("Y", "coord(mesh)[1]")
    vi.DefineScalarExpression("Z", "coord(mesh)[2]")
    vi.DefineScalarExpression("R", "sqrt(X*X + Y*Y + Z*Z)")
    print("Defined scalar expressions \n") if Visit_projector_1_log_level >= 2 else None

    # define plot
    if "Pseudocolor - Isosurface" in Plots:
        print("plotting Pseudocolor - Isosurface\n") if Visit_projector_1_log_level >= 2 else None
        
        vi.AddPlot("Pseudocolor", Pseudocolor_Variable, 1, 1)
        # SetActivePlots(0) is usually done to target the most recently added plot for operators
        vi.SetActivePlots(0) 

        vi.AddOperator("Isosurface", 1) # Apply to active plot (index 0)
        IsosurfaceAtts = vi.IsosurfaceAttributes()
        IsosurfaceAtts.contourNLevels = 10
        IsosurfaceAtts.contourValue = (Isosurface_ContourValue) if isinstance(Isosurface_ContourValue, tuple) else (Isosurface_ContourValue,) # Ensure it's a tuple
        IsosurfaceAtts.contourPercent = ()
        IsosurfaceAtts.contourMethod = IsosurfaceAtts.Value  # Level, Value, Percent
        IsosurfaceAtts.minFlag = 0
        IsosurfaceAtts.min = 0
        IsosurfaceAtts.maxFlag = 0
        IsosurfaceAtts.max = 1
        IsosurfaceAtts.scaling = IsosurfaceAtts.Linear  # Linear, Log
        IsosurfaceAtts.variable = Isosurface_Variable
        # Apply Isosurface attributes to the Isosurface operator on the current plot (index 0, operator index 0)
        vi.SetOperatorOptions(IsosurfaceAtts, 0) # The second argument 0 refers to the first operator on the plot


        # --- Custom Color Table Logic ---
        # Check if the requested color table is one of the custom ones
        if Pseudocolor_colortable == "CustomBW1":
            print("Creating CustomBW1 color table") if Visit_projector_1_log_level >= 1 else None
            ccpl = vi.ColorControlPointList()
            p1 = vi.ColorControlPoint(); p1.colors = (255, 255, 255, 255); p1.position = 0.0; ccpl.AddControlPoints(p1)
            p2 = vi.ColorControlPoint(); p2.colors = (255, 255, 255, 255); p2.position = 0.4; ccpl.AddControlPoints(p2)
            p3 = vi.ColorControlPoint(); p3.colors = (0, 0, 0, 255); p3.position = 0.6; ccpl.AddControlPoints(p3)
            p4 = vi.ColorControlPoint(); p4.colors = (0, 0, 0, 255); p4.position = 1.0; ccpl.AddControlPoints(p4)
            vi.AddColorTable("CustomBW1", ccpl)
        elif Pseudocolor_colortable == "CustomBW2":
            print("Creating CustomBW2 color table") if Visit_projector_1_log_level >= 1 else None
            ccpl = vi.ColorControlPointList()
            p1 = vi.ColorControlPoint(); p1.colors = (255, 255, 255, 255); p1.position = 0.0; ccpl.AddControlPoints(p1)
            p2 = vi.ColorControlPoint(); p2.colors = (255, 255, 255, 255); p2.position = 0.4; ccpl.AddControlPoints(p2)
            p3 = vi.ColorControlPoint(); p3.colors = (0, 0, 0, 255); p3.position = 0.6; ccpl.AddControlPoints(p3)
            p4 = vi.ColorControlPoint(); p4.colors = (0, 0, 0, 255); p4.position = 1.0; ccpl.AddControlPoints(p4)
            vi.AddColorTable("CustomBW2", ccpl)
        elif Pseudocolor_colortable == "PeriodicBW": # Check if it's the periodic one
            print(f"Creating PeriodicBW color table") if Visit_projector_1_log_level >= 1 else None
            # Create a subdirectory for color tables
            colortable_storage_dir = os.path.join(output_dir, "colortables")
            VPBWCT.create_periodic_bw_color_table(
                Pseudocolor_periodic_num_periods,
                distance_ww,
                distance_wb,
                distance_bb,
                distance_bw,
                visit_module_ref=vi,
                colortable_output_dir=colortable_storage_dir
                )
        # --- End Custom Color Table Logic ---

        PseudocolorAtts = vi.PseudocolorAttributes()
        PseudocolorAtts.scaling = PseudocolorAtts.Linear  # Linear, Log, Skew
        PseudocolorAtts.skewFactor = 1
        PseudocolorAtts.limitsMode = PseudocolorAtts.ActualData  # OriginalData, ActualData
        PseudocolorAtts.minFlag = 0
        PseudocolorAtts.min = 0
        PseudocolorAtts.useBelowMinColor = 0
        PseudocolorAtts.belowMinColor = (0, 0, 0, 255)
        PseudocolorAtts.maxFlag = 0
        PseudocolorAtts.max = 1
        PseudocolorAtts.useAboveMaxColor = 0
        PseudocolorAtts.aboveMaxColor = (0, 0, 0, 255)
        PseudocolorAtts.centering = PseudocolorAtts.Natural  # Natural, Nodal, Zonal
        PseudocolorAtts.colorTableName = Pseudocolor_colortable # Actual color table name to use
        PseudocolorAtts.invertColorTable = invertColorTable
        PseudocolorAtts.opacityType = PseudocolorAtts.FullyOpaque  # ColorTable, FullyOpaque, Constant, Ramp, VariableRange
        PseudocolorAtts.opacityVariable = ""
        PseudocolorAtts.opacity = 1
        PseudocolorAtts.opacityVarMin = 0
        PseudocolorAtts.opacityVarMax = 1
        PseudocolorAtts.opacityVarMinFlag = 0
        PseudocolorAtts.opacityVarMaxFlag = 0
        PseudocolorAtts.pointSize = 0.05
        PseudocolorAtts.pointType = PseudocolorAtts.Point  # Box, Axis, Icosahedron, Octahedron, Tetrahedron, SphereGeometry, Point, Sphere
        PseudocolorAtts.pointSizeVarEnabled = 0
        PseudocolorAtts.pointSizeVar = "default"
        PseudocolorAtts.pointSizePixels = 2
        PseudocolorAtts.lineType = PseudocolorAtts.Line  # Line, Tube, Ribbon
        PseudocolorAtts.lineWidth = 0
        PseudocolorAtts.tubeResolution = 10
        PseudocolorAtts.tubeRadiusSizeType = PseudocolorAtts.FractionOfBBox  # Absolute, FractionOfBBox
        PseudocolorAtts.tubeRadiusAbsolute = 0.125
        PseudocolorAtts.tubeRadiusBBox = 0.005
        PseudocolorAtts.tubeRadiusVarEnabled = 0
        PseudocolorAtts.tubeRadiusVar = ""
        PseudocolorAtts.tubeRadiusVarRatio = 10
        PseudocolorAtts.tailStyle = PseudocolorAtts.NONE  # NONE, Spheres, Cones
        PseudocolorAtts.headStyle = PseudocolorAtts.NONE  # NONE, Spheres, Cones
        PseudocolorAtts.endPointRadiusSizeType = PseudocolorAtts.FractionOfBBox  # Absolute, FractionOfBBox
        PseudocolorAtts.endPointRadiusAbsolute = 0.125
        PseudocolorAtts.endPointRadiusBBox = 0.05
        PseudocolorAtts.endPointResolution = 10
        PseudocolorAtts.endPointRatio = 5
        PseudocolorAtts.endPointRadiusVarEnabled = 0
        PseudocolorAtts.endPointRadiusVar = ""
        PseudocolorAtts.endPointRadiusVarRatio = 10
        PseudocolorAtts.renderSurfaces = 1
        PseudocolorAtts.renderWireframe = 0
        PseudocolorAtts.renderPoints = 0
        PseudocolorAtts.smoothingLevel = 0
        PseudocolorAtts.legendFlag = 1
        PseudocolorAtts.lightingFlag = 1
        PseudocolorAtts.wireframeColor = (0, 0, 0, 0) # Last component is alpha
        PseudocolorAtts.pointColor = (0, 0, 0, 0)   # Last component is alpha
        vi.SetPlotOptions(PseudocolorAtts) # Apply to the active plot (Pseudocolor)

    print("Added plot\n") if Visit_projector_1_log_level >= 2 else None

    # calculate plot
    vi.DrawPlots()
    print("Drew Plots\n") if Visit_projector_1_log_level >= 2 else None # Corrected typo "Drawed"

    # set view
    View3DAtts = vi.View3DAttributes()
    View3DAtts.viewNormal = (viewNormal[0], viewNormal[1], viewNormal[2])
    View3DAtts.focus = (0, 0, 0)
    View3DAtts.viewUp = (viewUp[0], viewUp[1], viewUp[2])
    View3DAtts.viewAngle = 30
    View3DAtts.parallelScale = parallelScale
    View3DAtts.nearPlane = -300.0 
    View3DAtts.farPlane = 300.0
    View3DAtts.imagePan = (0, 0)
    View3DAtts.imageZoom = imageZoom
    View3DAtts.perspective = perspective
    View3DAtts.eyeAngle = 2
    View3DAtts.centerOfRotationSet = 0
    View3DAtts.centerOfRotation = (0, 0, 0)
    View3DAtts.axis3DScaleFlag = 0
    View3DAtts.axis3DScales = (1, 1, 1)
    View3DAtts.shear = (0, 0, 1)
    View3DAtts.windowValid = 1 # It's good practice to set this if modifying the view
    vi.SetView3D(View3DAtts)
    print("view set\n") if Visit_projector_1_log_level >= 2 else None

    # no annotations
    if no_annotations == 1:
        AnnotationAtts = vi.AnnotationAttributes()
        # --- Simplified annotation settings for brevity ---
        AnnotationAtts.axes2D.visible = 0
        AnnotationAtts.axes3D.visible = 0
        AnnotationAtts.axes3D.triadFlag = 0
        AnnotationAtts.axes3D.bboxFlag = 0
        AnnotationAtts.userInfoFlag = 0
        AnnotationAtts.databaseInfoFlag = 0
        AnnotationAtts.timeInfoFlag = 0 # Typically you want time for an animation
        AnnotationAtts.legendInfoFlag = 0
        AnnotationAtts.backgroundMode = AnnotationAtts.Solid # Keep background solid white for no annotations
        AnnotationAtts.backgroundColor = (255,255,255,255) 
        AnnotationAtts.foregroundColor = (0,0,0,255)
        # --- End simplified ---
        vi.SetAnnotationAttributes(AnnotationAtts)

    # set windows attributes
    SaveWindowAtts = vi.SaveWindowAttributes()
    SaveWindowAtts.outputToCurrentDirectory = 0 # Save to specified output_dir
    SaveWindowAtts.outputDirectory = output_dir
    SaveWindowAtts.family = 0 # Set to 0 for explicit naming in loop
    SaveWindowAtts.format = SaveWindowAtts.PNG
    SaveWindowAtts.width = 1024
    SaveWindowAtts.height = 1024
    SaveWindowAtts.quality = 80 # For JPEG
    # No need to call SetSaveWindowAttributes here, will be set in loop for each filename
    print("window attributes set for loop\n") if Visit_projector_1_log_level >= 2 else None

    # States (t's) range
    if State_range_manual: 
        State_range = State_range_manual
    else:
        # Ensure database is open and engine is responsive before querying states
        if OpenSuccess == 1:
            try:
                num_states = vi.TimeSliderGetNStates()
                State_range = range(num_states)
            except Exception as e:
                print(f"Error getting number of states: {e}. Defaulting to empty range.")
                State_range = range(0) # Default to empty if error
        else:
            State_range = range(0) # Default to empty if database not open

    print("States = ", list(State_range)) if Visit_projector_1_log_level >= 1 else None # list() for printing

    Image_filenames_VisIt = []
    Times_VisIt = []
    R_SF_Average_VisIt = []

    start_time_loop = time.time() if Visit_projector_1_log_level >= 1 else None

    # Loop once through all states (t's)
    for state in State_range: 
        start_time_state = time.time() if Visit_projector_1_log_level >= 1 else None
        print(f"Working on state {state:06d}") if Visit_projector_1_log_level >= 1 else None
        
        try:
            print(f"Attempting: vi.SetTimeSliderState({state})") if Visit_projector_1_log_level >= 3 else None
            vi.SetTimeSliderState(state)
            print(f"Done: vi.SetTimeSliderState(state) with state = {state}") if Visit_projector_1_log_level >= 3 else None

            # save window as .png image
            Image_filenames_VisIt_state = f"visit_{state:06d}" # Use d for integer formatting
            SaveWindowAtts.fileName = Image_filenames_VisIt_state # Set specific filename for this state
            vi.SetSaveWindowAttributes(SaveWindowAtts) # Apply before saving
            vi.SaveWindow() 
            print(f"saved image for state {state:06d}", end='\r') if Visit_projector_1_log_level >= 1 else None

            # save properties
            Image_filenames_VisIt.append(Image_filenames_VisIt_state + ".png") # Add extension

            vi.Query("Time")
            Time_state = vi.GetQueryOutputValue()
            Times_VisIt.append(Time_state)

            # Ensure the Pseudocolor plot is active before changing its variable
            vi.SetActivePlots(0) # Assuming Pseudocolor is plot 0
            vi.ChangeActivePlotsVar("R")
            vi.Query("Average Value") # This query might not exist. Common queries are "Min", "Max", "Weighted Variable Sum"
            R_SF_Average_state = vi.GetQueryOutputValue()
            R_SF_Average_VisIt.append(R_SF_Average_state)
            # Change back to original variable for next iteration's plot setup if necessary
            vi.ChangeActivePlotsVar(Pseudocolor_Variable) 
            print(f"saved R_Average_state = {R_SF_Average_state} for file {state:06d}", end='\r') if Visit_projector_1_log_level >= 1 else None

        except Exception as e:
            print(f"Error processing state {state}: {e}")
            # Decide if you want to continue to next state or stop
            # For now, let's append placeholders and continue
            Image_filenames_VisIt.append(f"error_state_{state:06d}.png")
            Times_VisIt.append(float('nan'))
            R_SF_Average_VisIt.append(float('nan'))
            continue # Skip to next state

        # Attempt to clear VisIt's cache to free memory.
        if Visit_projector_1_log_level >= 2:
            print(f"Clearing VisIt cache for state {state}")
        try:
            vi.ClearCacheForAllEngines()
            if Visit_projector_1_log_level >= 2:
                print(f"Done clearing VisIt cache for state {state}")
        except Exception as e:
            if Visit_projector_1_log_level >= 0: 
                print(f"Error clearing VisIt cache for state {state}: {e}")

        if Visit_projector_1_log_level >= 1:
            elapsed_time_state = time.time() - start_time_state
            print(f"State {state:06d} Elapsed time: {elapsed_time_state:.2f} s")
    
    if Visit_projector_1_log_level >= 1 and len(State_range) > 0 : # Avoid division by zero if State_range is empty
        elapsed_time_loop = time.time() - start_time_loop
        print(f"\nTotal Loop Elapsed time: {elapsed_time_loop:.2f} s, Avg per state: {elapsed_time_loop/len(State_range):.2f} s")


    #################################################### save data

    VisIt_data_df = pd.DataFrame({
        'Plot_VisIt': [Plots[0]] * len(State_range) if State_range else [], # Handle empty State_range
        'Image_filename_VisIt': Image_filenames_VisIt,
        'State_range_VisIt': list(State_range), # Convert range to list for DataFrame
        'Time_VisIt': Times_VisIt,
        'R_SF_Average_VisIt': R_SF_Average_VisIt,
        })
    print(VisIt_data_df)

    # save to pickle
    pkl_filename = os.path.join(output_dir, "Visit_projector_1_data.pkl") # Use os.path.join
    VisIt_data_df.to_pickle(pkl_filename)
    print(f"Saved extracted DataFrame to {pkl_filename}") if Visit_projector_1_log_level >= 2 else None

    # save to csv
    csv_filename = os.path.join(output_dir, "Visit_projector_1_data.csv") # Use os.path.join
    VisIt_data_df.to_csv(csv_filename, sep='\t', index=False)
    print(f"Saved extracted DataFrame to {csv_filename}") if Visit_projector_1_log_level >= 2 else None

    # Clean up
    try:
        vi.DeleteAllPlots()
        if OpenSuccess == 1: # Only close if successfully opened
             vi.CloseDatabase(Database)
        vi.CloseComputeEngine("euler.ethz.ch") # Or use p.host if it's more general
        print("Closed all plots, database, and compute engine") if Visit_projector_1_log_level >= 1 else None
    except Exception as e:
        print(f"Error during VisIt cleanup: {e}") if Visit_projector_1_log_level >=0 else None


    #################################################### return
    return output_dir

# Example of how to call your function (assuming vi is imported as visit)
# if __name__ == "__main__":
#     import visit as vi # Example: if you run this script with python your_script.py
#     if not _VISIT_INITIALIZED:
#         sys.path.append(r"C:\Users\obs\LLNL\VisIt3.4.2\lib\site-packages")
#         _VISIT_INITIALIZED = True
#         print("imported visit in main")
#         vi.AddArgument("-nowin")
#         vi.AddArgument("-quiet -nopty")
#         vi.Launch()
#         print("launched visit in main")

#     # --- Call with default periodic color table ---
#     Visit_projector_1(
#         input_dir=".", # Or your actual input directory
#         Database=r"euler.ethz.ch:/cluster/scratch/orsob/your_database.nek5000", # Replace with actual path
#         Pseudocolor_colortable="PeriodicBW", # This will trigger the new function
#         Visit_projector_1_log_level=1
#     )

#     # --- Call with custom periodic color table parameters ---
#     Visit_projector_1(
#         input_dir=".",
#         Database=r"euler.ethz.ch:/cluster/scratch/orsob/your_database.nek5000", # Replace
#         Pseudocolor_colortable="MyCustomPeriodic", 
#         Pseudocolor_periodic_name="MyCustomPeriodic", # Match this with Pseudocolor_colortable
#         Pseudocolor_periodic_num_periods=3,
#         Pseudocolor_periodic_ww=1.0,
#         Pseudocolor_periodic_wb=0.5,
#         Pseudocolor_periodic_bb=1.0,
#         Pseudocolor_periodic_bw=0.5,
#         Visit_projector_1_log_level=1
#     )
#     vi.Close() # Close VisIt if launched from script
