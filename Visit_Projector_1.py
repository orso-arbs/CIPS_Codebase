import os
import sys
import pandas as pd

import Format_1 as F_1



@F_1.ParameterLog(max_size = 1024 * 10) # 10KB 
def Visit_projector_1(
    # input
    input_dir,

    # VisIt parameters
    Database, State_range_manual = [], # Data
    Plots = ["Pseudocolor-velocity_magnitude Isosurface-temperature3"], # Plots
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
        If 1, VisIt windows are shown. Default is 0. Does not work :(.

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
    """


    # visit import and launch
    sys.path.append(r"C:\Users\obs\LLNL\VisIt3.4.2\lib\site-packages")
    import visit as vi
    vi.AddArgument("-nowin") if Visit_projector_1_show_windows == 1 else None #-gui
    vi.Launch() # loads rest of visit functions
    print("launched visit") if Visit_projector_1_log_level > 0 else None
    import visit as vi # loads rest of visit functions
    print("imported visit \n") if Visit_projector_1_log_level > 0 else None

    #################################################### I/O
    output_dir = F_1.F_out_dir(input_dir, __file__, output_dir_comment = output_dir_comment) # Format_1 required definition of output directory


    #################################################### VisIt

    # launch compute engine
    print(
    "\nif prompted, please confirm visit window: 'Select options for 'euler.ethz.ch' with 'ok'\n" \
    "If prompted, please provide Euler password\n" \
    "Then wait for Euler to allocate resources for the job\n" \
    "DO NOT close the appearing terminal and visit window\n"
    ) if Visit_projector_1_log_level > 0 else None
    p = vi.GetMachineProfile("euler.ethz.ch")
    #print(p) # uncomment to see the machine profile
    p.userName="orsob"
    p.activeProfile = 1
    p.GetLaunchProfiles(1).numProcessors = 4
    p.GetLaunchProfiles(1).numNodes = 1
    p.GetLaunchProfiles(1).timeLimit = "02:00:00"
    vi.OpenComputeEngine(p)
    print("launched compute engine \n") if Visit_projector_1_log_level > 0 else None

    # open database
    vi.OpenDatabase(Database)     #i.e. Database = r"euler.ethz.ch:/cluster/scratch/orsob/MastersThesis/postProc/po_part1/po_s912k_post.nek5000"
    print("Opened Database\n") if Visit_projector_1_log_level > 0 else None

    # define Expressions
    vi.DefineScalarExpression("X", "coord(mesh)[0]")
    vi.DefineScalarExpression("Y", "coord(mesh)[1]")
    vi.DefineScalarExpression("Z", "coord(mesh)[2]")
    vi.DefineScalarExpression("R", "sqrt(X*X + Y*Y + Z*Z)")

    # define plot
    #vi.AddPlot("Contour", "temperature", 1, 1) 
    if "Pseudocolor-velocity_magnitude Isosurface-temperature" in Plots:
        print("plotting Pseudocolor-velocity_magnitude Isosurface-temperature\n") if Visit_projector_1_log_level > 0 else None
        
        vi.AddPlot("Pseudocolor", "velocity_magnitude", 1, 1)

        vi.SetActivePlots(0)

        vi.AddOperator("Isosurface", 1)
        IsosurfaceAtts = vi.IsosurfaceAttributes()
        IsosurfaceAtts.contourNLevels = 10
        IsosurfaceAtts.contourValue = (3)
        IsosurfaceAtts.contourPercent = ()
        IsosurfaceAtts.contourMethod = IsosurfaceAtts.Value  # Level, Value, Percent
        IsosurfaceAtts.minFlag = 0
        IsosurfaceAtts.min = 0
        IsosurfaceAtts.maxFlag = 0
        IsosurfaceAtts.max = 1
        IsosurfaceAtts.scaling = IsosurfaceAtts.Linear  # Linear, Log
        IsosurfaceAtts.variable = "temperature"
        vi.SetOperatorOptions(IsosurfaceAtts, 0, 1)

    if "Pseudocolor-velocity_magnitude Isosurface-temperature colorTableName-CustomBW" in Plots:
        print("plotting Pseudocolor-velocity_magnitude Isosurface-temperature colorTableName-CustomBW\n") if Visit_projector_1_log_level > 0 else None
        
        vi.AddPlot("Pseudocolor", "velocity_magnitude", 1, 1)

        vi.SetActivePlots(0)

        vi.SetActivePlots(0)

        vi.AddOperator("Isosurface", 1)
        IsosurfaceAtts = vi.IsosurfaceAttributes()
        IsosurfaceAtts.contourNLevels = 10
        IsosurfaceAtts.contourValue = (3)
        IsosurfaceAtts.contourPercent = ()
        IsosurfaceAtts.contourMethod = IsosurfaceAtts.Value  # Level, Value, Percent
        IsosurfaceAtts.minFlag = 0
        IsosurfaceAtts.min = 0
        IsosurfaceAtts.maxFlag = 0
        IsosurfaceAtts.max = 1
        IsosurfaceAtts.scaling = IsosurfaceAtts.Linear  # Linear, Log
        IsosurfaceAtts.variable = "temperature"


        ### create color table
        # Create a color control point list
        ccpl = vi.ColorControlPointList()
        # Point 1: white at 0.0 (fade from white)
        p1 = vi.ColorControlPoint()
        p1.colors = (255, 255, 255, 255)
        p1.position = 0.0
        ccpl.AddControlPoints(p1)
        # Point 2: white at 0.5
        p2 = vi.ColorControlPoint()
        p2.colors = (255, 255, 255, 255)
        p2.position = 0.4
        ccpl.AddControlPoints(p2)
        # Point 3: black at 0.8
        p3 = vi.ColorControlPoint()
        p3.colors = (0, 0, 0, 255)
        p3.position = 0.6
        ccpl.AddControlPoints(p3)
        # Point 4: black at 1.0 (fade to black)
        p4 = vi.ColorControlPoint()
        p4.colors = (0, 0, 0, 255)
        p4.position = 1.0
        ccpl.AddControlPoints(p4)
        # Add the color table to VisIt
        vi.AddColorTable("CustomBW", ccpl)


        vi.SetOperatorOptions(IsosurfaceAtts, 0, 1)
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
        PseudocolorAtts.colorTableName = "CustomBW"
        PseudocolorAtts.invertColorTable = 0
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
        PseudocolorAtts.wireframeColor = (0, 0, 0, 0)
        PseudocolorAtts.pointColor = (0, 0, 0, 0)
        vi.SetPlotOptions(PseudocolorAtts)





    print("Added plot\n") if Visit_projector_1_log_level > 0 else None


    # calculate plot
    vi.DrawPlots()
    print("Drawed Plots\n") if Visit_projector_1_log_level > 0 else None

    # set view
    View3DAtts = vi.View3DAttributes()
    View3DAtts.viewNormal = (viewNormal[0], viewNormal[1], viewNormal[2])  # the direction from the camera location to the focal point
    View3DAtts.focus = (0, 0, 0)                              # the focal point
    View3DAtts.viewUp = (viewUp[0], viewUp[1], viewUp[2])      # specifies the axis that goes along the height of the screen
    View3DAtts.viewAngle = 30                                 # specifies the "view angle", meaning the "angle" of the pyramid that defines the view frustum
    View3DAtts.parallelScale = parallelScale                  # scales how far the camera is located along the viewNormal from the focal point, which affects the scale of the data set in the screen
    View3DAtts.nearPlane = -300.0                             # specifies where the pyramid of the view frustum is truncated on the near side; relative to the focal point, often negative
    View3DAtts.farPlane = 300.0                               # specifies where the pyramid of the view frustum is truncated on the far side; relative to the focal point
    View3DAtts.imagePan = (0, 0)                              # allows the image to be translated without affecting the view frustum definition
    View3DAtts.imageZoom = imageZoom                          # allows the image to be zoomed in on without affecting the view frustum definition
    View3DAtts.perspective = perspective                      # a Boolean: 1 for perspective projection, 0 for orthographic
    View3DAtts.eyeAngle = 2                                   # specifies the angle of the eye for stereo viewing
    View3DAtts.centerOfRotationSet = 0                        # specifies whether or not rotations occur around the focal point (0) or another point (1)
    View3DAtts.centerOfRotation = (0, 0, 0)                   # the place to rotate around if we are not using the focal point
    View3DAtts.axis3DScaleFlag = 0                            # specifies whether or not the axes are scaled by a scale factor
    View3DAtts.axis3DScales = (1, 1, 1)                       # scale factor for each axis (x, y, z)
    View3DAtts.shear = (0, 0, 1)                              # used to support oblique projections (like cabinet or cavalier); default disables shear
    View3DAtts.windowValid = 1
    vi.SetView3D(View3DAtts)
    print("view set\n") if Visit_projector_1_log_level > 0 else None


    # no annotations
    if no_annotations == 1:
        AnnotationAtts = vi.AnnotationAttributes()
        AnnotationAtts.axes2D.visible = 0
        AnnotationAtts.axes2D.autoSetTicks = 1
        AnnotationAtts.axes2D.autoSetScaling = 1
        AnnotationAtts.axes2D.lineWidth = 0
        AnnotationAtts.axes2D.tickLocation = AnnotationAtts.axes2D.Outside  # Inside, Outside, Both
        AnnotationAtts.axes2D.tickAxes = AnnotationAtts.axes2D.BottomLeft  # Off, Bottom, Left, BottomLeft, All
        AnnotationAtts.axes2D.xAxis.title.visible = 1
        AnnotationAtts.axes2D.xAxis.title.font.font = AnnotationAtts.axes2D.xAxis.title.font.Courier  # Arial, Courier, Times
        AnnotationAtts.axes2D.xAxis.title.font.scale = 1
        AnnotationAtts.axes2D.xAxis.title.font.useForegroundColor = 1
        AnnotationAtts.axes2D.xAxis.title.font.color = (0, 0, 0, 255)
        AnnotationAtts.axes2D.xAxis.title.font.bold = 1
        AnnotationAtts.axes2D.xAxis.title.font.italic = 1
        AnnotationAtts.axes2D.xAxis.title.userTitle = 0
        AnnotationAtts.axes2D.xAxis.title.userUnits = 0
        AnnotationAtts.axes2D.xAxis.title.title = "X-Axis"
        AnnotationAtts.axes2D.xAxis.title.units = ""
        AnnotationAtts.axes2D.xAxis.label.visible = 1
        AnnotationAtts.axes2D.xAxis.label.font.font = AnnotationAtts.axes2D.xAxis.label.font.Courier  # Arial, Courier, Times
        AnnotationAtts.axes2D.xAxis.label.font.scale = 1
        AnnotationAtts.axes2D.xAxis.label.font.useForegroundColor = 1
        AnnotationAtts.axes2D.xAxis.label.font.color = (0, 0, 0, 255)
        AnnotationAtts.axes2D.xAxis.label.font.bold = 1
        AnnotationAtts.axes2D.xAxis.label.font.italic = 1
        AnnotationAtts.axes2D.xAxis.label.scaling = 0
        AnnotationAtts.axes2D.xAxis.tickMarks.visible = 1
        AnnotationAtts.axes2D.xAxis.tickMarks.majorMinimum = 0
        AnnotationAtts.axes2D.xAxis.tickMarks.majorMaximum = 1
        AnnotationAtts.axes2D.xAxis.tickMarks.minorSpacing = 0.02
        AnnotationAtts.axes2D.xAxis.tickMarks.majorSpacing = 0.2
        AnnotationAtts.axes2D.xAxis.grid = 0
        AnnotationAtts.axes2D.yAxis.title.visible = 1
        AnnotationAtts.axes2D.yAxis.title.font.font = AnnotationAtts.axes2D.yAxis.title.font.Courier  # Arial, Courier, Times
        AnnotationAtts.axes2D.yAxis.title.font.scale = 1
        AnnotationAtts.axes2D.yAxis.title.font.useForegroundColor = 1
        AnnotationAtts.axes2D.yAxis.title.font.color = (0, 0, 0, 255)
        AnnotationAtts.axes2D.yAxis.title.font.bold = 1
        AnnotationAtts.axes2D.yAxis.title.font.italic = 1
        AnnotationAtts.axes2D.yAxis.title.userTitle = 0
        AnnotationAtts.axes2D.yAxis.title.userUnits = 0
        AnnotationAtts.axes2D.yAxis.title.title = "Y-Axis"
        AnnotationAtts.axes2D.yAxis.title.units = ""
        AnnotationAtts.axes2D.yAxis.label.visible = 1
        AnnotationAtts.axes2D.yAxis.label.font.font = AnnotationAtts.axes2D.yAxis.label.font.Courier  # Arial, Courier, Times
        AnnotationAtts.axes2D.yAxis.label.font.scale = 1
        AnnotationAtts.axes2D.yAxis.label.font.useForegroundColor = 1
        AnnotationAtts.axes2D.yAxis.label.font.color = (0, 0, 0, 255)
        AnnotationAtts.axes2D.yAxis.label.font.bold = 1
        AnnotationAtts.axes2D.yAxis.label.font.italic = 1
        AnnotationAtts.axes2D.yAxis.label.scaling = 0
        AnnotationAtts.axes2D.yAxis.tickMarks.visible = 1
        AnnotationAtts.axes2D.yAxis.tickMarks.majorMinimum = 0
        AnnotationAtts.axes2D.yAxis.tickMarks.majorMaximum = 1
        AnnotationAtts.axes2D.yAxis.tickMarks.minorSpacing = 0.02
        AnnotationAtts.axes2D.yAxis.tickMarks.majorSpacing = 0.2
        AnnotationAtts.axes2D.yAxis.grid = 0
        AnnotationAtts.axes3D.visible = 0
        AnnotationAtts.axes3D.autoSetTicks = 1
        AnnotationAtts.axes3D.autoSetScaling = 1
        AnnotationAtts.axes3D.lineWidth = 0
        AnnotationAtts.axes3D.tickLocation = AnnotationAtts.axes3D.Inside  # Inside, Outside, Both
        AnnotationAtts.axes3D.axesType = AnnotationAtts.axes3D.ClosestTriad  # ClosestTriad, FurthestTriad, OutsideEdges, StaticTriad, StaticEdges
        AnnotationAtts.axes3D.triadFlag = 0
        AnnotationAtts.axes3D.bboxFlag = 0
        AnnotationAtts.axes3D.xAxis.title.visible = 1
        AnnotationAtts.axes3D.xAxis.title.font.font = AnnotationAtts.axes3D.xAxis.title.font.Arial  # Arial, Courier, Times
        AnnotationAtts.axes3D.xAxis.title.font.scale = 1
        AnnotationAtts.axes3D.xAxis.title.font.useForegroundColor = 1
        AnnotationAtts.axes3D.xAxis.title.font.color = (0, 0, 0, 255)
        AnnotationAtts.axes3D.xAxis.title.font.bold = 0
        AnnotationAtts.axes3D.xAxis.title.font.italic = 0
        AnnotationAtts.axes3D.xAxis.title.userTitle = 0
        AnnotationAtts.axes3D.xAxis.title.userUnits = 0
        AnnotationAtts.axes3D.xAxis.title.title = "X-Axis"
        AnnotationAtts.axes3D.xAxis.title.units = ""
        AnnotationAtts.axes3D.xAxis.label.visible = 1
        AnnotationAtts.axes3D.xAxis.label.font.font = AnnotationAtts.axes3D.xAxis.label.font.Arial  # Arial, Courier, Times
        AnnotationAtts.axes3D.xAxis.label.font.scale = 1
        AnnotationAtts.axes3D.xAxis.label.font.useForegroundColor = 1
        AnnotationAtts.axes3D.xAxis.label.font.color = (0, 0, 0, 255)
        AnnotationAtts.axes3D.xAxis.label.font.bold = 0
        AnnotationAtts.axes3D.xAxis.label.font.italic = 0
        AnnotationAtts.axes3D.xAxis.label.scaling = 0
        AnnotationAtts.axes3D.xAxis.tickMarks.visible = 1
        AnnotationAtts.axes3D.xAxis.tickMarks.majorMinimum = 0
        AnnotationAtts.axes3D.xAxis.tickMarks.majorMaximum = 1
        AnnotationAtts.axes3D.xAxis.tickMarks.minorSpacing = 0.02
        AnnotationAtts.axes3D.xAxis.tickMarks.majorSpacing = 0.2
        AnnotationAtts.axes3D.xAxis.grid = 0
        AnnotationAtts.axes3D.yAxis.title.visible = 1
        AnnotationAtts.axes3D.yAxis.title.font.font = AnnotationAtts.axes3D.yAxis.title.font.Arial  # Arial, Courier, Times
        AnnotationAtts.axes3D.yAxis.title.font.scale = 1
        AnnotationAtts.axes3D.yAxis.title.font.useForegroundColor = 1
        AnnotationAtts.axes3D.yAxis.title.font.color = (0, 0, 0, 255)
        AnnotationAtts.axes3D.yAxis.title.font.bold = 0
        AnnotationAtts.axes3D.yAxis.title.font.italic = 0
        AnnotationAtts.axes3D.yAxis.title.userTitle = 0
        AnnotationAtts.axes3D.yAxis.title.userUnits = 0
        AnnotationAtts.axes3D.yAxis.title.title = "Y-Axis"
        AnnotationAtts.axes3D.yAxis.title.units = ""
        AnnotationAtts.axes3D.yAxis.label.visible = 1
        AnnotationAtts.axes3D.yAxis.label.font.font = AnnotationAtts.axes3D.yAxis.label.font.Arial  # Arial, Courier, Times
        AnnotationAtts.axes3D.yAxis.label.font.scale = 1
        AnnotationAtts.axes3D.yAxis.label.font.useForegroundColor = 1
        AnnotationAtts.axes3D.yAxis.label.font.color = (0, 0, 0, 255)
        AnnotationAtts.axes3D.yAxis.label.font.bold = 0
        AnnotationAtts.axes3D.yAxis.label.font.italic = 0
        AnnotationAtts.axes3D.yAxis.label.scaling = 0
        AnnotationAtts.axes3D.yAxis.tickMarks.visible = 1
        AnnotationAtts.axes3D.yAxis.tickMarks.majorMinimum = 0
        AnnotationAtts.axes3D.yAxis.tickMarks.majorMaximum = 1
        AnnotationAtts.axes3D.yAxis.tickMarks.minorSpacing = 0.02
        AnnotationAtts.axes3D.yAxis.tickMarks.majorSpacing = 0.2
        AnnotationAtts.axes3D.yAxis.grid = 0
        AnnotationAtts.axes3D.zAxis.title.visible = 1
        AnnotationAtts.axes3D.zAxis.title.font.font = AnnotationAtts.axes3D.zAxis.title.font.Arial  # Arial, Courier, Times
        AnnotationAtts.axes3D.zAxis.title.font.scale = 1
        AnnotationAtts.axes3D.zAxis.title.font.useForegroundColor = 1
        AnnotationAtts.axes3D.zAxis.title.font.color = (0, 0, 0, 255)
        AnnotationAtts.axes3D.zAxis.title.font.bold = 0
        AnnotationAtts.axes3D.zAxis.title.font.italic = 0
        AnnotationAtts.axes3D.zAxis.title.userTitle = 0
        AnnotationAtts.axes3D.zAxis.title.userUnits = 0
        AnnotationAtts.axes3D.zAxis.title.title = "Z-Axis"
        AnnotationAtts.axes3D.zAxis.title.units = ""
        AnnotationAtts.axes3D.zAxis.label.visible = 1
        AnnotationAtts.axes3D.zAxis.label.font.font = AnnotationAtts.axes3D.zAxis.label.font.Arial  # Arial, Courier, Times
        AnnotationAtts.axes3D.zAxis.label.font.scale = 1
        AnnotationAtts.axes3D.zAxis.label.font.useForegroundColor = 1
        AnnotationAtts.axes3D.zAxis.label.font.color = (0, 0, 0, 255)
        AnnotationAtts.axes3D.zAxis.label.font.bold = 0
        AnnotationAtts.axes3D.zAxis.label.font.italic = 0
        AnnotationAtts.axes3D.zAxis.label.scaling = 0
        AnnotationAtts.axes3D.zAxis.tickMarks.visible = 1
        AnnotationAtts.axes3D.zAxis.tickMarks.majorMinimum = 0
        AnnotationAtts.axes3D.zAxis.tickMarks.majorMaximum = 1
        AnnotationAtts.axes3D.zAxis.tickMarks.minorSpacing = 0.02
        AnnotationAtts.axes3D.zAxis.tickMarks.majorSpacing = 0.2
        AnnotationAtts.axes3D.zAxis.grid = 0
        AnnotationAtts.axes3D.setBBoxLocation = 0
        AnnotationAtts.axes3D.bboxLocation = (0, 1, 0, 1, 0, 1)
        AnnotationAtts.axes3D.triadColor = (0, 0, 0)
        AnnotationAtts.axes3D.triadLineWidth = 0
        AnnotationAtts.axes3D.triadFont = 0
        AnnotationAtts.axes3D.triadBold = 1
        AnnotationAtts.axes3D.triadItalic = 1
        AnnotationAtts.axes3D.triadSetManually = 0
        AnnotationAtts.userInfoFlag = 0
        AnnotationAtts.userInfoFont.font = AnnotationAtts.userInfoFont.Arial  # Arial, Courier, Times
        AnnotationAtts.userInfoFont.scale = 1
        AnnotationAtts.userInfoFont.useForegroundColor = 1
        AnnotationAtts.userInfoFont.color = (0, 0, 0, 255)
        AnnotationAtts.userInfoFont.bold = 0
        AnnotationAtts.userInfoFont.italic = 0
        AnnotationAtts.databaseInfoFlag = 0
        AnnotationAtts.timeInfoFlag = 1
        AnnotationAtts.databaseInfoFont.font = AnnotationAtts.databaseInfoFont.Arial  # Arial, Courier, Times
        AnnotationAtts.databaseInfoFont.scale = 1
        AnnotationAtts.databaseInfoFont.useForegroundColor = 1
        AnnotationAtts.databaseInfoFont.color = (0, 0, 0, 255)
        AnnotationAtts.databaseInfoFont.bold = 0
        AnnotationAtts.databaseInfoFont.italic = 0
        AnnotationAtts.databaseInfoExpansionMode = AnnotationAtts.File  # File, Directory, Full, Smart, SmartDirectory
        AnnotationAtts.databaseInfoTimeScale = 1
        AnnotationAtts.databaseInfoTimeOffset = 0
        AnnotationAtts.legendInfoFlag = 0
        AnnotationAtts.backgroundColor = (255, 255, 255, 255)
        AnnotationAtts.foregroundColor = (0, 0, 0, 255)
        AnnotationAtts.gradientBackgroundStyle = AnnotationAtts.Radial  # TopToBottom, BottomToTop, LeftToRight, RightToLeft, Radial
        AnnotationAtts.gradientColor1 = (0, 0, 255, 255)
        AnnotationAtts.gradientColor2 = (0, 0, 0, 255)
        AnnotationAtts.backgroundMode = AnnotationAtts.Solid  # Solid, Gradient, Image, ImageSphere
        AnnotationAtts.backgroundImage = ""
        AnnotationAtts.imageRepeatX = 1
        AnnotationAtts.imageRepeatY = 1
        AnnotationAtts.axesArray.visible = 1
        AnnotationAtts.axesArray.ticksVisible = 1
        AnnotationAtts.axesArray.autoSetTicks = 1
        AnnotationAtts.axesArray.autoSetScaling = 1
        AnnotationAtts.axesArray.lineWidth = 0
        AnnotationAtts.axesArray.axes.title.visible = 1
        AnnotationAtts.axesArray.axes.title.font.font = AnnotationAtts.axesArray.axes.title.font.Arial  # Arial, Courier, Times
        AnnotationAtts.axesArray.axes.title.font.scale = 1
        AnnotationAtts.axesArray.axes.title.font.useForegroundColor = 1
        AnnotationAtts.axesArray.axes.title.font.color = (0, 0, 0, 255)
        AnnotationAtts.axesArray.axes.title.font.bold = 0
        AnnotationAtts.axesArray.axes.title.font.italic = 0
        AnnotationAtts.axesArray.axes.title.userTitle = 0
        AnnotationAtts.axesArray.axes.title.userUnits = 0
        AnnotationAtts.axesArray.axes.title.title = ""
        AnnotationAtts.axesArray.axes.title.units = ""
        AnnotationAtts.axesArray.axes.label.visible = 1
        AnnotationAtts.axesArray.axes.label.font.font = AnnotationAtts.axesArray.axes.label.font.Arial  # Arial, Courier, Times
        AnnotationAtts.axesArray.axes.label.font.scale = 1
        AnnotationAtts.axesArray.axes.label.font.useForegroundColor = 1
        AnnotationAtts.axesArray.axes.label.font.color = (0, 0, 0, 255)
        AnnotationAtts.axesArray.axes.label.font.bold = 0
        AnnotationAtts.axesArray.axes.label.font.italic = 0
        AnnotationAtts.axesArray.axes.label.scaling = 0
        AnnotationAtts.axesArray.axes.tickMarks.visible = 1
        AnnotationAtts.axesArray.axes.tickMarks.majorMinimum = 0
        AnnotationAtts.axesArray.axes.tickMarks.majorMaximum = 1
        AnnotationAtts.axesArray.axes.tickMarks.minorSpacing = 0.02
        AnnotationAtts.axesArray.axes.tickMarks.majorSpacing = 0.2
        AnnotationAtts.axesArray.axes.grid = 0
        vi.SetAnnotationAttributes(AnnotationAtts)


    # set windows attributes
    SaveWindowAtts = vi.SaveWindowAttributes()
    SaveWindowAtts.outputToCurrentDirectory = 0
    SaveWindowAtts.outputDirectory = output_dir
    #SaveWindowAtts.fileName = "visit"
    SaveWindowAtts.family = 0
    SaveWindowAtts.format = SaveWindowAtts.PNG  # BMP, CURVE, JPEG, OBJ, PNG, POSTSCRIPT, POVRAY, PPM, RGB, STL, TIFF, ULTRA, VTK, PLY, EXR
    SaveWindowAtts.width = 1024
    SaveWindowAtts.height = 1024
    SaveWindowAtts.quality = 80
    vi.SetSaveWindowAttributes(SaveWindowAtts)
    print("window set\n") if Visit_projector_1_log_level > 0 else None



    # States (t's) range
    if State_range_manual: # if it's not []
        State_range = State_range_manual
    else:
        State_range = range(vi.TimeSliderGetNStates())
        startSlide = 0
        tstep = 1
        State_range = range(startSlide, vi.TimeSliderGetNStates(), tstep)

    # check timeslider to troubleshoot 
    if Visit_projector_1_log_level >= 2:
        w = vi.GetWindowInformation()
        if len(w.timeSliders) > 0 and 0 <= w.activeTimeSlider < len(w.timeSliders):
            print(f"Active time slider: {w.timeSliders[w.activeTimeSlider]}")
        else:
            print("WARNING: No active time slider or sliders not initialized yet.")
            print(f"w.timeSliders = {w.timeSliders}")
            print(f"w.activeTimeSlider = {w.activeTimeSlider}")

    print("States = ", State_range) if Visit_projector_1_log_level >= 1 else None

    Image_filenames_VisIt = []
    Times_VisIt = []
    R_SF_Average_VisIt = []

    # Loop once through all states (t's)
    for state in State_range: 
        print("state = ", state) if Visit_projector_1_log_level > 0 else None
        
        # set state
        vi.SetTimeSliderState(state)

        # save window as .png image
        Image_filenames_VisIt_state = f"visit_{state:06}"
        SaveWindowAtts.fileName = Image_filenames_VisIt_state
        vi.SetSaveWindowAttributes(SaveWindowAtts)
        vi.SaveWindow() 
        print(f"saved image for file {state:06d}\n", end='\r') if Visit_projector_1_log_level > 0 else None

        # save properties
        Image_filenames_VisIt.append(Image_filenames_VisIt_state)

        vi.Query("Time")
        Time_state = vi.GetQueryOutputValue()
        Times_VisIt.append(Time_state)

        vi.ChangeActivePlotsVar("R")
        vi.Query("Average Value")
        R_SF_Average_state = vi.GetQueryOutputValue()
        R_SF_Average_VisIt.append(R_SF_Average_state)
        print(f"saved R_Average_state = {R_SF_Average_state} for file {state:06d}\n", end='\r') if Visit_projector_1_log_level > 0 else None


    #################################################### save data

    VisIt_data_df = pd.DataFrame({
        'Plot_VisIt': Plots[0] * len(State_range),
        'Image_filename_VisIt': Image_filenames_VisIt,
        'State_range_VisIt': State_range,
        'Time_VisIt': Times_VisIt,
        'R_SF_Average_VisIt': R_SF_Average_VisIt,
        })
    print(VisIt_data_df)

    # save to pickle
    pkl_filename = f"{output_dir}/Visit_projector_1_data.pkl"
    VisIt_data_df.to_pickle(pkl_filename)
    print(f"Saved extracted DataFrame to {pkl_filename}") if Visit_projector_1_log_level >= 1 else None

    # save to csv
    csv_filename = f"{output_dir}/Visit_projector_1_data.csv"
    VisIt_data_df.to_csv(csv_filename, sep='\t', index=False)
    print(f"Saved extracted DataFrame to {csv_filename}") if Visit_projector_1_log_level >= 1 else None


    # Clean up
    vi.DeleteAllPlots()
    vi.CloseDatabase(r"euler.ethz.ch:/cluster/scratch/cfrouzak/spher_H2/postProc/fields/po_part2/po_s912k_post.nek5000")


    #################################################### return

    return output_dir