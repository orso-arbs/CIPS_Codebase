import visit as vi
vi.Launch() # loads rest of visit functions
import visit as vi # loads rest of visit functions


### from visit GUI commands deduced

vi.OpenDatabase(r"euler.ethz.ch:/cluster/scratch/cfrouzak/spher_H2/postProc/fields/po_part2/po_s912k_post.nek5000", 0)
vi.AddPlot("Contour", "temperature", 1, 1)
vi.DrawPlots()
# Begin spontaneous state
View3DAtts = vi.View3DAttributes()
View3DAtts.viewNormal = (-0.313864, -0.493751, 0.810987)
View3DAtts.focus = (0, 0, 0)
View3DAtts.viewUp = (-0.0333566, 0.859356, 0.510289)
View3DAtts.viewAngle = 30
View3DAtts.parallelScale = 173.205
View3DAtts.nearPlane = -346.41
View3DAtts.farPlane = 346.41
View3DAtts.imagePan = (0, 0)
View3DAtts.imageZoom = 1
View3DAtts.perspective = 1
View3DAtts.eyeAngle = 2
View3DAtts.centerOfRotationSet = 0
View3DAtts.centerOfRotation = (0, 0, 0)
View3DAtts.axis3DScaleFlag = 0
View3DAtts.axis3DScales = (1, 1, 1)
View3DAtts.shear = (0, 0, 1)
View3DAtts.windowValid = 1
vi.SetView3D(View3DAtts)
# End spontaneous state


SaveWindowAtts = vi.SaveWindowAttributes()
SaveWindowAtts.outputToCurrentDirectory = 0
SaveWindowAtts.outputDirectory = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VisitOutput"
SaveWindowAtts.fileName = "visit"
SaveWindowAtts.family = 1
SaveWindowAtts.format = SaveWindowAtts.PNG  # BMP, CURVE, JPEG, OBJ, PNG, POSTSCRIPT, POVRAY, PPM, RGB, STL, TIFF, ULTRA, VTK, PLY, EXR
SaveWindowAtts.width = 1024
SaveWindowAtts.height = 1024
SaveWindowAtts.screenCapture = 0
SaveWindowAtts.saveTiled = 0
SaveWindowAtts.quality = 80
SaveWindowAtts.progressive = 0
SaveWindowAtts.binary = 0
SaveWindowAtts.stereo = 0
SaveWindowAtts.compression = SaveWindowAtts.NONE  # NONE, PackBits, Jpeg, Deflate, LZW
SaveWindowAtts.forceMerge = 0
SaveWindowAtts.resConstraint = SaveWindowAtts.ScreenProportions  # NoConstraint, EqualWidthHeight, ScreenProportions
SaveWindowAtts.pixelData = 1
SaveWindowAtts.advancedMultiWindowSave = 0
SaveWindowAtts.subWindowAtts.win1.position = (0, 0)
SaveWindowAtts.subWindowAtts.win1.size = (128, 128)
SaveWindowAtts.subWindowAtts.win1.layer = 0
SaveWindowAtts.subWindowAtts.win1.transparency = 0
SaveWindowAtts.subWindowAtts.win1.omitWindow = 0
SaveWindowAtts.opts.types = ()
SaveWindowAtts.opts.help = ""
vi.SetSaveWindowAttributes(SaveWindowAtts)
vi.SaveWindow()
