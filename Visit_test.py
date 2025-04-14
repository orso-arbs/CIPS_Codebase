import sys
sys.path.append(r"C:\Users\obs\LLNL\VisIt3.4.2\lib\site-packages")
import visit as vi
vi.Launch() # loads rest of visit functions
print("launched visit")
import visit as vi # loads rest of visit functions
print("imported visit \n")


### from visit GUI commands deduced
# vi.OpenComputeEngine("euler.ethz.ch", ("-l", "srun", "-np", "2", "-nn", "1", "-t", "1:00:00"))


#'''
print(
"if prompted, please confirm visit window: 'Select options for 'euler.ethz.ch' with 'ok'\n" \
"If prompted, please provide Euler password\n" \
"Then wait for euler to allocate resources for the job\n" \
)
p = vi.GetMachineProfile("euler.ethz.ch")
#print(p)

p.userName="orsob"
p.activeProfile = 1
p.GetLaunchProfiles(1).numProcessors = 4
p.GetLaunchProfiles(1).numNodes = 1
p.GetLaunchProfiles(1).timeLimit = "00:30:00"
vi.OpenComputeEngine(p)
#'''

print("launched compute engine \n")

Database = r"euler.ethz.ch:/cluster/scratch/cfrouzak/spher_H2/postProc/fields/po_part2/po_s912k_post.nek5000"
Database = r"euler.ethz.ch:/cluster/scratch/orsob/MastersThesis/postProc/po_part1/po_s912k_post.nek5000"

vi.OpenDatabase(Database)
print("Opened Database\n")


vi.AddPlot("Contour", "temperature", 1, 1)
print("Added plot\n")

vi.DrawPlots()
print("Drawed Plots\n")

# set view
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
print("view set\n")


# set windows attributes
SaveWindowAtts = vi.SaveWindowAttributes()
SaveWindowAtts.outputToCurrentDirectory = 0
SaveWindowAtts.outputDirectory = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\VisitOutput"
SaveWindowAtts.fileName = "visit"
SaveWindowAtts.family = 0
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
print("window set\n")




# Iterate through time and save window images

# Do time query
# startSlide = 0
# tstep = 1
# for i in range(startSlide, TimeSliderGetNStates(), tstep):
#     SetTimeSliderState(i)

#     Query("Time")
#     t = GetQueryOutputValue()
#     Query("Average Value")
#     vel_mag_avg = GetQueryOutputValue()
#     Query("Total Length")
#     length = GetQueryOutputValue()

States = range(vi.TimeSliderGetNStates())

startSlide = 0
tstep = 1
States = range(startSlide, vi.TimeSliderGetNStates(), tstep)

w = vi.GetWindowInformation()
if len(w.timeSliders) > 0 and 0 <= w.activeTimeSlider < len(w.timeSliders):
    print(f"Active time slider: {w.timeSliders[w.activeTimeSlider]}")
else:
    print("WARNING: No active time slider or sliders not initialized yet.")
    print(f"w.timeSliders = {w.timeSliders}")
    print(f"w.activeTimeSlider = {w.activeTimeSlider}")

print("States = ", States)
for state in States: 
    vi.SetTimeSliderState(state) 
    SaveWindowAtts.fileName = f"visit_{state:04d}"
    vi.SetSaveWindowAttributes(SaveWindowAtts)
    
    vi.SaveWindow() 
    print(f"saved image for file {state:04d}\n")



# States = range(vi.TimeSliderGetNStates())
# print("States = ", States)

# for state in States: 
#     vi.TimeSliderSetState(state) 
#     SaveWindowAtts.fileName = f"visit_{state:04d}"
#     vi.SetSaveWindowAttributes(SaveWindowAtts)
#     vi.SaveWindow() 
#     print(f"saved image for file {state:04d}\n")


#vi.SaveWindow()















# Clean up
vi.DeleteAllPlots()
vi.CloseDatabase(r"euler.ethz.ch:/cluster/scratch/cfrouzak/spher_H2/postProc/fields/po_part2/po_s912k_post.nek5000")


print("\ncode completely executed\n")