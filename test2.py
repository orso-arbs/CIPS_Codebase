import os

CP_model_type = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\Python Code\msc python cellpose\CP Models"
if os.path.exists(CP_model_type):
    print("Path exists!")
else:
    print("Path does not exist.")