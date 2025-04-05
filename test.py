from cellpose import models, plot, utils, io

import visit
visit.Launch()

import numpy as np
import pandas as pd

print(visit.__file__)


data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}

df = pd.DataFrame(data)
print(df)


import numpy as np
import time, os, sys
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100
from cellpose import utils, io


io.logger_setup()

file = r"C:\Users\obs\Desktop\img02.png"

img = io.imread(file)
plt.figure(figsize=(2,2))
plt.imshow(img)
plt.axis('off')
#plt.show()

from cellpose import models, io

model = models.Cellpose(gpu=True, model_type='cyto3')

channel = [[2,3]]


img = io.imread(file)
masks, flows, styles, diams = model.eval(img, diameter=None, channels=channel)
file = str(file)
print(f"file: {file}")


# save results as png
io.save_to_png(img, masks, flows, file)


from cellpose import plot

fig = plt.figure(figsize=(8,5))
plot.show_segmentation(fig, img, masks, flows[0], channels=channel)
plt.tight_layout()
plt.show()