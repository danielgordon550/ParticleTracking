import pims
import trackpy as tp
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import glob

@pims.pipeline
def gray(image):
    return image[:, :, 1]  # Take just the green channel


# Optionally, tweak styles.
mpl.rc('image', cmap='gray')
mpl.rcParams.update({'font.size': 24})
plt.rcParams["font.family"] = "sans-serif"



file = "F:/PacmenResults/Rheotaxis2/"


frames = gray(pims.open(file))

f = tp.locate(frames[150], 31, invert=True, minmass=5.0e3,  threshold=0.5)
tp.annotate(f, frames[150])
fig, ax = plt.subplots()
ax.hist(f['mass'], bins=20)
#Optionally, label the axes.
ax.set(xlabel='mass', ylabel='count')



