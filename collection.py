import trackpy as tp
import pims
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numba 
from matplotlib.ticker import FormatStrFormatter
import timeit
import glob
from pathlib import Path

plt.rcParams['figure.figsize'] = [12, 8] 
mpl.rcParams.update({'font.size': 24})




results_path= "F:/PacmenResults/PacmenVel/25-01-15"



# for all results in experiment resilts folder:
#     go through each of the results files
        
#         read in average velocity data
#         compile into list
        
#         source - particle number - value

all_data=[]
        
        
for folder in glob.glob(results_path + "/*"):
    for data_file in glob.glob(folder + "/*_results"):
        
        try:
            data = pd.read_csv(data_file + "/average_velocities_filtered.csv")
            
            print(data_file)
    
            data['Source File'] = data_file  # Add a column for the source file
            all_data.append(data)
            
        except FileNotFoundError:
            
            print( data_file + ": No file")
       
# Concatenate all data
all_data_df = pd.concat(all_data, ignore_index=True)    
all_data_df = all_data_df.sort_values(by=all_data_df.columns[2])


plt.hist(all_data_df['average_velocity'], bins=30)
plt.xlabel('Velocity') 
plt.ylabel('Frequency') 
plt.title(results_path) # Save the histogram to a file

plt.savefig(results_path + '/particle_velocities_histogram.jpg')
all_data_df.to_csv(results_path + '/collected_velocities.csv', index=False)









