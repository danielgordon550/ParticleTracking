import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob 
import os
from pathlib import Path
import matplotlib as mpl

"""
This code takes velocity data over concentrations and plots a velocity box plot to
represent them

As written the code requires a specific file structure:
    
    Ordered -> 0 -> X_concentration -> Data
            -> 1 -> Y_concentration -> Data
            -> 2 -> Z_concentration -> Data
            
Where X < Y < Z, this plots increasing concentration data.
"""


plt.rcParams['figure.figsize'] = [18, 24] 
mpl.rcParams.update({'font.size': 36})



"""
HAVE YOU SET THE RIGHT SCALE
"""
scale = 1/12.07   #micron per pixel

"""
HAVE YOU SET THE RIGHT FPS
"""
fps = 30          #frames per second
    

#for big runs
superfile="F:/PacmenResults/PacmenVel/CollectedData- Redo/Ordered_Full"

#Path to the results files
#pathtofiles ="F:/PacmenResults/PacmenVel/24-12-11/50x_30fps_pB7_0.0001H2O2"


alldata={}

for pathtofiles in sorted(glob.glob(superfile + "/*/")):
    
    alldata_fromfolder=[]
    
    for results_path in glob.glob(pathtofiles + "*/*_results"):
        try:
            
            
            p=Path(results_path)
            sampleID=p.parts[-2]
            
            t2= pd.read_csv(results_path + "/trackparticleoutput.csv")
            average_velocities=pd.read_csv(results_path + '/average_velocities_filtered.csv', header=None, names=['particle', sampleID])
       
        

            alldata_fromfolder.append(average_velocities.iloc[1: , 1])
    
    
    
    
        except ZeroDivisionError:
    
            print(results_path + " : No file")
        
    
        except FileNotFoundError:
    
            print(results_path + " : No file")
            
 
    alldata[sampleID] = pd.concat(alldata_fromfolder, ignore_index=True)
  
combined_df = pd.DataFrame(alldata)
combined_df=combined_df.astype(float)

#velocity_box_plot(combined_df)

fig3, ax = plt.subplots() 
# Create box plot 
box = combined_df.boxplot() 


lwidth=5
# Making the box plot lines thicker
for line in box.get_lines():
    line.set_linewidth(lwidth)

lwidth1=3
# Making the border lines thicker 
box.spines['top'].set_linewidth(lwidth1) 
box.spines['bottom'].set_linewidth(lwidth1) 
box.spines['left'].set_linewidth(lwidth1) 
box.spines['right'].set_linewidth(lwidth1)

# Customizing the plot 
ax.set_xticks([1, 2, 3, 4,5]) 
ax.set_xticklabels(combined_df.columns, rotation=45)
plt.xlabel('Concentration [% $H_2 O_2$]') 
plt.ylabel('Velocity [$\mu$m/s]')
plt.grid(True, linewidth=lwidth1) # Display the plot 
plt.ylim(0,5)
plt.subplots_adjust(bottom=0.2)
plt.show()
plt.tight_layout()


p=Path(superfile)
locale=p.parts[-2]
fig3.savefig(superfile + "/" + locale + "_velocityBoxplot.jpg")







           
            
            
            
            