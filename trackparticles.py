import pims
import trackpy as tp
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
import glob
import os
from pathlib import Path
import numpy as np
import numba


# Optionally, tweak styles.
mpl.rc('image', cmap='gray')
mpl.rcParams.update({'font.size': 24})
plt.rcParams["font.family"] = "sans-serif"



"""
HAVE YOU SET THE RIGHT SCALE
"""
scale =  1/12.07  #micron per pixel

"""
HAVE YOU SET THE RIGHT FPS
"""
fps = 30          #frames per second




@pims.pipeline
def gray(image):
    """
    This is a pipeline function, it coverts the frames read in by pims.open and converts them to greyscale by selecting one channel

    Parameters
    ----------
    image : Frame
        The image frames to be converted.

    Returns
    -------
    image[:,:,1]
        The grey scale version of image.

    """

    return image[:, :, 1]  # Take just the green channel

def batch(frames):
    """
    This section runs tracking look for a batch of frames (i.e. a video) and returns the tracking

    Parameters
    ----------
    frames : ImageSequence
        The frames of the sequence/video to be analysed.

    Returns
    -------
    t2 : DataFrame
        The tracking data of the video.

    """
    
    f= tp.batch(frames[:], 31, minmass=5.0e3, invert=True,  engine='numba', processes=1,  threshold=0.5)
    t = tp.link(f, 20, memory=10)
    t2 = tp.filter_stubs(t,110) #trackpy defaults for MSD and VACF time lags are 100 frames. Cutting trackes less than 110 ensures all tracks > 100
    return t2

def find_Velocities(t):
    """
    This function calculates the velocities of the particles tracks by finding the difference frame by frame.

    Parameters
    ----------
    t : DataFrame
        DataFrame containing the track values for the calulation

    Returns
    -------
    all_velocity_data : DataFrame
        Returns the track data with velocity data concatonated onto the end.

    """
    
    # Group the data by particle
    grouped = t.groupby('particle')
    
    #lists and dictionaries for holding data
    velocity_data = []
    average_velocities = {}
    
    #go through each particle and find the difference in x/y and divide by difference in frame (dy/dx)
    for particle, group in grouped:
        group = group.sort_values(by='frame')
        group['vx'] = group['x'].diff() / group['frame'].diff()
        group['vy'] = group['y'].diff() / group['frame'].diff()
        group['v'] = np.sqrt(group['vy']**2 + group['vx']**2)
        
        group['vx'] = group['vx'] * scale * fps     #correct units for velocity (microns per pixel)
        group['vy'] = group['vy'] * scale * fps
        group['v'] = group['v'] * scale * fps
        
        velocity_data.append(group)
        
        #average velocities are calculated through the mean
        average_vx = np.mean(group['vx'].dropna())
        average_vy = np.mean(group['vy'].dropna())
        average_v = np.sqrt( average_vx**2 + average_vy**2 ) 
        average_velocities[particle] = (particle, average_v)
        
        
    # Combine all the velocity data
    all_velocity_data = pd.concat(velocity_data)

    average_velocities= pd.DataFrame(average_velocities).T
    
    return all_velocity_data, average_velocities



filelocation="F:/PacmenResults/Rheotaxis2"

#folder= "F:/PacmenResults/amirPacman2"


for folder in glob.glob(filelocation + '/*'):
    for file in glob.glob(folder + "/*.avi"):
       
      
       
       #file=glob.glob(folder + "/*.avi")[3]
       print(file)
       frames = gray(pims.open(file))
       tp.quiet(suppress=False) #False to see frame numbers printed, True to remove them
       
       t2=batch(frames)
       
       #output file formatting for use in trackparticles.py
       t2= t2.drop(['mass','size','ecc','signal','raw_mass','ep'], axis=1)
       t2= t2.reset_index(drop=True)
       t2= t2.sort_values(['particle','frame'])
       
       p=Path(file)
       sampleID=p.parts[-1]
       results_path = folder+ "/" + sampleID + "_results"
       os.makedirs(results_path, exist_ok=True)
       
       with open(results_path + '/sampleID.txt', 'w') as file:
       # Write the variable content to the file
           file.write(sampleID)
       
       
       #Calculate the velocities, add to the track data, save files
       tracksandvelocities, average_velocities = find_Velocities(t2)
       tracksandvelocities.to_csv(results_path + '/trackparticleoutput.csv', index=False)
       average_velocities.to_csv(results_path + '/particle_average_velocities.csv', header=False, index=False)
