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


"""
HAVE YOU SET THE RIGHT SCALE
"""
scale = 1/12.07   #micron per pixel

"""
HAVE YOU SET THE RIGHT FPS
"""
fps = 30          #frames per second




def plot_loop(t2, frames, results_path, sampleID):
    """
    The plot loop takes the tracking data and plots track coloured by frame

    Parameters
    ----------
    t2 : DataFrame
        The tracking data of the video.

    Returns
    -------
    None.

    """
    
    fig2, ax2 = plt.subplots()
    ax2.imshow(frames[-1], cmap='gray')
    
    # Define the colormap and normalize
    cmap = mpl.cm.inferno
    norm = mpl.colors.Normalize(vmin=t2['frame'].min(), vmax=t2['frame'].max())
    
    cmap = mpl.cm.inferno
    norm = mpl.colors.Normalize(vmin=0, vmax=len(frames))
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) 
    
    
    # Iterate through each particle's trajectory
    for particle, traj in t2.groupby('particle'):
        # Get the colors based on the frame number
        colors = cmap(norm(traj['frame'].values))

        # Plot the trajectory with the specified linewidth and color for each segment
        for i in range(len(traj) - 1):
            ax2.plot(traj.iloc[i:i + 2]['x'], traj.iloc[i:i + 2]['y'],
                    color=colors[i], linewidth=2)
            

        ax2.text(traj.iloc[-1]['x'] - 30, traj.iloc[-1]['y'] - 30, str(particle), fontsize=24, ha='right', color='blue')
        
    
    
    ax2.set_title(sampleID)
    
    ax2.invert_yaxis()
    ax2.set_xlabel("x [$\mu$m]")
    ax2.set_ylabel("y [$\mu$m]")

    
    fig2.savefig(results_path + "/" + sampleID + "_trackplot.jpg")
    
def eMSD(t2, filename, sampleID):
    """
    Calculates and plots the MSD function using trackpy.emsd (ensemble MSD)

    Parameters
    ----------
    t2 : DataFrame
        Tracking data.
    filename : str
        File path for saving the resulting graph.
    savestring : str
        The name for the resulting file, set to the components of the file path.
    sampleID: str
        String added in just to give the plot a short clear name when looking visually

    Returns
    -------
    em : Series
        Data series containing the msd plot.

    """

    em = tp.emsd(t2, scale, fps) # microns per pixel , frames per second 
    fig, ax = plt.subplots()
    ax.plot(em.index, em, 'o')
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]', xlabel='lag time $t$')
    ax.set_title(sampleID)
    fig.savefig(filename + "/" + sampleID + "_ensembleMSDplot.jpg")
    
    return em
    
def iMSD(t2, filename, sampleID):
    """
    Calculates and plots the MSD function using trackpy.imsd (individual MSD)

    Parameters
    ----------
    t2 : DataFrame
        Tracking data.
    filename : str
        File path for saving the resulting graph.
    sampleID: str
        String added in just to give the plot a short clear name when looking visually

    Returns
    -------
    em : Series
        Data series containing the msd plot.

    """

    im = tp.imsd(t2, scale, fps) # microns per pixel , frames per second 
    fig, ax = plt.subplots()
    for column in im.columns:
        ax.plot(im.index, im[column],  'o', label=im[column].name)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]', xlabel='lag time $t$')
    ax.set_title(sampleID)
    fig.savefig(filename + "/" + sampleID + "_individualMSDplot.jpg")
    
    return im

def v_corr(all_velocity_data, filename, sampleID):
    """
    Parameters
    ----------
    all_velocity_data : DataFrame
        The tracking data frame with velocities attached
    filename : String
        The file path for saving the graph
    sampleID : String
        filename for labelling

    Returns
    -------
    vacf_Frame : DataFrame
        DataFrame containing the velocity autocorrelation function data

    """
    

    # Maximum time lag
    max_lag = 100  # set to match trackpy default
    
    # Dictionary to store VACF of each particle
    vacf_dict = {}
    
    #Group data into each particle if its not done already
    grouped = all_velocity_data.groupby('particle')
    for particle, frame_data in grouped:
        
        #Take the velocity data and prepare for calculation (remove nans etc.)
        velocities = frame_data[['vx', 'vy']].values
        n = len(velocities)
        velocities = velocities[~np.isnan(velocities).any(axis=1)]
        particle_vacf = []

        #For frames OVER the time lag chosen find the dot product then calculate the VACF
        for lag in range(1, max_lag + 1):
            if n > lag:
                v_dot = np.sum(velocities[:-lag] * velocities[lag:], axis=1)
                autocorrelation = np.mean(v_dot)
                particle_vacf.append(autocorrelation)

        # Normalize if needed
        particle_vacf = np.array(particle_vacf)
        particle_vacf /= particle_vacf[0]

        # Store in dictionary
        vacf_dict[particle] = particle_vacf


    # Create DataFrame
    vacf_Frame = pd.DataFrame(vacf_dict)
    vacf_Frame.index = vacf_Frame.index * 1/fps
    
    # Plotting
    fig3, ax3 = plt.subplots() 
    for column in vacf_Frame.columns:
        ax3.plot(vacf_Frame.index[::30], vacf_Frame[column][::30], label=vacf_Frame[column].name)


    ax3.set(ylabel="VACF", xlabel='lag time $t$')
    ax3.set_title(sampleID)
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig3.tight_layout()
    fig3.savefig(filename + "/" + sampleID + "_VACF.jpg")
    
    return vacf_dict

def plot_Velocities(velocity_data, avg_vel_data, filename, sampleID):
    
    # Find the top 3 particles with the highest average velocity
    average_velocities = dict(zip(avg_vel_data.iloc[:, 0], avg_vel_data.iloc[:, 1]))
    # Plotting the total velocity data for the top 3 particles
    fig, ax = plt.subplots()
    
    fig2, ax2 = plt.subplots()
    
    # Sort particles by average velocity and select the top 3
    top_particles = sorted(average_velocities, key=average_velocities.get, reverse=True)[:3]
    
    # Group the position and velocity data by particle
    grouped = velocity_data.groupby('particle')
    
    
    
    for particle, group in grouped:
        if particle in top_particles:
            group = group.sort_values(by='frame')
            
            #group['v'] = group['v'].interpolate(method='linear').dropna() 
            ax.plot(group['frame'][::10], group['v'][::10], label=f'Particle {particle} (v_avg={format(average_velocities[particle], ".2g")})')

            group['rolling_Avg'] = group['v'].rolling(window=10).mean()
         
            
            ax2.plot(group['frame'][::10], group['rolling_Avg'][::10])


    #ax.set_ylim(-1, 5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Total Velocity')
    ax.set_title(sampleID)
    ax.legend()
    fig.savefig(filename + "/" + sampleID + "_velocityplot.jpg")
    
    
    #ax2.set_ylim(-1, 5)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Velocity (rolling=10)')
    ax2.set_title(sampleID)
    ax2.legend()
    fig2.savefig(filename + "/" + sampleID + "_velocityplot_rollingaverage.jpg")
    
    
    return velocity_data

def filter_tracking_data(tracking_df, velocity_df, threshold=1):
    # Step 1: Get the columns from MSD DataFrame with final value greater than threshold
    msd_df = tp.imsd(tracking_df, scale, fps)
      
   # Calculate the average second derivative for each particle in MSD data 
    average_second_derivatives = {} 
    for particle in msd_df.columns[:]:
        msd_values = msd_df[particle] 
        second_derivative = np.gradient(np.gradient(msd_values)) 
        average_second_derivative = np.mean(second_derivative) 
        average_second_derivatives[particle] = average_second_derivative 
    
    # Create a DataFrame to store average second derivatives 
    average_second_derivative_df = pd.DataFrame(list(average_second_derivatives.items()), columns=['particle', 'average_second_derivative']) 
    
    # Find particles with an average second derivative greater than 0 
    positive_particles = average_second_derivative_df[average_second_derivative_df['average_second_derivative'] > 0]['particle'] 
    
    # Filter tracking data to include only particles with an average second derivative greater than 0 
    derivative_filtered_tracking_df = tracking_df[tracking_df['particle'].isin(positive_particles)] 
    derivative_filtered_velocity_df = velocity_df[velocity_df['particle'].isin(positive_particles)]
    
    
    #Final MSD value must be greater than a threshhold to filter stuck particles that evade dy2 MSD check
    final_msd_values = msd_df.iloc[-1]
    msd_filtered_columns = final_msd_values[final_msd_values > threshold].index
    
    filtered_tracking_df = derivative_filtered_tracking_df[derivative_filtered_tracking_df['particle'].isin(msd_filtered_columns)]
    filtered_velocity_df = derivative_filtered_velocity_df[derivative_filtered_velocity_df['particle'].isin(msd_filtered_columns)]
    
    return filtered_tracking_df, filtered_velocity_df

def filter_velocity_data(tracking_df, velocity_df, threshold=1):
    # Step 1: Get the columns from MSD DataFrame with final value greater than threshold
    msd_df = tp.imsd(tracking_df, scale, fps)
    final_msd_values = msd_df.iloc[-1]
    filtered_columns = final_msd_values[final_msd_values > threshold].index
    
    # Step 2: Filter the tracking DataFrame to include only rows with these particle IDs
    filtered_velocity_df = velocity_df[velocity_df['particle'].isin(filtered_columns)]
    
    return filtered_velocity_df


#for big runs
superfile="F:/PacmenResults/Rheotaxis2"

#Path to the results files
#pathtofiles ="F:/PacmenResults/amirPacman2"

for pathtofiles in glob.glob(superfile + "/*"):
    for results_path in glob.glob(pathtofiles + "/*_results"):
        
        try:
            #open data files. Need the sampleID, frames, and trackdata
            with open(results_path + '/sampleID.txt', 'r') as file:
                sampleID = file.read()
            filename = pathtofiles + "/" + sampleID
            
            t2= pd.read_csv(results_path + "/trackparticleoutput.csv")
            average_velocities=pd.read_csv(results_path + '/particle_average_velocities.csv', header=None, names=['particle', 'average_velocity'])
            frames = pims.open(filename)      
            
    
    
    
            #Unfiltered plotting
            plot_loop(t2, frames, results_path, sampleID+"_unfiltered")
            imsd=iMSD(t2, results_path, sampleID+"_unfiltered")
            
            
            filtered_data, filter_velocity=filter_tracking_data(t2, average_velocities, threshold=1)
            #Filtered plotting
            if filtered_data.empty:
                #Finish and save
                print(filename + " : No data (filtered)")
         
                
                #plt.show()
                continue
            else:
                plot_loop(filtered_data, frames, results_path, sampleID+"_filtered")
                v_data= plot_Velocities(filtered_data, average_velocities, results_path, sampleID+"_filtered")
                vacf = v_corr(filtered_data, results_path, sampleID+"_filtered")
                msd=eMSD(filtered_data, results_path, sampleID+"_filtered")
                imsd=iMSD(filtered_data, results_path, sampleID+"_filtered")
                
                
                # filter_velocity = filter_velocity_data(t2, average_velocities, threshold=1)
                filter_velocity.to_csv(results_path + '/average_velocities_filtered.csv', index=False)
                
                print(filename + " : Done")
    
            
    
            
        except ZeroDivisionError:
    
            print(filename + " : No file")
        
    
        except FileNotFoundError:
    
            print(filename + " : No file")