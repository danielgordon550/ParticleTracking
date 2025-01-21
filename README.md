These codes are short and sweet methods of performing particle tracking using the trackpy. 

To use these codes you will require the following extra python packages:
- trackpy: the main particle tracking software
- OpenCV: a video analysis software that, to my knowledge, helps open avi videos
- pims: sister package to trackpy for reading videos and images
- MoviePy: subpackage of pims, allows for avi video readings
- numba: A JIT compilier for python that translates script into fast machine code, really speeds things up
- glob: glob is a directory managing and file path package that can pull a wide array of file strings, used liberally in these codes
- pandas: pandas is a package to handle arrays, lists and file reading.
This is not an exhaustive list, these are just packages not commonly installed, refer to imports at the top of the script for more details and keep an
eye out for errors that indicate a missing package.

I use anaconda and spyder to run my code and most of these can be downloaded through the environment handler in the navigator. For me a new environment was required to
install cv2, so be careful. Any packages unavailable through the navigator packages list can be installed manually through conda and anaconda prompt. Make sure you're in
the correct environment when running prompt in this way.


These codes require a specific file structure to analyse videos as follows:

Path to files -> Local Director -> Video1
		                -> Video2
			        -> Video3

The codes create their own output directories giving,

Path to files -> Local Director -> Video1_results
		                -> Video2_results
			        -> Video3_results


The scripts includes are:
- trackparticles: This code does the bulk of the work and loops through all videos in the local directory and tracks the particles. the base version takes
			avi files and reads from that but pims has functions for reading images and all sorts of other formats as required. The output
			is "particletracking.csv" and contains the tracking data.

- plotparticles: This code plots the associated data by exploring the same file structure and finding the .csv files. The base version of this code plots the ensemble and 
			individual MSD plots for all tracks in the videos. The code also plots the particle track superimposed on the final frame of the video.

- dialinparticles: This code is a short check for before running the full trackparticles, allowing you to check if the parameters in the "batch" function are suitable for
			the video you want to track.

- collection: I wrote this code to select all the average_velocities_filtered files from the _results folders created by trackpy and plotparticles and collect them into a 
			csv file.

- velocity_Boxplot: This code takes velocity data over concentrations and plots a velocity box plot to represent them

			As written the code requires a specific file structure:
    
    				Ordered -> 0 -> X_concentration -> Data
            				-> 1 -> Y_concentration -> Data
         				-> 2 -> Z_concentration -> Data
            
			Where X < Y < Z, this plots increasing concentration data.

directory_map: This is just a little something I conjured up to visualise file structures better. It creates an aligned network graph which represents the folder
		structure specifically for the "_results" stuff I've kinda built my things around. This code in particular requires a few unique packages:
			
			- networkx (installed as pynetworkx)
			- a program called graphviz


		


For more information of what is available in trackpy read the documentation at: https://soft-matter.github.io/trackpy/dev/index.html
