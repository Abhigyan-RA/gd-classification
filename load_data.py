### Produced by Carolyn J. Swinney and John C. Woods for use with the 'DroneDetect' dataset ###

import os
import numpy as np
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal

file = "/path/to/file/file.dat_"                            # path to file location
f = open(file, "rb")                                        # open file
data = np.fromfile(f, dtype="float32",count=240000000)      # read the data into numpy array
data = data.astype(np.float32).view(np.complex64)           # view as complex
data_norm = (data-np.mean(data))/(np.sqrt(np.var(data)))    # normalise
newarr = np.array_split(data_norm, 100)	                    # split the array, 100 will equate to a sample length of 20ms


i=0	      						    # initialise counter
while i < 100:					            # loop through each split
    e = newarr[i]
    print(e)
     
# inside this loop you can save new smaller files or produce graphs

    i=i+1


