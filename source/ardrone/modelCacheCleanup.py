'''
    Vibration Modeling Cache Cleanup
    
    Background:
        Single cache file can reach ~300MB
        Accumulated training checkpoint cache can reach ~200GB on local disk space
    
    Action:
        Select cache file with every train time (logged in 'Vibration Modelling Log')
        Only save 2-3 best weight, remove all remaining bad metrics
'''

import os

cacheDir = 'D:/Dataset/ARDrone/ModelCheckpointLog'

dirList = os.listdir(cacheDir)

print(len(dirList))