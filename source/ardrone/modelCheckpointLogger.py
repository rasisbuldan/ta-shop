
'''
    Vibration Modeling Logger & Cache Cleanup
    
    Background:
        Single cache file can reach ~300MB
        Accumulated training checkpoint cache can reach ~200GB on local disk space
    
    Action:
        Select cache file with every train time (logged in 'Vibration Modelling Log')
        Only save 2-3 best weight, remove all remaining bad metrics
'''

import os
import glob
import sys

dirPath = 'D:/Dataset/ARDrone/ModelCheckpointLog'
dirList = os.listdir(dirPath)


# File level processing
def getTrainTime(filename):
    return filename[:17]

def getMetricsDesc(filename):
    fn = filename.split('_')
    mdString = fn[-1].split('-')
    return {
        'trainTime': '_'.join(fn[:6]),
        'epoch': int(mdString[0]),
        'loss': float(mdString[1]),
        'valLoss': float(mdString[2].replace('.h5',''))
    }


# Directory level processing
def getUniqueTrainTimeList(dir):
    return list(set([getTrainTime(filename) for filename in dir]))


def getMetricsDescList(dir):
    return [getMetricsDesc(filename) for filename in dir]


def getBestMetricsDescList(dir, metrics='valLoss', n_best=3):
    uniqueTrainTimeList = getUniqueTrainTimeList(dir)
    metricsDescList = getMetricsDescList(dir)
    
    bestMetricsDescList = []
    for trainTime in uniqueTrainTimeList:

        nData = len([md for md in metricsDescList if md['trainTime'] == trainTime])
        
        for _ in range(min(n_best, nData)):

            minVal = float('inf')
            minMD = {}
            mdList = [md for md in metricsDescList if md['trainTime'] == trainTime]
            
            for md in mdList:
                if md[metrics] < minVal:
                    minVal = md[metrics]
                    minMD = md
            
            #print(md)
            metricsDescList.remove(md)
            bestMetricsDescList.append(minMD)
    
    cleanupCache(metricsDescList)
    return listOfDictSort(bestMetricsDescList, 'trainTime')



def cleanupCache(metrics_desc_list):
    '''
        Delete cache with description listed in *metrics_desc_list*
        Delete on occurence trainTime + epoch only (must be unique)
    '''
    
    nData = len(metrics_desc_list)
    i = 0

    for metricsDesc in metrics_desc_list:
        i += 1
        print('Removing File [{}/{}] - {:.1f}%'.format(i, nData, 100*i/nData), end='\r')
        filePattern = dirPath + '/' + metricsDesc['trainTime'] + '_' + '{:03d}'.format(metricsDesc['epoch']) + '*.h5'
        fileList = glob.glob(filePattern)

        for filePath in fileList:
            os.remove(filePath)
        

    print('Number of files to delete:', i)


# Extras
def listOfDictSort(dictList, metrics='trainTime'):
    return sorted(dictList, key=lambda x: x[metrics])
        

#print(*dirList, sep='\n')
#print(*listTrainTime(dirList), sep='\n')
bestMetricsDescList = getBestMetricsDescList(dirList, n_best=2)
