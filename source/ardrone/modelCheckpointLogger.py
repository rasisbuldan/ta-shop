import os

dirList = os.listdir('D:/Dataset/ARDrone/ModelCheckpointLog')


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


def getBestMetricsDescList(dir, metrics='valLoss'):
    uniqueTrainTimeList = getUniqueTrainTimeList(dir)
    metricsDescList = getMetricsDescList(dir)
    
    bestMetricsDescList = []
    for trainTime in uniqueTrainTimeList:
        
        minVal = float('inf')
        minMD = {}
        mdList = [md for md in metricsDescList if md['trainTime'] == trainTime]
        
        for md in mdList:
            if md[metrics] < minVal:
                minVal = md[metrics]
                minMD = md
        
        bestMetricsDescList.append(minMD)
    
    return listOfDictSort(bestMetricsDescList, 'trainTime')


# Extras
def listOfDictSort(dictList, metrics='trainTime'):
    return sorted(dictList, key=lambda x: x[metrics])
        

#print(*dirList, sep='\n')
#print(*listTrainTime(dirList), sep='\n')
print(*getBestMetricsDescList(dirList), sep='\n')