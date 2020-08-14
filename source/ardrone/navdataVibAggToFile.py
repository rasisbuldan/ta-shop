from navdataVibProc import NavData, VibData, NavdataVib
import numpy as np
import scipy.stats
import math
import sys
import random
import traceback

#############################
##### Parameter Options #####
#############################

fileName = 'seq_agg_1k1s_aug14'
dumpPath = 'D:/Cloud/Google Drive/Tugas Akhir/data/dumps/seq_agg'

numOfStep = 6
nAggregates = 1000
timeWindow = 1000   # ms



#############################
###### Data Filtering #######
#############################

# Include substring
includeDesc = ['aug9_0_hover30s', 'aug10', 'aug11']

# Exclude subsstring
excludeDesc = ['crash', 'fail', 'test', '10s']

# Discard specific filename
""" discardDesc = [
    'aug10_testfail.json',
    'aug10_1_test.json',
    'aug10_1_test_2.json',
    'aug10_2_hover20s_4_crash.json',
    'aug10_3_hover20s_fail.json',
    'aug11_4_hover20s_2_crash.json',
] """

discardDesc = []



#############################
###### Initialization #######
#############################

Nav = NavData()
Vib = VibData()
NV = NavdataVib(verbose=False)

# Get description list
descList = Nav.listDescription()
print('Description List:')
print(*descList, sep='\n')
#sys.exit()



#############################
###### Helper Function ######
#############################

def parseDescription(desc):
    '''
        [!] Current implementation: group number --> imbalance step count

        Parse description with '_' separator
        [0] : date
        [1] : group number ([!] step count)
        [2] : maneuver description
        [3] : attempt number

    '''
    ds = desc.split('_')
    dsDict = {
        'date': ds[0],
        'stepCount': int(ds[1]),
        'maneuver': ds[2],
        'attemptNum': int(ds[3].replace('.json',''))
    }

    return dsDict


def getFeature(array):
    '''
        Calculate time domain features from *array* input
        Features:
        - RMS
        - Kurtosis
        - Skewness
        - Crest Factor
        - Peak-to-peak

        return: dict {
            'rms': ...,
            'kurtosis': ...,
            'skewness': ...,
            'crest-factor': ...,
            'peak-to-peak': ...,
        }
    '''
    featureValue = {
        'rms': math.sqrt(np.mean(np.square(array))),
        'kurtosis': scipy.stats.kurtosis(array),
        'skewness': scipy.stats.skew(array),
        'crest-factor': (max(array)/math.sqrt(np.mean(np.square(array)))),
        'peak-to-peak': (max(array) - min(array)),
    }

    return featureValue



#############################
####### Main Program ########
#############################

# Create new filedata array grouped per 
datasetArrayPerStep = []
for _ in range(numOfStep+1):
    datasetArrayPerStep.append([])

# Description traversal
for desc in descList:
    '''
        Description filtering (sorted by priority)
        - Discard: specific filename in *discardDesc*
        - Include: substring in *includeDesc*
        - Exclude: substring in *excludeDesc*
    '''
    if ((desc in discardDesc) 
            or not any(fd in desc for fd in includeDesc)
            or any(fd in desc for fd in excludeDesc)):
        
        continue

    #print('Processed:', desc)
    
    stepIdx = parseDescription(desc)['stepCount']
    datasetArrayPerStep[stepIdx].append(desc)


#print(*dataArrayPerStep, sep='\n')

stepDatasetArray = []

# Step count file set traversal
stepDatasetNum = 0
for stepDataset in datasetArrayPerStep:
    stepDatasetNum += 1

    stepDataArray = []

    # Single file traversal
    stepNum = 0
    for stepData in stepDataset:
        stepNum += 1
        print('[D({}/{}) F({}/{})] Processing {}...'.format(stepDatasetNum, len(datasetArrayPerStep), stepNum, len(stepDataset), stepData))
        
        # Get timestamp range
        tsStart, tsStop = Nav.getTimestampRangeByDescription(
            description=stepData,
            landed=True,
        )

        # Get vibration data
        vibData = Vib.getBetweenTimestamp(
            lower=tsStart,
            upper=tsStop
        )

        ##### CHECKPOINT 1 - Vibdata Empty #####
        if len(vibData) == 0:
            print('--> [!] Vibdata empty, skipping...')
            continue

        # Store navdata to local active data
        navId = Nav.storeData(
            data_array=Nav.getByDescription(
                description=stepData,
                landed=False),
            description=desc
        )

        # Store vibdata to local active data
        vibId = Vib.storeData(
            data_array=vibData,
            description=desc
        )

        # Combine Array
        combinedData = NV.combineDataMultiFeature(
            navdata=Nav.getMultiFeatureArray(navId),
            vibdata=Vib.getMultiFeatureArray(vibId)
        )

        # Aggregate Combined Arrya
        combinedAggData = NV.aggregateCombined(
            combined_data=combinedData,
            time_window=timeWindow
        )

        for dataIdx in range(len(combinedAggData)):
            combinedAggData[dataIdx]['desc'] = stepData
            combinedAggData[dataIdx]['timestart'] = combinedData[0]['timestamp']
            combinedAggData[dataIdx]['aggNum'] = dataIdx

        # Insert into step data array
        stepDataArray += combinedAggData

    # Insert into step dataset array
    #stepDatasetArray += stepDataArray

    # Convert into numpy array
    print('Converting to numpy array!')
    dataArray = np.array([]).reshape(0,35)
    dataNum = 0
    for data in stepDataArray:
        dataNum += 1
        hashCount = dataNum//(len(stepDataArray)//50)
        print('[' + ('#' * hashCount) + ('-' * (50-hashCount)) + '] - {:.1f}%'.format(dataNum/len(stepDataArray)*100), end='\r')

        dataRow = np.array([
            data['desc'],
            data['aggNum'],
            data['timestart'],
            data['mot1']['pwm'],
            *data['mot1']['rms'],
            *data['mot1']['kurtosis'],
            *data['mot1']['skewness'],
            *data['mot1']['crest-factor'],
            *data['mot1']['peak-to-peak'],
            data['mot2']['pwm'],
            *data['mot2']['rms'],
            *data['mot2']['kurtosis'],
            *data['mot2']['skewness'],
            *data['mot2']['crest-factor'],
            *data['mot2']['peak-to-peak'],
        ]).reshape(1,35)

        dataArray = np.append(dataArray, dataRow, axis=0)

    print('Converting to numpy array complete!')

    # Export to csv (using numpy)
    print('Exporting to dump/' + fileName + '_' + str(stepDatasetNum-1) + '.csv')
    np.savetxt(
        fname=dumpPath + '/' + fileName + '_' + str(stepDatasetNum-1) + '.csv',
        X=dataArray,
        fmt='%s',
        delimiter=',', 
        header='desc,aggNum,timestart,pwm1,rms1x,rms1y,rms1z,kurt1x,kurt1y,kurt1z,skew1x,skew1y,skew1z,crest1x,crest1y,crest1z,peak1x,peak1y,peak1z,pwm2,rms2x,rms2y,rms2z,kurt2x,kurt2y,kurt2z,skew2x,skew2y,skew2z,crest2x,crest2y,crest2z,peak2x,peak2y,peak2z'
    )
    print('------------------------')
