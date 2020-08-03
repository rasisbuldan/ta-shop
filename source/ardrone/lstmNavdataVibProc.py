# Data processing
import numpy as np
import sys
import matplotlib.pyplot as plt
from random import random, randint, seed

# LSTM
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from navdataVibProc import VibData, NavData, NavdataVib


### Parameter ###
queryDescription = "jul29_2_hover30s_1.json"
plotVibAxis = ['x','y','z']
stepWeight = 0.1
windowSize = 200
weight = np.arange(stepWeight, stepWeight*(windowSize+1), stepWeight)
discardDesc = ["jul29_up_down_02.json"]
filterDesc = ['jul29']

### Dataset split ###
nTrain = 10
nTest = 3
trainDataset = np.empty((0,4))
testDataset = np.empty((0,4))

# Reproducibility
seed(1234)

### Get Dataset ###
# Vibibibhelofrenaijeswantutelyudetaiemsopritiihiqtengkyu:>
Vib = VibData()
Nav = NavData()
NV = NavdataVib()

# List description
descArray = [nav['description'] for nav in Nav.listDescriptionTimestamp()[1:]]

datasetCombined = []

for desc in descArray:
    # Data filtering
    # skipping substring in *discardDesc* or *filterDesc* existed in *desc* string
    if (desc in discardDesc) or not all(fd in desc for fd in filterDesc):
        print('--------\nFiltered data:', desc)
        continue
    print('--------\nProcessing:', desc)
    
    # Get navdata by description *desc*
    navdataArr = Nav.getByDescription(
        description=desc,
        landed=True
    )

    # Get timestamp range value
    tsMin, tsMax = Nav.getTimestampRangeByDescription(
        description=desc,
        landed=True
    )

    # Get vibdata by timestamp range
    vibdataArr = Vib.getBetweenTimestamp(tsMin, tsMax)
    
    # Conditional: missing data 
    if len(navdataArr) <= 0:
        print('Skipping dataset (navdata empty)..')
        continue

    if len(vibdataArr) <= 0:
        print('Skipping dataset (vibdata empty)..')
        continue
    
    # Store navdata by description
    navId = Nav.storeData(
        data_array=navdataArr,
        description=desc
    )

    # Store vibdata by timestamp range
    vibId = Vib.storeData(
        data_array=vibdataArr,
        description=desc
    )

    # Combine data
    aggArray = NV.combineAggregatedArray(
        navdata_agg=Nav.getPWMAggregatedArray(navId, windowSize),
        vibdata_agg = Vib.getVibAggregatedArray(vibId, windowSize),
        time_window=windowSize
    )

    print('Processed data with id: {}|{} with size {}|{} => {}'.format(
        navId,
        vibId,
        len(Nav.activeDataArray[navId]['data']),
        len(Vib.activeDataArray[vibId]['data']),
        len(aggArray)
    ))

    datasetCombined.append(aggArray)


# Data sample check
print('\nCombined dataset: {} data sets'.format(len(datasetCombined)))

### Dataset Splitting
'''
    Currently extracting [pwm1, vib1-x, vib1-y, vib1-z]
    Shape expected: (len(datasetCombined),4)
'''

# Random pick test dataset from combined dataset with *nTrain* set count
for n in range(nTest):
    popId = randint(0, len(datasetCombined)-1)
    popData = datasetCombined.pop(popId)
    
    # Insert individual data point into train dataset
    for data in popData:
        testDataset = np.append(
            arr=testDataset,
            values=np.array([data[1][0], data[2][0], data[2][1], data[2][2]]).reshape(1,4),
            axis=0    
        )

# Store remaining combined dataset into train data
for dataset in datasetCombined:
    for data in dataset:
        trainDataset = np.append(
            arr=trainDataset,
            values=np.array([data[1][0], data[2][0], data[2][1], data[2][2]]).reshape(1,4),
            axis=0    
        )


### Dataset split sample head check
print('\n------------------------')
print('Train dataset shape:', trainDataset.shape)
print(trainDataset[:10])
print('\n------------------------')
print('Test dataset shape:', testDataset.shape)
print(testDataset[:10])


### Plot splitted dataset
# Train dataset
plt.figure()
plt.title('Train dataset')
plt.ylim(-10,270)
plt.tight_layout()
for plotNum in range(0,4):
    plt.plot(
        list(range(len(trainDataset))),
        trainDataset[:,plotNum],
        linewidth=0.8,
        color='C' + str(plotNum)
    )

# Test dataset
plt.figure()
plt.title('Test dataset')
plt.ylim(-10,270)
plt.tight_layout()
for plotNum in range(0,4):
    plt.plot(
        list(range(len(testDataset))),
        testDataset[:,plotNum],
        linewidth=0.8,
        color='C' + str(plotNum)
    )

plt.show()


# Terminate here
sys.exit()