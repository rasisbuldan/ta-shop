# Data processing
import numpy as np
import sys
import matplotlib.pyplot as plt
from random import random, randint, seed
import time

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
filterDesc = ['jul29'] # 'jul29'

### Dataset split ###
nTrain = 10
nTest = 3
trainDataset = np.empty((0,2,1))
testDataset = np.empty((0,2,1))

# Reproducibility
seed(int(time.time()))

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
            values=np.array([[data[1][0]], [data[2][0]]]).reshape(1,2,1),
            axis=0    
        )

# Store remaining combined dataset into train data
for dataset in datasetCombined:
    for data in dataset:
        trainDataset = np.append(
            arr=trainDataset,
            values=np.array([[data[1][0]], [data[2][0]]]).reshape(1,2,1),
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
""" plt.figure()
plt.title('Train dataset')
plt.ylim(-10,270)
plt.tight_layout()
for plotNum in range(0,2):
    plt.plot(
        list(range(len(trainDataset))),
        trainDataset[:,plotNum],
        linewidth=0.8,
        color='C' + str(plotNum)
    ) """

# Test dataset
""" plt.figure()
plt.title('Test dataset')
plt.ylim(-10,270)
plt.tight_layout()
for plotNum in range(0,2):
    plt.plot(
        list(range(len(testDataset))),
        testDataset[:,plotNum],
        linewidth=0.8,
        color='C' + str(plotNum)
    ) """

""" plt.show() """

# Terminate here
""" sys.exit() """


##### LSTM #####
# Parameter
nSteps = 10
nFeature = 1

# Function - helper
def splitSequences(array, n_steps):
    seqArray = []
    for i in range(len(array)-(n_steps-1)):
        seq = array[i:i+n_steps]
        seqArray.append(seq)

    return seqArray


### Data step sequence
trainX = np.array(splitSequences(trainDataset[:,0], nSteps))
trainY = trainDataset[nSteps-1:,1]
testX = np.array(splitSequences(testDataset[:,0], nSteps))
testY = testDataset[nSteps-1:,1]
print('Train dataset: ', trainX.shape, trainY.shape)
print('Test dataset: ', testX.shape, testY.shape)

# Create model
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(nSteps,nFeature)))
model.add(Dense(1))

# Print summary of model
model.summary()

# Compile model
#model.compile(optimizer='adam', loss='cosine_similarity')
model.compile(optimizer='adam', loss='mean_absolute_error')

# Model fitting
model.fit(trainX, trainY, epochs=20, verbose=0)
#model.fit(trainX, trainY, validation_data=(testX,testY), epochs=100, verbose=1)

# Predict
predArray = []

for i in range(testX.shape[0]):
    testOutput = model.predict(np.array(testX[i,:,:]).reshape(1,nSteps,1), verbose=0)
    predArray.append(float(testOutput))

print('Prediction array:', len(predArray))
#print(predArray)

plt.plot(list(range(len(predArray))), predArray, color='C0')
plt.plot(list(range(len(testY))), testY, color='C3')
plt.grid(True)
plt.show()