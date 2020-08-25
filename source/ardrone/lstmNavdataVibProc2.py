'''
    To do:
    [ ] Migrate to windowSizeSample

'''


# Data processing
from navdataVibProc import VibData, NavData, NavdataVib
import numpy as np
import sys
import matplotlib.pyplot as plt
from random import random, randint, seed
import time
import os

# LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


###################################################
#################### Parameter ####################
###################################################

# Dataset
nTrain = 10     # kinda unused
nTest = 1
windowSize = 100
windowSizeSample = 25

# Weighted Moving RMS
stepWeight = 0.1
weight = np.arange(stepWeight, stepWeight*(windowSize+1), stepWeight)

# LSTM
nSteps = 2
nFeatureInput = 7   # pwm1, pitch, roll, yaw, acc.x, acc.y, acc.z
nFeatureOutput = 1
epochNum = 100


###################################################
################## Data Filtering #################
###################################################

# Data filtering
discardDesc = ["jul29_up_down_02.json"]
filterDesc = ['aug11_4'] # jul_29


### Dataset split ###
trainDatasetInput = np.empty((0, 1, nFeatureInput))
trainDatasetOutput = np.empty((0, 1, nFeatureOutput))
testDatasetInput = np.empty((0, 1, nFeatureInput))
testDatasetOutput = np.empty((0, 1, nFeatureOutput))
trainDatasetDesc = []
testDatasetDesc = []


###################################################
#################### Settings #####################
###################################################
useCached = False
showPlot = True
earlyExit = True
plotVibAxis = ['x', 'y', 'z']

# Random reproducibility
seed(int(time.time()))

### Dataset and model save-load ###
data_path = 'C:/Users/rss75/Documents/GitHub/ta-shop/source/ardrone'
modelFilename = 'lstm_navdatavib_model_aug6.h5'
datasetFilename = 'lstm_navdatavib_dataset_agg_aug7_v2_multi_' + str(windowSize) + '_' + '_'.join(filterDesc)


###################################################
############### Dataset Preparation ###############
###################################################
# Vibibibhelofrenaijeswantutelyudetaiemsopritiihiqtengkyu:>
Vib = VibData()  # -2375
Nav = NavData()
NV = NavdataVib()


def getDirectoryList(data_dir):
    '''
        Get list of file in directory
    '''
    return [f for f in os.listdir(data_dir) if '.npy' in f]


# If dataset not cached
if (datasetFilename + '.npy') not in getDirectoryList(os.path.join(data_path,'cache')) or not useCached:
    print('[!] Cached dataset not found, doing file traversal')

    # List description
    print('[o] Getting description list...', end='\r')
    descArray = [nav['description'] for nav in Nav.listDescriptionTimestamp()[1:]]
    print('[v] Getting description list done!\n')
    print(*descArray, sep='\n')

    datasetCombined = []

    # Dataset traversal
    dArrCount = 0
    for desc in descArray:
        dArrCount += 1
        # Data filtering
        # skipping substring in *discardDesc* or *filterDesc* existed in *desc* string
        if (desc in discardDesc) or not any(fd in desc for fd in filterDesc):
            print(
                '--------\n[{}/{}] Filtered data:'.format(dArrCount, len(descArray)), desc)
            continue
        print(
            '--------\n[{}/{}] Processing:'.format(dArrCount, len(descArray)), desc)

        # Get navdata by description *desc*
        print('[o] Querying pwm data...', end='\r')
        navArr = Nav.getByDescription(
            description=desc,
            landed=True
        )

        # Get timestamp range value
        print('[o] Querying vib data...', end='\r')
        tsMin, tsMax = Nav.getTimestampRangeByDescription(
            description=desc,
            landed=True
        )

        # Get vibdata by timestamp range
        print('[o] Get vibration data...', end='\r')
        vibdataArr = Vib.getBetweenTimestamp(tsMin, tsMax)

        # Conditional: missing data
        if len(navArr) <= 0:
            # should not?
            print('Skipping dataset (pwmdata empty)..')
            continue

        if len(vibdataArr) <= 0:
            print('Skipping dataset (vibdata empty)..')
            continue

        # Store navdata by description
        print('[o] Storing static data...', end='\r')
        navId = Nav.storeData(
            data_array=navArr,
            description=desc
        )

        # Store vibdata by timestamp range
        print('[o] Storing vibration data...', end='\r')
        vibId = Vib.storeData(
            data_array=vibdataArr,
            description=desc
        )

        # Combine data

        print('[o] Combining static and vibration data...', end='\r')
        """ aggArray = SV.combineAggArray(
            pwmdata=[Static.getTimestampArray(pwmId)] + [Static.getPWMArray(pwmId)],
            vibdata=[Vib.getTimestampArray(vibId)] + [Vib.getVibArray(vibId)],
            time_window=windowSizeSample
        ) """

        aggArray = NV.combineMultiAggregatedArray(
            navdata_agg=Nav.getMultiAggregatedArray(navId, windowSize),
            vibdata_agg=Vib.getVibAggregatedArray(vibId, windowSize),
            time_window=windowSize
        )

        #print(aggArray[:5])

        print('[v] Processed data with id: {}|{} with size {}|{} => {}'.format(
            navId,
            vibId,
            len(Nav.activeDataArray[navId]['data']),
            len(Vib.activeDataArray[vibId]['data']),
            len(aggArray)
        ))

        # Breakpoint (debug)
        #sys.exit()

        datasetCombined.append([desc, aggArray])

    # Dataset save to file
    print('[o] Saving dataset to', datasetFilename, end='\r')
    np.array(datasetCombined)
    np.save(data_path + '/cache/' + datasetFilename, datasetCombined)
    print('[v] Saved dataset to', datasetFilename)

else:
    print('[v] Cached dataset found:', datasetFilename + '.npy')
    # Load dataset from file
    print('[o] Loading dataset from', datasetFilename + str('.npy'), end='\r')
    datasetCombined = list(
        np.load(data_path + '/cache/' + datasetFilename + str('.npy'), allow_pickle=True))
    print('[v] Loaded dataset from', datasetFilename)


# Data sample check
print('\n\n------------------------\nCombined dataset: {} data sets'.format(len(datasetCombined)))


# Dataset Splitting
'''
    Currently extracting [pwm1, vib1-x, vib1-y, vib1-z]
    Shape expected: (len(datasetCombined),4)
'''

# Random pick test dataset from combined dataset with *nTrain* set count
print('[o] Preparing test dataset...', end='\r')
for n in range(nTest):
    print('[o] Preparing test dataset [{}/{}] - {:.1f}%'.format(n,
                nTest, n/nTest*100), end='\r')
    popId = randint(0, len(datasetCombined)-1)
    popData = datasetCombined.pop(popId)
    testDatasetDesc.append(popData[0])

    # Insert individual data point into train dataset
    for data in popData[1]:
        inputArr = [data[1][0], *data[2], *data[3]]
        outputArr = [data[4][2]]    # x
        
        # Merge input data
        testDatasetInput = np.append(
            arr=testDatasetInput,
            values=np.array([inputArr]).reshape(1, 1, nFeatureInput),
            axis=0
        )
        
        # Merge output data
        testDatasetOutput = np.append(
            arr=testDatasetOutput,
            values=np.array([outputArr]).reshape(1, 1, nFeatureOutput),
            axis=0
        )
print('[v] Test dataset prepared                                  ')


# Store remaining combined dataset into train data
print('[o] Preparing train dataset...', end='\r')
i = 0
for dataset in datasetCombined:
    print('[o] Preparing train dataset [{}/{}] - {:.1f}%'.format(i,
                len(datasetCombined), i/len(datasetCombined)*100), end='\r')
    trainDatasetDesc.append(dataset[0])
    for data in dataset[1]:
        inputArr = [data[1][0], *data[2], *data[3]]
        outputArr = [data[4][2]]    # y

        # Merge input data
        trainDatasetInput = np.append(
            arr=trainDatasetInput,
            values=np.array(inputArr).reshape(1, 1, nFeatureInput),
            axis=0
        )

        # Merge output data
        trainDatasetOutput = np.append(
            arr=trainDatasetOutput,
            values=np.array([outputArr]).reshape(1, 1, nFeatureOutput),
            axis=0
        )
    
    i += 1

print('[v] Train dataset prepared                                 ')


# Dataset split sample head check
print('\n------------------------')
print('Train dataset shape:', trainDatasetInput.shape, trainDatasetOutput.shape)
print('\nTrain dataset description list: ')
print(*trainDatasetDesc, sep='\n')
print('\nTrain dataset head: ')
print(trainDatasetInput[:3])
print(trainDatasetOutput[:3])
print('\n------------------------')
print('Test dataset shape:', testDatasetInput.shape, testDatasetOutput.shape)
print('\nTest dataset description list: ')
print(*testDatasetDesc, sep='\n')
print('\nTest dataset head: ')
print(testDatasetInput[:3])
print(testDatasetOutput[:3])

# Plot splitted dataset
if showPlot:
    # Train dataset
    fig = plt.figure()
    plt.title('Train dataset (' +  ','.join(filterDesc) + ') - ' + str(windowSize))
    ax1_pwm = fig.add_subplot(111, frame_on=True)
    ax1_vibx = fig.add_subplot(111, frame_on=False)

    p1_pwm, = ax1_pwm.plot(
        list(range(len(trainDatasetInput))),
        trainDatasetInput[:, 0, 0],
        linewidth=0.8,
        color='C0'
    )
    ax1_pwm.set_ylim([0, 300])
    #print('Shape: ', trainDatasetInput.shape, trainDatasetOutput.shape)
    p1_vibx, = ax1_vibx.plot(
        list(range(len(trainDatasetOutput))),
        trainDatasetOutput[:, 0, 0],
        linewidth=0.5,
        color='C1'
    )
    ax1_vibx.set_ylim([0, 16])
    ax1_vibx.yaxis.tick_right()

    # Test dataset
    fig = plt.figure()
    plt.title('Test dataset (' +  ','.join(filterDesc) + ') - ' + str(windowSize))
    ax2_pwm = fig.add_subplot(111, frame_on=True)
    ax2_vibx = fig.add_subplot(111, frame_on=False)

    p2_pwm, = ax2_pwm.plot(
        list(range(len(testDatasetInput))),
        testDatasetInput[:, 0, 0],
        linewidth=0.8,
        color='C0'
    )
    ax2_pwm.set_ylim([0, 300])
    p2_vibx, = ax2_vibx.plot(
        list(range(len(testDatasetOutput))),
        testDatasetOutput[:, 0, 0],
        linewidth=0.5,
        color='C1'
    )
    ax2_vibx.set_ylim([0, 16])
    ax2_vibx.yaxis.tick_right()

    ax1_pwm.legend(
        (p1_pwm, p1_vibx),
        ('PWM', 'Vib - X'),
        loc='upper right'
    )
    ax2_pwm.legend(
        (p2_pwm, p2_vibx),
        ('PWM', 'Vib - X'),
        loc='upper right'
    )
    plt.show()

# Terminate here
if earlyExit:
    sys.exit()


###################################################
###################### LSTM #######################
###################################################

# Helper function
def splitSequences(array, n_steps):
    seqArray = []
    for i in range(array.shape[0] - (n_steps-1)):
        seq = array[i:i+n_steps,0]
        seqArray.append(seq)
    
    return seqArray


# Data step sequence
trainX = np.array(splitSequences(trainDatasetInput, nSteps))
trainY = trainDatasetOutput[nSteps-1:,0]
testX = np.array(splitSequences(testDatasetInput, nSteps))
testY = testDatasetOutput[nSteps-1:,0]
print('Train dataset: ', trainX.shape, trainY.shape)
print('Test dataset: ', testX.shape, testY.shape)


# Create model
model = Sequential()
model.add(LSTM(512, activation='relu', input_shape=(nSteps, nFeatureInput), return_sequences=True))
model.add(LSTM(512, activation='relu', input_shape=(nSteps, nFeatureInput), return_sequences=True))
model.add(LSTM(256, activation='relu', input_shape=(nSteps, nFeatureInput)))
model.add(Dense(1))

# Print summary of model
model.summary()

# Add early stopping and save best model
earlyStopping = EarlyStopping(monitor='mean_squared_error', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='mean_squared_error', mode='min')

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])

# Model fitting
model.fit(trainX, trainY, epochs=epochNum, callbacks=[earlyStopping, mcp_save], batch_size=128, verbose=1)

# Save model
print('Saving model to', modelFilename)
model.save(data_path + '/' + modelFilename)

# Predict
predArray = []

for i in range(testX.shape[0]):
    print('Predicting [{}/{}] - {:.1f}%'.format(i+1,
            testX.shape[0], (i+1)/testX.shape[0]*100), end='\r')
    testOutput = model.predict(
        np.array(testX[i, :, :]).reshape(1, nSteps, nFeatureInput), verbose=0)
    predArray.append(float(testOutput))

print('\nPrediction array:', len(predArray))
# print(predArray)

plt.plot(list(range(len(testY))), testY, color='C1', linewidth=0.5)
plt.plot(list(range(len(predArray))), predArray, color='C0', linewidth=1.5)
plt.grid(True)
plt.show()