'''
    To do:
    [v] Split dataset by one step loop
    [v] Stacked LSTM (2-3)
'''


# Data processing
from staticpwmVibProc import VibData, StaticPWMData, StaticVibData
import numpy as np
import sys
import matplotlib.pyplot as plt
from random import random, randint, seed
import time
import os

# LSTM\
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


###################################################
#################### Parameter ####################
###################################################

# Dataset
nTrain = 10
nTest = 3
windowSize = 75
windowSizeSample = 75

# Weighted Moving RMS
stepWeight = 0.1
weight = np.arange(stepWeight, stepWeight*(windowSize+1), stepWeight)

# LSTM
nSteps = 8
nStepsPred = 3
nFeature = 1
epochNum = 100


###################################################
#################### Settings #####################
###################################################
showPlot = False
earlyExit = False
plotVibAxis = ['x', 'y', 'z']

# Random reproducibility
seed(int(time.time()))

### Dataset and model save-load ###
data_path = 'C:/Users/rss75/Documents/GitHub/ta-shop/source/vib-static'
modelFilename = 'lstm_staticvib_model_aug6.h5'
datasetFilename = 'lstm_staticvib_dataset_agg_aug6_v2_' + str(windowSizeSample)


###################################################
################## Data Filtering #################
###################################################

# Data filtering
discardDesc = [
    'aug5_step_clean_0_300_900_3000_1_7',
    'aug5_step_clean_0_300_900_3000_2_9',
    'aug5_step_clean_0_300_900_3000_3_4',
    'aug5_step_clean_0_300_900_3000_4_14',
    'aug5_step_clean_0_300_900_3000_5_3',
    'aug6_triangle_clean_0_400_900_4000_100_1_10',
    'aug6_triangle_clean_0_400_900_4000_100_4_0',
    'aug6_triangle_clean_0_400_900_4000_100_5_12',
    'aug6_triangle_clean_0_400_900_4000_100_5_13'
]
filterDesc = ['aug6']  # 'jul29'


### Dataset split ###
trainDataset = np.empty((0, 2, 1))
testDataset = np.empty((0, 2, 1))
trainDatasetDesc = []
testDatasetDesc = []


###################################################
############### Dataset Preparation ###############
###################################################
# Vibibibhelofrenaijeswantutelyudetaiemsopritiihiqtengkyu:>
Vib = VibData(offset=0)  # -2375
Static = StaticPWMData()
SV = StaticVibData()


def getDirectoryList(data_dir):
    '''
        Get list of file in directory
    '''
    return [f for f in os.listdir(data_dir) if '.npy' in f]


# If dataset not cached
if (datasetFilename + '.npy') not in getDirectoryList(data_path):
    print('[!] Cached dataset not found, doing file traversal')

    # List description
    print('[o] Getting description list...', end='\r')
    descArray = [static['description']
                 for static in Static.listDescriptionTimestamp()[1:]]
    print('[v] Getting description list done!\n')

    datasetCombined = []

    # Dataset traversal
    dArrCount = 0
    for desc in descArray:
        dArrCount += 1
        # Data filtering
        # skipping substring in *discardDesc* or *filterDesc* existed in *desc* string
        if (desc in discardDesc) or not all(fd in desc for fd in filterDesc):
            print(
                '--------\n[{}/{}] Filtered data:'.format(dArrCount, len(descArray)), desc)
            continue
        print(
            '--------\n[{}/{}] Processing:'.format(dArrCount, len(descArray)), desc)

        # Get navdata by description *desc*
        print('[o] Querying pwm data...', end='\r')
        pwmArr = Static.getByDescription(
            description=desc,
            landed=True
        )

        # Get timestamp range value
        print('[o] Querying vib data...', end='\r')
        tsMin, tsMax = Static.getTimestampRangeByDescription(
            description=desc,
            landed=True
        )

        # Get vibdata by timestamp range
        print('[o] Get vibration data...', end='\r')
        vibdataArr = Vib.getBetweenTimestamp(tsMin, tsMax)

        # Conditional: missing data
        if len(pwmArr) <= 0:
            # should not?
            print('Skipping dataset (pwmdata empty)..')
            continue

        if len(vibdataArr) <= 0:
            print('Skipping dataset (vibdata empty)..')
            continue

        # Store navdata by description
        print('[o] Storing static data...', end='\r')
        pwmId = Static.storeData(
            data_array=pwmArr,
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

        aggArray = SV.combineAggArray(
            pwmdata=[Static.getTimestampArray(
                pwmId)] + [Static.getPWMArray(pwmId)],
            vibdata=[Vib.getTimestampArray(vibId)] + [Vib.getVibArray(vibId)],
            time_window=windowSizeSample
        )

        print('[v] Processed data with id: {}|{} with size {}|{} => {}'.format(
            pwmId,
            vibId,
            len(Static.activeDataArray[pwmId]['data']),
            len(Vib.activeDataArray[vibId]['data']),
            len(aggArray)
        ))

        datasetCombined.append([desc, aggArray])

    # Dataset save to file
    print('[o] Saving dataset to', datasetFilename, end='\r')
    np.array(datasetCombined)
    np.save(data_path + '/' + datasetFilename, datasetCombined)
    print('[v] Saved dataset to', datasetFilename)

else:
    print('[v] Cached dataset found:', datasetFilename + '.npy')
    # Load dataset from file
    print('[o] Loading dataset from', datasetFilename + str('.npy'), end='\r')
    datasetCombined = list(
        np.load(data_path + '/' + datasetFilename + str('.npy'), allow_pickle=True))
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
        testDataset = np.append(
            arr=testDataset,
            values=np.array([[data[1]], [data[2][0]]]).reshape(1, 2, 1),
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
        trainDataset = np.append(
            arr=trainDataset,
            values=np.array([[data[1]], [data[2][0]]]).reshape(1, 2, 1),
            axis=0
        )
    i += 1
print('[v] Train dataset prepared                                 ')


# Dataset split sample head check
print('\n------------------------')
print('Train dataset shape:', trainDataset.shape)
print('\nTrain dataset description list: ')
print(*trainDatasetDesc, sep='\n')
print('\nTrain dataset head: ')
print(trainDataset[:3])
print('\n------------------------')
print('Test dataset shape:', testDataset.shape)
print('\nTest dataset description list: ')
print(*testDatasetDesc, sep='\n')
print('\nTest dataset head: ')
print(testDataset[:3])


# Plot splitted dataset
if showPlot:
    # Train dataset
    fig = plt.figure()
    plt.title('Train dataset')
    ax_pwm = fig.add_subplot(111, frame_on=True)
    ax_vibx = fig.add_subplot(111, frame_on=False)

    ax_pwm.plot(
        list(range(len(trainDataset))),
        trainDataset[:, 0],
        linewidth=0.8,
        color='C0'
    )
    ax_pwm.set_ylim([0, 1000])
    ax_vibx.plot(
        list(range(len(trainDataset))),
        trainDataset[:, 1],
        linewidth=0.5,
        color='C1'
    )
    ax_vibx.set_ylim([0, 16])
    ax_vibx.yaxis.tick_right()

    # Test dataset
    fig = plt.figure()
    plt.title('Test dataset')
    ax_pwm = fig.add_subplot(111, frame_on=True)
    ax_vibx = fig.add_subplot(111, frame_on=False)

    ax_pwm.plot(
        list(range(len(testDataset))),
        testDataset[:, 0],
        linewidth=0.8,
        color='C0'
    )
    ax_pwm.set_ylim([0, 1000])
    ax_vibx.plot(
        list(range(len(testDataset))),
        testDataset[:, 1],
        linewidth=0.5,
        color='C1'
    )
    ax_vibx.set_ylim([0, 16])
    ax_vibx.yaxis.tick_right()

    plt.show()

# Terminate here
if earlyExit:
    sys.exit()


###################################################
###################### LSTM #######################
###################################################
# Function - helper
def splitSequences(array, n_steps):
    seqArray = []
    for i in range(len(array)-(n_steps-1)):
        seq = array[i:i+n_steps]
        seqArray.append(seq)

    return seqArray


# Data step sequence
trainX = np.array(splitSequences(trainDataset[:, 0], nSteps))
trainY = trainDataset[nSteps-1:, 1]
testX = np.array(splitSequences(testDataset[:, 0], nSteps))
testY = testDataset[nSteps-1:, 1]
print('Train dataset: ', trainX.shape, trainY.shape)
print('Test dataset: ', testX.shape, testY.shape)

sys.exit()

# Create model
model = Sequential()
model.add(LSTM(512, activation='relu', input_shape=(nSteps, nFeature), return_sequences=True))
model.add(LSTM(256, activation='relu', input_shape=(nSteps, nFeature), return_sequences=True))
model.add(LSTM(128, activation='relu', input_shape=(nSteps, nFeature)))
model.add(Dense(1))

# Print summary of model
model.summary()

# Add early stopping and save best model
earlyStopping = EarlyStopping(monitor='mean_squared_error', patience=3, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='mean_squared_error', mode='min')

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])

# Model fitting
model.fit(trainX, trainY, epochs=epochNum, callbacks=[earlyStopping, mcp_save], batch_size=32, verbose=1)

# Save model
print('Saving model to', modelFilename)
model.save(data_path + '/' + modelFilename)

# Predict
predArray = []

for i in range(testX.shape[0]):
    print('Predicting [{}/{}] - {:.1f}%'.format(i+1,
                                                testX.shape[0], (i+1)/testX.shape[0]*100), end='\r')
    testOutput = model.predict(
        np.array(testX[i, :, :]).reshape(1, nSteps, 1), verbose=0)
    predArray.append(float(testOutput))

print('\nPrediction array:', len(predArray))
# print(predArray)

plt.plot(list(range(len(testY))), testY, color='C3', linewidth=0.5)
plt.plot(list(range(len(predArray))), predArray, color='C0', linewidth=1.5)
plt.grid(True)
plt.show()