'''
    LSTM Nav-Vib Data with Multiple Output - Separate model
    - Fixed seed
'''


# Data processing
import numpy as np
import pandas as pd
import sys
import scipy
import sys
import matplotlib.pyplot as plt
import random
import time
import os
import math
import json
from datetime import datetime
import pickle

from navdataVibProc import VibData, NavData, NavdataVib
from evalMetrics import RMSE, SMAPE, MAE, RSquared


###################################################
#################### Parameter ####################
###################################################

# Dataset Preparation
nTest = 3
timeWindow =  250    # in ms


featureName = [
    'RMS (X)',
    'RMS (Y)',
    'RMS (Z)',
    'Kurtosis (X)',
    'Kurtosis (Y)',
    'Kurtosis (Z)',
    'Skewness (X)',
    'Skewness (Y)',
    'Skewness (Z)',
    'Crest Factor (X)',
    'Crest Factor (Y)',
    'Crest Factor (Z)',
    'Peak-To-Peak (X)',
    'Peak-To-Peak (Y)',
    'Peak-To-Peak (Z)',
]


# LSTM
nSequence = 10
nFeatureInput = 4   # pwm1, pitch, roll, yaw
nFeatureOutput = 15  # rms.x, rms.y, rms.z
epochNum = 100

###################################################
################## Data Filtering #################
###################################################

# Data filtering
discardDesc = [
    'aug10_testfail.json',
    'aug10_3_hover20s_fail.json',
    'aug11_4_hover20s_2_crash.json',
    'aug10_2_hover20s_4_crash.json',
    'aug10_1_test_2.json',
    'aug10_1_test.json',
    'aug9_2_hover20s_crash.json',
    'aug9_hover30s_calib0.json',
    'aug9_3_hover10s.json'
]
filterDesc = ['aug9_0', 'aug7_2', 'jul29_2'] # jul_29
excludeDesc = ['fail', 'up_down', 'test', 'crash', '10s', '20s']

""" filterDesc = ['aug10_1']
excludeDesc = ['fail', 'up_down', 'test', 'crash'] """


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

# Program flow
useCachedDataset = False
useCachedModel = False
train = True
predict = True
savePredictPlot = True
bypassCheckpoint = False
predictPlot = True
earlyExit = True
verboseFiltered = False
plotVibAxis = ['x', 'y', 'z']

# Random reproducibility
random.seed(1132) # 12.3

### Dataset and model save-load ###
loadFilename = 'lstm_navdatavib_model_multidenseoutfilter_aug20_20_08_21_10_22_29_100_aug9_0.h5'
loadHistoryFilename = 'lstm_navdatavib_model_multidenseoutfilter_aug20_20_08_21_10_22_29_100_aug9_0.pkl'
#cachePath = 'D:/Dataset/ARDrone/cache'
plotPath = 'D:/Cloud/Google Drive/Tugas Akhir/data/cache/Aug21/plot'
metricsPath = 'D:/Cloud/Google Drive/Tugas Akhir/data/cache/Aug21/metrics'
dataPath = 'D:/Cloud/Google Drive/Tugas Akhir/data/cache/Aug21/'
cachePath = 'D:/Cloud/Google Drive/Tugas Akhir/data/cache/Aug21'
modelPath = 'D:/Cloud/Google Drive/Tugas Akhir/data/cache/Aug21'
#modelPath = 'D:/Dataset/ARDrone/cache'
#modelPath = 'D:/Cloud/Google Drive/Tugas Akhir/data/model'

modelFilename = 'lstm_navdatavib_model_multidenseoutfilter_aug20_' + str(datetime.now().strftime('%y_%m_%d_%H_%M_%S'))
datasetFilename = 'lstm_navdatavib_dataset_agg_aug21_multioutfilter_' + str(timeWindow) + '_' + '_'.join(filterDesc)
predPlotFilename = '{}/plot_pred_{}_{}.jpg'
metricsFilename = 'metrics_{}_{}.csv'


###################################################
########## File Traversal & Aggregation ###########
###################################################
# Vibibibhelofrenaijeswantutelyudetaiemsopritiihiqtengkyu:>

NV = NavdataVib()


def getDirectoryList(data_dir):
    '''
        Get list of file in directory
    '''
    return [f for f in os.listdir(data_dir) if '.npy' in f]


# If dataset not cached
if (datasetFilename + '.npy') not in getDirectoryList(os.path.join(dataPath)) or not useCachedDataset:
    print('[!] Cached dataset not found, doing file traversal')

    # List description
    print('[o] Getting description list...', end='\r')
    descArray = NV.getDescriptionList()
    print('[v] Getting description list done!\n')
    #print(*descArray, sep='\n')

    combinedDataset = []

    # Dataset traversal
    dArrCount = 0
    for desc in descArray:
        dArrCount += 1
        # Data filtering
        # skipping substring in *discardDesc* or *filterDesc* existed in *desc* string
        if (desc in discardDesc) or not any(fd in desc for fd in filterDesc) or any(fd in desc for fd in excludeDesc):
            if verboseFiltered:
                print('--------\n[{}/{}] Filtered data:'.format(dArrCount, len(descArray)), desc)
            continue
        print('------------------------------------------------------------')
        print('[{}/{}] Processing:'.format(dArrCount, len(descArray)), desc)

        # Get combined data (from flightdatas)
        combinedData = NV.getCombined(
            description=desc,
            landed=True
        )


        # Get feature aggregation
        print('[o] Calculating feature aggregation data...', end='\r')
        combinedDataAgg = NV.aggregateCombined(
            combined_data=combinedData,
            time_window=timeWindow
        )


        print('[o] Appending to combined dataset...', end='\r')
        combinedDataset.append(
            {
                'description': desc,
                'data': combinedDataAgg
            }
        )

        print('--> Processed [{}/{}]'.format(dArrCount, len(descArray)), desc)


    # Dataset save to file
    print('[o] Saving dataset to', datasetFilename, end='\r')
    np.array(combinedDataset)
    np.save(dataPath + datasetFilename, combinedDataset)
    print('[v] Saved dataset to', datasetFilename)

else:
    print('[v] Cached dataset found:', datasetFilename + '.npy')

    # Load dataset from file
    print('[o] Loading dataset from', datasetFilename + str('.npy'), end='\r')
    combinedDataset = list(np.load(dataPath + datasetFilename + str('.npy'), allow_pickle=True))
    print('[v] Loaded dataset from', datasetFilename)



###################################################
############### Dataset Preparation ###############
###################################################
'''
    LSTM Expected Input Shape
    (nData, nSequence, nFeature)
    
    - nData: number of data in dataset (nSequence, nFeature)
    - nSequence: number of input sequence for prediction
    - nFeature: number of input feature
'''

def getSequenceArray(dataset, n_sequence, n_feature):
    '''
    ---------------
    Input parameters:
    ---------------
    dataset: array of combinedDataAgg
    n_sequence: int
    n_feature: (input,output)

    ---------------
    Output data:
    ---------------
    seqArrInput: numpy array with shape (nData, n_sequence, n_feature[0])
    seqArrOutput: numpy array with shape (nData, n_feature[1]) -> shifted n_sequence to right

    '''
    dataArr = []
    
    # Append all data in dataset into one data array
    for data in dataset:
        dataArr += data['data']

    # Iterate nData
    nData = len(dataArr) - n_sequence + 1

    seqSetArrInput = np.array([]).reshape(0,n_sequence,n_feature[0])
    seqSetArrOutput = np.array([]).reshape(0,n_feature[1])
    
    # Timestamp concat
    timestampArrOutput = []
    nSet = 0
    newSetTime = 0
    tsPrev = 0
    
    # Iterate over data points
    for nd in range(nData):

        sequenceArrInput = np.array([]).reshape(0,n_feature[0])

        # Iterate over sequence
        for ns in range(n_sequence):
            
            # Add input feature array
            featureArrInput = np.array([
                dataArr[nd+ns]['mot1']['pwm'],                      # 0
                *dataArr[nd+ns]['orientation'],
            ]).reshape(1,n_feature[0])

            sequenceArrInput = np.append(sequenceArrInput, featureArrInput, axis=0)
        
        # Add output feature array
        featureArrOutput = np.array([
            *dataArr[nd+n_sequence-1]['mot1']['rms'],               # 0
            *dataArr[nd+n_sequence-1]['mot1']['kurtosis'],
            *dataArr[nd+n_sequence-1]['mot1']['skewness'],
            *dataArr[nd+n_sequence-1]['mot1']['crest-factor'],
            *dataArr[nd+n_sequence-1]['mot1']['peak-to-peak'],
        ]).reshape(1,n_feature[1])

        #seqSetArrTimestamp.append(dataArr[nd+n_sequence-1]['timestamp'])

        # Append into sequence set
        seqSetArrInput = np.append(seqSetArrInput, sequenceArrInput.reshape(1,n_sequence,n_feature[0]), axis=0)
        seqSetArrOutput = np.append(seqSetArrOutput, featureArrOutput, axis=0)
        
        # Timestamp concat
        ts = dataArr[nd+n_sequence-1]['timestamp']

        if ts < tsPrev:
            newSetTime = timestampArrOutput[nd - (1 + nSet)]
            nSet += 1
            timestampArrOutput.append(ts + newSetTime + 5)
        else:
            timestampArrOutput.append(ts + newSetTime)
        
        tsPrev = ts

    return seqSetArrInput, seqSetArrOutput, timestampArrOutput


def getEWM(arr):
    '''
        Get exponential weighted moving average with pandas
        Currently supported for output with 15 features
    '''

    arrDf = pd.DataFrame({
        'd0': arr[:,0],
        'd1': arr[:,1],
        'd2': arr[:,2],
        'd3': arr[:,3],
        'd4': arr[:,4],
        'd5': arr[:,5],
        'd6': arr[:,6],
        'd7': arr[:,7],
        'd8': arr[:,8],
        'd9': arr[:,9],
        'd10': arr[:,10],
        'd11': arr[:,11],
        'd12': arr[:,12],
        'd13': arr[:,13],
        'd14': arr[:,14],
    })

    arrDfEWM = arrDf.ewm(alpha=0.1).mean()
    arrEWM = arrDfEWM.to_numpy()

    return arrEWM


combSum = 0
for data in combinedDataset:
    dataLen = len(data['data'])
    combSum += dataLen

##### Split data into train and test #####

# Split test data
testDataset = []
for _ in range(nTest):
    
    # Pick random data from dataset
    popData = combinedDataset.pop(random.randrange(len(combinedDataset)))
    testDataset.append(popData)

# Split train data
trainDataset = combinedDataset

##### Split data into input/output sequence #####
#testDatasetTimestamp = get
trainDatasetInput, trainDatasetOutput, trainDatasetTimestamp = getSequenceArray(trainDataset, nSequence, (nFeatureInput,nFeatureOutput))
testDatasetInput, testDatasetOutput, testDatasetTimestamp = getSequenceArray(testDataset, nSequence, (nFeatureInput,nFeatureOutput))
#trainDatasetOutput = getEWM(trainDatasetOutput)
#testDatasetOutput = getEWM(testDatasetOutput)
#for i in range(len(featureName)):
#    print(featureName[i],':',np.sqrt(np.mean(np.square(testDatasetOutput[:,i]))))

# Overview of train and test dataset
print('')
print('-'*40)
print('Dataset Overview Confirmation')
print('-'*40)
print('Number of data:', combSum)
print('Sequence length:', nSequence)
print('Number of Input Feature:', nFeatureInput)
print('Number of Output Feature:', nFeatureOutput)
print('')
print('-'*40)
print('Train dataset')
print('-'*40)
print('Number of dataset(s):', len(trainDataset))
print('Input:', trainDatasetInput.shape)
print('Output:', trainDatasetOutput.shape)
print('')
print('-'*40)
print('Test dataset')
print('-'*40)
print('Number of dataset(s):', len(testDataset))
print('Input:', testDatasetInput.shape)
print('Output:', testDatasetOutput.shape)
print('\n')

# Early exit
if earlyExit:
    sys.exit()

# Checkpoint confirmation
if not bypassCheckpoint:
    cont = input('Continue train dataset? [y/n] ')
    if cont != 'y':
        sys.exit()


###################################################
###################### LSTM #######################
###################################################
'''
    TO DO:
    [ ] 
'''

# Module imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import load_model


def createModel(early_stop=True, checkpoint=True):
    # Create model
    model = Sequential()
    #model.add(LSTM(256, activation='tanh', input_shape=(nSequence, nFeatureInput), return_sequences=True))
    #model.add(LSTM(512, activation='tanh', input_shape=(nSequence, nFeatureInput), return_sequences=True))
    model.add(LSTM(512, activation='tanh', input_shape=(nSequence, nFeatureInput)))
    model.add(Dense(256))
    model.add(Dense(64))
    model.add(Dense(nFeatureOutput))


    # Add Ons
    addOns = []

    if early_stop:
        addOns.append(EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='min', restore_best_weights=True))

    if checkpoint:
        trainTime = str(datetime.now().strftime('%y_%m_%d_%H_%M_%S'))
        addOns.append(ModelCheckpoint('D:/Dataset/ARDrone/modelCheckpointLog/' + trainTime + '_{epoch:03d}-{loss:03f}-{val_loss:03f}.h5', save_best_only=True, monitor='val_loss', mode='min'))
    

    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])

    return model, addOns


def loadModel(filename=None):
    if filename == None:
        filename = modelFilename + '_' + str(timeWindow) + '_' + '_'.join(filterDesc) + '.h5'
    

    print('-'*30)
    print('Loading model from {}'.format(filename))
    model = load_model(os.path.join(cachePath, filename))

    return model


def loadHistory(filename):
    if filename == None:
        filename = modelFilename + '_' + str(timeWindow) + '_' + '_'.join(filterDesc) + '.pkl'
    
    with open(os.path.join(dataPath, filename), 'rb') as historyFile:
        history = pickle.load(historyFile)

    return history


def exportModel(model):
    filename = modelFilename + '_' + str(timeWindow) + '_' + '_'.join(filterDesc) + '.h5'
    
    print('-'*30)
    print('Saving model  to {}'.format(filename))
    model.save(os.path.join(dataPath, filename))


def exportHistory(history):
    filename = modelFilename + '_' + str(timeWindow) + '_' + '_'.join(filterDesc) + '.pkl'
    with open(os.path.join(dataPath, filename), 'wb') as historyFile:
        pickle.dump(history.history, historyFile)


def plotPrediction(timestamp_arr, output_arr, pred_arr, idx, simple=False):
    '''
        Black and White graph (optimized for final report)
        idx: feature number in featureName
    '''
    # Set font to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    # Get timestamp array
    xData = [(ts/1000) for ts in timestamp_arr]
    #print(xData[:10])
    #print(xData[-1])

    # Set y-axis limit and ticks
    if 'RMS' in featureName[idx]:
        ylim = [0,20]
        yticks = list(range(ylim[0],ylim[1]+1,5))
    
    if 'Kurtosis' in featureName[idx]:
        ylim = [-10,30]
        yticks = list(range(ylim[0],ylim[1]+1,10))

    if 'Skewness' in featureName[idx]:
        ylim = [-10,10]
        yticks = list(range(ylim[0],ylim[1]+1,5))

    if 'Crest Factor' in featureName[idx]:
        ylim = [-5,10]
        yticks = list(range(ylim[0],ylim[1]+1,5))

    if 'Peak-To-Peak' in featureName[idx]:
        ylim = [0,40]
        yticks = list(range(ylim[0],ylim[1]+1,10))

    """ if simple:
        yticks = [] """

    # Set x-axis limit and ticks
    xlim = [xData[0], xData[-1]]
    if simple:
        xticks = []
    else:
        xticks = list(range(0, (int(xData[-1]) + 2), 5))

    # Plot
    fig = plt.figure(figsize=(16,2.2), dpi=120)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.97, bottom=0.15)

    ax1 = fig.add_subplot(111, frame_on=True)
    ax2 = fig.add_subplot(111, frame_on=False)

    p_test, = ax1.plot(xData, output_arr, 'r--', linewidth=0.8)
    ax1.tick_params(grid_alpha=0.6, grid_linewidth=0.4, labelsize=16)

    ax1.set_xticks(xticks)
    ax1.set_xlim(xlim)
    ax1.set_yticks(yticks)
    ax1.set_ylim(ylim)
    ax1.grid(True)
    if not simple:
        ax1.set_title(featureName[idx].title(), fontsize=20)
        ax1.set_xlabel('Waktu (ms)', fontsize=20)
    ax1.set_ylabel(featureName[idx].title(), fontsize=20)

    p_pred, = ax2.plot(xData, pred_arr, 'k-', linewidth=1)
    ax2.set_xticks([])
    ax2.set_xlim(xlim)
    ax2.set_yticks([])
    ax2.set_ylim(ylim)
    ax2.grid(True)

    """ ax1.legend(
        (p_test, p_pred),
        ('Real Data', 'Prediction'),
        loc='upper right',
        fontsize=18
    ) """

def plotMetricsHistory(history):
    '''
        Plot metrics (currently supported: mean squared error)
    '''
    # Set font to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    if train:
        plotData = history.history['loss']
    else:
        plotData = history['loss']

    if train:
        plotDataVal = history.history['val_loss']
    else:
        plotDataVal = history['val_loss']

    minVal = plotDataVal[0]
    minEpoch = 1
    for i in range(len(plotDataVal)):
        if plotDataVal[i] < minVal:
            minVal = plotDataVal[i]
            minEpoch = i+1


    fig = plt.figure(figsize=(16,3), dpi=120)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.25)

    p1, = plt.plot(plotData, 'r--', linewidth=1.2)
    p2, = plt.plot(plotDataVal, 'k-', linewidth=1.2)

    plt.xlim([0,100])

    plt.title('Mean Squared Error over Epoch ({}-{:.5f})'.format(minEpoch, minVal), fontsize=22)
    plt.xlabel('Epochs', fontsize=22)
    plt.ylabel('MSE', fontsize=22)
    plt.grid(True)
    plt.legend((p1, p2),('Training','Validation'))



###################################################
################ Train & Predict ##################
###################################################

trainTime = 0

# Model train iteration by output feature count
if train:
    print('Starting training...')

    #for outFeatureNum in range(3):
    if not useCachedModel:
        # Create model
        print('Creating model')
        model, addOns = createModel(
            early_stop=False,
            checkpoint=True
        )

        # Fit model
        print('Fitting model')
        history = model.fit(
            x=trainDatasetInput,
            y=trainDatasetOutput,
            validation_data=(testDatasetInput, testDatasetOutput),
            epochs=epochNum,
            callbacks=[*addOns],
            batch_size=64,
            verbose=1
        )

        # Export model
        print('Exporting model')
        exportModel(model)
        exportHistory(history)

    else:
        model = loadModel()


# Prediction
if predict:
    print('Starting predicting...')
    #testDatasetOutput = getEWM(testDatasetOutput)

    if not train:
        model = loadModel(loadFilename)
        history = loadHistory(loadHistoryFilename)

    predOutput = []

    if trainTime == 0:
        trainTime = str(datetime.now().strftime('%y_%m_%d_%H_%M_%S'))
    os.mkdir(plotPath + '/' + str(trainTime) + '_' + str(timeWindow))

    predArr = np.array([]).reshape(0,nFeatureOutput)
    i = 0
    for datapoint in range(testDatasetInput.shape[0]):
        i += 1
        print('Data Point: {}/{}'.format(i,testDatasetInput.shape[0]), end='\r')
        predArr = np.append(predArr, model.predict(testDatasetInput[datapoint,:,:].reshape(1,nSequence,nFeatureInput), verbose=0).reshape(1,nFeatureOutput), axis=0)


    ### Calculate Metrics ###
    metricsArr = np.array([]).reshape(0,4)
    for featNum in range(len(featureName)):
        trueData = testDatasetOutput[:, featNum]
        predData = predArr[:, featNum]

        metricsVal = np.array([
            RMSE(trueData, predData),
            SMAPE(trueData, predData),
            MAE(trueData, predData),
            RSquared(trueData, predData)
        ]).reshape(1,4)

        metricsArr = np.append(metricsArr, metricsVal, axis=0)

    np.savetxt(
        metricsPath + '/' + metricsFilename.format(trainTime, timeWindow),
        metricsArr,
        delimiter=',',
        header='rmse,smape,mae,rsquared'
    )


    ### Plotting Training Metrics ###
    plotMetricsHistory(
        history=history
    )

    if savePredictPlot:
        
        plt.savefig(
            fname=plotPath + '/' + predPlotFilename.format(trainTime + '_' + str(timeWindow), 'all', 'hist')
        )
    
    ### Plotting Predicted Value vs Real Value for each feature ###
    print('Predicting output')
    for featNum in range(len(featureName)):
        
        plotPrediction(
            timestamp_arr=testDatasetTimestamp,
            output_arr=testDatasetOutput[:,featNum],
            pred_arr=predArr[:,featNum],
            idx=featNum,
            simple=False
        )
        
        if savePredictPlot:
            plt.savefig(
                fname=plotPath + '/' + predPlotFilename.format(trainTime + '_' + str(timeWindow), featureName[featNum], 'pred')
            )
        else:
            plt.show()