'''
    LSTM combined load with load number as input
'''

import numpy as np
import pandas as pd
import sys
import os
import time
import pickle
import random
import matplotlib.pyplot as plt
from datetime import datetime
from flightdataProc import FlightData
from evalMetrics import RMSE, SMAPE, MAE, RSquared

###################################################
#################### Parameter ####################
###################################################

# Dataset Preparation
nTest = 0.3
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
nFeatureInput = 5   # pwm1, pitch, roll, yaw, loadNum
nFeatureOutput = 15  # rms, kurt, skew, crest, p2p
epochNum = 200

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

filterDesc = [
    'aug9_0',
    'aug10_1',
    'aug10_2',
    'aug10_3',
    'aug11_4',
    'aug11_5',
    'aug11_6',
]

excludeDesc = ['fail', 'up_down', 'test', 'crash']


###################################################
#################### Settings #####################
###################################################

# Program flow
useCachedDataset = False
useCachedModel = False
bypassCheckpoint = True
earlyExit = False
train = True
predict = True
predictPlot = True
savePredictPlot = True
verboseFiltered = False

# Random reproducibility
random.seed(1132) # 12.3


### Dataset and model caching ###
cachePath = 'D:/cache_new'
datasetFilename = 'vm_dataset_' + str(timeWindow) + '_' + '_'.join(filterDesc) + '.npy'
modelFilename = 'vm_model_' + str(datetime.now().strftime('%y_%m_%d_%H_%M_%S')) + '.h5'
metricsFilename = 'vm_metrics_' + str(datetime.now().strftime('%y_%m_%d_%H_%M_%S')) + '.csv'



FD = FlightData()



# Check cached dataset
if datasetFilename not in [f for f in os.listdir(cachePath) if '.npy' in f] or not useCachedDataset:

    # load number loop
    dataset = []
    timestampPrev = 0
    for loadNum in range(len(filterDesc)):
        dArrCount = 0
        descList = [desc for desc in FD.listDescription() if (filterDesc[loadNum] in desc) and not any(desc in disc for disc in discardDesc)]
        
        # Desc loop
        for desc in descList:
            dArrCount += 1
            flightData = FD.getFlightData(
                description=desc,
                landed=True
            )

            for fd in flightData:
                fd['loadNum'] = loadNum
                fd['timestamp'] = timestampPrev + fd['timestamp']
            
            timestampPrev = flightData[-1]['timestamp']

            dataset += flightData
            
            print('--> Processed [{}][{}/{}]'.format(loadNum, dArrCount, len(descList)), desc, end='\r')
            
    print('')
    print('--> Aggregating: {} data points'.format(len(dataset)))
    aggregateDataset = FD.aggregate(
        flight_data=dataset,
        time_window=timeWindow,
    )

    # Dataset save to file
    print('[o] Saving dataset to', datasetFilename, end='\r')
    np.array(aggregateDataset)
    np.save(cachePath + '/' + datasetFilename, dataset)
    print('[v] Saved dataset to', datasetFilename, '   ')

else:
    print('[v] Cached dataset found:', datasetFilename)

    # Load dataset from file
    print('[o] Loading dataset from', datasetFilename, end='\r')
    aggregateDataset = list(np.load(cachePath + '/' + datasetFilename, allow_pickle=True))
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
    dataArr = dataset

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
                dataArr[nd+ns]['loadNum'],
                dataArr[nd+ns]['pwm'],                      # 0
                *dataArr[nd+ns]['orientation']
            ]).reshape(1,n_feature[0])

            sequenceArrInput = np.append(sequenceArrInput, featureArrInput, axis=0)
        
        # Add output feature array
        featureArrOutput = np.array([
            *dataArr[nd+n_sequence-1]['rms'],               # 0
            *dataArr[nd+n_sequence-1]['kurtosis'],
            *dataArr[nd+n_sequence-1]['skewness'],
            *dataArr[nd+n_sequence-1]['crest-factor'],
            *dataArr[nd+n_sequence-1]['peak-to-peak'],
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


### Dataset splitting
testIdxStart = int(nTest * len(aggregateDataset))
testDataset = aggregateDataset[-testIdxStart:]
trainDataset = aggregateDataset[:-testIdxStart]

trainDatasetInput, trainDatasetOutput, trainDatasetTimestamp = getSequenceArray(trainDataset, nSequence, (nFeatureInput,nFeatureOutput))
testDatasetInput, testDatasetOutput, testDatasetTimestamp = getSequenceArray(testDataset, nSequence, (nFeatureInput,nFeatureOutput))

# Stateful -> dividable by batch size
#trainDatasetInput = trainDatasetInput[:4736,:,:]
#trainDatasetOutput = trainDatasetOutput[:4736,:]
#testDatasetInput = testDatasetInput[:1920,:,:]
#testDatasetOutput = testDatasetOutput[:1920,:]

# Overview of train and test dataset
print('')
print('-'*40)
print('Dataset Overview Confirmation')
print('-'*40)
print('Number of data:', len(testDataset)+len(trainDataset))
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

""" plt.plot(testDatasetOutput[:,2], 'k-')
plt.plot(testDatasetInput[:,-1,0], 'r-')
plt.grid(True)
plt.show() """

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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silent log
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

def createModel(early_stop=True, checkpoint=True):
    # Create model
    model = Sequential()
    model.add(LSTM(512, activation='tanh', input_shape=(nSequence, nFeatureInput), stateful=True))
    #model.add(LSTM(512, activation='tanh', batch_input_shape=(128, nSequence, nFeatureInput), stateful=True))
    model.add(Dense(nFeatureOutput))

    #model.summary()
    #sys.exit()

    # Add Ons
    addOns = []

    if early_stop:
        addOns.append(EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='min', restore_best_weights=True))

    if checkpoint:
        trainTime = str(datetime.now().strftime('%y_%m_%d_%H_%M_%S'))
        addOns.append(ModelCheckpoint('D:/cache_new/checkpoint/' + trainTime + '_{epoch:03d}-{loss:03f}-{val_loss:03f}.h5', save_best_only=True, monitor='val_loss', mode='min'))
    

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
    
    with open(os.path.join(cachePath, filename), 'rb') as historyFile:
        history = pickle.load(historyFile)

    return history


def exportModel(model):
    filename = modelFilename + '_' + str(timeWindow) + '_' + '_'.join(filterDesc) + '.h5'
    
    print('-'*30)
    print('Saving model  to {}'.format(filename))
    model.save(os.path.join(cachePath, filename))


def exportHistory(history):
    filename = modelFilename + '_' + str(timeWindow) + '_' + '_'.join(filterDesc) + '.pkl'
    with open(os.path.join(cachePath, filename), 'wb') as historyFile:
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
        xticks = list(range(0, (int(xData[-1]) + 2), 25))

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


###################################################
################ Train & Predict ##################
###################################################

trainTime = 0

### Model train iteration by output feature count
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
            batch_size=128,
            verbose=1
        )

        # Export model
        print('Exporting model')
        exportModel(model)
        exportHistory(history)

    else:
        model = loadModel()



### Prediction
if predict:
    print('Starting predicting...')
    #testDatasetOutput = getEWM(testDatasetOutput)

    # Load file only
    """ if not train:
        model = loadModel(loadFilename)
        history = loadHistory(loadHistoryFilename) """

    predOutput = []

    if trainTime == 0:
        trainTime = str(datetime.now().strftime('%y_%m_%d_%H_%M_%S'))


    ### Predict trained data
    predArr = np.array([]).reshape(0,nFeatureOutput)
    i = 0
    for datapoint in range(trainDatasetInput.shape[0]):
        i += 1
        print('Data Point: {}/{}'.format(i,trainDatasetInput.shape[0]), end='\r')
        predArr = np.append(predArr, model.predict(trainDatasetInput[datapoint,:,:].reshape(1,nSequence,nFeatureInput), verbose=0).reshape(1,nFeatureOutput), axis=0)


    ### Calculate Metrics - Train ###
    loadData = list(trainDatasetInput[:,-1,0])
    loadSet = sorted(set([int(ld) for ld in loadData]))
    loadIdxChange = [loadData.index(load) for load in loadSet] + [len(loadData)]
    print(loadSet)
    print(loadIdxChange)

    metricsLoad = np.array([]).reshape(0,4)
    loadIdx = 0
    for load in loadSet:
        metricsArr = np.array([]).reshape(0,4)
        for featNum in range(len(featureName)):
            trueData = trainDatasetOutput[loadIdxChange[loadIdx]:loadIdxChange[loadIdx+1], featNum]
            predData = predArr[loadIdxChange[loadIdx]:loadIdxChange[loadIdx+1], featNum]

            print('true: {} pred: {}'.format(len(trueData), len(predData)))

            metricsVal = np.array([
                load,
                RMSE(trueData, predData),
                MAE(trueData, predData),
                RSquared(trueData, predData)
            ]).reshape(1,4)

            metricsArr = np.append(metricsArr, metricsVal, axis=0)
        
        metricsLoad = np.append(metricsLoad, metricsArr, axis=0)
        loadIdx += 1


    ### Predict test data
    predArr = np.array([]).reshape(0,nFeatureOutput)
    i = 0
    for datapoint in range(testDatasetInput.shape[0]):
        i += 1
        print('Data Point: {}/{}'.format(i,testDatasetInput.shape[0]), end='\r')
        predArr = np.append(predArr, model.predict(testDatasetInput[datapoint,:,:].reshape(1,nSequence,nFeatureInput), verbose=0).reshape(1,nFeatureOutput), axis=0)

    ### Calculate Metrics - Test ###
    loadData = list(testDatasetInput[:,-1,0])
    loadSet = sorted(set([int(ld) for ld in loadData]))
    loadIdxChange = [loadData.index(load) for load in loadSet] + [len(loadData)]
    print(loadSet)
    print(loadIdxChange)

    loadIdx = 0
    for load in loadSet:
        metricsArr = np.array([]).reshape(0,4)
        for featNum in range(len(featureName)):
            trueData = testDatasetOutput[loadIdxChange[loadIdx]:loadIdxChange[loadIdx+1], featNum]
            predData = predArr[loadIdxChange[loadIdx]:loadIdxChange[loadIdx+1], featNum]

            print('true: {} pred: {}'.format(len(trueData), len(predData)))

            metricsVal = np.array([
                load,
                RMSE(trueData, predData),
                MAE(trueData, predData),
                RSquared(trueData, predData)
            ]).reshape(1,4)

            metricsArr = np.append(metricsArr, metricsVal, axis=0)
        
        metricsLoad = np.append(metricsLoad, metricsArr, axis=0)
        loadIdx += 1

    np.savetxt(
        cachePath + '/metrics/' + metricsFilename.format(trainTime, timeWindow),
        metricsLoad,
        delimiter=',',
        header='load,rmse,mae,rsquared'
    )

    ### Plotting Training Metrics ###
    """ plotMetricsHistory(
        history=history
    ) """

    ### Save prediction plot
    """ if savePredictPlot:
        
        plt.savefig(
            fname=plotPath + '/' + predPlotFilename.format(trainTime + '_' + str(timeWindow), 'all', 'hist')
        ) """
    
    ### Plotting Predicted Value vs Real Value for each feature ###
    print('Predicting output')
    os.mkdir(cachePath + '/plot/{}'.format(trainTime))
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
                fname= cachePath + '/plot/{}/vm_plot_{}.jpg'.format(trainTime, featureName[featNum])
            )
        else:
            plt.show()