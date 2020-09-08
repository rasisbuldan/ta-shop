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
from sklearn.metrics import recall_score, precision_score, f1_score

###################################################
#################### Parameter ####################
###################################################

# Dataset Preparation
nTest = 0.3
timeWindow =  1000    # in ms

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
nSequence = 60
nFeatureInput = 5   # (average) rms, kurt, skew, crest, p2p
nFeatureOutput = 1
epochNum = 600

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
train = False
predict = True
predictPlot = True
savePredictPlot = True
verboseFiltered = False

# Random reproducibility
random.seed(1132) # 12.3


### Dataset and model caching ###
cachePath = 'D:/cache_new'
dataPath = 'D:/mc'
datasetFilename = 'mc_dataset_' + str(timeWindow) + '_' + '_'.join(filterDesc) + '.npy'
modelFilename = 'mc_model_' + str(datetime.now().strftime('%y_%m_%d_%H_%M_%S')) + '.h5'
metricsFilename = 'mc_metrics_' + str(datetime.now().strftime('%y_%m_%d_%H_%M_%S')) + '.csv'

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
                landed=False
            )

            for fd in flightData:
                fd['loadNum'] = loadNum
                fd['timestamp'] = timestampPrev + fd['timestamp']
            
            timestampPrev = flightData[-1]['timestamp']

            dataset += flightData
            
            print('--> Processed [{}][{}/{}]'.format(loadNum, dArrCount, len(descList)), desc, end='\r')
            
    print('')
    print('--> Aggregating: {} data points'.format(len(dataset)))
    aggregateDataset = FD.aggregateMulti(
        flight_data=dataset,
        time_window=timeWindow,
        load_num=True
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
    seqArrInput: numpy array with shape (nData, n_sequence, n_feature[0]+1) # plus loadNum on beginning
    seqArrOutput: numpy array with shape (nData, n_feature[1]) -> shifted n_sequence to right

    '''
    dataArr = dataset

    # Iterate nData
    nData = len(dataArr) - n_sequence + 1

    seqSetArrInput = np.array([]).reshape(0,n_sequence,n_feature[0]+1)
    seqSetArrOutput = np.array([]).reshape(0,n_feature[1])
    
    # Timestamp concat
    timestampArrOutput = []
    nSet = 0
    newSetTime = 0
    tsPrev = 0
    
    # Iterate over data points
    for nd in range(nData):

        sequenceArrInput = np.array([]).reshape(0,n_feature[0]+1)

        # Iterate over sequence
        
        # Abnormal motor
        if dataArr[nd+n_sequence-1]['loadNum'] >= 4 or dataArr[nd+n_sequence-1]['loadNum'] == 0:
            if dataArr[nd+n_sequence-1]['loadNum'] >= 4:
                motKey = 'mot1'
                loadNum = dataArr[nd+n_sequence-1]['loadNum']
            else:
                motKey = 'mot2'
                loadNum = 7

            for ns in range(n_sequence):

                # Add input feature array
                featureArrInput = np.array([
                    loadNum,
                    dataArr[nd+ns][motKey]['rms'][1],
                    dataArr[nd+ns][motKey]['kurtosis'][1],
                    dataArr[nd+ns][motKey]['skewness'][1],
                    dataArr[nd+ns][motKey]['crest-factor'][1],
                    dataArr[nd+ns][motKey]['peak-to-peak'][1]
                ]).reshape(1,n_feature[0]+1)    # plus loadNum

                sequenceArrInput = np.append(sequenceArrInput, featureArrInput, axis=0)
            
            featureArrOutput = np.array([1]).reshape(1,n_feature[1])

            # Append into sequence set
            seqSetArrInput = np.append(seqSetArrInput, sequenceArrInput.reshape(1,n_sequence,n_feature[0]+1), axis=0)
            seqSetArrOutput = np.append(seqSetArrOutput, featureArrOutput, axis=0)

        sequenceArrInput = np.array([]).reshape(0,n_feature[0]+1)

        # Normal motor
        if dataArr[nd+n_sequence-1]['loadNum'] < 4:
            for ns in range(n_sequence):

                # Add input feature array
                featureArrInput = np.array([
                    dataArr[nd+n_sequence-1]['loadNum'],
                    dataArr[nd+ns]['mot1']['rms'][1],
                    dataArr[nd+ns]['mot1']['kurtosis'][1],
                    dataArr[nd+ns]['mot1']['skewness'][1],
                    dataArr[nd+ns]['mot1']['crest-factor'][1],
                    dataArr[nd+ns]['mot1']['peak-to-peak'][1]
                ]).reshape(1,n_feature[0]+1)    # plus loadNum

                sequenceArrInput = np.append(sequenceArrInput, featureArrInput, axis=0)
            
            featureArrOutput = np.array([0]).reshape(1,n_feature[1])

            # Append into sequence set
            seqSetArrInput = np.append(seqSetArrInput, sequenceArrInput.reshape(1,n_sequence,n_feature[0]+1), axis=0)
            seqSetArrOutput = np.append(seqSetArrOutput, featureArrOutput, axis=0)

        # Append into sequence set
        #seqSetArrInput = np.append(seqSetArrInput, sequenceArrInput.reshape(1,n_sequence,n_feature[0]+1), axis=0)
        #seqSetArrOutput = np.append(seqSetArrOutput, featureArrOutput, axis=0)
        
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


# Get sequence
datasetInput, datasetOutput, datasetTimestamp = getSequenceArray(aggregateDataset, nSequence, (nFeatureInput,nFeatureOutput))

# Get random sample
rowsTrain = list(range(datasetInput.shape[0]))
rowsTest = []
testDataCount = int(datasetInput.shape[0] * nTest)
for _ in range(testDataCount):
    rowsTest.append(rowsTrain.pop(random.randrange(len(rowsTrain))))

trainDatasetInput = datasetInput[rowsTrain, :, :]
trainDatasetOutput = datasetOutput[rowsTrain, :]
testDatasetInput = datasetInput[rowsTest, :, :]
testDatasetOutput = datasetOutput[rowsTest, :]
#trainDatasetInput, trainDatasetOutput, trainDatasetTimestamp = getSequenceArray(trainDataset)
#testDatasetInput, testDatasetOutput, testDatasetTimestamp = getSequenceArray(testDataset, nSequence, (nFeatureInput,nFeatureOutput))

# Overview of train and test dataset
print('')
print('-'*40)
print('Dataset Overview Confirmation')
print('-'*40)
#print('Number of data:', len(testDataset)+len(trainDataset))
print('Sequence length:', nSequence)
print('Number of Input Feature:', nFeatureInput)
print('Number of Output Feature:', nFeatureOutput)
print('')
print('-'*40)
print('Train dataset')
print('-'*40)
#print('Number of dataset(s):', len(trainDataset))
print('Input:', trainDatasetInput.shape)
print('Output:', trainDatasetOutput.shape)
print('')
print('-'*40)
print('Test dataset')
print('-'*40)
#print('Number of dataset(s):', len(testDataset))
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
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json
#from tensorflow.keras import initializers
from tensorflow.keras.initializers import glorot_uniform

def createModel(early_stop=True, checkpoint=True):
    # Create model
    model = Sequential()
    #model.add(LSTM(512, activation='tanh', input_shape=(nSequence, nFeatureInput), stateful=True))
    model.add(LSTM(5, activation='tanh', input_shape=(nSequence, nFeatureInput)))
    model.add(Dense(nFeatureOutput, activation='sigmoid'))

    #model.summary()
    #sys.exit()

    #with open('C:/Users/rss75/Documents/GitHub/ta-shop/source/ardrone/new/model/modelclass.json', 'r') as modelFile:
    #    loadedModel = model_from_json(modelFile.read())

    # Compile model
    #loadedModel.summary()
    #sys.exit()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def loadModel(filename=None):
    if filename == None:
        filename = modelFilename + '_' + str(timeWindow) + '_' + '_'.join(filterDesc) + '.h5'
        filename = os.path.join(dataPath, filename)
    

    print('-'*30)
    print('Loading model from {}'.format(filename))
    model = load_model(filename)

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
        model = createModel(
            early_stop=False,
            checkpoint=True
        )

        # Fit model
        print('Fitting model')
        history = model.fit(
            x=trainDatasetInput[:,:,1:],
            y=trainDatasetOutput,
            validation_data=(testDatasetInput[:,:,1:], testDatasetOutput),
            epochs=epochNum,
            batch_size=128,
            verbose=1
        )

        # Export model
        print('Exporting model')
        exportModel(model)
        exportHistory(history)

    else:
        model = loadModel('C:/Users/rss75/Documents/GitHub/ta-shop/source/ardrone/new/model/')



### Prediction
if predict:
    print('Starting predicting...')
    #testDatasetOutput = getEWM(testDatasetOutput)

    # Load file only
    if not train:
        model = loadModel('D:/mc/mc_model_20_09_05_02_59_02.h5_1000_aug9_0_aug10_1_aug10_2_aug10_3_aug11_4_aug11_5_aug11_6.h5')
        #history = loadHistory(loadHistoryFilename)

    predOutput = []

    if trainTime == 0:
        trainTime = str(datetime.now().strftime('%y_%m_%d_%H_%M_%S'))

    ### Predict test data
    predArr = np.array([]).reshape(0,nFeatureOutput)
    i = 0
    for datapoint in range(testDatasetInput.shape[0]):
        i += 1
        print('Predicting Data Point: {}/{}'.format(i,testDatasetInput.shape[0]), end='\r')
        predArr = np.append(predArr, model.predict(testDatasetInput[datapoint,:,1:].reshape(1,nSequence,nFeatureInput), verbose=0).reshape(1,nFeatureOutput), axis=0)

    ### Calculate metrics
    precision = precision_score(testDatasetOutput, np.around(predArr, decimals=0), average='binary')
    recall = recall_score(testDatasetOutput, np.around(predArr, decimals=0), average='binary')
    score = f1_score(testDatasetOutput, np.around(predArr, decimals=0), average='binary')
    print('precision: {:.4f} | recall: {:.4f} | score: {:.4f}'.format(precision, recall, score))

    ### Box plot
    loadMask = np.around(testDatasetInput[:,-1,0]).reshape(-1,1)
    print(loadMask.shape, predArr.shape)
    predArr = np.append(loadMask, predArr, axis=1)
    predArr = predArr[np.argsort(predArr[:,0])]
    

    # Split into load
    print('set', set(list(predArr[:,0])))
    #print(predArr[:50,:])
    #print(np.diff(predArr[:,0]))
    loadArray = np.split(predArr, np.where(np.diff(predArr[:,0]))[0]+1)
    loadArrayFiltered = [la[:,1] for la in loadArray]
    #print(loadArray)
    #print(len(loadArray))
    #print(len(loadArrayFiltered), loadArrayFiltered[0].shape)
    fig = plt.figure(figsize=(10,6))
    plt.boxplot(loadArrayFiltered)
    plt.ylabel('Kondisi', fontsize=18)
    plt.xticks(
        [1,2,3,4,5,6,7,8],
        ['0\n(Baru)', '1', '2', '3', '4', '5', '6', '0\n(Lama)'],
        fontsize=16
    )
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1], fontsize=14)
    plt.savefig('boxplot_rev4.png')
    plt.show()

    #plt.scatter(predArr[:,0], predArr[:,1])
    #plt.show()