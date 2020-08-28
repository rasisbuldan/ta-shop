from navdataVibProc import NavData, VibData, NavdataVib
import numpy as np
import scipy.stats
import math
import sys
import random
import traceback
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from keras.models import load_model

#############################
##### Parameter Options #####
#############################

fileName = 'seq_agg_pred_1k1s_aug14'
modelPath = 'D:/Cloud/Google Drive/Tugas Akhir/data/cache/Aug21'
dumpPath = 'D:/Cloud/Google Drive/Tugas Akhir/data/dumps/seq_agg_load_pred'
dataPath = 'D:/Cloud/Google Drive/Tugas Akhir/data/cache/Aug21/'
modelFilename = [
    'lstm_navdatavib_model_multidenseoutfilter_aug20_20_08_26_09_24_23_250_aug9_0_aug7_2_jul29_2.h5',
    'lstm_navdatavib_model_multidenseoutfilter_aug20_20_08_26_09_34_20_250_aug10_1.h5',
    'lstm_navdatavib_model_multidenseoutfilter_aug20_20_08_26_09_39_16_250_aug10_2.h5',
    'lstm_navdatavib_model_multidenseoutfilter_aug20_20_08_26_09_40_33_250_aug10_3.h5',
    'lstm_navdatavib_model_multidenseoutfilter_aug20_20_08_26_09_43_05_250_aug11_4.h5',
    'lstm_navdatavib_model_multidenseoutfilter_aug20_20_08_26_09_46_51_250_aug11_5.h5',
    'lstm_navdatavib_model_multidenseoutfilter_aug20_20_08_26_09_49_23_250_aug11_6.h5'
]
datasetFilename = [
    'lstm_navdatavib_dataset_agg_aug21_multioutfilter_250_aug9_0',
    'lstm_navdatavib_dataset_agg_aug21_multioutfilter_250_aug10_1',
    'lstm_navdatavib_dataset_agg_aug21_multioutfilter_250_aug10_2',
    'lstm_navdatavib_dataset_agg_aug21_multioutfilter_250_aug10_3',
    'lstm_navdatavib_dataset_agg_aug21_multioutfilter_250_aug11_4',
    'lstm_navdatavib_dataset_agg_aug21_multioutfilter_250_aug11_5',
    'lstm_navdatavib_dataset_agg_aug21_multioutfilter_250_aug11_6'
]

numOfStep = 6
timeWindow = 1000   # ms

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
epochNum = 200

###################################################
################## Data Filtering #################
###################################################

# Include substring
filterDesc = [
    ['aug9_0', 'aug7_2', 'jul29_2'],
    ['aug10_1'],
    ['aug10_2'],
    ['aug10_3'],
    ['aug11_4'],
    ['aug11_5'],
    ['aug11_6']
]

# Exclude subsstring
excludeDesc = ['crash', 'fail', 'test', '10s']

# Discard specific filename
discardDesc = [
    'aug10_testfail.json',
    'aug10_1_test.json',
    'aug10_1_test_2.json',
    'aug10_2_hover20s_4_crash.json',
    'aug10_3_hover20s_fail.json',
    'aug11_4_hover20s_2_crash.json',
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
useCachedDataset = True
useCachedModel = False
train = True
predict = True
savePredictPlot = True
bypassCheckpoint = True
predictPlot = True
earlyExit = True
verboseFiltered = False
plotVibAxis = ['x', 'y', 'z']



###################################################
########## File Traversal & Aggregation ###########
###################################################

NV = NavdataVib()


def getDirectoryList(data_dir):
    '''
        Get list of file in directory
    '''
    return [f for f in os.listdir(data_dir) if '.npy' in f]


combinedDatasetArr = []
for i in range(7):
    # Load dataset from file
    print('[o] Loading dataset from', datasetFilename[i] + str('.npy'), end='\r')
    combinedDataset = list(np.load(dataPath + datasetFilename[i] + str('.npy'), allow_pickle=True))
    print('[v] Loaded dataset from', datasetFilename[i])
    combinedDatasetArr.append(combinedDataset)



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

combSum = 0
for data in combinedDataset:
    dataLen = len(data['data'])
    combSum += dataLen

# Overview of train and test dataset
print('')
print('-'*40)
print('Dataset Overview Confirmation')
print('-'*40)
print('Number of data:', combSum)
print('Sequence length:', nSequence)
print('Number of Input Feature:', nFeatureInput)
print('Number of Output Feature:', nFeatureOutput)
print('-'*40)
print('Dataset')
print('-'*40)
print('Number of dataset(s):', len(combinedDatasetArr))

##### Split data into input/output sequence #####
datasetArr = []
for i in range(7):
    datasetInput, datasetOutput, datasetTimestamp = getSequenceArray(combinedDatasetArr[i], nSequence, (nFeatureInput,nFeatureOutput))

    datasetArr.append({
        'input': datasetInput,
        'output': datasetOutput
    })

    print(i, 'Input:', datasetInput.shape)
    print(i, 'Output:', datasetOutput.shape)


###################################################
################ Model Prediction #################
###################################################

for i in range(7):
    print('Predicting', modelFilename[i])
    print('-> Loading Model')
    model = load_model(os.path.join(modelPath, modelFilename[i]))

    predArr = np.array([]).reshape(0, nFeatureOutput)
    datasetInput = datasetArr[i]['input']
    datasetOutput = datasetArr[i]['output']

    print('-> Getting prediction')
    for datapoint in range(datasetInput.shape[0]):
        # Get prediction
        predArr = np.append(
            predArr, 
            model.predict(
                datasetInput[datapoint,:,:].reshape(1,nSequence,nFeatureInput), 
                verbose=0).reshape(1,nFeatureOutput), 
            axis=0)

    # Add pwm
    pwmArr = datasetInput[:,-1,0].reshape(datasetInput.shape[0],1)
    predArr = np.append(pwmArr, predArr, axis=1)
    
    print('-> Aggregating')
    predArrWindow = np.array([]).reshape(0, nFeatureOutput+1)
    buf = np.array([]).reshape(0, nFeatureOutput+1)
    for d in range(predArr.shape[0]):
        buf = np.append(buf, predArr[d,:].reshape(1, nFeatureOutput+1), axis=0)
        if d % 4 == 0:
            predArrWindow = np.append(predArrWindow, np.mean(buf, axis=0).reshape(1,nFeatureOutput+1), axis=0)
            buf = np.array([]).reshape(0, nFeatureOutput+1)

    print('-> Saving')
    np.savetxt(
        fname=dumpPath + '/' + fileName + '_' + str(i) + '.csv',
        X=predArrWindow,
        fmt='%s',
        delimiter=',',
        header='pwm1,rms1x,rms1y,rms1z,kurt1x,kurt1y,kurt1z,skew1x,skew1y,skew1z,crest1x,crest1y,crest1z,peak1x,peak1y,peak1z'
    )