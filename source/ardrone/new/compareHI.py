from flightdataProc import FlightData
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
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
cachePath = 'D:/cache_new'
modelFilename = 'D:/cache_new/vm_model_20_09_02_14_19_15.h5_250_aug9_0_aug10_1_aug10_2_aug10_3_aug11_4_aug11_5_aug11_6.h5'
datasetFilename = 'D:/cache_new/vm_dataset_250_aug9_0_aug10_1_aug10_2_aug10_3_aug11_4_aug11_5_aug11_6.npy'

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

FD = FlightData()

# Load dataset from file
print('[o] Loading dataset from', datasetFilename[i] + str('.npy'), end='\r')
combinedDataset = list(np.load(cachePath + '/' + datasetFilename, allow_pickle=True))
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

combDataset = np.array([]).reshape(0, (nFeatureOutput*2)+2)

# Iterate over load level
for i in range(7):
    print('Predicting', modelFilename[i])
    print('-> Loading Model')
    model = load_model(os.path.join(modelPath, modelFilename[i]))


    datasetInput = datasetArr[i]['input']
    datasetOutput = datasetArr[i]['output']
    predArr = np.array([]).reshape(0, nFeatureOutput)
    realArr = datasetOutput

    print('-> Getting prediction')
    for datapoint in range(datasetInput.shape[0]):
        # Get prediction
        predArr = np.append(
            predArr, 
            model.predict(
                datasetInput[datapoint,:,:].reshape(1,nSequence,nFeatureInput), 
                verbose=0).reshape(1,nFeatureOutput), 
            axis=0)

    # Combine array
    try:
        combArr = np.append(realArr, predArr, axis=1)
    except:
        print(realArr.shape, predArr.shape)
    
    pwmArr = datasetInput[:,-1,0].reshape(datasetInput.shape[0],1)
    loadNumArr = np.full((datasetInput.shape[0],1), i)
    combArr = np.append(pwmArr, combArr, axis=1)
    combArr = np.append(loadNumArr, combArr, axis=1)

    
    # Aggregating into 1000ms time window
    print('-> Aggregating')
    combWindow = np.array([]).reshape(0, (nFeatureOutput*2)+2)
    combBuf = np.array([]).reshape(0, (nFeatureOutput*2)+2)
    for d in range(predArr.shape[0]):
        combBuf = np.append(combBuf, combArr[d,:].reshape(1, (nFeatureOutput*2)+2), axis=0)
        if d % 4 == 0:
            combWindow = np.append(combWindow, np.mean(combBuf, axis=0).reshape(1, (nFeatureOutput*2)+2), axis=0)
            combBuf = np.array([]).reshape(0, (nFeatureOutput*2)+2)

    combDataset = np.append(combDataset, combWindow, axis=0)
    print(i, combDataset.shape)

    """ print('-> Saving')
    np.savetxt(
        fname=dumpPath + '/' + fileName + '_' + str(i) + '.csv',
        X=predArrWindow,
        fmt='%s',
        delimiter=',',
        header='pwm1,rms1x,rms1y,rms1z,kurt1x,kurt1y,kurt1z,skew1x,skew1y,skew1z,crest1x,crest1y,crest1z,peak1x,peak1y,peak1z'
    ) """


# Dataframe processing
cd = pd.DataFrame(
    data=combDataset,
    columns=[
        'loadNum',
        'pwm',
        'rms1xr',
        'rms1yr',
        'rms1zr',
        'kurt1xr',
        'kurt1yr',
        'kurt1zr',
        'skew1xr',
        'skew1yr',
        'skew1zr',
        'crest1xr',
        'crest1yr',
        'crest1zr',
        'p2p1xr',
        'p2p1yr',
        'p2p1zr',
        'rms1xp',
        'rms1yp',
        'rms1zp',
        'kurt1xp',
        'kurt1yp',
        'kurt1zp',
        'skew1xp',
        'skew1yp',
        'skew1zp',
        'crest1xp',
        'crest1yp',
        'crest1zp',
        'p2p1xp',
        'p2p1yp',
        'p2p1zp',
    ]
)

# Add HI value
cd = cd.iloc[100:]
cd = cd[cd['pwm'] != 0]
realSelected = cd[['rms1yr', 'kurt1yr', 'skew1yr', 'crest1yr']]
predSelected = cd[['rms1yp', 'kurt1yp', 'skew1yp', 'crest1yp']]

# PCA (real)
realPca = PCA(n_components=1)
realPca.fit(realSelected)
print(realPca.get_covariance())
sys.exit()
cd['rHI'] = realPca.transform(realSelected)
cd['rHIs'] = cd[['rHI']].ewm(alpha=0.01).mean()['rHI']

# Scaler (real)
realScaler = MinMaxScaler()
realScaler.fit(cd['rHIs'].to_numpy().reshape(-1,1))
cd['rHIss'] = realScaler.transform(cd['rHIs'].to_numpy().reshape(-1,1))


# PCA (pred)
predPca = PCA(n_components=1)
predPca.fit(predSelected)
cd['pHI'] = predPca.transform(predSelected)
cd['pHIs'] = cd[['pHI']].ewm(alpha=0.01).mean()['pHI']

# Scaler (pred)
predScaler = MinMaxScaler()
predScaler.fit(cd['pHIs'].to_numpy().reshape(-1,1))
cd['pHIss'] = predScaler.transform(cd['pHIs'].to_numpy().reshape(-1,1))

# Manual scaler
def rescale(x):
    return (x+5)*100/15

cd['pHIsm'] = cd[['pHIs']].applymap(rescale)
cd['rHIsm'] = cd[['rHIs']].applymap(rescale)


plt.plot(cd['pHIsm'], 'k-')
plt.plot(cd['rHIsm'], 'r-')
plt.ylim(0,100)

plt.plot(cd['loadNum']*10, 'g-')
plt.xlabel('Waktu (s)', fontsize=22)
plt.ylabel('Indikator Kesehatan (%)', fontsize=22)
plt.tick_params(grid_alpha=0.6, grid_linewidth=0.4, labelsize=18)
plt.grid(True)
plt.show()

#cd.to_csv('comparedataHI2.csv')