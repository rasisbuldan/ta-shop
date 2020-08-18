'''
    LSTM Nav-Vib Data with Multiple Output - Separate model
    - Fixed seed
'''


# Data processing
import sys
print(sys.version)
from navdataVibProc import VibData, NavData, NavdataVib
import scipy
import numpy as np
import sys
import matplotlib.pyplot as plt
import random
import time
import os
import math
import json
from datetime import datetime
import pickle


###################################################
#################### Parameter ####################
###################################################

# Dataset Preparation
nTest = 3
timeWindow =  250    # in ms


featureName = [
    'rms.x',
    'rms.y',
    'rms.z',
    'kurtosis.x',
    'kurtosis.y',
    'kurtosis.z',
    'skewness.x',
    'skewness.y',
    'skewness.z',
    'crest-factor.x',
    'crest-factor.y',
    'crest-factor.z',
    'peak-to-peak.x',
    'peak-to-peak.y',
    'peak-to-peak.z',
]


""" featureName = [
    'rms.x',
    'rms.y',
    'rms.z'
] """

# LSTM
nSequence = 8
nFeatureInput = 4   # pwm1, pitch, roll, yaw, acc.x, acc.y, acc.z
nFeatureOutput = 15  # rms.x, rms.y, rms.z
epochNum = 1000

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
filterDesc = ['aug9_0'] # jul_29


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
useCachedModel = True
train = False
predict = True
savePredictPlot = True
predictPlot = False
earlyExit = True
verboseFiltered = False
plotVibAxis = ['x', 'y', 'z']

# Random reproducibility
#random.seed(int(time.time()))
random.seed(1132)

### Dataset and model save-load ###
dataPath = 'C:/Users/rss75/Documents/GitHub/ta-shop/source/ardrone'
modelFilename = 'cache/lstm_navdatavib_model_aug16'
datasetFilename = 'lstm_navdatavib_dataset_agg_aug16_multiout_' + str(timeWindow) + '_' + '_'.join(filterDesc)
predPlotFilename = '/plot/{}/plot_pred_{}_{}_{}.jpg'


###################################################
########## File Traversal & Aggregation ###########
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


""" print('[o] Getting description list...', end='\r')
descArray = [nav['description'] for nav in Nav.listDescriptionTimestamp()[1:]]
print('[v] Getting description list done!\n')
print(*descArray, sep='\n')
sys.exit() """

# If dataset not cached
if (datasetFilename + '.npy') not in getDirectoryList(os.path.join(dataPath,'cache')) or not useCachedDataset:
    print('[!] Cached dataset not found, doing file traversal')

    # List description
    print('[o] Getting description list...', end='\r')
    descArray = [nav['description'] for nav in Nav.listDescriptionTimestamp()[1:]]
    print('[v] Getting description list done!\n')
    print(*descArray, sep='\n')

    combinedDataset = []

    # Dataset traversal
    dArrCount = 0
    for desc in descArray:
        dArrCount += 1
        # Data filtering
        # skipping substring in *discardDesc* or *filterDesc* existed in *desc* string
        if (desc in discardDesc) or not any(fd in desc for fd in filterDesc):
            if verboseFiltered:
                print('--------\n[{}/{}] Filtered data:'.format(dArrCount, len(descArray)), desc)
            continue
        print('------------------------------------------------------------')
        print('[{}/{}] Processing:'.format(dArrCount, len(descArray)), desc)

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
        print('[o] Combining navigation and vibration data...', end='\r')
        combinedData = NV.combineDataMultiFeature(
            navdata=Nav.getMultiFeatureArray(navId),
            vibdata=Vib.getMultiFeatureArray(vibId)
        )

        # Get feature aggregation
        print('[o] Calculating feature aggregation data...', end='\r')
        combinedDataAgg = NV.aggregateCombined(
            combined_data=combinedData,
            time_window=timeWindow
        )

        #print('\n\nCombined data agg')
        #print(json.dumps(combinedDataAgg[0], indent=2))

        print('[o] Appending to combined dataset...', end='\r')
        combinedDataset.append(
            {
                'description': desc,
                'data': combinedDataAgg
            }
        )

        print('--> Processed:'.format(dArrCount, len(descArray)), desc)


    # Dataset save to file
    print('[o] Saving dataset to', datasetFilename, end='\r')
    np.array(combinedDataset)
    np.save(dataPath + '/cache/' + datasetFilename, combinedDataset)
    print('[v] Saved dataset to', datasetFilename)

else:
    print('[v] Cached dataset found:', datasetFilename + '.npy')

    # Load dataset from file
    print('[o] Loading dataset from', datasetFilename + str('.npy'), end='\r')
    combinedDataset = list(np.load(dataPath + '/cache/' + datasetFilename + str('.npy'), allow_pickle=True))
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
    seqSetArrTimestamp = []
    
    # Iterate over data points
    for nd in range(nData):

        sequenceArrInput = np.array([]).reshape(0,n_feature[0])

        # Iterate over sequence
        for ns in range(n_sequence):
            
            # Add input feature array
            featureArrInput = np.array([
                dataArr[nd+ns]['mot1']['pwm'],                      # 0
                dataArr[nd+ns]['orientation'][0],
                dataArr[nd+ns]['orientation'][1],
                dataArr[nd+ns]['orientation'][2],
            ]).reshape(1,n_feature[0])

            """ dataArr[nd+ns]['orientation'][0],
                dataArr[nd+ns]['orientation'][1],
                dataArr[nd+ns]['orientation'][2],
                dataArr[nd+ns]['rawAccel'][0],
                dataArr[nd+ns]['rawAccel'][1],                      # 5
                dataArr[nd+ns]['rawAccel'][2],                      # 6 """

            sequenceArrInput = np.append(sequenceArrInput, featureArrInput, axis=0)
        
        # Add output feature array
        featureArrOutput = np.array([
            dataArr[nd+n_sequence-1]['mot1']['rms'][0],               # 0
            dataArr[nd+n_sequence-1]['mot1']['rms'][1],
            dataArr[nd+n_sequence-1]['mot1']['rms'][2],
            dataArr[nd+n_sequence-1]['mot1']['kurtosis'][0],
            dataArr[nd+n_sequence-1]['mot1']['kurtosis'][1],
            dataArr[nd+n_sequence-1]['mot1']['kurtosis'][2],          # 5
            dataArr[nd+n_sequence-1]['mot1']['skewness'][0],
            dataArr[nd+n_sequence-1]['mot1']['skewness'][1],
            dataArr[nd+n_sequence-1]['mot1']['skewness'][2],
            dataArr[nd+n_sequence-1]['mot1']['crest-factor'][0],
            dataArr[nd+n_sequence-1]['mot1']['crest-factor'][1],      # 10
            dataArr[nd+n_sequence-1]['mot1']['crest-factor'][2],
            dataArr[nd+n_sequence-1]['mot1']['peak-to-peak'][0],
            dataArr[nd+n_sequence-1]['mot1']['peak-to-peak'][1],
            dataArr[nd+n_sequence-1]['mot1']['peak-to-peak'][2],      # 14
        ]).reshape(1,n_feature[1])

        #seqSetArrTimestamp.append(dataArr[nd+n_sequence-1]['timestamp'])

        # Append into sequence set
        seqSetArrInput = np.append(seqSetArrInput, sequenceArrInput.reshape(1,n_sequence,n_feature[0]), axis=0)
        seqSetArrOutput = np.append(seqSetArrOutput, featureArrOutput, axis=0)

    return seqSetArrInput, seqSetArrOutput

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
trainDatasetInput, trainDatasetOutput = getSequenceArray(trainDataset, nSequence, (nFeatureInput,nFeatureOutput))
testDatasetInput, testDatasetOutput = getSequenceArray(testDataset, nSequence, (nFeatureInput,nFeatureOutput))

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

#cont = input('Continue train dataset? [y/n] ')
cont = 'y' # skip confirmation
if cont != 'y':
    sys.exit()



""" plt.plot(list(range(testDatasetOutput.shape[0])), testDatasetInput[:,0,0], color='C0')
plt.plot(list(range(nSequence, nSequence + testDatasetOutput.shape[0])), testDatasetOutput[:,0], color='C1')
plt.show() """


###################################################
###################### LSTM #######################
###################################################
'''
    TO DO:
    [ ] Custom model for different feature (variable model params)
        -> createModel()
    [ ] Add model load/save history
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
    model.add(LSTM(512, activation='tanh', input_shape=(nSequence, nFeatureInput), return_sequences=True))
    model.add(LSTM(512, activation='tanh', input_shape=(nSequence, nFeatureInput), return_sequences=True))
    model.add(LSTM(256, activation='tanh', input_shape=(nSequence, nFeatureInput)))
    model.add(Dense(1))


    # Add Ons
    addOns = []

    if early_stop:
        addOns.append(EarlyStopping(monitor='mean_squared_error', patience=30, verbose=0, mode='min'))

    if checkpoint:
        addOns.append(ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='mean_squared_error', mode='min'))
    

    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])

    return model, addOns


def loadModel(feature_num):
    filename = modelFilename + '_' + str(feature_num) + '_' + str(timeWindow) + '_' + '_'.join(filterDesc) + '.h5'

    print('-'*30)
    print('Loading model of feature {} from {}'.format(featureName[feature_num], filename))
    model = load_model(os.path.join(dataPath, filename))

    return model


def loadHistory(feature_num):
    filename = modelFilename + '_' + str(feature_num) + '_' + str(timeWindow) + '_' + '_'.join(filterDesc) + '.pkl'
    with open(os.path.join(dataPath, filename), 'rb') as historyFile:
        history = pickle.load(historyFile)

    return history


def exportModel(model, feature_num):
    filename = modelFilename + '_' + str(feature_num) + '_' + str(timeWindow) + '_' + '_'.join(filterDesc) + '.h5'
    
    print('-'*30)
    print('Saving model of feature {} to {}'.format(featureName[feature_num], filename))
    model.save(os.path.join(dataPath, filename))


def exportHistory(history, feature_num):
    filename = modelFilename + '_' + str(feature_num) + '_' + str(timeWindow) + '_' + '_'.join(filterDesc) + '.pkl'
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
    xData = [ts - timestamp_arr[0] for ts in timestamp_arr]

    # Set y-axis limit and ticks
    if 'rms' in featureName[idx]:
        ylim = [0,20]
        yticks = list(range(ylim[0],ylim[1]+1,5))
    
    if 'kurtosis' in featureName[idx]:
        ylim = [-10,30]
        yticks = list(range(ylim[0],ylim[1]+1,10))

    if 'skewness' in featureName[idx]:
        ylim = [-10,10]
        yticks = list(range(ylim[0],ylim[1]+1,5))

    if 'crest-factor' in featureName[idx]:
        ylim = [-5,10]
        yticks = list(range(ylim[0],ylim[1]+1,5))

    if 'peak-to-peak' in featureName[idx]:
        ylim = [0,40]
        yticks = list(range(ylim[0],ylim[1]+1,10))

    if simple:
        yticks = []

    # Set x-axis limit and ticks
    xlim = [0, xData[-1]]
    if simple:
        xticks = []
    else:
        xticks = list(range(0, (((xData[-1] // 1000) + 1) * 1000) + 1, 5000))
    
    # Plot
    fig = plt.figure(figsize=(16,8), dpi=120)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.35, bottom=0.1)
    plt.get_current_fig_manager().window.state('zoomed')

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
        ax1.set_title(featureName[idx].title(), fontsize=22)
        ax1.set_xlabel('Waktu (ms)', fontsize=22)
    ax1.set_ylabel(featureName[idx].title(), fontsize=22)

    p_pred, = ax2.plot(xData, pred_arr, 'k-', linewidth=1)
    ax2.set_xticks([])
    ax2.set_xlim(xlim)
    ax2.set_yticks([])
    ax2.set_ylim(ylim)
    ax2.grid(True)

    ax1.legend(
        (p_test, p_pred),
        ('Real Data', 'Prediction'),
        loc='upper right',
        fontsize=18
    )

def plotMetricsHistory(history):
    '''
        Plot metrics (currently supported: mean squared error)
    '''

    plotData = history['mean_squared_error']
    #print(*['Epoch {}: {:.4f}'.format(i+1,plotData[i]) for i in range(0,1000,100)], sep='\n')
    print('[{}] min: {}'.format(featureName[featNum], min(plotData)))

    fig = plt.figure(figsize=(16,3), dpi=120)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.25)

    ax1 = fig.add_subplot(111, frame_on=True)

    p_test, = ax1.plot(plotData, 'k-', linewidth=1.2)
    ax1.tick_params(grid_alpha=0.6, grid_linewidth=0.4, labelsize=16)

    ax1.set_xlim([0,len(plotData)])
    ax1.set_xticks(list(range(0, len(plotData)+1, 200)))
    ax1.grid(True)
    ax1.set_ylabel('MSE', fontsize=22)
    ax1.set_xlabel('Epochs', fontsize=22)
    ax1.set_title('Mean Squared Error over Epoch', fontsize=22)

# Model train iteration by output feature count
if train:

    modelArr = []
    #for outFeatureNum in range(3):
    for outFeatureNum in range(len(featureName)):
        if not useCachedModel:
            # Create model
            model, addOns = createModel(
                early_stop=False,
                checkpoint=True
            )

            # Fit model
            history = model.fit(trainDatasetInput, trainDatasetOutput[:,outFeatureNum], epochs=epochNum, callbacks=[*addOns], batch_size=64, verbose=1)

            # Export model
            exportModel(model, outFeatureNum)
            exportHistory(history, outFeatureNum)

        else:
            model = loadModel(outFeatureNum)

        modelArr.append({
            'feature': featureName[outFeatureNum],
            'model': model,
            'history': loadHistory(outFeatureNum)
        })


# Prediction
if predict:
    if not train:
        modelArr = []
        #for outFeatureNum in range(3):
        for outFeatureNum in range(len(featureName)):
            modelArr.append({
                'feature': featureName[outFeatureNum],
                'model': loadModel(outFeatureNum),
                'history': loadHistory(outFeatureNum)
            })

    predOutput = []
    plotTime = str(datetime.now().strftime('%y_%m_%d_%H_%M_%S'))
    os.mkdir(os.path.join(dataPath,'plot',plotTime + '_' + str(timeWindow)))

    #for featNum in range(3):
    for featNum in range(len(featureName)):
        print('Predicting output feature {}'.format(featureName[featNum]))
        
        predArr = []
        for datapoint in range(testDatasetInput.shape[0]):
            predArr.append(float(modelArr[featNum]['model'].predict(testDatasetInput[datapoint,:,:].reshape(1,nSequence,nFeatureInput), verbose=0)))

        predOutput.append({
            'feature': featureName[featNum],
            'data': predArr
        })

        plotMetricsHistory(
            history=modelArr[featNum]['history']
        )

        if savePredictPlot:
            plt.savefig(
                fname=dataPath + predPlotFilename.format(plotTime + '_' + str(timeWindow), timeWindow, featureName[featNum], 'hist')
            )

        # Plot Predicted Value vs Real Value
        """ plotPrediction(
            timestamp_arr=list(range(len(predArr))),
            output_arr=testDatasetOutput[:,featNum],
            pred_arr=predArr,
            idx=featNum,
            simple=True
        )
        
        if savePredictPlot:
            plt.savefig(
                fname=dataPath + predPlotFilename.format(plotTime + '_' + str(timeWindow), timeWindow, featureName[featNum], 'pred')
            ) """