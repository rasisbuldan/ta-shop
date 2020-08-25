'''
    TO DO:
    [v] Nav, Vib aggregation split timestamp delta spike into smaller
        - Dynamic split by window size
    [v] Output data aggregation into RNN-LSTM compatible dimension
    [ ] Merge training data by all description
    [v] Formal plot

'''


import pymongo
import numpy as np
import matplotlib.pyplot as plt
import time
import collections
import scipy.stats
import sys
from datetime import datetime
import os


class VibData:
    def __init__(self, verbose=False):
        """
        [DOCUMENT TEMPLATE]
        {
            "timestamp": 0,
            "mpu1" : {
                "x": 0,
                "y": 0,
                "z": 0,
            },
            "mpu2" : {
                "x": 0,
                "y": 0,
                "z": 0,
            }
        }
        """
        self.verbose = verbose

        self.initializeConnectionDB()
        self.activeDataArray = []


    def initializeConnectionDB(self, host='localhost', port=27017):
        ''' Connect to database with default localhost '''
        self.clientDB = pymongo.MongoClient(host, port)

        self.vibCollection = self.clientDB['test-db']['acceldatas']
        if self.verbose:
            print('Connection to test-db.acceldatas successful with', self.numOfDocuments(), 'documents')


    ##### Analytics #####
    def numOfDocuments(self):
        ''' Number of all documents in collection '''
        return self.vibCollection.count_documents({})


    def countBetweenTimestamp(self, lower, upper):
        ''' Number of documents between specified timestamp (boundary included) '''
        return len(self.getBetweenTimestamp(lower, upper))


    def timestampArray(self, obj_array):
        ''' Timestamp array queried document list '''
        return [obj['timestamp'] for obj in obj_array]


    def xArray(self, obj_array, mpu):
        ''' X-Acceleration array of specific *mpu* queried document list '''
        return [obj[mpu]['x'] for obj in obj_array]


    def yArray(self, obj_array, mpu):
        ''' Y-Acceleration array of specific *mpu* queried document list '''
        return [obj[mpu]['y'] for obj in obj_array]


    def zArray(self, obj_array, mpu):
        ''' Z-Acceleration array of specific *mpu* queried document list '''
        return [obj[mpu]['z'] for obj in obj_array]


    ##### Query #####
    def storeData(self, data_array, description):
        '''
            Store queried document(s) result to class attribute (main memory)
            for further processing without db query
            
            return: activeDataArray index (id) of queried result
        '''

        self.activeDataArray.append({
            'desc': description,
            'data': data_array
        })

        active_id = len(self.activeDataArray) - 1

        if self.verbose:
            print('Store vibration data with description "{}" success with id:{}'.format(description, active_id))
        return active_id

    
    def deleteData(self, active_id):
        '''
            Delete stored queried document(s) with index *active_id*.
            Replace data with empty array, so active_id of other stored document won't change
        '''

        self.activeDataArray[active_id]['data'] = []

        return active_id


    def getAllDocuments(self):
        '''
            [WARNING!!!] Variable saved in main memory
            
            Get all documents in collection.
        '''

        return list(self.vibCollection.find({}))


    def getBetweenTimestamp(self, lower, upper):
        ''' Get document(s) as list of object with specified timestamp boundary '''

        #print('[lower] {} [upper] {}'.format(lower,upper))
        return list(self.vibCollection.find({
            "timestamp": { "$gte": lower, "$lte": upper }
        }))


    ##### Active data processing #####
    '''
        Further processing of stored queried document(s) of method storeData().
        active_id required as arguments.
    '''

    ### Timestamp Array
    def getTimestampArray(self, active_id):
        ''' Get timestamp array of stored documents with index *active_id* '''

        timestampArray = [int(obj['timestamp']) for obj in self.activeDataArray[active_id]['data']]
        return timestampArray

    
    def getTimestampDeltaArray(self, active_id):
        ''' Get timestamp delta array of stored documents with index *active_id* '''

        tsArray = self.getTimestampArray(active_id)

        tdArray = [(tsArray[i] - tsArray[i-1]) for i in range(len(tsArray))]
        return tdArray



    def getTimestampFilledArray(self, active_id):
        ''' 
            Get timestamp with evenly spaced 1ms per data, missing values replaced by zero
            array of stored documents with index *active_id*
        '''
        # Get timestamp array
        timestampArray = self.getTimestampArray(active_id)
        tsMin = min(timestampArray)
        tsMax = max(timestampArray)
        
        # Create offset-ed array
        tsOffsetArray = [(ts - tsMin) for ts in timestampArray]
        timestampFilledArray = [0 for i in range(tsMax-tsMin+2)]

        for ts in tsOffsetArray:
            timestampFilledArray[ts] = ts
        
        return timestampFilledArray

    
    def getMovingTimestampArray(self, time_array, window_size):
        '''
            Get Moving Timestamp array of stored documents (used for Moving Average / Moving RMS).
            *time_array* passed from getTimestampArray() method. 
        '''
        
        return time_array[window_size : (len(time_array)-window_size)]


    ### Data Array
    def getVibArray(self, active_id):
        '''
            Get acceleration array in all axis of stored documents with specified *active_id*.
            
            return: list of list [mpu1.x, mpu1.y, mpu1.z, mpu2.x, mpu2.y, mpu2.z]
        '''

        def getVib(obj):
            return [
                obj['mpu1']['x'],
                obj['mpu1']['y'],
                obj['mpu1']['z'],
                obj['mpu2']['x'],
                obj['mpu2']['y'],
                obj['mpu2']['z'],
            ]

        vibArray = [[],[],[],[],[],[]]

        objArray = self.activeDataArray[active_id]['data']
        objProcessed = 0
        objArrayLen = len(objArray)
        # Document Array
        for obj in objArray:
            objProcessed += 1
            if self.verbose:
                print('Getting Vibration [{}/{}] ({:.1f})%'.format(objProcessed, objArrayLen, 100*objProcessed/objArrayLen), end='\r')
            vib = getVib(obj)

            # Motor Number
            for i in range(6):
                vibArray[i].append(vib[i])
        print('')
        return vibArray

    
    def getMultiFeatureArray(self, active_id):
        '''
            Aggregated array with window *time_window*

            return: list of dict
            {
                timestamp
                mpu1: [x,y,z],
                mpu2: [x,y,z]
            }
        '''

        # Extract data
        activeData = self.activeDataArray[active_id]['data']

        multiFeatureArray = []

        # Aggregation loop
        for data in activeData:
            featureData = {
                'timestamp': data['timestamp'],
                'mpu1': [a[1] for a in data['mpu1'].items()],
                'mpu2': [a[1] for a in data['mpu2'].items()]
            }

            multiFeatureArray.append(featureData)

        return multiFeatureArray

    
    def getVibAggregatedArray(self, active_id, time_window):
        '''
            Aggregated array with window *time_window*

            return: list of tuple [(ts, [[mpu1.x],..,[mpu2.x],..])]
        '''

        # Extract data
        activeData = self.activeDataArray[active_id]['data']
        tsMin = activeData[0]['timestamp']
        tsMax = activeData[-1]['timestamp']

        # Aggregation loop
        tsPrev = tsMin
        vibBuf = [[],[],[],[],[],[]]
        vibAgg = []
        for i in range(len(activeData)):
            cursor = activeData[i]
            tsCursor = cursor['timestamp']
            
            # Aggregate if delta time exceed time_window
            if (tsCursor > (tsPrev + time_window)) or (i+1) == len(activeData):
                vibAgg.append(
                    [
                        int(tsCursor), 
                        [np.sqrt(np.mean(np.square(v))) for v in vibBuf]
                    ]
                )
                tsPrev = tsCursor
                vibBuf = [[],[],[],[],[],[]]

            # Add data point to buffer
            else:
                vibBuf[0].append(cursor['mpu1']['x'])
                vibBuf[1].append(cursor['mpu1']['y'])
                vibBuf[2].append(cursor['mpu1']['z'])
                vibBuf[3].append(cursor['mpu2']['x'])
                vibBuf[4].append(cursor['mpu2']['y'])
                vibBuf[5].append(cursor['mpu2']['z'])

        return vibAgg

    
    def getMultiFeatureAggregatedArray(self, active_id, time_window):
        '''
            Aggregated array with window *time_window*

            return: list of dict
            {
                timestamp
                rms: [[mpu1.x,...],...[mpu2.z,...]],
                kurtosis: [[mpu1.x,...],...[mpu2.z,...]],
                skewness: [[mpu1.x,...],...[mpu2.z,...]],
                crest-factor: [[mpu1.x,...],...[mpu2.z,...]],
                peak-to-peak: [[mpu1.x,...],...[mpu2.z,...]],
            }
        '''

        # Feature calculation by *array* of values
        def rms(array):
            return np.sqrt(np.mean(np.square(array)))
        
        def kurtosis(array):
            return scipy.stats.kurtosis(array)

        def skewness(array):
            return scipy.stats.skew(array)

        def crest_factor(array):
            return (max(array)/rms(array))

        def peak_to_peak(array):
            return (max(array) - min(array))


        # Extract data
        activeData = self.activeDataArray[active_id]['data']
        tsMin = activeData[0]['timestamp']
        tsMax = activeData[-1]['timestamp']


        # Aggregation variable buffer
        vibBuf = [[],[],[],[],[],[]]  # mpu1.x, ..., mpu2.z
        aggArray = []


        # Aggregation loop
        tsPrev = tsMin
        for i in range(len(activeData)):
            cursor = activeData[i]
            tsCursor = cursor['timestamp']

            # Add data point to buffer
            vibBuf[0].append(cursor['mpu1']['x'])
            vibBuf[1].append(cursor['mpu1']['y'])
            vibBuf[2].append(cursor['mpu1']['z'])
            vibBuf[3].append(cursor['mpu2']['x'])
            vibBuf[4].append(cursor['mpu2']['y'])
            vibBuf[5].append(cursor['mpu2']['z'])
            
            # Aggregate if delta time exceed time_window
            if (tsCursor > (tsPrev + time_window)) or (i+1) == len(activeData):

                # Add aggregated value
                aggArray.append(
                    {
                        'timestamp': int(tsCursor),
                        'rms': [rms(vArr) for vArr in vibBuf],
                        'kurtosis': [kurtosis(vArr) for vArr in vibBuf],
                        'skewness': [skewness(vArr) for vArr in vibBuf],
                        'crest-factor': [crest_factor(vArr) for vArr in vibBuf],
                        'peak-to-peak': [peak_to_peak(vArr) for vArr in vibBuf]
                    }
                )

                # Reset buffer
                tsPrev = tsCursor
                vibBuf = [[],[],[],[],[],[]]

        return aggArray


    def getVibRMSArray(self, active_id):
        '''
            Get RMS in all axis of stored documents with specified *active_id*.
            
            return: list of list [mpu1.x, mpu1.y, mpu1.z, mpu2.x, mpu2.y, mpu2.z]
        '''

        def getRMS(arr):
            return np.sqrt(np.mean(np.square(arr)))

        vibArray = self.getVibArray(active_id)
        vibRMSArray = [[],[],[],[],[],[]]
        batchSize = 50

        for batchNum in range(0, len(vibArray[0])//batchSize):
            if self.verbose:
                print('Getting Vibration RMS [{}/{}] ({:.1f})%'.format(batchNum+1, len(vibArray[0])//batchSize, 100*(batchNum+1)/(len(vibArray[0])//batchSize)), end='\r')
            for i in range(6):
                vibRMSArray[i].append(getRMS(vibArray[i][(batchSize*batchNum):(batchSize*(batchNum+1))]))

        return vibRMSArray

    
    def getVibRMSFilledArray(self, active_id):
        '''
            Get RMS in all axis of stored documents with specified *active_id*.
            
            return: list of list [mpu1.x, mpu1.y, mpu1.z, mpu2.x, mpu2.y, mpu2.z]
        '''

        vibArray = self.getVibArray(active_id)


    def getMovingRMSArray(self, data_array, window_size):
        def getRMS(window_arr):
            N = len(window_arr)
            return np.sqrt(np.sum(np.square(window_arr))/N)

        vibMovingRMSArray = [[],[],[],[],[],[]]

        for window in range(0, len(data_array[0])-window_size, window_size):
            for i in range(6):
                vibMovingRMSArray[i].append(getRMS(data_array[i][window : (window + window_size)]))

        return vibMovingRMSArray


    def getMovingAverageArray(self, data_array, window_size):
        def getAverage(window_arr):
            return np.mean(window_arr)

        vibMovingAverageArray = [[],[],[],[],[],[]]

        for window in range(0, len(data_array[0])-window_size, window_size):
            for i in range(6):
                vibMovingAverageArray[i].append(getAverage(data_array[i][window : (window + window_size)]))

        return vibMovingAverageArray


    
    def getMovingRMSWeightedArray(self, data_array, window_size, weight):
        ''' 
            Moving RMS Weighted with linear weight factor
        '''
        vibMovingWMSWeightedArray = []
        for vibdata in data_array:
            dataArraySquared = np.square(vibdata)
            vibMovingWMSWeighted = []
            for d in range(window_size, len(vibdata)-window_size):
                sumWeighted = 0
                for i in range(window_size):
                    sumWeighted += weight[i] * dataArraySquared[d-window_size+i]
                
                vibMovingWMSWeighted.append(np.sqrt(sumWeighted/np.sum(weight)))
            vibMovingWMSWeightedArray.append(vibMovingWMSWeighted)

        return vibMovingWMSWeightedArray


    ##### Plotting #####
    def plotDataMulti(self, data_array, title=None):
        rowCount = len(data_array)
        xData = list(range(len(data_array[0])))

        for r in range(rowCount):
            plt.subplot(100*rowCount + 11 + r)
            plt.plot(xData, data_array[r])
            plt.ylim(0,16)
            plt.grid(True)

            if title == None:
                plt.title('Data Array id:{}'.format(r))
            else:
                plt.title(title[r])

        plt.tight_layout(0,0,0,(0,0,1,1))
        plt.show()


    def plotDataMultiMovingRMS(self, time_array, data_array, window_size, label=None):
        rowCount = len(data_array)
        minTimestamp = min(time_array)
        maxTimestamp = max(time_array)
        rmsTimestampArray = self.getMovingTimestampArray(time_array, window_size)
        rmsDataArray = self.getMovingRMSArray(data_array, window_size)

        xData = list(range(len(data_array[0])))
        
        fig = plt.figure()
        ax = []
        for r in range(rowCount):
            ax.append([])

        for r in range(rowCount):
            ax[r].append(fig.add_subplot(100*rowCount + 11 + r))
            ax[r].append(fig.add_subplot(100*rowCount + 11 + r, frame_on=False))

            # Raw data
            ax[r][0].plot(time_array, data_array[r], color='C1')
            ax[r][0].set_xlim([minTimestamp, maxTimestamp])
            ax[r][0].set_ylim([-16,16])

            # Moving RMS data
            ax[r][0].plot(rmsTimestampArray, rmsDataArray[r], color='C2')
            ax[r][0].set_xlim([minTimestamp, maxTimestamp])
            ax[r][0].set_ylim([-16,16])

            """ if title == None:
                plt.title('Data Array id:{}'.format(r))
            else:
                plt.title(title[r]) """

        plt.show()

    
    def plotTimedelta(self, active_id):
        '''
            Plot time delta from stored data by getTimesampDeltaArray
        '''
        tdArray = self.getTimestampDeltaArray(active_id)

        fig = plt.figure()
        plt.scatter(list(range(len(tdArray)-1)), tdArray[1:], s=2)
        plt.plot([np.mean(tdArray[1:]) for _ in range(len(tdArray)-1)], color='C3')
        plt.title('MQTT')
        plt.ylim(0,20)
        plt.show()


class NavData:
    def __init__(self, verbose=False, host='localhost', port=27017):
        """
        [DOCUMENT TEMPLATE]
        {
            "description": "",
            "timestart": 0,
            "timestamp": 0,
            "state": {
                "controlState": "",
                "batteryPercentage": 0,
                "batteryMillivolt": 0,
            },
            "navdata": {
                "altitude": 0,
                "orientation": {
                    "roll": 0,
                    "pitch": 0,
                    "yaw": 0,
                },
                "pwm": {
                    "mot1": 0,
                    "mot2": 0,
                    "mot3": 0,
                    "mot4": 0,
                },
                "input": {
                    "uroll": 0,
                    "upitch": 0,
                    "uyaw": 0,
                },
                "rawMeasures": {
                    "accelerometers": {
                        "x": 0,
                        "y": 0,
                        "z": 0,
                    }
                }
            }
        }
        """

        """ Connect to database """
        self.verbose = verbose
        self.initializeConnectionDB(host, port)
        
        self.activeDataArray = []
    

    def initializeConnectionDB(self, host='localhost', port=27017):
        self.clientDB = pymongo.MongoClient(host, port)

        self.navCollection = self.clientDB['test-db']['navdatas']
        if self.verbose:
            print('Connection to test-db.navdatas successful with', self.numOfDocuments(), 'documents')


    ##### Analytics #####
    def numOfDocuments(self):
        return self.navCollection.count_documents({})


    def countBetweenTimestamp(self, lower, upper):
        return len(self.getBetweenTimestamp(lower, upper))


    ##### Query #####
    def storeData(self, data_array, description):
        self.activeDataArray.append({
            'desc': description,
            'data': data_array
        })

        active_id = len(self.activeDataArray) - 1

        if self.verbose:
            print('Store navigation data with description "{}" success with id:{}'.format(description, active_id))
        return active_id

    
    def listDescription(self):
        return list(self.navCollection.distinct('description'))


    def listDescriptionTimestamp(self):
        descArray = self.listDescription()
        descTimestampArray = []

        for desc in descArray:
            minTs, maxTs = self.getTimestampRangeByDescription(desc, True)
            descTimestampArray.append({
                "description": desc,
                "timestampStart": minTs,
                "timestampStop": maxTs
            })

        return descTimestampArray


    def getAllDocuments(self):
        '''
            WARNING!!! Variable saved in main memory
        '''
        return list(self.navCollection.find({}))


    def getBetweenTimestamp(self, lower, upper):
        return list(self.navCollection.find({
            "timestamp": { "$gte": lower, "$lte": upper }
        }))


    def getByDescription(self, description, landed=True):
        if landed:
            return list(self.navCollection.find({
                "$and": [
                    {
                        "description": { "$eq": description }
                    }
                ]
            }))

        else:
            return list(self.navCollection.find({
                "$and": [
                    {
                        "description": { "$eq": description }
                    },
                    {
                        "state.controlState": { "$ne": "CTRL_LANDED" }
                    }
                ]
            }))


    def getTimestampRangeByDescription(self, description, landed=True):
        if landed:
            tsArray = [n['timestamp'] for n in list(self.navCollection.find({
                "description": { "$eq": description }
            },
            {
                "_id": 0,
                "timestamp": 1
            }))]

        else:
            tsArray = [n['timestamp'] for n in list(self.navCollection.find({
                "$and" : [
                    {
                        "description": { "$eq": description }
                    },
                    {
                        "state.controlState": { "$ne": "CTRL_LANDED" }
                    },
                ]
            },
            {
                "_id": 0,
                "timestamp": 1
            }))]

        return min(tsArray), max(tsArray)


    ##### Active data processing #####
    def getTimestampArray(self, active_id):
        timestampArray = [obj['timestamp'] for obj in self.activeDataArray[active_id]['data']]
        return timestampArray


    def getTimestampDeltaArray(self, active_id):
        ''' Get timestamp delta array of stored documents with index *active_id* '''

        tsArray = self.getTimestampArray(active_id)

        tdArray = [(tsArray[i] - tsArray[i-1]) for i in range(len(tsArray))]
        return tdArray


    def getPWMArray(self, active_id):
        def getPWM(obj):
            return [
                obj['navdata']['pwm']['mot1'],
                obj['navdata']['pwm']['mot2'],
                obj['navdata']['pwm']['mot3'],
                obj['navdata']['pwm']['mot4']
            ]

        pwmArray = [[],[],[],[]]

        objArray = self.activeDataArray[active_id]['data']
        objProcessed = 0
        objArrayLen = len(objArray)
        # Document Array
        for obj in objArray:
            objProcessed += 1
            if self.verbose:
                print('Getting PWM [{}/{}] ({:.1f})%'.format(objProcessed, objArrayLen, 100*objProcessed/objArrayLen), end='\r')
            pwm = getPWM(obj)

            # Motor Number
            for i in range(4):
                pwmArray[i].append(pwm[i])

        return pwmArray

    
    def getAltitudeArray(self, active_id):
        objArray = self.activeDataArray[active_id]['data']

        altitudeArray = [obj['navdata']['altitude'] for obj in objArray]

        return altitudeArray

    
    def getControlStateArray(self, active_id):
        objArray = self.activeDataArray[active_id]['data']

        controlStateArray = [obj['state']['controlState'] for obj in objArray]

        return controlStateArray


    def getMultiFeatureArray(self, active_id):
        '''
            Get multi feature array of stored active data with index *active_id*

            return: list of dict
            {
                timestamp,
                state: {
                    controlState,
                    batteryPercentage,
                    batteryMillivolt
                },
                pwm: [mot1,mot2,mot3,mot4],
                orientation: [roll,pitch,yaw],
            }
        '''

        # Extract data
        activeData = self.activeDataArray[active_id]['data']

        multiFeatureArray = []
        
        for data in activeData:
            featureData = {
                'timestamp': data['timestamp'],
                'state': {
                    'controlState': data['state']['controlState'],
                    'batteryPercentage': data['state']['batteryPercentage'],
                    'batteryMillivolt': data['state']['batteryMillivolt']
                },
                'pwm': [p[1] for p in data['navdata']['pwm'].items()],
                'orientation': [p[1] for p in data['navdata']['orientation'].items()],
            }
            
            multiFeatureArray.append(featureData)

        return multiFeatureArray

    
    def getPWMAggregatedArray(self, active_id, time_window):
        '''
            Aggregated array with window *time_window*

            return: list of tuple [(ts, [[navdata.pwm.mot1],..,..,[navdata.pwm.mot4]]),...]
        '''

        # Extract data
        activeData = self.activeDataArray[active_id]['data']
        tsMin = activeData[0]['timestamp']
        tsMax = activeData[-1]['timestamp']

        # Aggregation loop
        tsPrev = tsMin
        pwmBuf = [[],[],[],[]]
        pwmAgg = []
        for i in range(len(activeData)):
            cursor = activeData[i]
            tsCursor = cursor['timestamp']

            # Add data point to buffer
            for idx, (key, val) in enumerate(cursor['navdata']['pwm'].items()):
                    pwmBuf[idx].append(val)
            
            # Aggregate if delta time exceed time_window
            if (tsCursor > (tsPrev + time_window)) or (i+1) == len(activeData):

                # Add aggregated value
                pwmAgg.append(
                    [
                        int(tsCursor), 
                        [np.sqrt(np.mean(np.square(v))) for v in pwmBuf]
                    ]
                )

                # Reset buffer
                tsPrev = tsCursor
                pwmBuf = [[],[],[],[]]

        return pwmAgg

    
    def getMultiAggregatedArray(self, active_id, time_window):
        '''
            Aggregated array with window *time_window*

            return: list of tuple [(ts, [pwm1,..], [pitch,...], [acc.x,...]),...]
        '''

        # Extract data
        activeData = self.activeDataArray[active_id]['data']
        tsMin = activeData[0]['timestamp']
        tsMax = activeData[-1]['timestamp']

        # Aggregation variable buffer
        pwmBuf = [[],[],[],[]]  # pwm1,...,pwm4
        orientBuf = [[],[],[]]  # roll, pitch, yaw
        rawBuf = [[],[],[]]     # acc.x, acc.y, acc.z

        aggArray = []
        
        # Aggregation loop
        tsPrev = tsMin
        for i in range(len(activeData)):
            cursor = activeData[i]
            tsCursor = cursor['timestamp']

            # Add data point to buffer
            for idx, (key, val) in enumerate(cursor['navdata']['pwm'].items()):
                pwmBuf[idx].append(val)
            for idx, (key, val) in enumerate(cursor['navdata']['orientation'].items()):
                orientBuf[idx].append(val)
            for idx, (key, val) in enumerate(cursor['navdata']['rawMeasures']['accelerometers'].items()):
                rawBuf[idx].append(val)
            
            # Aggregate if delta time exceed time_window
            if (tsCursor > (tsPrev + time_window)) or (i+1) == len(activeData):

                # Add aggregated value to aggArray
                aggArray.append(
                    {
                        'timestamp': int(tsCursor),
                        'pwm': [np.mean(pArr) for pArr in pwmBuf],
                        'orientation': [np.mean(oArr) for oArr in orientBuf],
                        'rawaccel': [np.mean(rArr) for rArr in rawBuf],
                    }
                )

                # Reset buffer
                tsPrev = tsCursor
                pwmBuf = [[],[],[],[]]
                orientBuf = [[],[],[]]
                rawBuf = [[],[],[]]

        return aggArray


    ##### Plotting #####
    def plotDataMulti(self, data_array, label=None):
        rowCount = len(data_array)
        #rowCount = 2
        xData = list(range(len(data_array[0])))

        for r in range(rowCount):
            plt.subplot(100*rowCount + 11 + r)
            plt.plot(xData, data_array[r])
            plt.ylim(-10,310)
            plt.grid(True)

            if label == None:
                plt.title('Data Array id:{}'.format(r))
            else:
                plt.title(label[r])

        plt.tight_layout(0,0,0,(0,0,1,1))
        plt.grid(True)
        plt.show()

    def plotTimedelta(self, active_id):
        '''
            Plot time delta from stored data by getTimesampDeltaArray
        '''
        tdArray = self.getTimestampDeltaArray(active_id)

        fig = plt.figure()
        plt.scatter(list(range(len(tdArray)-1)), tdArray[1:], s=2)
        plt.plot([np.mean(tdArray[1:]) for _ in range(len(tdArray)-1)], color='C3')
        plt.title('Local')
        plt.yticks(range(0,20,2))
        plt.ylim(0,20)
        plt.show()



class NavdataVib:
    def __init__(self, verbose=False):
        self.verbose = verbose
        
        self.initializeConnectionDB()


    def initializeConnectionDB(self, host='localhost', port=27017):
        self.clientDB = pymongo.MongoClient(host, port)

        self.flightCollection = self.clientDB['test-db']['flightdatas']
        if self.verbose:
            print('Connection to test-db.navdatas successful with', self.numOfDocuments(), 'documents')


    def getDescriptionList(self):
        '''
            Get list of description in class attribute flightCollection
        '''

        desc = list(self.flightCollection.distinct('description'))

        return desc

    
    def getCombined(self, description, landed=True):
        '''
        ---------------
        Return format:
        ---------------
        list of dict
        {
            'description',
            'timestamp',
            'state': {
                'controlState',
                'batteryPercentage',
                'batteryMillivolt',
            },
            'pwm': [mot1,mot2,mot3,mot4],
            'orientation': [roll,pitch,yaw],
            'mpu1': [x,y,z],
            'mpu2': [x,y,z]
        }

        '''

        # Get by description
        if landed:
            docs = list(self.flightCollection.find({
                "$and": [
                    {
                        "description": { "$eq": description }
                    }
                ]
            }))

        else:
            docs = list(self.flightCollection.find({
                "$and": [
                    {
                        "description": { "$eq": description }
                    },
                    {
                        "state.controlState": { "$ne": "CTRL_LANDED" }
                    }
                ]
            }))

        return docs

    
    def combineData(self, pwmdata, vibdata):
        '''
            Combine based on smallest timestamp delta
            (Currently pwmdata to match vibdata timestamp)

            Input format:
            pwmdata = [[timestamp,...], [pwm,...]]
            vibdata = [[timestamp,...], [[vib.x,...],...]]

            Return format:
            [timestamp, pwm, vibx, viby, vibz]
        '''

        tsPWM = pwmdata[0]
        tsVib = vibdata[0]
        pwmArray = pwmdata[1]
        vibArray = vibdata[1]

        # Get average timestamp
        tsPWMAvg = (max(tsPWM) - min(tsPWM)) / len(tsPWM)
        tsVibAvg = (max(tsVib) - min(tsVib)) / len(tsVib)

        if self.verbose:
            print('Timestamp avg: {} | {}'.format(tsPWMAvg, tsVibAvg))

        
        # Interpolate PWM data into Vib data
        if tsVibAvg < tsPWMAvg:
            newPWMArray = []

            for ts in tsVib:

                # Vibration data traversal
                i = len(tsPWM) - 1
                while tsPWM[i] > ts:
                    i -= 1
                #print('idx:', i, end='\r')
                
                # Append inrange data into new pwm array
                newPWMArray.append(pwmArray[i])

        #print('Combined data result: {} | {}'.format(len(newPWMArray), len(tsVib)))
        
        combinedArray = []
        for i in range(len(tsVib)):
            combinedArray.append([
                tsVib[i],
                newPWMArray[i],
                [
                    vibArray[0][i],
                    vibArray[1][i],
                    vibArray[2][i]
                ]
            ])

        #combinedArray = [tsVib, newPWMArray, *vibArray]
        return combinedArray

    
    def combineDataMultiFeature(self, navdata, vibdata):
        '''
            Combine navdata and vibdata based on smallest timestamp delta,
            Value of feature duplicated / shifted to matching timestamp

            Compatible feed function: 
            - NavData.getMultiFeatureArray
            - VibData.getMultiFeatureArray

            ---------------
            Input format:
            ---------------
            navdata = list of dict
                {
                timestamp,
                state: {
                    controlState,
                    batteryPercentage,
                    batteryMillivolt
                },
                pwm: [mot1,mot2,mot3,mot4],
                orientation: [roll,pitch,yaw]
            }

            vibdata = list of dict
                {
                    timestamp,
                    mpu1: [x,y,z],
                    mpu2: [x,y,z],
                }

            ---------------
            Return format:
            ---------------
            list of dict
                {
                    timestamp,
                    state: {
                        controlState,
                        batteryPercentage,
                        batteryMillivolt
                    },
                    pwm: [mot1,mot2,mot3,mot4],
                    orientation: [roll,pitch,yaw],
                    mpu1: [x,y,z],
                    mpu2: [x,y,z],
                }
        '''

        tsNav = [nav['timestamp'] for nav in navdata]
        tsVib = [vib['timestamp'] for vib in vibdata]

        # Check for data duplication (skip if tsNow <= tsPrev)
        tsNavPrev = 0
        tsVibPrev = 0

        # Get average timestamp
        tsNavAvg = (max(tsNav) - min(tsNav)) / len(tsNav)
        tsVibAvg = (max(tsVib) - min(tsVib)) / len(tsVib)

        #print('tsNav', len(tsNav))
        #print('tsVib', len(tsVib))

        #print('tsPWM: {} | tsVib: {}'.format(tsNavAvg, tsVibAvg))
        #print('tsPWM: {} - {}'.format(min(tsNav), max(tsNav)))
        #print('tsVib: {} - {}'.format(min(tsVib), max(tsVib)))

        if self.verbose:
            print('tsPWM: {} | tsVib: {}'.format(tsNavAvg, tsVibAvg))
            print('tsPWM: {} - {}'.format(min(tsNav), max(tsNav)))
            print('tsVib: {} - {}'.format(min(tsVib), max(tsVib)))
        
        combinedArray = []

        # Interpolate Nav data into Vib data
        if tsVibAvg < tsNavAvg:
            
            # Iterate over vib timestamp
            tsNavPrev = 0
            tsVibPrev = 0
            for t in range(0, len(tsVib)):

                # Get insert index (nav timestamp less than current vib timestamp)
                i = len(tsNav) - 1
                while tsNav[i] > tsVib[t]:
                    i -= 1
                
                # Conditional for duplicated timestamp or backward timestamp
                # (may be indicate of duplicate data point / data set)
                if (tsNav[i] < tsNavPrev) or (tsVib[t] < tsVibPrev):
                    continue

                # Append data into combined array
                combinedArray.append(
                    {
                        'timestamp': tsVib[t],
                        'state': {
                            'controlState': navdata[i]['state']['controlState'],
                            'batteryPercentage': navdata[i]['state']['batteryPercentage'],
                            'batteryMillivolt': navdata[i]['state']['batteryMillivolt']
                        },
                        'pwm': navdata[i]['pwm'],
                        'orientation': navdata[i]['orientation'],
                        'mpu1': vibdata[t]['mpu1'],
                        'mpu2': vibdata[t]['mpu2']
                    }
                )

                tsNavPrev = tsNav[i]
                tsVibPrev = tsVib[i]


        # Interpolate vib data into PWM data
        elif tsNavAvg < tsVibAvg:
            #print('\n\ntsnav < tsvib\n\n')
            # Special case: trim boundary data from Nav array
            # start(tsNav) <= start(tsVib)
            startTrim = 0
            while tsNav[startTrim] < tsVib[0]:
                startTrim += 1

            # stop(tsNav) >= stop(tsVib)
            stopTrim = len(tsNav) - 1
            while tsNav[stopTrim] > tsVib[-1]:
                stopTrim -= 1
            
            # Iterate over nav timestamp (trimmed based on special case)
            tsNavPrev = 0
            tsVibPrev = 0
            for t in range(startTrim, stopTrim):
                # Check possibility of data repetition
                if tsNav[t] < tsNavPrev:
                    continue

                # Get insert index (vib timestamp less than current nav timestamp)
                i = len(tsVib) - 1
                while tsVib[i] > tsNav[t]:
                    i -= 1

                # Append data into combined array
                combinedArray.append(
                    {
                        'timestamp': tsNav[t],
                        'state': {
                            'controlState': navdata[t]['state']['controlState'],
                            'batteryPercentage': navdata[t]['state']['batteryPercentage'],
                            'batteryMillivolt': navdata[t]['state']['batteryMillivolt']
                        },
                        'pwm': navdata[t]['pwm'],
                        'orientation': navdata[t]['orientation'],
                        'mpu1': vibdata[i]['mpu1'],
                        'mpu2': vibdata[i]['mpu2']
                    }
                )

                tsNavPrev = tsNav[i]
                tsVibPrev = tsVib[i]

        # Return combined array
        #print('Combined data result: {} | {}'.format(len(newVibArray[0]), len(tsPWM)))
        return combinedArray

    
    def aggregateCombined(self, combined_data, time_window):
        '''
            Compatible input data feed: NavdataVib.combineDataMultiFeature
            ---------------
            Input format:
            ---------------
            list of dict
                {
                    timestamp,
                    pwm: [mot1,mot2,mot3,mot4],
                    orientation: [roll,pitch,yaw],
                    mpu1: [x,y,z],
                    mpu2: [x,y,z],
                }

            ---------------
            Return format:
            ---------------
            list of dict
                {
                    timestamp,
                    orientation: [roll,pitch,yaw],
                    mot1: {
                        'pwm',
                        'rms': [x,y,z],
                        'kurtosis': [x,y,z],
                        'skewness': [x,y,z],
                        'crest-factor': [x,y,z],
                        'peak-to-peak': [x,y,z],
                    },
                    mot2: {
                        'pwm',
                        'rms': [x,y,z],
                        'kurtosis': [x,y,z],
                        'skewness': [x,y,z],
                        'crest-factor': [x,y,z],
                        'peak-to-peak': [x,y,z],
                    }
                }
        '''

        # Feature calculation by *array* of values
        def rms(array):
            return np.sqrt(np.mean(np.square(array)))
        
        def kurtosis(array):
            return scipy.stats.kurtosis(array)

        def skewness(array):
            return scipy.stats.skew(array)

        def crest_factor(array):
            return (max(array)/rms(array))

        def peak_to_peak(array):
            return (max(array) - min(array))


        combinedData = combined_data.copy()

        # Loop over timestamp and aggregate if buffer time exceed time window
        timePrev = combinedData[0]['timestamp']
        cursorIdx = 0

        # Aggregate buffer to store combined data point
        aggBuf = []
        aggDataArray = []

        while cursorIdx < len(combinedData):
            # Calculate time delta
            timeCursor = combinedData[cursorIdx]['timestamp']
            
            # Calculate feature
            if (timeCursor - timePrev) >= time_window:

                aggBuf.append(combinedData[cursorIdx])

                # Empty dictionary format
                aggData = {
                    'timestamp': timeCursor,
                    'orientation': [0,0,0],
                    'mot1': {
                        'pwm': 0,
                        'rms': [0,0,0],
                        'kurtosis': [0,0,0],
                        'skewness': [0,0,0],
                        'crest-factor': [0,0,0],
                        'peak-to-peak': [0,0,0],
                    },
                    'mot2': {
                        'pwm': 0,
                        'rms': [0,0,0],
                        'kurtosis': [0,0,0],
                        'skewness': [0,0,0],
                        'crest-factor': [0,0,0],
                        'peak-to-peak': [0,0,0],
                    }
                }

                # Calculate feature aggregation
                aggData['orientation'] = [np.mean([data['orientation'][axis] for data in aggBuf]) for axis in range(3)]
                
                aggData['mot1']['pwm'] = np.mean([data['pwm'][0] for data in aggBuf])
                aggData['mot2']['pwm'] = np.mean([data['pwm'][1] for data in aggBuf])

                aggData['mot1']['rms'] = [rms([data['mpu1'][axis] for data in aggBuf]) for axis in range(3)]
                aggData['mot2']['rms'] = [rms([data['mpu2'][axis] for data in aggBuf]) for axis in range(3)]
                aggData['mot1']['kurtosis'] = [kurtosis([data['mpu1'][axis] for data in aggBuf]) for axis in range(3)]
                aggData['mot2']['kurtosis'] = [kurtosis([data['mpu2'][axis] for data in aggBuf]) for axis in range(3)]
                aggData['mot1']['skewness'] = [skewness([data['mpu1'][axis] for data in aggBuf]) for axis in range(3)]
                aggData['mot2']['skewness'] = [skewness([data['mpu2'][axis] for data in aggBuf]) for axis in range(3)]
                aggData['mot1']['crest-factor'] = [crest_factor([data['mpu1'][axis] for data in aggBuf]) for axis in range(3)]
                aggData['mot2']['crest-factor'] = [crest_factor([data['mpu2'][axis] for data in aggBuf]) for axis in range(3)]
                aggData['mot1']['peak-to-peak'] = [peak_to_peak([data['mpu1'][axis] for data in aggBuf]) for axis in range(3)]
                aggData['mot2']['peak-to-peak'] = [peak_to_peak([data['mpu2'][axis] for data in aggBuf]) for axis in range(3)]

                aggDataArray.append(aggData)
                timePrev = timeCursor

                # Reset buffer
                aggBuf = []


            # Add to buffer
            else:
                aggBuf.append(combinedData[cursorIdx])

            cursorIdx += 1

        return aggDataArray

    
    def combineMultiAggregatedArray(self, navdata_agg, vibdata_agg, time_window=None):
        '''
            If aggregated array length not equal, required arg *time_window*:
                same as previous value with specified timeWindow
        '''

        # Get aggregated vibration and pwm data
        tsArrayNavdata = [n[0] for n in navdata_agg]
        tsArrayVibdata = [v[0] for v in vibdata_agg]
        nNavdata = len(tsArrayNavdata)
        nVibdata = len(tsArrayVibdata)

        # Debug
        #print([tsArrayNavdata[i] - tsArrayNavdata[i-1] for i in range(1,nNavdata)])
        
        # Aggregated array length not equal
        if nNavdata != nVibdata:
            if self.verbose:
                print('Array dimension not equal [nav: {} | vib: {}], processing...'.format(nNavdata, nVibdata))

            # Plot timestamp (debug)
            """ plt.plot(list(range(nNavdata)), tsArrayNavdata, color='C0')
            plt.plot(list(range(nVibdata)), tsArrayVibdata, color='C1')
            plt.show() """

            # Value insertion
            if nNavdata < nVibdata:
                tdNavdata = [(tsArrayNavdata[i] - tsArrayNavdata[i-1]) for i in range(1,nNavdata)]

                # Get timedelta index bigger than timeWindow (>30%)
                tdOverrange = [(idx,td) for (idx,td) in zip(list(range(len(tdNavdata))), tdNavdata) if td >= (time_window * 1.2)]

                # Before state (debug)
                #print('Before:', *navdata_agg, sep='\n')
                #print('Timedelta overrange: ', tdOverrange)

                for td in tdOverrange[::-1]:
                    tdCount = td[1] // time_window
                    tdIdx = td[0]
                    prevNavdata = navdata_agg[tdIdx]

                    # Insert with value 
                    for i in range(tdCount-1):
                        # Copy previous navdata
                        insertedNavdata = prevNavdata.copy()

                        # Update timestamp to match time_window step value
                        insertedNavdata[0] = prevNavdata[0] + ((tdCount-1-i)*time_window)

                        # Insert
                        navdata_agg.insert(tdIdx+1, insertedNavdata)


                # Validation (debug)
                #tsArrayNavdata = [n[0] for n in navdata_agg]
                #tdNavdata = [(tsArrayNavdata[i] - tsArrayNavdata[i-1]) for i in range(1,nNavdata)]
                #print('After:', *navdata_agg, sep='\n')
                #print(tdNavdata)
                #plt.plot(list(range(len(tsArrayNavdata))), tsArrayNavdata)
                #plt.show()

        # Aggregated array length equal
        aggArray = []

        for i in range(nNavdata):
            try:
                aggArray.append(
                    [
                        navdata_agg[i][0],
                        navdata_agg[i][1],
                        navdata_agg[i][2],
                        navdata_agg[i][3],
                        vibdata_agg[i][1]
                    ]
                )
            
            except IndexError:
                try:
                    print('vibdata_agg[i]')
                    print(vibdata_agg[i])

                except IndexError:
                    print('vibdata_agg')
                    print(vibdata_agg)



        return aggArray
        
    
    def combineAggregatedArray(self, navdata_agg, vibdata_agg, time_window=None):
        '''
            If aggregated array length not equal, required arg *time_window*: 
                same as previous value with specified timeWindow
        '''

        # Get aggregated vibration and pwm data
        tsArrayNavdata = [n[0] for n in navdata_agg]
        tsArrayVibdata = [v[0] for v in vibdata_agg]
        nNavdata = len(tsArrayNavdata)
        nVibdata = len(tsArrayVibdata)

        # Debug
        #print([tsArrayNavdata[i] - tsArrayNavdata[i-1] for i in range(1,nNavdata)])
        
        # Abort if aggregated array length not equal
        if nNavdata != nVibdata:
            if self.verbose:
                print('Array dimension not equal [nav: {} | vib: {}], processing...'.format(nNavdata, nVibdata))

            # Plot timestamp (debug)
            """ plt.plot(list(range(nNavdata)), tsArrayNavdata, color='C0')
            plt.plot(list(range(nVibdata)), tsArrayVibdata, color='C1')
            plt.show() """

            # Value insertion
            if nNavdata < nVibdata:
                tdNavdata = [(tsArrayNavdata[i] - tsArrayNavdata[i-1]) for i in range(1,nNavdata)]

                # Get timedelta index bigger than timeWindow (>10%)
                tdOverrange = [(idx,td) for (idx,td) in zip(list(range(len(tdNavdata))), tdNavdata) if td >= (time_window * 1.2)]

                # Before state (debug)
                #print('Before:', *navdata_agg, sep='\n')
                #print('Timedelta overrange: ', tdOverrange)

                for td in tdOverrange[::-1]:
                    tdCount = td[1] // time_window
                    tdIdx = td[0]
                    prevNavdata = navdata_agg[tdIdx]

                    # Insert with value 
                    for i in range(tdCount-1):
                        # Copy previous navdata
                        insertedNavdata = prevNavdata.copy()

                        # Update timestamp to match time_window step value
                        insertedNavdata[0] = prevNavdata[0] + ((tdCount-1-i)*time_window)

                        # Insert
                        navdata_agg.insert(tdIdx+1, insertedNavdata)


                # Validation (debug)
                #tsArrayNavdata = [n[0] for n in navdata_agg]
                #tdNavdata = [(tsArrayNavdata[i] - tsArrayNavdata[i-1]) for i in range(1,nNavdata)]
                #print('After:', *navdata_agg, sep='\n')
                #print(tdNavdata)
                #plt.plot(list(range(len(tsArrayNavdata))), tsArrayNavdata)
                #plt.show()

        # Aggregated array length equal
        aggArray = []

        for i in range(nNavdata):
            try:
                aggArray.append(
                    [
                        navdata_agg[i][0],
                        navdata_agg[i][1],
                        vibdata_agg[i][1]
                    ]
                )
            
            except IndexError:
                try:
                    print('vibdata_agg[i]')
                    print(vibdata_agg[i])

                except IndexError:
                    print('vibdata_agg')
                    print(vibdata_agg)



        return aggArray

    
    def getTimestampAnalysis(self, navdata, vibdata):
        # Extract timestamp
        timestampNav = navdata[0]
        timestampVib = vibdata[0]

        # Get number of data
        tsNavLen = len(timestampNav)
        tsVibLen = len(timestampVib)

        # Get boundary value
        tsNavMin = min(timestampNav)
        tsNavMax = max(timestampNav)
        tsVibMin = min(timestampVib)
        tsVibMax = max(timestampVib)

        # Get average
        tsNavSamplingAvg = (tsNavMax - tsNavMin) / tsNavLen
        tsVibSamplingAvg = (tsVibMax - tsVibMin) / tsVibLen

        # Get delta array
        tsNavDelta = [0] + [(timestampNav[i] - timestampNav[i-1]) for i in (range(1,len(timestampNav)))]
        tsVibDelta = [0] + [(timestampVib[i] - timestampVib[i-1]) for i in (range(1,len(timestampVib)))]

        print('[runtime] nav: {} | {}'.format((tsNavMax-tsNavMin)/1000, (tsVibMax-tsVibMin)/1000))
        print('[count] nav: {} | vib: {}'.format(tsNavLen, tsVibLen))
        print('[bound] nav: {} - {} | vib: {} - {}'.format(tsNavMin, tsNavMax, tsVibMin, tsVibMax))
        print('[avg] nav: {} | vib: {}'.format(tsNavSamplingAvg, tsVibSamplingAvg))
        #print('[outlier] nav: {}'.format(tsVibOutlier))

        # Plot timestamp delta
        plt.scatter(list(range(len(tsNavDelta))), tsNavDelta, s=5, color='C0')
        plt.plot(list(range(len(tsNavDelta))), [tsNavSamplingAvg for i in range(len(tsNavDelta))], color='C1')
        plt.title('Timestamp Delta - Navigation Data')
        plt.ylim(0,50)
        plt.grid(True)
        plt.show()
        
        plt.scatter(list(range(len(tsVibDelta))), tsVibDelta, s=5, color='C0')
        plt.plot(list(range(len(tsVibDelta))), [tsVibSamplingAvg for i in range(len(tsVibDelta))], color='C1')
        plt.title('Timestamp Delta - Vibration Data')
        plt.grid(True)
        plt.show()

        plt.scatter(list(range(len(tsVibDelta))), tsVibDelta, s=5, color='C0')
        plt.plot(list(range(len(tsVibDelta))), [tsVibSamplingAvg for i in range(len(tsVibDelta))], color='C1')
        plt.title('Timestamp Delta - Vibration Data')
        plt.grid(True)
        plt.show()


    def plotNavdataVibTimestamp(self, navdata, vibdata, axis=None):
        '''
            Fixed for 2 motors - each 3 accelerometer axis
            - ts_range = tuple(min_ts, max_ts)
            - navdata = [[timestamp],[pwm1],[pwm2],[pwm3],[pwm4]]   => currently pwm only
            - vibdata = [[timestamp],[mpu1.x],[mpu1.y],[mpu1.z],[mpu2.x],[mpu2.y],[mpu2.z]]

            Relation:
            - pwm1 -> mpu1
            - pwm2 -> mpu2
        '''
        if axis == None:
            axis = ['x','y','z']

        # Timestamp
        timestampNav = navdata[0]
        timestampVib = vibdata[0]
        tsComb = timestampNav + timestampVib
        tsMin = (min(tsComb)//1000) * 1000
        tsMax = (max(tsComb)//1000) * 1000

        ##### Setup plot #####
        fig = plt.figure()

        # Subplot 1
        if 'x' in axis:
            ax1_vib_x = fig.add_subplot(211, label='Vib Motor 1 - X', frame_on=True)
        if 'y' in axis:
            ax1_vib_y = fig.add_subplot(211, label='Vib Motor 1 - Y', frame_on=False)
        if 'z' in axis:
            ax1_vib_z = fig.add_subplot(211, label='Vib Motor 1 - Z', frame_on=False)
        ax1_nav_pwm1 = fig.add_subplot(211, label='PWM Motor 1', frame_on=False)

        # Subplot 2
        if 'x' in axis:
            ax2_vib_x = fig.add_subplot(212, label='Vib Motor 2 - X', frame_on=True)
        if 'y' in axis:
            ax2_vib_y = fig.add_subplot(212, label='Vib Motor 2 - Y', frame_on=False)
        if 'z' in axis:
            ax2_vib_z = fig.add_subplot(212, label='Vib Motor 2 - Z', frame_on=False)
        ax2_nav_pwm2 = fig.add_subplot(212, label='PWM Motor 2', frame_on=False)


        ##### Plot #####
        ### Subplot 1 ###

        # PWM
        p1_nav_pwm1, = ax1_nav_pwm1.plot(timestampNav, navdata[1], label='PWM1', color='C0', linewidth=1.5)
        ax1_nav_pwm1.set_ylabel('PWM Motor 1', color='C0')
        ax1_nav_pwm1.set_ylim([-10,310])
        ax1_nav_pwm1.set_xlim([tsMin, tsMax])
        #ax1_nav_pwm1.set_xticks(np.arange(tsMin, tsMax, 2500))

        # Vib - X
        if 'x' in axis:
            p1_vib_x, = ax1_vib_x.plot(timestampVib, vibdata[1], label='Vib1-X', color='C1', linewidth=0.5)
            ax1_vib_x.yaxis.tick_right()
            ax1_vib_x.set_ylim([-16,16])
            ax1_vib_x.set_xlim([tsMin, tsMax])
            #ax1_vib_x.set_xticks(np.arange(tsMin, tsMax, 2500))

        # Vib - Y
        if 'y' in axis:
            p1_vib_y, = ax1_vib_y.plot(timestampVib, vibdata[2], label='Vib1-Y', color='C2', linewidth=0.5)
            ax1_vib_y.yaxis.tick_right()
            ax1_vib_y.set_ylim([-16,16])
            ax1_vib_y.set_xlim([tsMin, tsMax])
            #ax1_vib_y.set_xticks(np.arange(tsMin, tsMax, 2500))

        # Vib - Z
        if 'z' in axis:
            p1_vib_z, = ax1_vib_z.plot(timestampVib, vibdata[3], label='Vib1-Z', color='C3', linewidth=0.5)
            ax1_vib_z.yaxis.tick_right()
            ax1_vib_z.set_ylim([-16,16])
            ax1_vib_z.set_xlim([tsMin, tsMax])
            #ax1_vib_z.set_xticks(np.arange(tsMin, tsMax, 2500))


        ### Subplot 2 ###

        # PWM
        p2_nav_pwm2, = ax2_nav_pwm2.plot(timestampNav, navdata[2], label='PWM2', color='C0', linewidth=1.5)
        ax2_nav_pwm2.set_ylabel('PWM Motor 2', color='C0')
        ax2_nav_pwm2.set_ylim([-10,310])
        ax2_nav_pwm2.set_xlim([tsMin, tsMax])

        # Vib - X
        if 'x' in axis:
            p2_vib_x, = ax2_vib_x.plot(timestampVib, vibdata[4], label='Vib2-X', color='C1', linewidth=0.5)
            ax2_vib_x.yaxis.tick_right()
            ax2_vib_x.set_ylim([-16,16])
            ax2_vib_x.set_xlim([tsMin, tsMax])

        # Vib - Y
        if 'y' in axis:
            p2_vib_y, = ax2_vib_y.plot(timestampVib, vibdata[5], label='Vib2-Y', color='C2', linewidth=0.5)
            ax2_vib_y.yaxis.tick_right()
            ax2_vib_y.set_ylim([-16,16])
            ax2_vib_y.set_xlim([tsMin, tsMax])

        # Vib - Z
        if 'z' in axis:
            p2_vib_z, = ax2_vib_z.plot(timestampVib, vibdata[6], label='Vib2-Z', color='C3', linewidth=0.5)
            ax2_vib_z.yaxis.tick_right()
            ax2_vib_z.set_ylim([-16,16])
            ax2_vib_z.set_xlim([tsMin, tsMax])


        ### Config ###
        ''' ax1_nav_pwm1.legend(
            (p1_nav_pwm1, p1_vib_x, p1_vib_y, p1_vib_z),
            ('PWM1', 'Vib1-X', 'Vib1-Y', 'Vib1-Z'),
            loc='upper right'
        )
        ax2_nav_pwm2.legend(
            (p2_nav_pwm2, p2_vib_x, p2_vib_y, p2_vib_z),
            ('PWM2', 'Vib2-X', 'Vib2-Y', 'Vib2-Z'),
            loc='upper right'
        ) '''
        #plt.grid(True)
        plt.show()



    def plotNavdataVibAggregated(self, navdata_agg, vibdata_agg, axis=None, time_window=None):
        aggArray = self.combineAggregatedArray(navdata_agg, vibdata_agg, time_window)

        # Default axis selection
        if axis == None:
            axis = ['x','y','z']

        # Create list from sequence
        timestampArray = [a[0] for a in aggArray]
        pwm1Array = [a[1][0] for a in aggArray]
        pwm2Array = [a[1][1] for a in aggArray]
        mpu1xArray = [a[2][0] for a in aggArray]
        mpu1yArray = [a[2][1] for a in aggArray]
        mpu1zArray = [a[2][2] for a in aggArray]
        mpu2xArray = [a[2][3] for a in aggArray]
        mpu2yArray = [a[2][4] for a in aggArray]
        mpu2zArray = [a[2][5] for a in aggArray]

        tsMin = (min(timestampArray)//1000) * 1000
        tsMax = (max(timestampArray)//1000) * 1000

        ##### Setup plot #####
        fig = plt.figure()

        # Subplot 1
        if 'x' in axis:
            ax1_vib_x = fig.add_subplot(211, label='Vib Motor 1 - X', frame_on=True)
        if 'y' in axis:
            ax1_vib_y = fig.add_subplot(211, label='Vib Motor 1 - Y', frame_on=False)
        if 'z' in axis:
            ax1_vib_z = fig.add_subplot(211, label='Vib Motor 1 - Z', frame_on=False)
        ax1_nav_pwm1 = fig.add_subplot(211, label='PWM Motor 1', frame_on=False)

        # Subplot 2
        if 'x' in axis:
            ax2_vib_x = fig.add_subplot(212, label='Vib Motor 2 - X', frame_on=True)
        if 'y' in axis:
            ax2_vib_y = fig.add_subplot(212, label='Vib Motor 2 - Y', frame_on=False)
        if 'z' in axis:
            ax2_vib_z = fig.add_subplot(212, label='Vib Motor 2 - Z', frame_on=False)
        ax2_nav_pwm2 = fig.add_subplot(212, label='PWM Motor 2', frame_on=False)


        ##### Plot #####
        ### Subplot 1 ###

        # PWM
        p1_nav_pwm1, = ax1_nav_pwm1.plot(timestampArray, pwm1Array, label='PWM1', color='C0', linewidth=2)
        ax1_nav_pwm1.set_ylabel('PWM Motor 1', color='C0')
        ax1_nav_pwm1.set_ylim([-10,310])
        ax1_nav_pwm1.set_xlim([tsMin, tsMax])
        ax1_nav_pwm1.set_xticks(np.arange(tsMin, tsMax, 2500))

        # Vib - X
        if 'x' in axis:
            p1_vib_x, = ax1_vib_x.plot(timestampArray, mpu1xArray, label='Vib1-X', color='C1', linewidth=0.5)
            ax1_vib_x.yaxis.tick_right()
            ax1_vib_x.set_ylim([-16,16])
            ax1_vib_x.set_xlim([tsMin, tsMax])
            ax1_vib_x.set_xticks(np.arange(tsMin, tsMax, 2500))

        # Vib - Y
        if 'y' in axis:
            p1_vib_y, = ax1_vib_y.plot(timestampArray, mpu1yArray, label='Vib1-Y', color='C2', linewidth=0.5)
            ax1_vib_y.yaxis.tick_right()
            ax1_vib_y.set_ylim([-16,16])
            ax1_vib_y.set_xlim([tsMin, tsMax])
            ax1_vib_y.set_xticks(np.arange(tsMin, tsMax, 2500))

        # Vib - Z
        if 'z' in axis:
            p1_vib_z, = ax1_vib_z.plot(timestampArray, mpu1zArray, label='Vib1-Z', color='C3', linewidth=0.5)
            ax1_vib_z.yaxis.tick_right()
            ax1_vib_z.set_ylim([-16,16])
            ax1_vib_z.set_xlim([tsMin, tsMax])
            ax1_vib_z.set_xticks(np.arange(tsMin, tsMax, 2500))


        ### Subplot 2 ###

        # PWM
        p2_nav_pwm2, = ax2_nav_pwm2.plot(timestampArray, pwm2Array, label='PWM2', color='C0', linewidth=2)
        ax2_nav_pwm2.set_ylabel('PWM Motor 2', color='C0')
        ax2_nav_pwm2.set_ylim([-10,310])
        ax2_nav_pwm2.set_xlim([tsMin, tsMax])

        # Vib - X
        if 'x' in axis:
            p2_vib_x, = ax2_vib_x.plot(timestampArray, mpu2xArray, label='Vib2-X', color='C1', linewidth=0.5)
            ax2_vib_x.yaxis.tick_right()
            ax2_vib_x.set_ylim([-16,16])
            ax2_vib_x.set_xlim([tsMin, tsMax])

        # Vib - Y
        if 'y' in axis:
            p2_vib_y, = ax2_vib_y.plot(timestampArray, mpu2yArray, label='Vib2-Y', color='C2', linewidth=0.5)
            ax2_vib_y.yaxis.tick_right()
            ax2_vib_y.set_ylim([-16,16])
            ax2_vib_y.set_xlim([tsMin, tsMax])

        # Vib - Z
        if 'z' in axis:
            p2_vib_z, = ax2_vib_z.plot(timestampArray, mpu2zArray, label='Vib2-Z', color='C3', linewidth=0.5)
            ax2_vib_z.yaxis.tick_right()
            ax2_vib_z.set_ylim([-16,16])
            ax2_vib_z.set_xlim([tsMin, tsMax])


        ### Config ###
        plt.grid(True)
        plt.show()

    
    def plotRawCombined(self, combined_data, data_key, idx, plot_title, max_ts=None):
        '''
            Plot raw data from combined multi feature data
            Black and White graph (optimized for final report)

            Compatible feed function:
            - NavdataVib.combineDataMultiFeature
                - NavData.getMultiFeatureArray
                - VibData.getMultiFeatureArray

            ---------------
            Input format:
            ---------------
            list of dict
                {
                    timestamp,
                    pwm: [mot1,mot2,mot3,mot4],
                    orientation: [roll,pitch,yaw],
                    rawAccel: [x,y,z],
                    mpu1: [x,y,z],
                    mpu2: [x,y,z],
                }

            ---------------
            Argument format:
            ---------------
            combined_data: input format
            key: key of dict from combined_data
            idx: array index (to select axis) from combined_data[key]
        '''

        # Set font to Times New Roman
        plt.rcParams["font.family"] = "Times New Roman"

        # Get timestamp array
        xData = [(data['timestamp'] - combined_data[0]['timestamp']) for data in combined_data]
        yData = [data[data_key][idx] for data in combined_data]
        print(xData[-1])

        # Slice until max timestamp supplied in arg
        if (max_ts != None) and (max_ts < xData[-1]):
            xDataNew = []

            i = 0
            while True:
                if xData[i] > max_ts:
                    break
                xDataNew.append(xData[i])
                i += 1

            xData = xDataNew

            yData = yData[:i]

        # Set x-axis limit and ticks
        xlim = [0, xData[-1]]
        xticks = list(range(0, (((xData[-1] // 1000) + 1) * 1000) + 1, 5000))

        # Set y-axis limit and ticks
        if data_key == 'pwm':
            ylim = [0,280]  # pwm
            yticks = list(range(0,280,100))
            ylabel = 'PWM'

        if data_key == 'orientation':
            if idx == 0:
                ylabel = 'Roll (deg)'
            if idx == 1:
                ylabel = 'Pitch (deg)'
            if idx == 2:
                ylabel = 'Yaw (deg)'
            ylim = [-180,180]  # orientation.roll
            yticks = list(range(-180,181,90))

        if data_key == 'mpu1' or data_key == 'mpu2':
            if idx == 0:
                ylabel = 'Akselerasi (g)'
            if idx == 1:
                ylabel = 'Akselerasi (g)'
            if idx == 2:
                ylabel = 'Akselerasi (g)'
            ylim = [-16,16]  # orientation.roll
            yticks = list(range(-16,17,8))

        fig = plt.figure(figsize=(16,8), dpi=120)
        fig.subplots_adjust(left=0.07, right=0.97, top=0.35, bottom=0.1)
        plt.get_current_fig_manager().window.state('zoomed')

        ax1 = fig.add_subplot(111, frame_on=True)

        p1, = ax1.plot(xData, yData, 'k-', linewidth=1)
        ax1.tick_params(grid_alpha=0.6, grid_linewidth=0.4, labelsize=16)
        ax1.set_xticks(xticks)
        ax1.set_xlim(xlim)
        ax1.set_yticks(yticks)
        ax1.set_ylim(ylim)
        ax1.grid(True)
        ax1.set_title(plot_title, fontsize=22)
        ax1.set_xlabel('Waktu (ms)', fontsize=22)
        ax1.set_ylabel(ylabel, fontsize=22)

        """ ax2 = fig.add_subplot(111, frame_on=False)
        p2, = ax2.plot(xData, yData, 'k-', linewidth=2)
        ax2.set_xticks(list(range(0,21,2)))
        ax2.set_xlim(xlim)
        ax2.set_ylim([0,100])
        ax2.grid(True) """

        """ ax1.legend(
            (p1,p2),
            ('Plot 1', 'Plot 2'),
            loc='upper right',
            fontsize=16
        ) """

        plt.show()

    
    def plotAggregateCombined(self, combined_agg, mot, data_key, idx, max_ts=None, save_only=False):
        '''
            Compatible input data feed: NavdataVib.combineDataMultiFeature
            ---------------
            Input format:
            ---------------
            list of dict
                {
                    timestamp,
                    orientation: [roll,pitch,yaw],
                    rawAccel: [x,y,z],
                    mot1: {
                        'pwm',
                        'rms': [x,y,z],
                        'kurtosis': [x,y,z],
                        'skewness': [x,y,z],
                        'crest-factor': [x,y,z],
                        'peak-to-peak': [x,y,z],
                    },
                    mot2: {
                        'pwm',
                        'rms': [x,y,z],
                        'kurtosis': [x,y,z],
                        'skewness': [x,y,z],
                        'crest-factor': [x,y,z],
                        'peak-to-peak': [x,y,z],
                    }
                }

            ---------------
            Argument format:
            ---------------
            combined: input format
            mot: motor number [mot1,mot2]
            key: key of dict from combined_data
            idx: array index (to select axis) from combined_data[key]
                -> currently supported mot1
        '''

        # # Set font to Times New Roman
        plt.rcParams["font.family"] = "Times New Roman"

        # Get timestamp array
        xData = [(data['timestamp'] - combined_agg[0]['timestamp']) for data in combined_agg]
        
        # Get feature data array
        if data_key == 'pwm':
            yData = [data[mot][data_key] for data in combined_agg]
        else:
            yData = [data[mot][data_key][idx] for data in combined_agg]

        # Slice until max timestamp supplied in arg
        if (max_ts != None) and (max_ts < xData[-1]):
            xDataNew = []

            i = 0
            while True:
                if xData[i] > max_ts:
                    break
                xDataNew.append(xData[i])
                i += 1

            xData = xDataNew

            yData = yData[:i]

        # Set x-axis limit and ticks
        xlim = [0, xData[-1]]
        xticks = list(range(0, (((xData[-1] // 1000) + 1) * 1000) + 1, 5000))

        # Set y-axis limit and ticks
        if data_key == 'pwm':
            ylim = [0,280]
            yticks = list(range(ylim[0],ylim[1]+1,50))
            ylabel = 'PWM'

        if data_key == 'rms':
            ylim = [0,20]
            yticks = list(range(ylim[0],ylim[1]+1,5))
            ylabel = 'RMS'
        
        if data_key == 'kurtosis':
            ylim = [-10,30]
            yticks = list(range(ylim[0],ylim[1]+1,10))
            ylabel = 'Kurtosis'

        if data_key == 'skewness':
            ylim = [-10,10]
            yticks = list(range(ylim[0],ylim[1]+1,5))
            ylabel = 'Skewness'

        if data_key == 'crest-factor':
            ylim = [-5,10]
            yticks = list(range(ylim[0],ylim[1]+1,5))
            ylabel = 'Crest Factor'

        if data_key == 'peak-to-peak':
            ylim = [0,40]
            yticks = list(range(ylim[0],ylim[1]+1,10))
            ylabel = 'Peak-to-Peak'

        if data_key != 'pwm':
            if idx == 0:
                ylabel = ylabel + ' (X)'
            if idx == 1:
                ylabel = ylabel + ' (Y)'
            if idx == 2:
                ylabel = ylabel + ' (Z)'

        # Plot
        fig = plt.figure(figsize=(16,8), dpi=120)
        fig.subplots_adjust(left=0.07, right=0.97, top=0.35, bottom=0.1)
        plt.get_current_fig_manager().window.state('zoomed')

        ax1 = fig.add_subplot(111, frame_on=True)

        p1, = ax1.plot(xData, yData, 'k-', linewidth=1.5)
        ax1.tick_params(grid_alpha=0.6, grid_linewidth=0.4, labelsize=16)
        ax1.set_xticks(xticks)
        ax1.set_xlim(xlim)
        ax1.set_yticks(yticks)
        ax1.set_ylim(ylim)
        ax1.grid(True)
        #ax1.set_xlabel('Waktu (ms)', fontsize=22)
        ax1.set_ylabel(ylabel, fontsize=22)

        if save_only:
            plt.savefig(
                fname=save_only + 
                    predPlotFilename.format(
                        timePlot,
                        timeWindow,
                        mot,
                        timeWindow,
                        data_key,
                        idx)
            )
        else:
            plt.show()

    
    def plotAggregateCombinedMotDiff(self, combined_agg, data_key, idx, max_ts=None, save_only=False):
        '''
            Compatible input data feed: NavdataVib.combineDataMultiFeature
            Print between two dataset

            ---------------
            Input format:
            ---------------
            list of dict
                {
                    timestamp,
                    orientation: [roll,pitch,yaw],
                    rawAccel: [x,y,z],
                    mot1: {
                        'pwm',
                        'rms': [x,y,z],
                        'kurtosis': [x,y,z],
                        'skewness': [x,y,z],
                        'crest-factor': [x,y,z],
                        'peak-to-peak': [x,y,z],
                    },
                    mot2: {
                        'pwm',
                        'rms': [x,y,z],
                        'kurtosis': [x,y,z],
                        'skewness': [x,y,z],
                        'crest-factor': [x,y,z],
                        'peak-to-peak': [x,y,z],
                    }
                }

            ---------------
            Argument format:
            ---------------
            combined: input format
            mot: motor number [mot1,mot2]
            key: key of dict from combined_data
            idx: array index (to select axis) from combined_data[key]
                -> currently supported mot1
        '''

        # # Set font to Times New Roman
        plt.rcParams["font.family"] = "Times New Roman"

        # Get timestamp array
        xData = [(data['timestamp'] - combined_agg[0]['timestamp']) for data in combined_agg]
        
        # Get feature data array
        if data_key == 'pwm':
            yData1 = [data['mot2'][data_key] for data in combined_agg]
            yData2 = [data['mot1'][data_key] for data in combined_agg]
        else:
            yData1 = [data['mot1'][data_key][idx] for data in combined_agg]
            yData2 = [data['mot2'][data_key][idx] for data in combined_agg]

        # Slice until max timestamp supplied in arg
        if (max_ts != None) and (max_ts < xData[-1]):
            xDataNew = []

            i = 0
            while True:
                if xData[i] > max_ts:
                    break
                xDataNew.append(xData[i])
                i += 1

            xData = xDataNew

            yData1 = yData1[:i]
            yData2 = yData2[:i]

        # Set x-axis limit and ticks
        xlim = [0, xData[-1]]
        xticks = list(range(0, (((xData[-1] // 1000) + 1) * 1000) + 1, 5000))

        # Set y-axis limit and ticks
        if data_key == 'pwm':
            ylim = [0,280]
            yticks = list(range(ylim[0],ylim[1]+1,50))
            ylabel = 'PWM'

        if data_key == 'rms':
            ylim = [0,20]
            yticks = list(range(ylim[0],ylim[1]+1,5))
            ylabel = 'RMS'
        
        if data_key == 'kurtosis':
            ylim = [-10,30]
            yticks = list(range(ylim[0],ylim[1]+1,10))
            ylabel = 'Kurtosis'

        if data_key == 'skewness':
            ylim = [-10,10]
            yticks = list(range(ylim[0],ylim[1]+1,5))
            ylabel = 'Skewness'

        if data_key == 'crest-factor':
            ylim = [-5,10]
            yticks = list(range(ylim[0],ylim[1]+1,5))
            ylabel = 'Crest Factor'

        if data_key == 'peak-to-peak':
            ylim = [0,40]
            yticks = list(range(ylim[0],ylim[1]+1,10))
            ylabel = 'Peak-to-Peak'

        if data_key != 'pwm':
            if idx == 0:
                ylabel = ylabel + ' (X)'
            if idx == 1:
                ylabel = ylabel + ' (Y)'
            if idx == 2:
                ylabel = ylabel + ' (Z)'

        # Plot
        fig = plt.figure(figsize=(16,8), dpi=120)
        fig.subplots_adjust(left=0.07, right=0.97, top=0.35, bottom=0.1)
        plt.get_current_fig_manager().window.state('zoomed')

        ax1 = fig.add_subplot(111, frame_on=True)
        ax2 = fig.add_subplot(111, frame_on=False)

        p1, = ax1.plot(xData, yData1, 'k-', linewidth=1.5)
        ax1.tick_params(grid_alpha=0.6, grid_linewidth=0.4, labelsize=16)
        ax1.set_xticks(xticks)
        ax1.set_xlim(xlim)
        ax1.set_yticks(yticks)
        ax1.set_ylim(ylim)
        ax1.grid(True)
        ax1.set_xlabel('Waktu (ms)', fontsize=22)
        ax1.set_ylabel(ylabel, fontsize=22)

        p2, = ax2.plot(xData, yData2, 'r-', linewidth=1.5)
        ax2.set_xticks([])
        ax2.set_xlim(xlim)
        ax2.set_yticks([])
        ax2.set_ylim(ylim)

        ax1.legend(
            (p1,p2),
            ('Normal', 'Abnormal'),
            loc='upper left',
            fontsize=18
        )

        if save_only:
            plt.savefig(
                fname=save_only + 
                    predPlotFilename.format(
                        timePlot,
                        timeWindow,
                        timeWindow,
                        data_key,
                        idx)
            )
        else:
            plt.show()

    
    def plotAggregateCombinedMotDiff(self, combined_agg, data_key, idx, max_ts=None, save_only=False):
        '''
            Compatible input data feed: NavdataVib.combineDataMultiFeature
            Print between two dataset

            ---------------
            Input format:
            ---------------
            list of dict
                {
                    timestamp,
                    orientation: [roll,pitch,yaw],
                    rawAccel: [x,y,z],
                    mot1: {
                        'pwm',
                        'rms': [x,y,z],
                        'kurtosis': [x,y,z],
                        'skewness': [x,y,z],
                        'crest-factor': [x,y,z],
                        'peak-to-peak': [x,y,z],
                    },
                    mot2: {
                        'pwm',
                        'rms': [x,y,z],
                        'kurtosis': [x,y,z],
                        'skewness': [x,y,z],
                        'crest-factor': [x,y,z],
                        'peak-to-peak': [x,y,z],
                    }
                }

            ---------------
            Argument format:
            ---------------
            combined: input format
            mot: motor number [mot1,mot2]
            key: key of dict from combined_data
            idx: array index (to select axis) from combined_data[key]
                -> currently supported mot1
        '''

        # # Set font to Times New Roman
        plt.rcParams["font.family"] = "Times New Roman"

        # Get timestamp array
        xData = [(data['timestamp'] - combined_agg[0]['timestamp']) for data in combined_agg]
        
        # Get feature data array
        if data_key == 'pwm':
            yData1 = [data['mot2'][data_key] for data in combined_agg]
            yData2 = [data['mot1'][data_key] for data in combined_agg]
        else:
            yData1 = [data['mot1'][data_key][idx] for data in combined_agg]
            yData2 = [data['mot2'][data_key][idx] for data in combined_agg]

        # Slice until max timestamp supplied in arg
        if (max_ts != None) and (max_ts < xData[-1]):
            xDataNew = []

            i = 0
            while True:
                if xData[i] > max_ts:
                    break
                xDataNew.append(xData[i])
                i += 1

            xData = xDataNew

            yData1 = yData1[:i]
            yData2 = yData2[:i]

        # Set x-axis limit and ticks
        xlim = [0, xData[-1]]
        xticks = list(range(0, (((xData[-1] // 1000) + 1) * 1000) + 1, 5000))

        # Set y-axis limit and ticks
        if data_key == 'pwm':
            ylim = [0,280]
            yticks = list(range(ylim[0],ylim[1]+1,50))
            ylabel = 'PWM'

        if data_key == 'rms':
            ylim = [0,20]
            yticks = list(range(ylim[0],ylim[1]+1,5))
            ylabel = 'RMS'
            print(np.mean(yData1), np.mean(yData2))
        
        if data_key == 'kurtosis':
            ylim = [-10,30]
            yticks = list(range(ylim[0],ylim[1]+1,10))
            ylabel = 'Kurtosis'

        if data_key == 'skewness':
            ylim = [-10,10]
            yticks = list(range(ylim[0],ylim[1]+1,5))
            ylabel = 'Skewness'

        if data_key == 'crest-factor':
            ylim = [-5,10]
            yticks = list(range(ylim[0],ylim[1]+1,5))
            ylabel = 'Crest Factor'

        if data_key == 'peak-to-peak':
            ylim = [0,40]
            yticks = list(range(ylim[0],ylim[1]+1,10))
            ylabel = 'Peak-to-Peak'

        if data_key != 'pwm':
            if idx == 0:
                ylabel = ylabel + ' (X)'
            if idx == 1:
                ylabel = ylabel + ' (Y)'
            if idx == 2:
                ylabel = ylabel + ' (Z)'

        # Plot
        fig = plt.figure(figsize=(16,2.2), dpi=120)
        fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.15)
        #plt.get_current_fig_manager().window.state('zoomed')

        ax1 = fig.add_subplot(111, frame_on=True)
        ax2 = fig.add_subplot(111, frame_on=False)

        p1, = ax1.plot(xData, yData1, 'k-', linewidth=1.5)
        ax1.tick_params(grid_alpha=0.6, grid_linewidth=0.4, labelsize=16)
        ax1.set_xticks(xticks)
        ax1.set_xlim(xlim)
        ax1.set_yticks(yticks)
        ax1.set_ylim(ylim)
        ax1.grid(True)
        ax1.set_xlabel('Waktu (ms)', fontsize=22)
        ax1.set_ylabel(ylabel, fontsize=22)

        p2, = ax2.plot(xData, yData2, 'r-', linewidth=1.5)
        ax2.set_xticks([])
        ax2.set_xlim(xlim)
        ax2.set_yticks([])
        ax2.set_ylim(ylim)

        ax1.legend(
            (p1,p2),
            ('Normal', 'Abnormal'),
            loc='upper left',
            fontsize=18
        )

        if save_only:
            plt.savefig(
                fname=save_only + 
                    predPlotFilename.format(
                        timePlot,
                        timeWindow,
                        timeWindow,
                        data_key,
                        idx)
            )
        else:
            plt.show()

    
    def plotAltitude(self, nav_id, max_ts=None):
        '''
            [!!!] Not finished
        '''

        plt.rcParams["font.family"] = "Times New Roman"

        xData = self.getTimestampArray(nav_id)
        xData = [x - xData[0] for x in xData]
        yData = self.getAltitudeArray(nav_id)

        fig = plt.figure(figsize=(16,8), dpi=120)
        fig.subplots_adjust(left=0.07, right=0.97, top=0.35, bottom=0.1)
        plt.get_current_fig_manager().window.state('zoomed')

        ax1 = fig.add_subplot(111, frame_on=True)

        xlim = [0, xData[-1]]
        xticks = list(range(0, (((xData[-1] // 1000) + 1) * 1000) + 1, 5000))

        ylim = [0,1.5]  # orientation.roll
        yticks = [0,0.5,1,1.5]
        ylabel = 'Ketinggian (m)'

        p1, = ax1.plot(xData, yData, 'k-', linewidth=0.5)
        ax1.tick_params(grid_alpha=0.6, grid_linewidth=0.4, labelsize=16)
        ax1.set_xticks(xticks)
        ax1.set_xlim(xlim)
        ax1.set_yticks(yticks)
        ax1.set_ylim(ylim)
        ax1.grid(True)
        ax1.set_xlabel('Waktu (ms)', fontsize=22)
        ax1.set_ylabel(ylabel, fontsize=22)

        plt.show()

    
    def plotAggregateCombinedLoadDiff(self, dataset, data_key, idx, max_ts=None, save_only=False):
        '''
            Compatible input data feed: NavdataVib.combineDataMultiFeature
            Print between two dataset

            ---------------
            Input format:
            ---------------
            list of dict    -> stricted to 7 loadCount (0-6)
                {
                    'description',
                    'loadCount',
                    'combinedAgg',
                    'shiftTime'     -> time shift to uniform takeoff starting time
                }

            'combinedAgg': list of dict
                {
                    timestamp,
                    orientation: [roll,pitch,yaw],
                    rawAccel: [x,y,z],
                    mot1: {
                        'pwm',
                        'rms': [x,y,z],
                        'kurtosis': [x,y,z],
                        'skewness': [x,y,z],
                        'crest-factor': [x,y,z],
                        'peak-to-peak': [x,y,z],
                    },
                    mot2: {
                        'pwm',
                        'rms': [x,y,z],
                        'kurtosis': [x,y,z],
                        'skewness': [x,y,z],
                        'crest-factor': [x,y,z],
                        'peak-to-peak': [x,y,z],
                    }
                }

            ---------------
            Argument format:
            ---------------
            dataset: input format
            mot: motor number [mot1,mot2]
            key: key of dict from combined_data
            idx: array index (to select axis) from combined_data[key]
                -> currently supported mot1
            save_only: don't show plot
        '''

        print('Plotting Combined Load Diff')

        # # Set font to Times New Roman
        plt.rcParams["font.family"] = "Times New Roman"

        xDataArr = []
        yDataArr = []
        for data in dataset:
            combinedAgg = data['combinedAgg']
            
            # Get timestamp array
            xData = [(comb['timestamp'] - combinedAgg[0]['timestamp']) for comb in combinedAgg]
        
            # Get feature data array
            if data_key == 'pwm':
                yData = [comb['mot1'][data_key] for comb in combinedAgg]
            else:
                yData = [comb['mot1'][data_key][idx] for comb in combinedAgg]

            # Slice data until timestamp *max_ts* (unnecessary bc already plot range??)
            """ if (max_ts != None) and (max_ts < xData[-1]):
                xDataNew = []

                # Get index not exceeding *max_ts*
                i = len(xData)
                while True:
                    if xData[i] < max_ts:
                        break
                    i -= 1

                # Slice array
                xData = xData[:i]
                yData = yData[:i]
            """

            # Insert into data array
            xDataArr.append(xData)
            yDataArr.append(yData)


        # Set x-axis limit and ticks
        xlim = [0, xData[-1]]
        xticks = list(range(0, (((xData[-1] // 1000) + 1) * 1000) + 1, 5000))

        # Set y-axis limit and ticks
        if data_key == 'pwm':
            ylim = [0,280]
            yticks = list(range(ylim[0],ylim[1]+1,50))
            ylabel = 'PWM'

        if data_key == 'rms':
            ylim = [0,20]
            yticks = list(range(ylim[0],ylim[1]+1,5))
            ylabel = 'RMS'
        
        if data_key == 'kurtosis':
            ylim = [-10,30]
            yticks = list(range(ylim[0],ylim[1]+1,10))
            ylabel = 'Kurtosis'

        if data_key == 'skewness':
            ylim = [-10,10]
            yticks = list(range(ylim[0],ylim[1]+1,5))
            ylabel = 'Skewness'

        if data_key == 'crest-factor':
            ylim = [-5,10]
            yticks = list(range(ylim[0],ylim[1]+1,5))
            ylabel = 'Crest Factor'

        if data_key == 'peak-to-peak':
            ylim = [0,40]
            yticks = list(range(ylim[0],ylim[1]+1,10))
            ylabel = 'Peak-to-Peak'

        if data_key != 'pwm':
            if idx == 0:
                ylabel = ylabel + ' (X)'
            if idx == 1:
                ylabel = ylabel + ' (Y)'
            if idx == 2:
                ylabel = ylabel + ' (Z)'

        # Plot
        fig = plt.figure(figsize=(16,8), dpi=120)
        fig.subplots_adjust(left=0.07, right=0.97, top=0.35, bottom=0.1)
        plt.get_current_fig_manager().window.state('zoomed')

        axArr = [0,0,0,0,0,0,0]
        for i in range(7):
            if i == 0:
                axArr[i] = fig.add_subplot(111, label=str(i), frame_on=True)
            else:
                axArr[i] = fig.add_subplot(111, label=str(i), frame_on=False)

        pArr = []

        # 7 line
        for i in range(7):
            p, = axArr[i].plot(xDataArr[i], yDataArr[i], linewidth=0.5, color='C' + str(i))
            
            if i == 0:
                axArr[i].tick_params(grid_alpha=0.6, grid_linewidth=0.4, labelsize=16)
                axArr[i].set_xticks(xticks)
                axArr[i].set_xlim(xlim)
                axArr[i].set_yticks(yticks)
                axArr[i].set_ylim(ylim)
                axArr[i].grid(True)
                axArr[i].set_xlabel('Waktu (ms)', fontsize=22)
                axArr[i].set_ylabel(ylabel, fontsize=22)

            else:
                axArr[i].set_xticks([])
                axArr[i].set_xlim(xlim)
                axArr[i].set_yticks([])
                axArr[i].set_ylim(ylim)

            pArr.append(p)

        axArr[0].legend(
            tuple(pArr),
            tuple([('Load '+str(i)) for i in range(7)]), #('0','1','2','3','4','5','6'),         
            loc='upper right',
            fontsize=14
        )

        if save_only:
            plt.savefig(
                fname=save_only + 
                    predPlotFilename.format(
                        timePlot,
                        timeWindow,
                        timeWindow,
                        data_key,
                        idx)
            )
        else:
            plt.show()


##### Driver #####
if __name__ == '__main__':
    ### Parameter ###
    queryDescription = 'aug9_0_hover30s_5.json'
    dataSaveDir = 'D:/Cloud/Google Drive/Tugas Akhir/Laporan/plot/mot-diff'
    predPlotFilename = '/{}_{}_motdiff/plot_pred_{}_{}_{}.svg'
    plotVibAxis = ['x','y','z']
    stepWeight = 0.1
    timeWindow = 500
    
    ### Object Declaration ###
    Vib = VibData()
    Nav = NavData()
    NV = NavdataVib()

    # List description
    descList = [nav['description'] for nav in Nav.listDescriptionTimestamp()[1:]]
    #print('List description:', *descList, sep='\n')

    # Get timestamp range
    tstart, tstop = Nav.getTimestampRangeByDescription(description=queryDescription, landed=True)

    # Store data to active
    navId = Nav.storeData(Nav.getByDescription(description=queryDescription, landed=True), 'test by description')
    vibId = Vib.storeData(Vib.getBetweenTimestamp(tstart, tstop), 'test between timestamp')

    # Combine data
    navMultiArray = Nav.getMultiFeatureArray(navId)
    vibMultiArray = Vib.getMultiFeatureArray(vibId)
    combinedMultiArray = NV.combineDataMultiFeature(
        navdata=navMultiArray,
        vibdata=vibMultiArray
    )
    combinedAgg = NV.aggregateCombined(
        combined_data=combinedMultiArray,
        time_window=timeWindow
    )


    """ NV.plotRawCombined(
        combined_data=combinedMultiArray,
        data_key='pwm',
        idx=0,
        plot_title='Akselerasi (z) Motor 1',
        max_ts=40000
    ) """
    
    ##### Plot Aggregation #####
    timePlot = str(datetime.now().strftime('%y_%m_%d_%H_%M_%S'))
    os.mkdir(dataSaveDir + '/' + timePlot + '_' + str(timeWindow) + '_motdiff')

    # Plot combined aggregate
    """ NV.plotAggregateCombined(
            combined_agg=combinedAgg,
            mot='mot2',
            data_key='pwm',
            idx=0,
            max_ts=40000,
            save_only=dataSaveDir
        ) """

    # Aggregate for 2 different motor (normal v. abnormal)
    
    NV.plotAggregateCombinedMotDiff(
        combined_agg=combinedAgg,
        data_key='pwm',
        idx=0,
        max_ts=40000,
        save_only=dataSaveDir
    )
    
    for dataKey in ['rms', 'kurtosis', 'skewness', 'crest-factor', 'peak-to-peak']:
        for idx in range(3):
            # Aggregate only
            """ NV.plotAggregateCombined(
                combined_agg=combinedAgg,
                mot='mot2',
                data_key=dataKey,
                idx=idx,
                max_ts=40000,
                save_only=dataSaveDir
            ) """

            # Aggregate for 2 different motor (normal v. abnormal)
            NV.plotAggregateCombinedMotDiff(
                combined_agg=combinedAgg,
                data_key=dataKey,
                idx=idx,
                max_ts=40000,
                save_only=dataSaveDir
            )


    ##### Plot Load 0-6 (Motor1) #####
    """ descList = [
        'aug9_0_hover20s_2.json',
        'aug10_1_hover20s_6.json',
        'aug10_2_hover20s_9.json',
        'aug10_3_hover20s_2.json',
        'aug11_4_hover20s_6.json',
        'aug11_5_hover20s_4.json',
        'aug11_6_hover20s_5.json'
    ]

    feedDataset = []

    timePlot = str(datetime.now().strftime('%y_%m_%d_%H_%M_%S'))
    os.mkdir(os.path.join(dataSaveDir,timePlot+'_'+str(timeWindow)+'_loaddiff'))

    for desc in descList:
        print('Loading', desc)
        feedData = {
            'description': desc,
            'loadCount': int(desc.split('_')[1]),
            'combinedAgg': [],
            'shiftTime': 0
        }

        # Load data
        tstart, tstop = Nav.getTimestampRangeByDescription(
                                description=desc, 
                                landed=True)

        # Store data to active
        navId = Nav.storeData(Nav.getByDescription(description=desc, landed=True), 'test by description')
        vibId = Vib.storeData(Vib.getBetweenTimestamp(tstart, tstop), 'test between timestamp')

        # Combine data
        navMultiArray = Nav.getMultiFeatureArray(navId)
        vibMultiArray = Vib.getMultiFeatureArray(vibId)
        combinedMultiArray = NV.combineDataMultiFeature(
            navdata=navMultiArray,
            vibdata=vibMultiArray
        )
        combinedAgg = NV.aggregateCombined(
            combined_data=combinedMultiArray,
            time_window=timeWindow
        )

        feedData['combinedAgg'] = combinedAgg

        feedDataset.append(feedData)

    NV.plotAggregateCombinedLoadDiff(
        dataset=feedDataset,
        data_key='pwm',
        idx=0,
        max_ts=30000,
        save_only=dataSaveDir
    )

    for dataKey in ['rms','kurtosis','skewness','crest-factor','peak-to-peak']:
        for idx in range(3):
            NV.plotAggregateCombinedLoadDiff(
                dataset=feedDataset,
                data_key=dataKey,
                idx=idx,
                max_ts=30000,
                save_only=dataSaveDir
            ) """

    ##### Plot PWM Masking Control State #####
    """ csArray = Nav.getControlStateArray(navId)
    #print(*list(set(csArray)), sep='\n')

    pwmArray = Nav.getPWMArray(navId)

    # Masking number for Control State Array
    csArrayMask = []
    for cs in csArray:

        if cs == 'CTRL_LANDED':
            csArrayMask.append(0)
        elif cs == 'CTRL HOVERING':
            csArrayMask.append(5)
        elif cs == 'CTRL_TRANS_GOTOFIX':
            csArrayMask.append(2)
        elif cs == 'CTRL_TRANS_LOOPING':
            csArrayMask.append(3)
        elif cs == 'CTRL_TRANS_LANDING':
            csArrayMask.append(1)
        elif cs == 'CTRL_FLYING':
            csArrayMask.append(4)

    fig = plt.figure()

    ax1 = fig.add_subplot(111, frame_on=True)
    ax2 = fig.add_subplot(111, frame_on=False)

    xData = list(range(max(len(csArray),len(pwmArray))))

    p1, = ax1.plot(xData, pwmArray[0])
    ax1.set_xlim([xData[0],xData[-1]])
    ax1.set_ylim([0,280])
    ax1.set_yticks(list(range(0,280,50)))
    ax1.grid(True)
    
    p2, = ax2.plot(xData, csArray, color='C3')
    ax2.set_xlim([xData[0],xData[-1]])
    ax2.set_ylim([0,6])
    ax2.yaxis.tick_right()
    ax2.set_yticks([0,1,2,3,4,5])
    ax2.grid(True)

    plt.show()
 """