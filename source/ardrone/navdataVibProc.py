'''
    TO DO:
    [v] Nav, Vib aggregation split timestamp delta spike into smaller
        - Dynamic split by window size
    [ ] Output data aggregation into RNN-LSTM compatible dimension
    [ ] Merge training data by all description

'''


import pymongo
import numpy as np
import matplotlib.pyplot as plt
import time
import collections
import scipy.stats
import sys


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
                rms: [mpu1.x,...,mpu2.z],
                kurtosis: [mpu1.x,...,mpu2.z],
                skewness: [mpu1.x,...,mpu2.z],
                crest-factor: [mpu1.x,...,mpu2.z],
                peak-to-peak: [mpu1.x,...,mpu2.z]
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


    def getMultiFeatureArray(self, active_id):
        '''
            Get multi feature array of stored active data with index *active_id*

            return: list of dict
            {
                timestamp,
                pwm: [mot1,mot2,mot3,mot4],
                orientation: [roll,pitch,yaw],
                rawAccel: [x,y,z]
            }
        '''

        # Extract data
        activeData = self.activeDataArray[active_id]['data']

        multiFeatureArray = []
        
        for data in activeData:
            featureData = {
                'timestamp': data['timestamp'],
                'pwm': [p[1] for p in data['navdata']['pwm'].items()],
                'orientation': [p[1] for p in data['navdata']['orientation'].items()],
                'rawAccel': [p[1] for p in data['navdata']['rawMeasures']['accelerometers'].items()]
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
                    pwm: [mot1,mot2,mot3,mot4],
                    orientation: [roll,pitch,yaw],
                    rawAccel: [x,y,z]
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
                    pwm: [mot1,mot2,mot3,mot4],
                    orientation: [roll,pitch,yaw],
                    rawAccel: [x,y,z],
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
                        'pwm': [p[1] for p in navdata[i]['pwm'].items()],
                        'orientation': [o[1] for o in navdata[i]['orientation'].items()],
                        'rawAccel': [r[1] for r in navdata[i]['rawAccel'].items()],
                        'mpu1': [v[1] for v in vibdata[i]['mpu1']],
                        'mpu2': [v[1] for v in vibdata[i]['mpu2']]
                    }
                )

                tsNavPrev = tsNav[i]
                tsVibPrev = tsVib[i]


        # Interpolate vib data into PWM data
        elif tsNavAvg < tsVibAvg:
            #print('tsnav < tsvib')
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

                # Get insert index (vib timestamp less than current nav timestamp)
                i = len(tsVib) - 1
                while tsVib[i] > tsNav[t]:
                    i -= 1

                # Conditional for duplicated timestamp or backward timestamp
                # (may be indicate of duplicate data point / data set)
                if (tsVib[i] < tsVibPrev) or (tsNav[t] < tsNavPrev):
                    #print('{} <= {} | {} <= {}'.format(tsNav[i], tsNavPrev, tsVib[t], tsVibPrev))
                    continue

                # Append data into combined array
                """ {
                    'pwm': [p[1] for p in navdata[i]['pwm'].items()],
                    'orientation': [o[1] for o in navdata[i]['orientation'].items()],
                    'rawAccel': [r[1] for r in navdata[i]['rawAccel'].items()],
                    'mpu1': [v[1] for v in vibdata[i]['mpu1']],
                    'mpu2': [v[1] for v in vibdata[i]['mpu2']]
                } """
                combinedArray.append(
                    {
                        'timestamp': tsNav[t],
                        'pwm': navdata[t]['pwm'],
                        'orientation': navdata[t]['orientation'],
                        'rawAccel': navdata[t]['rawAccel'],
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
                    rawAccel: [x,y,z],
                    mpu1: [x,y,z],
                    mpu2: [x,y,z],
                }

            ---------------
            Return format:
            ---------------
            list of dict
                {
                    orientation: [roll,pitch,yaw],
                    rawAccel: [x,y,z],
                    mot1: {
                        'pwm': 0,
                        'rms': [x,y,z],
                        'kurtosis': [x,y,z],
                        'skewness': [x,y,z],
                        'crest-factor': [x,y,z],
                        'peak-to-peak': [x,y,z],
                    },
                    mot2: {
                        'pwm': 0,
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
                    'orientation': [0,0,0],
                    'rawAccel': [0,0,0],
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
                aggData['rawAccel'] = [np.mean([data['rawAccel'][axis] for data in aggBuf]) for axis in range(3)]
                
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

    
def combineArray(pwmdata, vibdata):
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

    # Check for data duplication
    tsPWMUnique = []
    tsPWMUnique = [tsPWMUnique.append(ts) for ts in tsPWM if ts not in tsPWMUnique] 
    tsVibUnique = []
    tsVibUnique = [tsVibUnique.append(ts) for ts in tsVib if ts not in tsVibUnique]
    
    # PWMData duplicate (most likely)
    if len(tsPWMUnique) != len(tsPWM):
        print('Duplicated data (PWM)! {} -> {}'.format(len(tsPWMUnique), len(tsPWM)))
        tsPWM = tsPWM[:len(tsPWMUnique)]
        pwmArray = pwmArray[:len(tsPWMUnique)]
    
    # Vibdata duplicate (not likely, but ok)
    if len(tsVibUnique) != len(tsVib):
        print('Duplicated data (Vib)!')


    # Get average timestamp
    tsPWMAvg = (max(tsPWM) - min(tsPWM)) / len(tsPWM)
    tsVibAvg = (max(tsVib) - min(tsVib)) / len(tsVib)

    print('tsPWM: {} | tsVib: {}'.format(tsPWMAvg, tsVibAvg))
    print('tsPWM: {} - {}'.format(min(tsPWM), max(tsPWM)))
    print('tsVib: {} - {}'.format(min(tsVib), max(tsVib)))
    
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

        # Restructure data
        combinedArray = []
        for i in range(len(tsVib)):
            combinedArray.append([
                tsVib[i]-tsVib[0],
                newPWMArray[i],
                [
                    vibArray[0][i],
                    vibArray[1][i],
                    vibArray[2][i]
                ]
            ])

    # Interpolate vib data into PWM data
    elif tsPWMAvg < tsVibAvg:
        # Special case: trim boundary data from tsPWM and pwmArray
        startTrim = 0
        while tsPWM[startTrim] < tsVib[0]:
            startTrim += 1

        stopTrim = -1
        while tsPWM[stopTrim] > tsVib[-1]:
            stopTrim -= 1
        
        tsPWM = tsPWM[startTrim:stopTrim]
        pwmArray = pwmArray[startTrim:stopTrim]

        newVibArray = [[],[],[]]

        for ts in tsPWM:
            i = 0
            try:
                while tsVib[i] < ts:
                    i += 1
            except:
                print(i, ':', tsVib[i-1], '<', ts)
                sys.exit()


            # Append inrange data into new pwm array
            newVibArray[0].append(vibArray[0][i])
            newVibArray[1].append(vibArray[1][i])
            newVibArray[2].append(vibArray[2][i])
        
        # Debug
        #print(len(newVibArray[0]), newVibArray[0][:5])
        #print(len(newVibArray[1]), newVibArray[1][:5])
        #print(len(newVibArray[2]), newVibArray[2][:5])

        # Restructure data
        combinedArray = []
        for i in range(len(tsPWM)):
            combinedArray.append([
                tsPWM[i]-tsPWM[0],
                pwmArray[i],
                [
                    newVibArray[0][i],
                    newVibArray[1][i],
                    newVibArray[2][i]
                ]
            ])

        #print('Combined data result: {} | {}'.format(len(newVibArray[0]), len(tsPWM)))
        # combinedArray = [tsVib, newPWMArray, *vibArray]
        return combinedArray
        
    
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


##### Driver #####
if __name__ == '__main__':
    ### Parameter ###
    queryDescription = 'aug9_0_hover10s.json'
    plotVibAxis = ['x','y','z']
    stepWeight = 0.1
    windowSize = 50
    weight = np.arange(stepWeight, stepWeight*(windowSize+1), stepWeight)
    
    #weight = np.power(weight, 2)

    ### Object Declaration ###
    Vib = VibData()
    Nav = NavData()
    NV = NavdataVib()

    # List description
    #print('List description:', Nav.listDescriptionTimestamp()[1:])

    # Load navigation data by timestamp
    """ desc_id_nav = Nav.storeData(Nav.getByDescription(description=queryDescription, landed=True), 'test by description')

    # Load vibration data by same timestamp range as navidation data
    tstart, tstop = Nav.getTimestampRangeByDescription(description=queryDescription, landed=True)
    print('tstart {} tstop {}'.format(tstart, tstop))
    desc_id_vib = Vib.storeData(Vib.getBetweenTimestamp(tstart, tstop), 'test between timestamp')
    #Vib.plotDataMultiMovingRMS(Vib.getTimestampArray(desc_id_vib), Vib.getVibArray(desc_id_vib), 10)

    # Plotting
    # Raw
    label = ['New Motor', 'Old Motor', 'Motor3', 'Motor4']
    #Nav.plotDataMulti(Nav.getPWMArray(desc_id_nav), label)
    label = ['New Motor - X', 'New Motor - Y', 'New Motor - Z', 'Old Motor- X', 'Old Motor- Y', 'Old Motor- Z']
    #Vib.plotDataMulti(Vib.getVibArray(desc_id_vib), label)
    #Vib.plotTimedelta(desc_id_vib)
    Nav.plotTimedelta(desc_id_nav) """

    """ Vib.plotDataMultiMovingRMS(
        time_array=Vib.getTimestampArray(desc_id_vib),
        data_array=Vib.getVibArray(desc_id_vib),
        window_size=windowSize,
        label=label
    ) """
    #Vib.plotDataMulti(Vib.getMovingRMSWeightedArray(Vib.getVibArray(desc_id_vib), windowSize, weight), label)

    """ movingTs = Vib.getMovingTimestampArray(Vib.getTimestampArray(desc_id_vib), windowSize)
    movingVibRMS = Vib.getMovingRMSWeightedArray(Vib.getVibArray(desc_id_vib), windowSize, weight)

    plt.subplot(211)
    plt.title('New Motor')
    plt.plot(movingTs, movingVibRMS[0])
    plt.plot(movingTs, movingVibRMS[1])
    plt.plot(movingTs, movingVibRMS[2])
    plt.ylim(0,16)
    plt.grid(True)

    plt.subplot(212)
    plt.title('Old Motor')
    plt.plot(movingTs, movingVibRMS[3])
    plt.plot(movingTs, movingVibRMS[4])
    plt.plot(movingTs, movingVibRMS[5])
    plt.ylim(0,16)
    plt.grid(True)

    plt.show() """

    ##### Navdatavib #####
    """ NV = NavdataVib()
    NV.getTimestampAnalysis(
        navdata=Nav.getPWMArray(desc_id_nav),
        vibdata=Vib.getVibArray(desc_id_vib)
    )
    NV.plotNavdataVibTimestamp(
        navdata=([Nav.getTimestampArray(desc_id_nav)] + Nav.getPWMArray(desc_id_nav)),
        vibdata=(
            [Vib.getMovingTimestampArray(Vib.getTimestampArray(desc_id_vib), windowSize)] 
            + Vib.getMovingRMSWeightedArray(Vib.getVibArray(desc_id_vib), windowSize, weight)
        ),
        axis=['x','y','z']
    ) """

    """ NV.plotNavdataVibAggregated(
        navdata_agg = Nav.getPWMAggregatedArray(desc_id_nav, windowSize),
        vibdata_agg = Vib.getVibAggregatedArray(desc_id_vib, windowSize),
        axis=['x','y','z'],
        time_window=windowSize
    ) """


    """ NV.plotNav(
        navdata=([Nav.getTimestampArray(desc_id_nav)] + Nav.getPWMArray(desc_id_nav)),
        vibdata=(
            [Vib.getTimestampArray(desc_id_vib)] 
            + Vib.getVibArray(desc_id_vib)
        ),
        axis=['y']
    ) """


    """ ### Timestamp sync ###
    NV.getTimestampAnalysis(
        navdata=([Nav.getTimestampArray(desc_id_nav)]),
        vibdata=([Vib.getTimestampFilledArray(desc_id_vib)])
    ) """

    # Get timestamp range
    tstart, tstop = Nav.getTimestampRangeByDescription(description=queryDescription, landed=True)

    # Store data to active
    navId = Nav.storeData(Nav.getByDescription(description=queryDescription, landed=True), 'test by description')
    vibId = Vib.storeData(Vib.getBetweenTimestamp(tstart, tstop), 'test between timestamp')

    navMultiArray = Nav.getMultiFeatureArray(navId)
    vibMultiArray = Vib.getMultiFeatureArray(vibId)

    print('navmulti', len(navMultiArray))
    print('vibmulti', len(vibMultiArray))

    combinedMultiArray = NV.combineDataMultiFeature(
        navdata=navMultiArray,
        vibdata=vibMultiArray
    )

    # Plot sample
    plt.plot(list(range(len(combinedMultiArray))), [data['pwm'][0] for data in combinedMultiArray])
    plt.plot(list(range(len(combinedMultiArray))), [data['mpu1'][0] for data in combinedMultiArray])
    plt.grid(True)
    plt.show()


    print('combmulti', len(combinedMultiArray))