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
import sys


class VibData:
    def __init__(self, verbose=False, offset=0):
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
        self.offset = offset

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
    def storeData(self, data_array, description, trim=None):
        '''
            Store queried document(s) result to class attribute (main memory)
            for further processing without db query
            
            return: activeDataArray index (id) of queried result
        '''

        if trim != None:
            data_array = data_array[trim[0]:trim[1]]

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



        return list(self.vibCollection.find({
            "timestamp": { 
                "$gte": lower - self.offset,
                "$lte": upper - self.offset
            }
        }))


    ##### Active data processing #####
    '''
        Further processing of stored queried document(s) of method storeData().
        active_id required as arguments.
    '''

    ### Timestamp Array
    def getTimestampArray(self, active_id):
        ''' Get timestamp array of stored documents with index *active_id* '''
    

        timestampArray = [int(obj['timestamp'])+self.offset for obj in self.activeDataArray[active_id]['data']]
        return timestampArray


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
        rowCount = 3
        xData = list(range(len(data_array[0])))

        for r in range(3):
            plt.subplot(100*rowCount + 11 + r)
            plt.plot(xData, data_array[r])
            plt.ylim(0,16)
            plt.grid(True)

            if title == None:
                plt.title('Data Array id:{}'.format(r))
            else:
                plt.title(title[r])

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



class StaticPWMData:
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

        self.pwmCollection = self.clientDB['test-db']['staticpwm2s']
        if self.verbose:
            print('Connection to test-db.staticpwms successful with', self.numOfDocuments(), 'documents')


    ##### General #####
    def numOfDocuments(self):
        ''' Number of all documents in collection '''
        return self.vibCollection.count_documents({})


    def countBetweenTimestamp(self, lower, upper):
        ''' Number of documents between specified timestamp (boundary included) '''
        return len(self.getBetweenTimestamp(lower, upper))


    def timestampArray(self, obj_array):
        ''' Timestamp array queried document list '''
        return [obj['timestamp'] for obj in obj_array]


    def pwmArray(self, obj_array):
        ''' PWM array queried document list '''
        return [obj['pwm'] for obj in obj_array]


    ##### Query #####
    def storeData(self, data_array, description, trim=None):
        '''
            Store queried document(s) result to class attribute (main memory)
            for further processing without db query
            
            return: activeDataArray index (id) of queried result
        '''

        if trim != None:
            data_array = data_array[trim[0]:trim[1]]

        self.activeDataArray.append({
            'desc': description,
            'data': data_array
        })

        active_id = len(self.activeDataArray) - 1

        if self.verbose:
            print('Store static pwm data with description "{}" success with id:{}'.format(description, active_id))
        return active_id


    def listDescription(self):
        return list(self.pwmCollection.distinct('description'))

    
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

        return list(self.pwmCollection.find({}))

    
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


    def getBetweenTimestamp(self, lower, upper):
        ''' Get document(s) as list of object with specified timestamp boundary '''

        return list(self.pwmCollection.find({
            "timestamp": { "$gte": lower, "$lte": upper }
        }))

    
        def getBetweenTimestamp(self, lower, upper):
          return list(self.pwmCollection.find({
            "timestamp": { "$gte": lower, "$lte": upper }
        }))


    def getByDescription(self, description, landed=True):
        if landed:
            return list(self.pwmCollection.find({
                "$and": [
                    {
                        "description": { "$eq": description }
                    }
                ]
            }))

        else:
            return list(self.pwmCollection.find({
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
            tsArray = [n['timestamp'] for n in list(self.pwmCollection.find({
                "description": { "$eq": description }
            },
            {
                "_id": 0,
                "timestamp": 1
            }))]

        else:
            tsArray = [n['timestamp'] for n in list(self.pwmCollection.find({
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
    '''
        Further processing of stored queried document(s) of method storeData().
        active_id required as arguments.
    '''

    ### Timestamp Array
    def getTimestampArray(self, active_id):
        ''' Get timestamp array of stored documents with index *active_id* '''

        timestampArray = [int(obj['timestamp']) for obj in self.activeDataArray[active_id]['data']]
        return timestampArray

    
    def getMovingTimestampArray(self, time_array, window_size):
        '''
            Get Moving Timestamp array of stored documents (used for Moving Average / Moving RMS).
            *time_array* passed from getTimestampArray() method. 
        '''
        
        return time_array[window_size : (len(time_array)-window_size)]


    ### Data Array
    def getPWMArray(self, active_id):
        '''
            Get PWM array of specified *active_id*
        '''

        pwmArray = []

        objArray = self.activeDataArray[active_id]['data']
        objProcessed = 0
        objArrayLen = len(objArray)
        for obj in objArray:
            objProcessed += 1
            if self.verbose:
                print('Getting PWM [{}/{}] ({:.1f})%'.format(objProcessed, objArrayLen, 100*objProcessed/objArrayLen), end='\r')

            pwmArray.append(obj['pwm'])

        return pwmArray


    def getPWMAggregatedArray(self, active_id, time_window):
        '''
            Aggregated (RMS) PWM array with window *time_window*
        '''

        # Extract data
        activeData = self.activeDataArray[active_id]['data']
        tsMin = activeData[0]['timestamp']
        tsMax = activeData[-1]['timestamp']

        tsPrev = tsMin
        vibBuf = []
        vibAgg = []

        for i in range(len(activeData)):
            cursor = activeData[i]
            tsCursor = cursor['timestamp']
            
            # Aggregate if delta time exceed time_window
            if (tsCursor > (tsPrev + time_window)) or (i+1) == len(activeData):
                vibAgg.append([
                    int(tsCursor), 
                    np.sqrt(np.mean(np.square(vibBuf)))
                ])
                tsPrev = tsCursor
                vibBuf = []

            # Add data point to buffer
            else:
                vibBuf.append(cursor['pwm'])

        return vibAgg

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


    def getMovingRMSWeightedArray(self, data_array, window_size, weight):
        ''' 
            Moving RMS Weighted with linear weight factor
        '''

        pwmMovingWMSWeightedArray = []
        for pwmdata in data_array:
            dataArraySquared = np.square(pwmdata)
            pwmMovingWMSWeighted = []
            for d in range(window_size, len(pwmdata)-window_size):
                sumWeighted = 0
                for i in range(window_size):
                    sumWeighted += weight[i] * dataArraySquared[d-window_size+i]
                
                pwmMovingWMSWeighted.append(np.sqrt(sumWeighted/np.sum(weight)))
            pwmMovingWMSWeightedArray.append(pwmMovingWMSWeighted)

        return pwmMovingWMSWeightedArray


    ##### Plotting #####
    def plotData(self, data_array, title=None):
        xData = list(range(len(data_array)))

        plt.plot(xData, data_array)
        plt.ylim(0,1000)
        plt.grid(True)

        plt.title('PWM Data Array')

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



class StaticVibData:
    def __init__(self, verbose=False):
        self.verbose = verbose


    def combineArray(self, pwmdata, vibdata):
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

    
    def combineAggArray(self, pwmdata, vibdata, time_window):
        '''
            Combine based on smallest timestamp delta
            (Currently pwmdata to match vibdata timestamp)

            Input format:
            pwmdata = [[timestamp,...], [pwm,...]]
            vibdata = [[timestamp,...], [[vib.x,...],...]]

            Return format:
            [timestamp, pwm, vibx, viby, vibz]
        '''

        def rms(array):
            return np.sqrt(np.mean(np.square(array)))

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
                
                # Append inrange data into new pwm array
                newPWMArray.append(pwmArray[i])


        # Aggregate traversal (time_window by number of samples)
        aggArray = []
        for i in range(0, len(tsVib)-time_window, time_window):
            aggArray.append([
                tsVib[i],
                rms(newPWMArray[i:i+time_window]),
                [
                    rms(vibArray[0][i:i+time_window]),
                    rms(vibArray[1][i:i+time_window]),
                    rms(vibArray[2][i:i+time_window])
                ]
            ])
        
        return aggArray



        #combinedArray = [tsVib, newPWMArray, *vibArray]
        return combinedArray
        
    
    def combineAggregatedArray(self, pwmdata_agg, vibdata_agg, time_window=None):
        '''
            If aggregated array length not equal, required arg *time_window*: 
                same as previous value with specified timeWindow
        '''

        # Get aggregated vibration and pwm data
        tsArrayPWMdata = [n[0] for n in pwmdata_agg]
        tsArrayVibdata = [v[0] for v in vibdata_agg]
        nPWMdata = len(tsArrayPWMdata)
        nVibdata = len(tsArrayVibdata)

        # Debug
        #print([tsArrayNavdata[i] - tsArrayNavdata[i-1] for i in range(1,nNavdata)])
        
        # Abort if aggregated array length not equal
        if nPWMdata != nVibdata:
            if self.verbose:
                print('Array dimension not equal [nav: {} | vib: {}], processing...'.format(nPWMdata, nVibdata))

            # Plot timestamp (debug)
            """ plt.plot(list(range(nNavdata)), tsArrayNavdata, color='C0')
            plt.plot(list(range(nVibdata)), tsArrayVibdata, color='C1')
            plt.show() """

            # Value insertion to PWM data
            if nPWMdata < nVibdata:
                tdPWMdata = [(tsArrayPWMdata[i] - tsArrayPWMdata[i-1]) for i in range(1,nPWMdata)]

                # Get timedelta index bigger than timeWindow (>10%)
                tdOverrange = [(idx,td) for (idx,td) in zip(list(range(len(tdPWMdata))), tdPWMdata) if td >= (time_window * 1.2)]

                # Before state (debug)
                #print('Before:', *navdata_agg, sep='\n')
                #print('Timedelta overrange: ', tdOverrange)

                for td in tdOverrange[::-1]:
                    tdCount = td[1] // time_window
                    tdIdx = td[0]
                    prevPWMdata = pwmdata_agg[tdIdx]

                    # Insert with value 
                    for i in range(tdCount-1):
                        # Copy previous navdata
                        insertedPWMdata = prevPWMdata.copy()

                        # Update timestamp to match time_window step value
                        insertedPWMdata[0] = prevPWMdata[0] + ((tdCount-1-i)*time_window)

                        # Insert
                        pwmdata_agg.insert(tdIdx+1, insertedPWMdata)


                # Validation (debug)
                #tsArrayNavdata = [n[0] for n in navdata_agg]
                #tdNavdata = [(tsArrayNavdata[i] - tsArrayNavdata[i-1]) for i in range(1,nNavdata)]
                #print('After:', *navdata_agg, sep='\n')
                #print(tdNavdata)
                #plt.plot(list(range(len(tsArrayNavdata))), tsArrayNavdata)
                #plt.show()

        # Aggregated array length equal
        aggArray = []

        for i in range(nPWMdata):
            try:
                aggArray.append([
                    pwmdata_agg[i][0],
                    pwmdata_agg[i][1],
                    vibdata_agg[i][1]
                ])
            
            except IndexError:
                try:
                    print('vibdata_agg[i]')
                    print(vibdata_agg[i])

                except IndexError:
                    print('vibdata_agg')
                    print(vibdata_agg)

        return aggArray

    
    def getTimestampAnalysis(self, pwmdata, vibdata):
        # Extract timestamp
        timestampPWM = pwmdata[0]
        timestampVib = vibdata[0]

        # Get number of data
        tsPWMLen = len(timestampPWM)
        tsVibLen = len(timestampVib)

        # Get boundary value
        tsPWMMin = min(timestampPWM)
        tsPWMMax = max(timestampPWM)
        tsVibMin = min(timestampVib)
        tsVibMax = max(timestampVib)

        # Get average
        tsPWMSamplingAvg = (tsPWMMax - tsPWMMin) / tsPWMLen
        tsVibSamplingAvg = (tsVibMax - tsVibMin) / tsVibLen

        # Get delta array
        tsPWMDelta = [0] + [(timestampPWM[i] - timestampPWM[i-1]) for i in (range(1,len(timestampPWM)))]
        tsVibDelta = [0] + [(timestampVib[i] - timestampVib[i-1]) for i in (range(1,len(timestampVib)))]

        print('[runtime] pwm: {} | {}'.format((tsPWMMax-tsPWMMin)/1000, (tsVibMax-tsVibMin)/1000))
        print('[count] pwm: {} | vib: {}'.format(tsPWMLen, tsVibLen))
        print('[bound] pwm: {} - {} | vib: {} - {}'.format(tsPWMMin, tsPWMMax, tsVibMin, tsVibMax))
        print('[avg] pwm: {} | vib: {}'.format(tsPWMSamplingAvg, tsVibSamplingAvg))

        # Plot timestamp delta
        plt.scatter(list(range(len(tsPWMDelta))), tsPWMDelta, s=5, color='C0')
        plt.plot(list(range(len(tsPWMDelta))), [tsPWMSamplingAvg for i in range(len(tsPWMDelta))], color='C0')
        plt.title('Timestamp Delta - Static PWM Data')
        plt.ylim(0,50)
        plt.grid(True)
        plt.show()
        
        plt.scatter(list(range(len(tsVibDelta))), tsVibDelta, s=5, color='C0')
        plt.plot(list(range(len(tsVibDelta))), [tsVibSamplingAvg for i in range(len(tsVibDelta))], color='C1')
        plt.title('Timestamp Delta - Vibration Data')
        plt.grid(True)
        plt.show()


    def plotPWMVibTimestamp(self, pwmdata, vibdata, axis=None):
        '''
            Fixed for 1 motors - each 3 accelerometer axis
            - ts_range = tuple(min_ts, max_ts)
            - pwmdata = [[timestamp],pwm]
            - vibdata = [[timestamp],[mpu.x],[mpu.y],[mpu.z]]
        '''
        if axis == None:
            axis = ['x','y','z']

        pwmTrim = (0,40800)
        vibTrim = (0,163000)

        # Timestamp
        timestampPWM = pwmdata[0]
        print(len(timestampPWM))
        timestampVib = vibdata[0]
        print(len(timestampVib))
        tsComb = timestampPWM + timestampVib
        tsMin = (min(tsComb)//1000) * 1000
        tsMax = (max(tsComb)//1000) * 1000

        ##### Setup plot #####
        fig = plt.figure()

        # Subplot 1
        if 'x' in axis:
            ax_vib_x = fig.add_subplot(111, label='Vib Motor - X', frame_on=True)
        if 'y' in axis:
            ax_vib_y = fig.add_subplot(111, label='Vib Motor - Y', frame_on=False)
        if 'z' in axis:
            ax_vib_z = fig.add_subplot(111, label='Vib Motor - Z', frame_on=False)
        ax_pwm = fig.add_subplot(111, label='PWM Motor', frame_on=False)


        ##### Plot #####
        ### Subplot 1 ###

        # PWM
        p_pwm, = ax_pwm.plot(timestampPWM[pwmTrim[0]:pwmTrim[1]], pwmdata[1:][pwmTrim[0]:pwmTrim[1]], label='PWM1', color='C0', linewidth=1.5)
        ax_pwm.set_ylabel('PWM Motor 1', color='C0')
        ax_pwm.set_ylim([0,1000])
        ax_pwm.set_xlim([tsMin, tsMax])
        #ax1_nav_pwm1.set_xticks(np.arange(tsMin, tsMax, 2500))

        # Vib - X
        if 'x' in axis:
            p1_vib_x, = ax_vib_x.plot(timestampVib[vibTrim[0]:vibTrim[1]], vibdata[1][vibTrim[0]:vibTrim[1]], label='Vib1-X', color='C1', linewidth=0.5)
            ax_vib_x.yaxis.tick_right()
            ax_vib_x.set_ylim([-16,16])
            ax_vib_x.set_xlim([tsMin, tsMax])
            #ax1_vib_x.set_xticks(np.arange(tsMin, tsMax, 2500))

        # Vib - Y
        if 'y' in axis:
            p1_vib_y, = ax_vib_y.plot(timestampVib[vibTrim[0]:vibTrim[1]], vibdata[2][vibTrim[0]:vibTrim[1]], label='Vib1-Y', color='C2', linewidth=0.5)
            ax_vib_y.yaxis.tick_right()
            ax_vib_y.set_ylim([-16,16])
            ax_vib_y.set_xlim([tsMin, tsMax])
            #ax1_vib_y.set_xticks(np.arange(tsMin, tsMax, 2500))

        # Vib - Z
        if 'z' in axis:
            p1_vib_z, = ax_vib_z.plot(timestampVib[vibTrim[0]:vibTrim[1]], vibdata[3][vibTrim[0]:vibTrim[1]], label='Vib1-Z', color='C3', linewidth=0.5)
            ax_vib_z.yaxis.tick_right()
            ax_vib_z.set_ylim([-16,16])
            ax_vib_z.set_xlim([tsMin, tsMax])
            #ax1_vib_z.set_xticks(np.arange(tsMin, tsMax, 2500))

        #plt.grid(True)
        plt.show()



    def plotPWMVibAggregated(self, pwmdata_agg, vibdata_agg, axis=None, time_window=None):
        aggArray = self.combineAggregatedArray(pwmdata_agg, vibdata_agg, time_window)

        # Default axis selection
        if axis == None:
            axis = ['x','y','z']

        # Create list from sequence
        timestampArray = [a[0] for a in aggArray]
        pwmArray = [a[1] for a in aggArray]
        mpuxArray = [a[2][0] for a in aggArray]
        mpuyArray = [a[2][1] for a in aggArray]
        mpuzArray = [a[2][2] for a in aggArray]

        tsMin = (min(timestampArray)//1000) * 1000
        tsMax = (max(timestampArray)//1000) * 1000

        ##### Setup plot #####
        fig = plt.figure()

        # Subplot 1
        if 'x' in axis:
            ax_vib_x = fig.add_subplot(111, label='Vib Motor Static - X', frame_on=True)
        if 'y' in axis:
            ax_vib_y = fig.add_subplot(111, label='Vib Motor Static - Y', frame_on=False)
        if 'z' in axis:
            ax_vib_z = fig.add_subplot(111, label='Vib Motor Static - Z', frame_on=False)
        ax_pwm = fig.add_subplot(111, label='PWM Motor Static', frame_on=False)

        ##### Plot #####
        ### Subplot 1 ###

        # PWM
        p_nav_pwm, = ax_pwm.plot(timestampArray, pwmArray, label='PWM', color='C0', linewidth=2)
        ax_pwm.set_ylabel('PWM Motor 1', color='C0')
        ax_pwm.set_ylim([0,1000])
        ax_pwm.set_xlim([tsMin, tsMax])
        ax_pwm.set_xticks(np.arange(tsMin, tsMax, 20000))

        # Vib - X
        if 'x' in axis:
            p_vib_x, = ax_vib_x.plot(timestampArray, mpuxArray, label='Vib-X', color='C1', linewidth=0.5)
            ax_vib_x.yaxis.tick_right()
            ax_vib_x.set_ylim([0,16])
            ax_vib_x.set_xlim([tsMin, tsMax])
            ax_vib_x.set_xticks(np.arange(tsMin, tsMax, 20000))

        # Vib - Y
        if 'y' in axis:
            p_vib_y, = ax_vib_y.plot(timestampArray, mpuyArray, label='Vib-Y', color='C2', linewidth=0.5)
            ax_vib_y.yaxis.tick_right()
            ax_vib_y.set_ylim([0,16])
            ax_vib_y.set_xlim([tsMin, tsMax])
            ax_vib_y.set_xticks(np.arange(tsMin, tsMax, 20000))

        # Vib - Z
        if 'z' in axis:
            p_vib_z, = ax_vib_z.plot(timestampArray, mpuzArray, label='Vib-Z', color='C3', linewidth=0.5)
            ax_vib_z.yaxis.tick_right()
            ax_vib_z.set_ylim([0,16])
            ax_vib_z.set_xlim([tsMin, tsMax])
            ax_vib_z.set_xticks(np.arange(tsMin, tsMax, 20000))


        ### Config ###
        plt.grid(True)
        plt.show()


##### Driver #####
if __name__ == '__main__':
    ### Parameter ###
    queryDescription = 'aug5_step_clean_0_300_900_3000_1_6'
    plotVibAxis = ['x','y','z']
    stepWeight = 0.1
    windowSize = 10
    weight = np.arange(stepWeight, stepWeight*(windowSize+1), stepWeight)

    ### Object Declaration ###
    Vib = VibData(offset=-2375)
    Static = StaticPWMData()

    # List description
    """ print('List description:', Nav.listDescriptionTimestamp()[1:]) """

    # Load navigation data by timestamp
    desc_id_pwm = Static.storeData(
                    Static.getByDescription(
                        description=queryDescription,
                        landed=True),
                    'test by description'
    )

    #Static.plotData(data_array=Static.getPWMArray(desc_id_pwm))

    # Load vibration data by same timestamp range as navidation data
    tstart, tstop = Static.getTimestampRangeByDescription(
                        description=queryDescription,
                        landed=True)
    print('tstart {} tstop {}'.format(tstart, tstop))
    desc_id_vib = Vib.storeData(Vib.getBetweenTimestamp(tstart, tstop), 'test between timestamp')
    #print(Vib.getTimestampArray(desc_id_vib))
    """ Vib.plotDataMultiMovingRMS(Vib.getTimestampArray(desc_id_vib), Vib.getVibArray(desc_id_vib), 10) """

    SV = StaticVibData()
    arrayCombined = SV.combineArray(
        pwmdata=[Static.getTimestampArray(desc_id_pwm)] + [Static.getPWMArray(desc_id_pwm)],
        vibdata=[Vib.getTimestampArray(desc_id_vib)] + [Vib.getVibArray(desc_id_vib)]
    )

    SV.plotPWMVibTimestamp(
        pwmdata=[arrayCombined[0], *arrayCombined[1]],
        vibdata=[arrayCombined[0], *arrayCombined[2:]]
    )

    """ plt.plot(arrayCombined[0], arrayCombined[1])
    plt.grid(True)
    plt.show() """

    """ SV.getTimestampAnalysis(
        pwmdata=[Static.getTimestampArray(desc_id_pwm)],
        vibdata=[Vib.getTimestampArray(desc_id_vib)]
    ) """

    """ SV.plotPWMVibTimestamp(
        pwmdata=[Static.getTimestampArray(desc_id_pwm)] + Static.getPWMArray(desc_id_pwm),
        vibdata=[Vib.getTimestampArray(desc_id_vib)] + Vib.getVibArray(desc_id_vib)
    ) """

    """ SV.plotPWMVibAggregated(
        pwmdata_agg=Static.getPWMAggregatedArray(desc_id_pwm, windowSize),
        vibdata_agg=Vib.getVibAggregatedArray(desc_id_vib, windowSize),
        axis=['x','y','z'],
        time_window=windowSize
    ) """