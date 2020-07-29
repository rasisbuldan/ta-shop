from navdataProc import NavData, FlightData
import pymongo
import numpy as np
import matplotlib.pyplot as plt
import time


class VibData:
    def __init__(self):
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

        """ Connect to database """
        self.initializeConnectionDB()
        self.activeDataArray = []


    def initializeConnectionDB(self, host='localhost', port=27017):
        self.clientDB = pymongo.MongoClient(host, port)

        self.vibCollection = self.clientDB['test-db']['acceldatas']
        print('Connection to test-db.acceldatas successful with', self.numOfDocuments(), 'documents')


    ##### Analytics #####
    def numOfDocuments(self):
        return self.vibCollection.count_documents({})


    def countBetweenTimestamp(self, lower, upper):
        return len(self.getBetweenTimestamp(lower, upper))


    def timestampArray(self, obj_array):
        return [obj['timestamp'] for obj in obj_array]


    def xArray(self, obj_array, mpu):
        return [obj[mpu]['x'] for obj in obj_array]


    def yArray(self, obj_array, mpu):
        return [obj[mpu]['y'] for obj in obj_array]


    def zArray(self, obj_array, mpu):
        return [obj[mpu]['z'] for obj in obj_array]


    ##### Query #####
    def storeData(self, data_array, description):
        self.activeDataArray.append({
            'desc': description,
            'data': data_array
        })

        active_id = len(self.activeDataArray) - 1

        print('Store vibration data with description "{}" success with id:{}'.format(description, active_id))
        return active_id


    def getAllDocuments(self):
        '''
            WARNING!!! Variable saved in main memory
        '''
        return list(self.vibCollection.find({}))


    def getBetweenTimestamp(self, lower, upper):
        print('[lower] {} [upper] {}'.format(lower,upper))
        return list(self.vibCollection.find({
            "timestamp": { "$gte": lower, "$lte": upper }
        }))


    ##### Active data processing #####
    ### Timestamp Array
    def getTimestampArray(self, active_id):
        timestampArray = [obj['timestamp'] for obj in self.activeDataArray[active_id]['data']]
        return timestampArray

    
    def getMovingRMSTimestampArray(self, time_array, window_size):
        vibMovingRMSTimeArray = []

        for i in range(0, len(time_array)-window_size, window_size):
            vibMovingRMSTimeArray.append(time_array[i+window_size])

        return vibMovingRMSTimeArray

    
    def getMovingTimestampArray(self, time_array, window_size):
        return time_array[window_size : (len(time_array)-window_size)]


    ### Data Array
    def getVibArray(self, active_id):
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
            print('Getting Vibration [{}/{}] ({:.1f})%'.format(objProcessed, objArrayLen, 100*objProcessed/objArrayLen), end='\r')
            vib = getVib(obj)

            # Motor Number
            for i in range(6):
                vibArray[i].append(vib[i])
        print('')
        return vibArray


    def getVibRMSArray(self, active_id):
        def getRMS(arr):
            return np.sqrt(np.mean(np.square(arr)))

        vibArray = self.getVibArray(active_id)
        vibRMSArray = [[],[],[],[],[],[]]
        batchSize = 50

        for batchNum in range(0, len(vibArray[0])//batchSize):
            print('Getting Vibration RMS [{}/{}] ({:.1f})%'.format(batchNum+1, len(vibArray[0])//batchSize, 100*(batchNum+1)/(len(vibArray[0])//batchSize)), end='\r')
            for i in range(6):
                vibRMSArray[i].append(getRMS(vibArray[i][(batchSize*batchNum):(batchSize*(batchNum+1))]))

        return vibRMSArray


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
        rmsTimestampArray = self.getMovingRMSTimestampArray(time_array, window_size)
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



class NavData:
    def __init__(self):
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
        self.initializeConnectionDB()
        self.activeDataArray = []


    def initializeConnectionDB(self, host='localhost', port=27017):
        self.clientDB = pymongo.MongoClient(host, port)

        self.navCollection = self.clientDB['test-db']['navdatas']
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

        print('Store navigation data with description "{}" success with id:{}'.format(description, active_id))
        return active_id


    def getAllDocuments(self):
        '''
            WARNING!!! Variable saved in main memory
        '''
        return list(self.vibCollection.find({}))


    def getBetweenTimestamp(self, lower, upper):
        print('[lower] {} [upper] {}'.format(lower,upper))
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
        print('[getTimestampArray] range: {} - {}'.format(min(timestampArray), max(timestampArray)))
        return timestampArray


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
            print('Getting PWM [{}/{}] ({:.1f})%'.format(objProcessed, objArrayLen, 100*objProcessed/objArrayLen), end='\r')
            pwm = getPWM(obj)

            # Motor Number
            for i in range(4):
                pwmArray[i].append(pwm[i])

        return pwmArray


    ##### Plotting #####
    def plotDataMulti(self, data_array, label=None):
        rowCount = len(data_array)
        xData = list(range(len(data_array[0])))

        for r in range(rowCount):
            plt.subplot(100*rowCount + 11 + r)
            plt.plot(xData, data_array[r])
            plt.grid(True)

            if label == None:
                plt.title('Data Array id:{}'.format(r))
            else:
                plt.title(label[r])

        plt.tight_layout(0,0,0,(0,0,1,1))
        plt.show()


class NavdataVib:
    def __init__(self):
        pass

    def loadNavdataVibTimestamp(self):
        pass

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
        tsMin = min(tsComb)
        tsMax = max(tsComb)

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
        p1_nav_pwm1, = ax1_nav_pwm1.plot(timestampNav, navdata[1], label='PWM1', color='C0', linewidth=2)
        ax1_nav_pwm1.set_ylabel('PWM Motor 1', color='C0')
        ax1_nav_pwm1.set_ylim([-10,310])
        ax1_nav_pwm1.set_xlim([tsMin, tsMax])
        ax1_nav_pwm1.set_xticks(np.arange(tsMin, tsMax, 1000))

        # Vib - X
        if 'x' in axis:
            p1_vib_x, = ax1_vib_x.plot(timestampVib, vibdata[1], label='Vib1-X', color='C1', linewidth=0.5)
            ax1_vib_x.yaxis.tick_right()
            ax1_vib_x.set_ylim([-16,16])
            ax1_vib_x.set_xlim([tsMin, tsMax])

        # Vib - Y
        if 'y' in axis:
            p1_vib_y, = ax1_vib_y.plot(timestampVib, vibdata[2], label='Vib1-Y', color='C2', linewidth=0.5)
            ax1_vib_y.yaxis.tick_right()
            ax1_vib_y.set_ylim([-16,16])
            ax1_vib_y.set_xlim([tsMin, tsMax])

        # Vib - Z
        if 'z' in axis:
            p1_vib_z, = ax1_vib_z.plot(timestampVib, vibdata[3], label='Vib1-Z', color='C3', linewidth=0.5)
            ax1_vib_z.yaxis.tick_right()
            ax1_vib_z.set_ylim([-16,16])
            ax1_vib_z.set_xlim([tsMin, tsMax])


        ### Subplot 2 ###

        # PWM
        p2_nav_pwm2, = ax2_nav_pwm2.plot(timestampNav, navdata[2], label='PWM2', color='C0', linewidth=2)
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
        plt.grid(True)
        plt.show()



##### Driver #####
if __name__ == '__main__':
    ### Parameter ###
    queryDescription = "jul29_2_hover30s_1.json"
    plotVibAxis = ['x','y','z']
    stepWeight = 0.1
    windowSize = 10
    weight = np.arange(stepWeight, stepWeight*(windowSize+1), stepWeight)

    ### Object Declaration ###
    Vib = VibData()
    Nav = NavData()

    # Load navigation data by timestamp
    desc_id_nav = Nav.storeData(Nav.getByDescription(description=queryDescription, landed=True), 'test by description')

    # Load vibration data by same timestamp range as navidation data
    tstart, tstop = Nav.getTimestampRangeByDescription(description=queryDescription, landed=True)
    print('tstart {} tstop {}'.format(tstart, tstop))
    desc_id_vib = Vib.storeData(Vib.getBetweenTimestamp(tstart, tstop), 'test between timestamp')
    #Vib.plotDataMultiMovingRMS(Vib.getTimestampArray(desc_id_vib), Vib.getVibArray(desc_id_vib), 10)

    # Plotting
    """ label = ['Motor1', 'Motor2', 'Motor3', 'Motor4']
    Nav.plotDataMulti(Nav.getPWMArray(desc_id_2), label)
    label = ['mpu1 - X', 'mpu1- Y', 'mpu1- Z', 'mpu2- X', 'mpu2- Y', 'mpu2- Z']
    Vib.plotDataMulti(Vib.getVibArray(desc_vib_id_1), label) """

    ##### Navdatavib #####
    NV = NavdataVib()
    NV.plotNavdataVibTimestamp(
        navdata=([Nav.getTimestampArray(desc_id_nav)] + Nav.getPWMArray(desc_id_nav)),
        vibdata=(
            [Vib.getMovingTimestampArray(Vib.getTimestampArray(desc_id_vib), windowSize)] 
            + Vib.getMovingRMSWeightedArray(Vib.getVibArray(desc_id_vib), windowSize, weight)
        ),
        axis=['y']
    )