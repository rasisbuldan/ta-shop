from navdataProc import NavData, FlightData
import pymongo
import numpy as np
import matplotlib.pyplot as plt


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
        print('Connection to test-db.acceldatas successful with', self.numOfDocuments(), 'data')


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
        return active_id


    def getAllDocuments(self):
        ''' 
            WARNING!!! Variable saved in main memory
        '''
        return list(self.vibCollection.find({}))


    def getBetweenTimestamp(self, lower, upper):
        return list(self.vibCollection.find({
            "timestamp": { "$gte": lower, "$lte": upper }
        }))


    ##### Active data processing #####
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
            print('Getting PWM [{}/{}] ({:.1f})%'.format(objProcessed, objArrayLen, 100*objProcessed/objArrayLen), end='\r')
            vib = getVib(obj)
            
            # Motor Number
            for i in range(6):
                vibArray[i].append(vib[i])

        return vibArray

    
    def getVibRMSArray(self, active_id):
        def getRMS(arr):
            return np.sqrt(np.mean(np.square(arr)))

        vibArray = self.getVibArray(active_id)
        vibRMSArray = [[],[],[],[],[],[]]
        batchSize = 50

        for batchNum in range(0, len(vibArray[0])//batchSize):
            for i in range(6):
                vibRMSArray[i].append(getRMS(vibArray[i][(batchSize*batchNum):(batchSize*(batchNum+1))]))
        
        return vibRMSArray


    
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
        print('Connection to test-db.navdatas successful with', self.numOfDocuments(), 'data')
    

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
        return active_id

    
    def getAllDocuments(self):
        ''' 
            WARNING!!! Variable saved in main memory
        '''
        return list(self.vibCollection.find({}))


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

        plt.show()
        
    
##### Driver #####
if __name__ == '__main__':
    Vib = VibData()
    Nav = NavData()

    #print(Vib.countBetweenTimestamp(1595914928926, 1595914993296))
    #print(Nav.countBetweenTimestamp(1595914928926, 1595914993296))
    print(len(Nav.getByDescription(description="hover30s_3.json")))
    print(len(Nav.getByDescription(description="hover30s_3.json", landed=False)))

    # Get by timestamp
    #ts_id = Nav.storeData(Nav.getBetweenTimestamp(1595914928926, 1595914993296), 'test between timestamp')
    #Nav.plotDataMulti(Nav.getPWMArray(ts_id))

    # Get by description
    #desc_id_1 = Nav.storeData(Nav.getByDescription(description="hover30s_3.json", landed=False), 'test by description')
    #Nav.plotDataMulti(Nav.getPWMArray(desc_id_1))
    #desc_id_2 = Nav.storeData(Nav.getByDescription(description="hover30s_3.json", landed=True), 'test by description')
    #Nav.plotDataMulti(Nav.getPWMArray(desc_id_2))

    # Get timestamp range by description
    print(Nav.getTimestampRangeByDescription(description="hover30s_3.json", landed=False))
    print(Nav.getTimestampRangeByDescription(description="hover30s_3.json", landed=True))

    tstart, tstop = Nav.getTimestampRangeByDescription(description="hover30s_3.json", landed=True)
    desc_vib_id_1 = Vib.storeData(Vib.getBetweenTimestamp(tstart, tstop), 'test between timestamp')
    Vib.plotDataMulti(Vib.getVibRMSArray(desc_vib_id_1))
