import pymongo
import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt
import scipy.stats

class FlightData:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.initializeConnectionDB()


    def initializeConnectionDB(self, host='localhost', port=27017):
        ''' Connect to database with default localhost '''
        self.clientDB = pymongo.MongoClient(host, port)

        self.flightCollection = self.clientDB['test-db']['flightdatas']
        if self.verbose:
            print('Connection to test-db.flightdatas successful with', self.numOfDocuments(), 'documents')


    def listDescription(self):
        ''' Unique description list in collection '''
        return list(self.flightCollection.distinct('description'))


    def numOfDocuments(self):
        ''' Number of all documents in collection '''
        return self.flightCollection.count_documents({})


    def numOfDocumentsByDesc(self, desc):
        ''' Number of documents by description '''
        return self.flightCollection.count_documents({
            "description": {"$eq" : desc}
        })


    def getFlightTimeByDesc(self, desc, landed=True):
        if landed:
            tsArray = [n['timestamp'] for n in list(self.flightCollection.find({
                "description": { "$eq": desc }
            },
            {
                "_id": 0,
                "timestamp": 1
            }))]

        else:
            tsArray = [n['timestamp'] for n in list(self.flightCollection.find({
                "$and" : [
                    {
                        "description": { "$eq": desc }
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

        return max(tsArray) - min(tsArray)

    
    def getFlightData(self, description, landed=True):
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

    
    def aggregate(self, flight_data, time_window):
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


        flightData = flight_data.copy()

        # Loop over timestamp and aggregate if buffer time exceed time window
        timePrev = flightData[0]['timestamp']
        cursorIdx = 0

        # Aggregate buffer to store combined data point
        aggBuf = []
        aggDataArray = []

        while cursorIdx < len(flightData):
            # Calculate time delta
            timeCursor = flightData[cursorIdx]['timestamp']
            
            # Calculate feature
            if (timeCursor - timePrev) >= time_window:

                aggBuf.append(flightData[cursorIdx])

                # Empty dictionary format
                aggData = {
                    'timestamp': timeCursor,
                    'loadNum': 0,
                    'pwm': 0,
                    'orientation': [0,0,0],
                    'rms': [0,0,0],
                    'kurtosis': [0,0,0],
                    'skewness': [0,0,0],
                    'crest-factor': [0,0,0],
                    'peak-to-peak': [0,0,0],
                }

                # Calculate feature aggregation
                aggData['loadNum'] = np.mean([data['loadNum'] for data in aggBuf])
                aggData['pwm'] = np.mean([data['pwm'][0] for data in aggBuf])
                aggData['orientation'] = [np.mean([data['orientation'][axis] for data in aggBuf]) for axis in range(3)]
                aggData['rms'] = [rms([data['mpu1'][axis] for data in aggBuf]) for axis in range(3)]
                aggData['kurtosis'] = [kurtosis([data['mpu1'][axis] for data in aggBuf]) for axis in range(3)]
                aggData['skewness'] = [skewness([data['mpu1'][axis] for data in aggBuf]) for axis in range(3)]
                aggData['crest-factor'] = [crest_factor([data['mpu1'][axis] for data in aggBuf]) for axis in range(3)]
                aggData['peak-to-peak'] = [peak_to_peak([data['mpu1'][axis] for data in aggBuf]) for axis in range(3)]

                aggDataArray.append(aggData)
                timePrev = timeCursor

                # Reset buffer
                aggBuf = []


            # Add to buffer
            else:
                aggBuf.append(flightData[cursorIdx])

            cursorIdx += 1

        return aggDataArray

    
    def plotRawVib(self, desc, save_dir=None):
        '''
            Currently 3 axis, save only
        '''
        rawFlightArr = list(self.flightCollection.find({
            "$and": [
                {
                    "description": { "$eq": desc }
                }
            ]
        }))

        timestampData = [(rf['timestamp']/1000) for rf in rawFlightArr]

        for axis in range(3):
            rawData = [rf['mpu1'][axis] for rf in rawFlightArr]

            fig = plt.figure(figsize=(16,2.2), dpi=120)
            fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.15)

            ax1 = fig.add_subplot(111, frame_on=True)

            p_test, = ax1.plot(timestampData, rawData, 'k-', linewidth=0.5)
            ax1.tick_params(grid_alpha=0.6, grid_linewidth=0.4, labelsize=16)

            #ax1.set_xticks(list(range(0,36,2)))
            ax1.set_xticks([])
            ax1.set_xlim([0,35])
            ax1.set_yticks(list(range(-16,17,8)))
            ax1.set_ylim([-16,16])

            if axis == 0:
                ax1.set_ylabel('Vibration X (g)', fontsize=16)
            if axis == 1:
                ax1.set_ylabel('Vibration Y (g)', fontsize=16)
            if axis == 2:
                ax1.set_ylabel('Vibration Z (g)', fontsize=16)

            ax1.grid(True)

            plt.savefig('vib_nt_{}.png'.format(axis))


    def plotRawPWM(self, desc, save_dir=None):
        '''
            Currently 3 axis, save only
        '''
        rawFlightArr = list(self.flightCollection.find({
            "$and": [
                {
                    "description": { "$eq": desc }
                }
            ]
        }))

        timestampData = [(rf['timestamp']/1000) for rf in rawFlightArr]

        for axis in range(4):
            rawData = [rf['pwm'][axis] for rf in rawFlightArr]

            fig = plt.figure(figsize=(16,2.2), dpi=120)
            fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.15)

            ax1 = fig.add_subplot(111, frame_on=True)

            p_test, = ax1.plot(timestampData, rawData, 'k-', linewidth=1.2)
            ax1.tick_params(grid_alpha=0.6, grid_linewidth=0.4, labelsize=16)

            #ax1.set_xticks(list(range(0,36,2)))
            ax1.set_xticks([])
            ax1.set_xlim([0,35])
            ax1.set_yticks(list(range(0,301, 50)))
            ax1.set_ylim([0,300])
            ax1.set_ylabel('PWM', fontsize=16)
            
            ax1.grid(True)

            plt.savefig('pwm_nt_{}.png'.format(axis))

    def plotRawOrientation(self, desc, save_dir=None):
        '''
            Currently 3 axis, save only
        '''
        rawFlightArr = list(self.flightCollection.find({
            "$and": [
                {
                    "description": { "$eq": desc }
                }
            ]
        }))

        timestampData = [(rf['timestamp']/1000) for rf in rawFlightArr]

        for axis in range(3):
            rawData = [rf['orientation'][axis] for rf in rawFlightArr]

            fig = plt.figure(figsize=(16,2.2), dpi=120)
            fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.15)

            ax1 = fig.add_subplot(111, frame_on=True)

            p_test, = ax1.plot(timestampData, rawData, 'k-', linewidth=1.2)
            ax1.tick_params(grid_alpha=0.6, grid_linewidth=0.4, labelsize=16)

            #ax1.set_xticks(list(range(0,36,2)))
            ax1.set_xticks([])
            ax1.set_xlim([0,35])
            ax1.set_yticks(list(range(0,301, 50)))
            ax1.set_ylim([0,300])

            if axis == 0:
                ax1.set_ylabel('Roll (deg)', fontsize=16)
                ax1.set_yticks(list(range(-20, 21, 10)))
                ax1.set_ylim([-20,20])
            if axis == 1:
                ax1.set_ylabel('Pitch (deg)', fontsize=16)
                ax1.set_yticks(list(range(-20, 21, 10)))
                ax1.set_ylim([-20,20])
            if axis == 2:
                ax1.set_ylabel('Yaw (deg)', fontsize=16)
                ax1.set_yticks(list(range(-180, 181, 90)))
                ax1.set_ylim([-180,180])
            
            ax1.grid(True)

            plt.savefig('orientation_nt_{}.png'.format(axis))


##################
##### Driver #####
##################

if __name__ == '__main__':
    FD = FlightData()
    
    listDesc = FD.listDescription()
    #print('Description List')
    #print(*listDesc, sep='\n')
    #print('\n--------------------\n')
    
    """ includeDesc = ['aug11_6'] # , 'aug11'
    excludeDesc = ['fail', 'up_down', 'test', 'crash']

    totalNumOfDesc = 0
    totalNumOfDocs = 0
    totalFlightTime = 0
    for desc in listDesc:

        # Selected 
        if not any(fd in desc for fd in includeDesc) or any(fd in desc for fd in excludeDesc):
            continue
        else:
            totalNumOfDesc += 1
            numOfDocs = FD.numOfDocumentsByDesc(desc)
            flightTime = FD.getFlightTimeByDesc(desc)
            print('{}: {} documents in {:.2f}s'.format(desc, numOfDocs, flightTime/1000))
            totalNumOfDocs += numOfDocs
            totalFlightTime += flightTime
    
    print('Total number of documents: {} in {} attempts in {:.2f}s'.format(totalNumOfDocs, totalNumOfDesc, totalFlightTime/1000)) """

    FD.plotRawVib(
        desc='aug9_0_hover30s_5.json'
    )

    FD.plotRawPWM(
        desc='aug9_0_hover30s_5.json'
    )

    FD.plotRawOrientation(
        desc='aug9_0_hover30s_5.json'
    )