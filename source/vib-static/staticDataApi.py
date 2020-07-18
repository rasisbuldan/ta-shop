import os
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime, time
import pickle
import traceback
import sys

class Vibration:
    def __init__(self, filename):
        #print("Initializing", filename)
        filearr = np.genfromtxt(filename, delimiter=",", dtype='float')

        self.filename = filename
        self.timeDelta = filearr[:,0]
        self.timeArray = np.cumsum(self.timeDelta)
        self.fileTime = self.getTotalTime()
        self.xAccel = filearr[:,1]
        self.yAccel = filearr[:,2]
        self.zAccel = filearr[:,3]

    ## Selector ##
    def getMeasureTime(self):
        fn = self.filename.split("_")
        filetime = datetime(
            year=int('20' + fn[-7]),
            month=int(fn[-6]),
            day=int(fn[-5]),
            hour=int(fn[-4]),
            minute=int(fn[-3]),
            second=int(fn[-2])
        )

        return filetime

    def getTimeDelta(self):
        return self.timeDelta
    
    def getTotalTime(self):
        ''' Get total time in seconds '''
        return (np.sum(self.timeDelta) / 1000000000)

    def getAccel(self):
        return (self.xAccel, self.yAccel, self.zAccel)

    def calculateFeature(self, func):
        return func(self.timeArray, self.xAccel, self.yAccel, self.zAccel)


class Power:
    def __init__(self, filename):
        #print("Initializing", filename)
        filearr = np.genfromtxt(filename, delimiter=",", dtype='float')

        self.filename = filename
        self.timeDelta = filearr[:,0]
        self.timeArray = np.cumsum(self.timeDelta)
        self.fileTime = self.getTotalTime()
        self.voltage = filearr[:,1]
        self.current = filearr[:,2]

    ## Selector ##
    def getMeasureTime(self):
        fn = self.filename.split("_")
        filetime = datetime(
            year=int('20'+fn[-7]),
            month=int(fn[-6]),
            day=int(fn[-5]),
            hour=int(fn[-4]),
            minute=int(fn[-3]),
            second=int(fn[-2])
        )

        return filetime

    def getTimeDelta(self):
        return self.timeDelta
    
    def getTotalTime(self):
        ''' Get total time in seconds '''
        return (np.sum(self.timeDelta) / 1000000000)

    def getPower(self):
        return (self.voltage, self.current)
    
    def calculateFeature(self, func):
        return func(self.timeArray, self.voltage, self.current)


class StaticData:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        # File traversal
        self.numberOfFiles = {
            'vibration': 0,
            'power': 0
        }
        self.unprocessedFiles = {
            'vibration': self.listFiles(os.path.join(self.data_dir, 'accel')),
            'power': self.listFiles(os.path.join(self.data_dir, 'vi'))
        }
        self.processedFiles = {
            'vibration': [],
            'power': []
        }
        
        # Feature data
        self.totalTime = 0
        self.totalTimeOn = 0
        self.featureData = {
            'vibration': {
                'measuretime': [],
                'rms': {
                    'x': [],
                    'y': [],
                    'z': []
                }
            },
            'power': {
                'measuretime': [],
                'rms': {
                    'voltage': [],
                    'current': [],
                }
            },
            'error': []
        }
        self.dataError = []

    ## File Traversal ##
    def listFiles(self, dir):
        return [f for f in os.listdir(dir)]
    
    def addDataPoint(self, filename, data_key, feature_key):
        def rms(array):
            return math.sqrt(np.mean(np.square(array)))


        if data_key == 'vibration':
            vib = Vibration(self.data_dir + '/accel/' + filename)
            
            if feature_key == 'rms':
                vibRaw = vib.getAccel()
                
                self.featureData['vibration']['measuretime'].append(vib.getMeasureTime())
                self.featureData['vibration']['rms']['x'].append(rms(vibRaw[0]))
                self.featureData['vibration']['rms']['y'].append(rms(vibRaw[1]))
                self.featureData['vibration']['rms']['z'].append(rms(vibRaw[2]))
        
        elif data_key == 'power':
            vi = Power(self.data_dir + '/vi/' + filename)

            if feature_key == 'rms':
                powerRaw = vi.getPower()
                
                self.featureData['power']['measuretime'].append(vi.getMeasureTime())
                self.featureData['power']['rms']['voltage'].append(rms(powerRaw[0]))
                self.featureData['power']['rms']['current'].append(rms(powerRaw[1]))

    def loadDataset(self, key):
        self.numberOfFiles[key] = len(self.unprocessedFiles[key])
        stepSize = self.numberOfFiles[key] // 25
        print('\nProcessing dataset with key', key)

        filenum = 0
        while len(self.unprocessedFiles[key]) > 0:
            filenum += 1
            step = filenum // stepSize
            print('Loading dataset', key, '[' + '#' * step + '-' * (25-step) + ']', '{:.1f}%'.format(filenum/self.numberOfFiles[key]*100), end='\r')

            # Pop from unprocessed files list
            f = self.unprocessedFiles[key].pop(0)
            
            try:
                # Extract data from file
                self.addDataPoint(filename=f, data_key=key, feature_key='rms')
            
            except ValueError:
                timeException = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
                print('\nValueError on', f, 'at', timeException)
                self.featureData['error'].append(('ValueError', key, f, timeException, len(self.processedFiles[key])))

            except IndexError:
                timeException = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
                print('\nIndexError on', f, 'at', timeException)
                self.featureData['error'].append(('IndexError', key, f, timeException, len(self.processedFiles[key])))

            except KeyboardInterrupt:
                sys.exit()
            
            except:
                timeException = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
                self.saveFeatureData(filename=('static_' + timeException + '.pkl'))
                print('\nException Occured on', timeException)
                print('----------------------------------')
                traceback.print_exc()

            # Add to processed files list
            self.processedFiles[key].append(f)

    ## Feature data from/to file ##
    def saveFeatureData(self, filename):
        print('\nSaving feature data to', filename)
        with open(filename, 'wb') as f:
            pickle.dump(self.featureData, f, pickle.HIGHEST_PROTOCOL)

    def loadFeatureData(self, filename):
        print('\nLoading feature data from', filename)
        with open(filename, 'rb') as f:
            self.featureData = pickle.load(f)

if __name__ == "__main__":
    Dataset = StaticData("D:/Cloud/Google Drive/Tugas Akhir/data/accel-vi-data-cont")
    Dataset.loadDataset(key='vibration')
    Dataset.loadDataset(key='power')
    Dataset.saveFeatureData(filename=('static_' + datetime.now().strftime('%y_%m_%d_%H_%M_%S') + '.pkl'))
    #Dataset.loadFeatureData(filename='static_20_07_18_16_35_22.pkl')
    #print(Dataset.featureData)

#############################################