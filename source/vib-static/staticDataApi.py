'''
    TO DO:
        - Implement total time using timedelta
        - Implement motor on lifetime
'''

import os
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime, time
import pickle
import traceback
import sys
import scipy.stats

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

    # Sweep vibration measurement (data-acq/accel-data)
    def getThrottleValue(self):
        fn = self.filename.split("_")
        throttle_val = fn[-1][:-4]

        return throttle_val

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
    def __init__(self, data_dir, data_type, accel_dir='accel', power_dir='vi'):
        self.data_dir = data_dir
        self.data_type = data_type
        self.accel_dir = accel_dir
        self.power_dir = power_dir
        
        if data_type == 'accel-power-cont':
            # File traversal
            self.numberOfFiles = {
                'vibration': 0,
                'power': 0
            }
            self.unprocessedFiles = {
                'vibration': self.listFiles(os.path.join(self.data_dir, accel_dir)),
                'power': self.listFiles(os.path.join(self.data_dir, power_dir))
            }
            self.processedFiles = {
                'vibration': [],
                'power': []
            }
            
            # Feature data
            self.totalTime = 0
            self.totalTimeOn = 0
            self.featureData = {
                'source': self.data_dir,
                'type': 'accel-power-cont',
                'numberOfData': {
                    'vibration': 0,
                    'power': 0
                },
                'vibration': {
                    'measuretime': [],
                    'rms': {
                        'x': [],
                        'y': [],
                        'z': []
                    },
                    'kurtosis': {
                        'x': [],
                        'y': [],
                        'z': []
                    },
                    'skewness': {
                        'x': [],
                        'y': [],
                        'z': [],
                    },
                    'crest-factor': {
                        'x': [],
                        'y': [],
                        'z': [],
                    },
                    'peak-to-peak': {
                        'x': [],
                        'y': [],
                        'z': [],
                    }
                },
                'power': {
                    'measuretime': [],
                    'rms': {
                        'voltage': [],
                        'current': [],
                    },
                    'kurtosis': {
                        'voltage': [],
                        'current': [],
                    },
                    'skewness': {
                        'voltage': [],
                        'current': [],
                    },
                    'crest-factor': {
                        'voltage': [],
                        'current': [],
                    },
                    'peak-to-peak': {
                        'voltage': [],
                        'current': [],
                    }
                },
                'error': []
            }
        
        if data_type == 'accel-sweep':
            # File traversal
            self.numberOfFiles = {
                'vibration': 0
            }
            self.unprocessedFiles = {
                'vibration': self.listFiles(self.data_dir)
            }
            self.processedFiles = {
                'vibration': []
            }

            # Feature data
            self.featureData = {
                'source': self.data_dir,
                'type': 'accel-sweep',
                'numberOfData': 0,
                'vibration': {
                    'throttle': [],
                    'rms': {
                        'x': [],
                        'y': [],
                        'z': []
                    },
                    'kurtosis': {
                        'x': [],
                        'y': [],
                        'z': []
                    },
                    'skewness': {
                        'x': [],
                        'y': [],
                        'z': [],
                    },
                    'crest-factor': {
                        'x': [],
                        'y': [],
                        'z': [],
                    },
                    'peak-to-peak': {
                        'x': [],
                        'y': [],
                        'z': [],
                    }
                },
                'error': []
            }


    ## File Traversal ##
    def listFiles(self, dir):
        return [f for f in os.listdir(dir)]
    
    def addDataPoint(self, filename, data_key, feature_key):
        def rms(array):
            return math.sqrt(np.mean(np.square(array)))


        if 'vibration' in data_key:
            if self.data_type == 'accel-power-cont':
                vib = Vibration(self.data_dir + '/' + self.accel_dir + '/' + filename)
            elif self.data_type == 'accel-sweep':
                vib = Vibration(self.data_dir + '/' + filename)
            
            vibRaw = vib.getAccel()

            # Horizontal Axis data
            if self.data_type == 'accel-power-cont':
                self.featureData['vibration']['measuretime'].append(vib.getMeasureTime())
            elif self.data_type == 'accel-sweep':
                self.featureData['vibration']['throttle'].append(vib.getThrottleValue())

            # Vertical Axis data
            if 'rms' in feature_key:
                self.featureData['vibration']['rms']['x'].append(rms(vibRaw[0]))
                self.featureData['vibration']['rms']['y'].append(rms(vibRaw[1]))
                self.featureData['vibration']['rms']['z'].append(rms(vibRaw[2]))

            if 'kurtosis' in feature_key:
                self.featureData['vibration']['kurtosis']['x'].append(scipy.stats.kurtosis(vibRaw[0]))
                self.featureData['vibration']['kurtosis']['y'].append(scipy.stats.kurtosis(vibRaw[1]))
                self.featureData['vibration']['kurtosis']['z'].append(scipy.stats.kurtosis(vibRaw[2]))

            if 'skewness' in feature_key:
                self.featureData['vibration']['skewness']['x'].append(scipy.stats.skew(vibRaw[0]))
                self.featureData['vibration']['skewness']['y'].append(scipy.stats.skew(vibRaw[1]))
                self.featureData['vibration']['skewness']['z'].append(scipy.stats.skew(vibRaw[2]))
        
        if 'power' in data_key:
            vi = Power(self.data_dir + '/' + self.power_dir + '/' + filename)
            powerRaw = vi.getPower()
            
            # Horizontal Axis data
            if self.data_type == 'accel-power-cont':
                self.featureData['power']['measuretime'].append(vib.getMeasureTime())

            # Vertical Axis data
            if 'rms' in feature_key:
                self.featureData['power']['rms']['voltage'].append(rms(powerRaw[0]))
                self.featureData['power']['rms']['current'].append(rms(powerRaw[1]))

            if 'kurtosis' in feature_key:
                self.featureData['power']['kurtosis']['voltage'].append(scipy.stats.kurtosis(powerRaw[0]))
                self.featureData['power']['kurtosis']['current'].append(scipy.stats.kurtosis(powerRaw[1]))
            
            if 'skewness' in feature_key:
                self.featureData['power']['skewness']['voltage'].append(scipy.stats.skew(powerRaw[0]))
                self.featureData['power']['skewness']['current'].append(scipy.stats.skew(powerRaw[1]))


    def loadDataset(self, data_key, feature_key):
        print('\nProcessing data key', data_key, 'and feature key', feature_key)

        # Data key traversal
        for key in data_key:
            print('\nProcessing dataset with key', key)

            self.numberOfFiles[key] = len(self.unprocessedFiles[key])
            
            stepSize = self.numberOfFiles[key] // 40
            filenum = 0
            
            # File traversal for specific data key
            while len(self.unprocessedFiles[key]) > 0:
                filenum += 1
                step = filenum // stepSize
                print('Loading dataset', key, '[' + '#' * step + '-' * (40-step) + ']', '{:.1f}%'.format(filenum/self.numberOfFiles[key]*100), end='\r')

                # Pop from unprocessed files list
                f = self.unprocessedFiles[key].pop(0)
                
                try:
                    # Extract data from file
                    self.addDataPoint(filename=f, data_key=key, feature_key=feature_key)
                
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
        self.featureData['numberOfData'] = len(self.processedFiles['vibration'])
        with open(filename, 'wb') as f:
            pickle.dump(self.featureData, f, pickle.HIGHEST_PROTOCOL)

    def loadFeatureData(self, filename):
        print('\nLoading feature data from', filename)
        with open(filename, 'rb') as f:
            self.featureData = pickle.load(f)
    

    ## Feature data processing demo ##
    def plotSimple(self, data_key, feature_key):
        if data_key == 'vibration':
            xdata = range(len(self.featureData[data_key][feature_key]['x']))
            
            # Subplot 1 : X
            plt.subplot(311)
            plt.plot(xdata, self.featureData[data_key][feature_key]['x'])
            plt.title('X over time')
            plt.grid(True)

            # Subplot 2 : Y
            plt.subplot(312)
            plt.plot(xdata, self.featureData[data_key][feature_key]['y'])
            plt.title('Y over time')
            plt.grid(True)

            # Subplot 3 : Z
            plt.subplot(313)
            plt.plot(xdata, self.featureData[data_key][feature_key]['z'])
            plt.title('Z over time')
            plt.grid(True)

        if data_key == 'power':
            xdata = range(len(self.featureData[data_key][feature_key]['voltage']))
            
            # Subplot 1 : Voltage
            plt.subplot(211)
            plt.plot(xdata, self.featureData[data_key][feature_key]['voltage'])
            plt.title('Voltage over time')
            plt.grid(True)

            # Subplot 2 : Current
            plt.subplot(212)
            plt.plot(xdata, self.featureData[data_key][feature_key]['current'])
            plt.title('Current over time')
            plt.grid(True)

        plt.show()

    
    def plotSweep(self, feature_key, data_key='vibration'):
        xdata = self.featureData['vibration']['throttle']
        print(self.featureData['vibration']['throttle'])
            
        # Subplot 1 : X
        plt.subplot(311)
        plt.plot(xdata, self.featureData[data_key][feature_key]['x'])
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))
        plt.grid(True)
        plt.title('X-Axis vibration vs Throttle')

        # Subplot 2 : Y
        plt.subplot(312)
        plt.plot(xdata, self.featureData[data_key][feature_key]['y'])
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))
        plt.grid(True)
        plt.title('Y-Axis vibration vs Throttle')

        # Subplot 3 : Z
        plt.subplot(313)
        plt.plot(xdata, self.featureData[data_key][feature_key]['z'])
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))
        plt.grid(True)
        plt.title('Z-Axis vibration vs Throttle')

        plt.show()


if __name__ == "__main__":
    Dataset = StaticData(data_dir='C:/Users/rss75/Documents/GitHub/ta-shop/data-acq/accel-data', data_type='accel-sweep')
    Dataset.loadDataset(data_key=['vibration'], feature_key=['rms','kurtosis','skewness'])
    Dataset.saveFeatureData(filename=('static_sweep_' + datetime.now().strftime('%y_%m_%d_%H_%M_%S') + '.pkl'))
    #Dataset.loadFeatureData(filename='static_sweep_20_07_21_00_50_02.pkl')
    #print(Dataset.featureData)
    #Dataset.plotSimple(data_key='vibration', feature_key='rms')
    #Dataset.plotSimple(data_key='power', feature_key='rms')
    #Dataset.plotSweep(feature_key='rms')