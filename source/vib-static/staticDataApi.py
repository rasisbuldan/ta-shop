'''
    TO DO:
        - Implement total time using timedelta
        - Implement motor on lifetime
        - Add voltage/current plot
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

    def getThrottleValue(self):
        fn = self.filename.split("_")
        throttle_val = fn[-1][:-4]

        return throttle_val

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
    def __init__(self, data_dir, dataset_dir, data_type, accel_dir='accel', power_dir='vi'):
        self.data_dir = data_dir
        self.dataset_dir = dataset_dir
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
        
        if data_type == 'accel-power-sweep':
            # File traversal
            self.numberOfFiles = {
                'vibration': 0
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
            self.featureData = {
                'source': self.data_dir,
                'type': 'accel-power-sweep',
                'step': 0,
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
                'power': {
                    'throttle': [],
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


    ## File Traversal ##
    def listFiles(self, dir):
        return [f for f in os.listdir(dir) if 'accel' in f or 'vi' in f]
    
    def addDataPoint(self, filename, data_key, feature_key):
        def rms(array):
            return math.sqrt(np.mean(np.square(array)))

        #print(data_key, feature_key)
        #sys.exit()

        if 'vibration' in data_key:
            vib = Vibration(self.data_dir + '/' + self.accel_dir + '/' + filename)
            vibRaw = vib.getAccel()

            # Horizontal Axis data
            if self.data_type == 'accel-power-cont':
                self.featureData['vibration']['measuretime'].append(vib.getMeasureTime())
            elif self.data_type == 'accel-power-sweep':
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
            
            if 'crest-factor' in feature_key:
                self.featureData['vibration']['crest-factor']['x'].append(max(vibRaw[0])/rms(vibRaw[0]))
                self.featureData['vibration']['crest-factor']['y'].append(max(vibRaw[1])/rms(vibRaw[1]))
                self.featureData['vibration']['crest-factor']['z'].append(max(vibRaw[2])/rms(vibRaw[2]))
            
            if 'peak-to-peak' in feature_key:
                self.featureData['vibration']['peak-to-peak']['x'].append(max(vibRaw[0]) - min(vibRaw[0]))
                self.featureData['vibration']['peak-to-peak']['y'].append(max(vibRaw[1]) - min(vibRaw[1]))
                self.featureData['vibration']['peak-to-peak']['z'].append(max(vibRaw[2]) - min(vibRaw[2]))
        
        if 'power' in data_key:
            vi = Power(self.data_dir + '/' + self.power_dir + '/' + filename)
            powerRaw = vi.getPower()
            
            # Horizontal Axis data
            if self.data_type == 'accel-power-cont':
                self.featureData['power']['measuretime'].append(vi.getMeasureTime())
            elif self.data_type == 'accel-power-sweep':
                self.featureData['power']['throttle'].append(vi.getThrottleValue())
            

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

            if 'crest-factor' in feature_key:
                self.featureData['power']['crest-factor']['voltage'].append(max(powerRaw[0])/rms(powerRaw[0]))
                self.featureData['power']['crest-factor']['current'].append(max(powerRaw[1])/rms(powerRaw[1]))
            
            if 'peak-to-peak' in feature_key:
                self.featureData['power']['peak-to-peak']['voltage'].append(max(powerRaw[0]) - min(powerRaw[0]))
                self.featureData['power']['peak-to-peak']['current'].append(max(powerRaw[1]) - min(powerRaw[1]))


    def loadDataset(self, data_key, feature_key):
        print('Processing data key', data_key, 'and feature key', feature_key)

        # Data key traversal
        for key in data_key:
            print('\nProcessing dataset with key', key)

            self.numberOfFiles[key] = len(self.unprocessedFiles[key])
            
            filenum = 0
            
            # File traversal for specific data key
            while len(self.unprocessedFiles[key]) > 0:
                filenum += 1
                step = filenum // (self.numberOfFiles[key] // 50)
                print('Loading dataset', key, '[' + '#' * step + '-' * (50-step) + ']', '{:.1f}%'.format(filenum/self.numberOfFiles[key]*100), end='\r')

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
                    #self.saveFeatureData(filename=('static_' + timeException + '.pkl'))
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

    def loadFeatureData(self, filename=None):
        # If no filename specified, load feature data from latest
        if filename == None:
            featureFileList = [f for f in os.listdir(self.dataset_dir) if '.pkl' in f]
            #print('Feature file list:', *featureFileList, sep='\n')
            filename = os.path.join(self.dataset_dir, featureFileList[-1])

        print('\nLoading feature data from', filename)
        with open(filename, 'rb') as f:
            self.featureData = pickle.load(f)

    def saveToCSV(self, filename, step_num):

        if self.data_type == 'accel-power-sweep':
            self.featureData['step'] = step_num

            dataArray = np.array([]).reshape(0,28)

            for dataIdx in range(len(self.featureData['vibration']['throttle'])):
                data = np.array([
                    self.featureData['source'].split('/')[-1],                                         # 0
                    self.featureData['step'],
                    self.featureData['vibration']['throttle'][dataIdx],
                    self.featureData['vibration']['rms']['x'][dataIdx],
                    self.featureData['vibration']['rms']['y'][dataIdx],
                    self.featureData['vibration']['rms']['z'][dataIdx],                 # 5
                    self.featureData['vibration']['kurtosis']['x'][dataIdx],
                    self.featureData['vibration']['kurtosis']['y'][dataIdx],
                    self.featureData['vibration']['kurtosis']['z'][dataIdx],
                    self.featureData['vibration']['skewness']['x'][dataIdx],
                    self.featureData['vibration']['skewness']['y'][dataIdx],            # 10
                    self.featureData['vibration']['skewness']['z'][dataIdx],
                    self.featureData['vibration']['crest-factor']['x'][dataIdx],
                    self.featureData['vibration']['crest-factor']['y'][dataIdx],
                    self.featureData['vibration']['crest-factor']['z'][dataIdx],
                    self.featureData['vibration']['peak-to-peak']['x'][dataIdx],        # 15
                    self.featureData['vibration']['peak-to-peak']['y'][dataIdx],
                    self.featureData['vibration']['peak-to-peak']['z'][dataIdx],
                    self.featureData['power']['rms']['voltage'][dataIdx],
                    self.featureData['power']['rms']['current'][dataIdx],
                    self.featureData['power']['kurtosis']['voltage'][dataIdx],          # 20
                    self.featureData['power']['kurtosis']['current'][dataIdx],
                    self.featureData['power']['skewness']['voltage'][dataIdx],
                    self.featureData['power']['skewness']['current'][dataIdx],
                    self.featureData['power']['crest-factor']['voltage'][dataIdx],
                    self.featureData['power']['crest-factor']['current'][dataIdx],      # 25
                    self.featureData['power']['peak-to-peak']['voltage'][dataIdx],
                    self.featureData['power']['peak-to-peak']['current'][dataIdx],      # 27
                ]).reshape(1,28)

                dataArray = np.append(dataArray, data, axis=0)

        np.savetxt(
            fname=filename,
            X=dataArray,
            fmt='%s',
            delimiter=',',
            header='src,imstep,throttle,vrmsx,vrmsy,vrmsz,vkurtx,vkurty,vkurtz,vskewx,vskewy,vskewz,vcrestx,vcresty,vcrestz,vpeakx,vpeaky,vpeakz,prmsv,prmsi,pkurtv,pkurti,pskewv,pskewi,pcrestv,pcresti,ppeakv,ppeaki'
        )
    

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


    def plotSimpleAggregate(self, data_key, feature_key):
        def rms(array):
            return np.sqrt(np.mean(np.square(array)))
        
        if data_key == 'vibration':
            data_buf = []
            for i in range(0, len(self.featureData['vibration']['rms']['x']), 3):
                data_buf.append(rms([self.featureData['vibration']['rms']['x'][i:i+3]]))

            xdata = range(len(self.featureData[data_key][feature_key]['x']))
            
            # Subplot 1 : X
            plt.subplot(311)
            plt.plot(len(data_buf), data_buf)
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

    
    def plotSweep(self, data_key, feature_key):
        if 'vibration' in data_key:
            xdata = [int(t) for t in self.featureData['vibration']['throttle']]
            #print(xdata)

            # Add multiplot
            fig = plt.figure()

            ax_vib_x = fig.add_subplot(311, label='VibX', frame_on=True)
            ax_vib_y = fig.add_subplot(312, label='VibY', frame_on=True)
            ax_vib_z = fig.add_subplot(313, label='VibZ', frame_on=True)

            #ax_v_x = fig.add_subplot(311, label='VoltageX', frame_on=False)
            ax_i_x = fig.add_subplot(311, label='CurrentX', frame_on=False)
            #ax_v_y = fig.add_subplot(312, label='VoltageY', frame_on=False)
            ax_i_y = fig.add_subplot(312, label='CurrentY', frame_on=False)
            #ax_v_z = fig.add_subplot(313, label='VoltageZ', frame_on=False)
            ax_i_z = fig.add_subplot(313, label='CurrentZ', frame_on=False)
            
            # Subplot 1
            p_vib_x, = ax_vib_x.plot(xdata, self.featureData['vibration'][feature_key]['x'], color='C0')
            ax_vib_x.set_ylabel('RMS Vibration - X', color='C0')
            ax_vib_x.set_ylim([0,20])
            ax_vib_x.set_xlim([100,1800])
            ax_vib_x.set_xticks(np.arange(100, 1800, 100))

            if 'power' in data_key:
                """ p_v_x, = ax_v_x.plot(xdata, self.featureData['power']['rms']['voltage'], color='C1')
                ax_v_x.yaxis.tick_right()
                ax_v_x.set_ylim([0,15])
                ax_v_x.set_xlim([100,1800])
                ax_v_x.set_xticks(np.arange(100, 1800, 100)) """

                p_i_x, = ax_i_x.plot(xdata, self.featureData['power']['rms']['current'], color='C2')
                ax_i_x.yaxis.tick_right()
                ax_i_x.set_ylim([0,10000])
                ax_i_x.set_xlim([100,1800])
                ax_i_x.set_xticks(np.arange(100, 1800, 100))

            ax_vib_x.grid(True)

            # Subplot 2
            p_vib_y, = ax_vib_y.plot(xdata, self.featureData['vibration'][feature_key]['y'], color='C0')
            ax_vib_y.set_ylabel('RMS Vibration - Y', color='C0')
            ax_vib_y.set_ylim([0,20])
            ax_vib_y.set_xlim([100,1800])
            ax_vib_y.set_xticks(np.arange(100, 1800, 100))

            if 'power' in data_key:
                """ p_v_y, = ax_v_y.plot(xdata, self.featureData['power']['rms']['voltage'], color='C1')
                ax_v_y.yaxis.tick_right()
                ax_v_y.set_ylim([0,15])
                ax_v_y.set_xlim([100,1800])
                ax_v_y.set_xticks(np.arange(100, 1800, 100)) """

                p_i_y, = ax_i_y.plot(xdata, self.featureData['power']['rms']['current'], color='C2')
                ax_i_y.yaxis.tick_right()
                ax_i_y.set_ylim([0,10000])
                ax_i_y.set_xlim([100,1800])
                ax_i_y.set_xticks(np.arange(100, 1800, 100))

            ax_vib_y.grid(True)

            # Subplot 2
            p_vib_z, = ax_vib_z.plot(xdata, self.featureData['vibration'][feature_key]['z'], color='C0')
            ax_vib_z.set_ylabel('RMS Vibration - Z', color='C0')
            ax_vib_z.set_ylim([0,20])
            ax_vib_z.set_xlim([100,1800])
            ax_vib_z.set_xticks(np.arange(100, 1800, 100))

            if 'power' in data_key:
                """ p_v_z, = ax_v_z.plot(xdata, self.featureData['power']['rms']['voltage'], color='C1')
                ax_v_z.yaxis.tick_right()
                ax_v_z.set_ylim([0,15])
                ax_v_z.set_xlim([100,1800])
                ax_v_z.set_xticks(np.arange(100, 1800, 100)) """

                p_i_z, = ax_i_z.plot(xdata, self.featureData['power']['rms']['current'], color='C2')
                ax_i_z.yaxis.tick_right()
                ax_i_z.set_ylim([0,10000])
                ax_i_z.set_xlim([100,1800])
                ax_i_z.set_xticks(np.arange(100, 1800, 100))

            ax_vib_z.grid(True)

        ax_vib_x.legend((p_vib_x, p_i_x), ('RMS-X','Current'), loc='upper right')
        ax_vib_y.legend((p_vib_y, p_i_y), ('RMS-Y','Current'), loc='upper right')
        ax_vib_z.legend((p_vib_z, p_i_z), ('RMS-Z','Current'), loc='upper right')

        dstr = self.data_dir.split('-')
        propNum = dstr[-3]
        if dstr[-1] == 'nooffset/':
            offsetType = 'No Offset'
        elif dstr[-1] == 'offset/':
            offsetType = 'Offset'
        else:
            offsetType = dstr[-1].title()

        plt.suptitle('Vibration and Power over Throttle (' + offsetType + ' - ' + propNum + ')')
        plt.show()



if __name__ == "__main__":
    filenameFormat = 'static_sweep_im_{}.csv'
    dataDir = 'C:/Users/rss75/Documents/GitHub/ta-shop/data-acq/accel-vi-data-varspeed-8045-2-im5'
    dataDirList = [
        ('C:/Users/rss75/Documents/GitHub/ta-shop/data-acq/accel-vi-data-varspeed-8045-2-im0',0),
        ('C:/Users/rss75/Documents/GitHub/ta-shop/data-acq/accel-vi-data-varspeed-8045-2-im1',1),
        ('C:/Users/rss75/Documents/GitHub/ta-shop/data-acq/accel-vi-data-varspeed-8045-2-im2',2),
        ('C:/Users/rss75/Documents/GitHub/ta-shop/data-acq/accel-vi-data-varspeed-8045-2-im3',3),
        ('C:/Users/rss75/Documents/GitHub/ta-shop/data-acq/accel-vi-data-varspeed-8045-2-im4',4),
        ('C:/Users/rss75/Documents/GitHub/ta-shop/data-acq/accel-vi-data-varspeed-8045-2-im5',5),
    ]
    datasetDir = 'C:/Users/rss75/Documents/GitHub/ta-shop/source/vib-static/dump/sweep-im/'

    for data in dataDirList:
        print('\n\n-----------------------')
        print('Processing', data[0].split('/')[-1])

        filename = filenameFormat.format(data[1])
        Dataset = StaticData(data_dir=data[0], dataset_dir=datasetDir, data_type='accel-power-sweep')
        Dataset.loadDataset(data_key=['vibration', 'power'], feature_key=['rms','kurtosis','skewness','crest-factor','peak-to-peak'])
        Dataset.saveToCSV(filename=datasetDir + filename, step_num=data[1])
    #Dataset.saveFeatureData(filename=(datasetDir + 'static_sweep_' + datetime.now().strftime('%y_%m_%d_%H_%M_%S') + '.pkl'))
    #Dataset.loadFeatureData()


    #Dataset.plotSimple(data_key=['vibration','power'], feature_key='rms')
    Dataset.plotSweep(data_key=['vibration','power'], feature_key='rms')