import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import cross_val_score
#from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import StratifiedKFold
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline

# Global Variable
vib_m1 = vib_m2 = []

# Data class as object
class Vibration():
    name = ''
    time = []
    x = []
    y = []
    z = []
    n_fft = 0
    n_data = 0

    def __init__(self, filename, name):
        print("Initializing " + filename)
        filearr = np.genfromtxt(filename, delimiter=",", dtype='float')

        self.time = filearr[:,0]
        self.x = filearr[:,1]
        self.y = filearr[:,2]
        self.z = filearr[:,3]
        self.name = name
        #print(self.time)
        self.n_data = self.time.shape[0]
        print(self.n_data)

        # Auto-preprocessing Data
        self.preProcessTime()
        self.preProcessZ()

    # Calculate offset and update time array to start from zero
    def preProcessTime(self):
        time_0 = self.time[0]
        for i in range(1, self.time.shape[0]):
            #print(self.time[i])
            self.time[i] = (self.time[i] - time_0)/10000
        self.time[0] = 0

    # Remove gravity offset from Z-axis
    def preProcessZ(self):
        for i in range(0, self.time.shape[0]):
            self.z[i] -= 9.98
        
    # Calculate average sampling rate from time array
    def getSamplingTimeAvg(self):
        dt = []
        #print('ndata: ',self.n_data)
        #print(self.time)
        for i in range(1, self.time.shape[0]):
            dt.append(self.time[i] - self.time[i-1])
        avg_dt = sum(dt)/len(dt)
        print(avg_dt)
        return (avg_dt)

    # Plot frequency magnitude spectrum (FFT), selectable axis
    def plotFreq(self, axis=None):
        #plt.clf()
        T = self.getSamplingTimeAvg()
        if axis == None:
            # Plot all axis
            plt.title('Vibration Magnitude Spectrum (all axis)')
            plt.magnitude_spectrum(self.x, Fs=1/T, label=self.name + ' x')
            plt.magnitude_spectrum(self.y, Fs=1/T, label=self.name + ' y')
            plt.magnitude_spectrum(self.z, Fs=1/T, label=self.name + ' z')
        elif axis == 'x':
            plt.title('Vibration Magnitude Spectrum (x axis)')
            plt.magnitude_spectrum(self.x, Fs=1/T, label=self.name + ' x')
        elif axis == 'y':
            plt.title('Vibration Magnitude Spectrum (y axis)')
            plt.magnitude_spectrum(self.y, Fs=1/T, label=self.name + ' y')
        elif axis == 'z':
            plt.title('Vibration Magnitude Spectrum (z axis)')
            plt.magnitude_spectrum(self.z, Fs=1/T, label=self.name + ' z')
        else:
            return 'Invalid parameter'
        #plt.show()

    # Plot Power Spectral Density (PSD), selectable axis
    def plotPSD(self, axis=None):
        #plt.clf()
        plt.title('Power Spectral Density')
        T = self.getSamplingTimeAvg()
        if axis == None:
            # Plot all axis
            plt.psd(self.x, Fs=1/T, label=self.name + ' x')
            plt.psd(self.y, Fs=1/T, label=self.name + ' y')
            plt.psd(self.z, Fs=1/T, label=self.name + ' z')
        elif axis == 'x':
            plt.psd(self.x, Fs=1/T, label=self.name + ' x')
        elif axis == 'y':
            plt.psd(self.y, Fs=1/T, label=self.name + ' y')
        elif axis == 'z':
            plt.psd(self.z, Fs=1/T, label=self.name + ' z')
        else:
            return 'Invalid parameter'
        #plt.show()

    # Plot time series vibration signal, selectable accelerometer axis
    def plot(self, ndata, axis=None, color=None):
        #plt.clf()
        plt.title('Vibration data in axis')
        if color == None:
            if axis == None:
                # Plot all axis
                plt.plot(self.time[0:ndata], self.x[0:ndata], 'r-', label=self.name + ' x')
                plt.plot(self.time[0:ndata], self.y[0:ndata], 'b-', label=self.name + ' y')
                plt.plot(self.time[0:ndata], self.z[0:ndata], 'g-', label=self.name + ' z')
            elif axis == 'x':
                plt.plot(self.time[0:ndata], self.x[0:ndata], 'r-', label=self.name + ' x')
            elif axis == 'y':
                plt.plot(self.time[0:ndata], self.y[0:ndata], 'b-', label=self.name + ' y')
            elif axis == 'z':
                plt.plot(self.time[0:ndata], self.z[0:ndata], 'g-', label=self.name + ' z')
            else:
                return 'Invalid parameter'
        else:
            if axis == 'x':
                plt.plot(self.time[0:ndata], self.x[0:ndata], color, label=self.name + ' x')
            elif axis == 'y':
                plt.plot(self.time[0:ndata], self.y[0:ndata], color, label=self.name + ' y')
            elif axis == 'z':
                plt.plot(self.time[0:ndata], self.z[0:ndata], color, label=self.name + ' z')
            else:
                return 'Invalid parameter'
        plt.legend([self.name])
        #plt.show()

    # Non-builtin function plot for FFT
    def plotFFT(self):
        frq = self.getFFTFreq()
        Y = self.getFFT(self.x)

        plt.plot(frq,abs(Y)) # plotting the spectrum
        plt.grid('on')

    # Get FFT Freq
    def getFFTFreq(self):
        Fs = 1 / 0.0012
        k = np.arange(self.time.shape[0])
        Ts = self.time.shape[0]/Fs
        frq = k/Ts            #Frequency range two sided
        frq = frq[range(self.time.shape[0]//2)] #Frequency range one sided
        frq = frq.reshape(frq.shape[0],1)
        print(frq.shape)
        return frq

    # Convert time to FFT
    def getFFT(self,axis):
        F = np.fft.rfft(axis)/self.time.shape[0]          #Fast fourier transfors
        F = F[range(self.time.shape[0]//2)]     #Normalise
        F = F.reshape(F.shape[0],1)
        print(F.shape)
        return F
    
    # Add label to data
    def getLabel(self,label):
        if label == 0:
            labels = np.zeros((self.time.shape[0]//2,1), dtype=int)
        elif label == 1:
            labels = np.ones((self.time.shape[0]//2,1), dtype=int)
        else:
            labels = np.repeat(label, self.time.shape[0]//2, axis=0)
            labels.reshape(self.time.shape[0]//2,1)
        return labels
    
    # Combine data
    def combine_FFT_label(self,label):
        buf = np.array([]).reshape(self.time.shape[0]//2,0)
        buf = np.append(buf, self.getFFTFreq(), axis=1)
        buf = np.append(buf, self.getFFT(self.x), axis=1)
        buf = np.append(buf, self.getFFT(self.y), axis=1)
        buf = np.append(buf, self.getFFT(self.z), axis=1)
        buf = np.append(buf, self.getLabel(label), axis=1)

        self.n_fft = buf.shape[0]
        return buf

class BinaryClassifier():
    # Class Attributes
    
    def __init__(self):
        self.data_array_train = np.array([]).reshape(0,5)
        self.data_array_test = np.array([]).reshape(0,5)
        self.n_data_test = 0
        self.n_data_train = 0

    # Add train data from vibration object
    def addDataTrain(self,data,label):
        a = data.combine_FFT_label(label)
        self.data_array_train = np.append(self.data_array_train, a, axis=0)
        self.n_data_train = self.data_array_train.shape[0]

    # Add test data from vibration object
    def addDataTest(self,data,label):
        a = data.combine_FFT_label(label)
        self.data_array_test = np.append(self.data_array_test, a, axis=0)
        self.n_data_test = self.data_array_test.shape[0]

    def create_nn(self):
        # Create neural network
        model = Sequential()
        model.add(Dense(4, input_dim=4, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def estimate(self):
        x_train = self.data_array_train[:,0:4]
        y_train = self.data_array_train[:,4]
        x_test  = self.data_array_test[:,0:4]
        y_test  = self.data_array_test[:,4]

        model = self.create_nn()
        model.fit(x_train, y_train, epochs=100)
        test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
        print('\nTest accuracy: {0:.3f} %'.format(float(test_acc*100)))

    def estimate2(self):
        print('Estimating...')
        x_train = self.data_array_train[:,0:4]
        y_train = self.data_array_train[:,4]
        x_test  = self.data_array_test[:,0:4]
        y_test  = self.data_array_test[:,4]

        # KerasClassifier
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasClassifier(build_fn=self.create_nn, epochs=10, batch_size=5, verbose=1)))
        pipeline = Pipeline(estimators)
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        results = cross_val_score(pipeline, x_train, y_train, cv=kfold)
        print("Result: acc:%.2f%% stdev:(%.2f%%)" % (results.mean()*100, results.std()*100))

# Execute if invoked directly from script
if __name__ == "__main__":
    # Loading data to object
    #m1_vib1_train = Vibration('data/processed/m1_vib1_train.csv','Healthy Train')
    #m1_vib1_test = Vibration('data/processed/m1_vib1_test.csv','Healthy Test')
    #m2_vib3_train = Vibration('data/processed/m2_vib3_train.csv','Broken Train')
    #m2_vib3_test = Vibration('data/processed/m2_vib3_test.csv','Broken Test')

    m_vib = Vibration('source/data/accel_00_42_30.txt','Test')
    #m1_vib1_test.plotFFT()
    #m1_vib1_train.plotFFT()
    
    m_vib.plot(150,'x')
    plt.show()

    m_vib.plotFFT()
    plt.show()

    #binClass = BinaryClassifier()
    #binClass.addDataTrain(m1_vib1_train,1)
    #binClass.addDataTest(m1_vib1_test,1)
    #binClass.addDataTrain(m2_vib3_train,0)
    #binClass.addDataTest(m2_vib3_test,0)
    #binClass.estimate2()


    #m1_vib1 = Vibration('data/m1_vib1.csv', 'Healthy')
    #m2_vib3 = Vibration('data/m2_vib3.csv', 'Broken')
    #m1_vib1.preProcessTime()
    #m2_vib3.preProcessTime()
    #m2_vib3.plotFFT()
    #m1_vib1.plotPSD('x')
    #m1_vib1.plotPSD('y')
    #m2_vib3.plotPSD('x')
    #m2_vib3.plotPSD('y')
    #m1_vib1.preProcessZ()
    #m2_vib3.plot(150,'x')
    #m2_vib3.plot(150)
    #plt.show()
    #m2_vib3.plotFreq('x')
    #m2_vib3.plotFreq('y')
    #m2_vib3.plotFreq('z')
    #m1_vib1.plotFreq('x')
    #m1_vib1.plotFreq('y')
    #m1_vib1.plotFreq('z')
    #plt.ylim(0,0.5)
    #legend = plt.legend(loc='upper right', fontsize='large')
    #plt.show()
    #vib_m2[0] = Vibration('data/m2_vib1.csv').preProcessTime()
    #vib_m2[1] = Vibration('data/m2_vib2.csv').preProcessTime()
    #vib_m2[2] = Vibration('data/m2_vib3.csv').preProcessTime()