import numpy as np
import matplotlib.pyplot as plt

# Global Variable
vib_m1 = vib_m2 = []

### File extraction ###
def getData(filename):
    print("Processing " + filename)
    filearr = np.genfromtxt(filename, delimiter=",", dtype='float')
    print(filearr.shape)
    return filearr

# Time data preprocessing
def preProcessData(data):
    time_0 = data[0,0]
    for i in range(1,data.shape[0]):
        data[i,0] = data[i,0] - time_0
    data[0,0] = 0  

    # Return data.time data.x data.y data.z
    return [data[:,0], data[:,1], data[:,2], data[:,3]] 

# Data class as object
class Vibration():
    name = ''
    time = []
    x = []
    y = []
    z = []

    def __init__(self, filename, name):
        print("Initializing " + filename)
        filearr = np.genfromtxt(filename, delimiter=",", dtype='float')

        self.time = filearr[:,0]
        self.x = filearr[:,1]
        self.y = filearr[:,2]
        self.z = filearr[:,3]
        self.name = name

    # Calculate offset and update time array to start from zero
    def preProcessTime(self):
        time_0 = self.time[0]
        for i in range(1, self.time.shape[0]):
            self.time[i] = (self.time[i] - time_0)/10000
        self.time[0] = 0

    # Remove gravity offset from Z-axis
    def preProcessZ(self):
        for i in range(0, self.time.shape[0]):
            self.z[i] -= 9.98
        
    # Calculate average sampling rate from time array
    def getSamplingTimeAvg(self):
        dt = []
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
        Fs = 1 / self.getSamplingTimeAvg()
        n = self.time.shape[0]              #no.Samples
        k = np.arange(n)
        Ts = n/Fs
        frq = k/Ts            #Frequency range two sided
        frq = frq[range(n//2)] #Frequency range one sided
        Y = np.fft.fft(self.y)/n          #Fast fourier transfors
        Y = Y[range(n//2)]     #Normalise

        plt.plot(frq,abs(Y),'r') # plotting the spectrum
        plt.grid('on')
        plt.show()

# Execute if invoked directly from script
if __name__ == "__main__":
    m1_vib1 = Vibration('data/m1_vib1.csv', 'Healthy')
    m2_vib3 = Vibration('data/m2_vib3.csv', 'Broken')
    m1_vib1.preProcessTime()
    m2_vib3.preProcessTime()
    #m2_vib3.plotFFT()
    #m1_vib1.plotPSD('x')
    #m1_vib1.plotPSD('y')
    #m2_vib3.plotPSD('x')
    #m2_vib3.plotPSD('y')
    #m1_vib1.preProcessZ()
    #m2_vib3.plot(150,'x')
    #m2_vib3.plot(150)
    #plt.show()
    m2_vib3.plotFreq('x')
    m2_vib3.plotFreq('y')
    #m2_vib3.plotFreq('z')
    m1_vib1.plotFreq('x')
    m1_vib1.plotFreq('y')
    #m1_vib1.plotFreq('z')
    #plt.ylim(0,0.5)
    legend = plt.legend(loc='upper right', fontsize='large')
    plt.show()
    #vib_m2[0] = Vibration('data/m2_vib1.csv').preProcessTime()
    #vib_m2[1] = Vibration('data/m2_vib2.csv').preProcessTime()
    #vib_m2[2] = Vibration('data/m2_vib3.csv').preProcessTime()