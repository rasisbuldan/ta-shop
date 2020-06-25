from navdataJSONParser import getDataFromFile, getJSONArray
import numpy as np
import matplotlib.pyplot as plt
import math

class NavData:
    '''
    Add more navdata,
        - raw accelerometer processing
    '''
    def __init__(self, data):
        self.navdata = data

    def getRollYawPitch(self, unit='deg'):
        roll = self.navdata['demo']['rotation']['roll']
        yaw = self.navdata['demo']['rotation']['yaw']
        pitch = self.navdata['demo']['rotation']['pitch']
        
        if unit == 'rad':
            roll = roll * math.pi / 180
            yaw = yaw * math.pi / 180
            pitch = pitch * math.pi / 180

        return roll, yaw, pitch

    def getBatteryPercentage(self):
        return self.navdata['demo']['batteryPercentage']
    
    def getVelocity(self):
        vx = self.navdata['demo']['velocity']['x']
        vy = self.navdata['demo']['velocity']['y']
        vz = self.navdata['demo']['velocity']['z']

        return vx, vy, vz
    
    def getTimeDelta(self):
        return self.navdata['timeDelta']


class FlightData:
    def __init__(self, filename):
        self.navdataArray = [NavData(n) for n in getJSONArray(filename)]
        self.timeDeltaArray = [nav.getTimeDelta() for nav in self.navdataArray]
        self.timeArray = np.cumsum(self.timeDeltaArray)
    

    def plotTimeDelta(self):
        timeDeltaAvg = np.average(self.timeDeltaArray)
        avgArray = np.full(len(self.timeDeltaArray), timeDeltaAvg)
        nbPacketArray = np.arange(len(self.timeDeltaArray))

        plt.plot(nbPacketArray, self.timeDeltaArray, label='Data')
        plt.plot(nbPacketArray, avgArray, linestyle='--', label='Mean: {:.2f}ms'.format(timeDeltaAvg))
        plt.title('Latency between navdata packet')
        plt.xlabel('# of packet')
        plt.ylabel('latency (ms)')
        plt.grid(which='major')
        plt.legend()
        plt.tight_layout()
        plt.show()


    def plotRollYawPitch(self, unit='deg'):
        rollArray = [nav.getRollYawPitch(unit)[0] for nav in self.navdataArray]
        yawArray = [(nav.getRollYawPitch(unit)[1]) for nav in self.navdataArray]
        pitchArray = [nav.getRollYawPitch(unit)[2] for nav in self.navdataArray]

        # Subplot 1 : Roll
        plt.subplot(311)
        plt.plot(self.timeArray, rollArray)
        plt.title('Roll over time')
        plt.ylabel('Roll ({})'.format(unit))
        plt.grid(True)

        # Subplot 2 : Yaw
        plt.subplot(312)
        plt.plot(self.timeArray, yawArray)
        plt.title('Yaw over time')
        plt.ylabel('Yaw ({})'.format(unit))
        plt.grid(True)

        # Subplot 3 : Pitch
        plt.subplot(313)
        plt.plot(self.timeArray, pitchArray)
        plt.title('Pitch over time')
        plt.ylabel('Pitch ({})'.format(unit))
        plt.grid(True)

        plt.tight_layout()
        plt.show()


    def plotTrajectory(self):
        '''
        Approximation using x(t) = x(t-1) + v(t-1) * dT
                            y(t) = y(t-1) + v(t-1) * dT
        '''
        vxArray = [nav.getVelocity()[0] for nav in self.navdataArray]
        vyArray = [nav.getVelocity()[1] for nav in self.navdataArray]

        xposArray = [0]
        yposArray = [0]

        for i in range(1, len(min(vxArray, vyArray))):
            xposArray.append(xposArray[i-1] + (vxArray[i-1] * self.timeDeltaArray[i-1]/1000000))
            yposArray.append(yposArray[i-1] + (vyArray[i-1] * self.timeDeltaArray[i-1]/1000000))
        
        plt.plot(xposArray, yposArray)
        plt.title('Drone trajectory over time')
        plt.xlim(min(xposArray) * 1.5, max(xposArray) * 1.5)
        plt.ylim(min(yposArray) * 1.5, max(yposArray) * 1.5)
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


    def plot(self, rypunit='deg'):
        # Orientation
        rollArray = [nav.getRollYawPitch(rypunit)[0] for nav in self.navdataArray]
        yawArray = [(nav.getRollYawPitch(rypunit)[1]) for nav in self.navdataArray]
        pitchArray = [nav.getRollYawPitch(rypunit)[2] for nav in self.navdataArray]

        # Velocity
        vxArray= [(nav.getVelocity()[0]/1000) for nav in self.navdataArray]
        vyArray= [(nav.getVelocity()[1]/1000) for nav in self.navdataArray]
        vzArray= [(nav.getVelocity()[2]/1000) for nav in self.navdataArray]

        # Subplot 1 : Roll
        plt.subplot(321)
        plt.plot(self.timeArray, rollArray)
        plt.title('Roll over time')
        plt.ylabel('Roll ({})'.format(rypunit))
        plt.ylim(-190,190)
        plt.grid(True)

        # Subplot 3 : Yaw
        plt.subplot(323)
        plt.plot(self.timeArray, yawArray)
        plt.title('Yaw over time')
        plt.ylabel('Yaw ({})'.format(rypunit))
        plt.ylim(-190,190)
        plt.grid(True)

        # Subplot 5 : Pitch
        plt.subplot(325)
        plt.plot(self.timeArray, pitchArray)
        plt.title('Pitch over time')
        plt.ylabel('Pitch ({})'.format(rypunit))
        plt.ylim(-190,190)
        plt.grid(True)

        # Subplot 2 : X velocity
        plt.subplot(322)
        plt.plot(self.timeArray, vxArray)
        plt.title('X velocity over time')
        plt.ylabel('Velocity (m/s)')
        plt.grid(True)

        # Subplot 4 : Y velocity
        plt.subplot(324)
        plt.plot(self.timeArray, vyArray)
        plt.title('Y velocity over time')
        plt.ylabel('Velocity (m/s)')
        plt.grid(True)

        # Subplot 6 : Z velocity
        plt.subplot(326)
        plt.plot(self.timeArray, vzArray)
        plt.title('Z velocity over time')
        plt.ylabel('Velocity (m/s)')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    FD = FlightData('C:/Users/rss75/Documents/GitHub/ta-shop/source/ardrone/nodejs/flight-data/data_23_03.json')
    print('Processing Flight Data with {} points'.format(len(FD.navdataArray)))
    FD.plot('deg')
    #FD.plotTrajectory()
    FD.plotTimeDelta()