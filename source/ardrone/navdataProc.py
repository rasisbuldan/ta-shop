from navdataJSONParser import getDataFromFile, getJSONArray
import numpy as np
import matplotlib.pyplot as plt
import math

class NavData:
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
    
    def plotRollYawPitch(self, unit='deg'):
        rollArray= [nav.getRollYawPitch(unit)[0] for nav in self.navdataArray]
        yawArray= [(nav.getRollYawPitch(unit)[1] + 90) % 180 for nav in self.navdataArray]
        pitchArray= [nav.getRollYawPitch(unit)[2] for nav in self.navdataArray]

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


if __name__ == '__main__':
    FD = FlightData('C:/Users/rss75/Documents/GitHub/ta-shop/source/ardrone/nodejs/flight-data/ardrone-flight-test3.json')
    FD.plotRollYawPitch('deg')