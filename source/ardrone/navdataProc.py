from navdataJSONParser import getDataFromFile, getJSONArray
import math
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


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

    def getAltitude(self):
        return self.navdata['demo']['altitude']

    def getTranslation(self):
        x_trans = self.navdata['demo']['drone']['camera']['translation']['x']
        y_trans = self.navdata['demo']['drone']['camera']['translation']['y']
        z_trans = self.navdata['demo']['drone']['camera']['translation']['z']

        return [x_trans, y_trans, z_trans]

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

    def plotTranslationManeuver(self):
        transArray = [nav.getTranslation() for nav in self.navdataArray[120:-250]]
        xTransArray = [t[0] for t in transArray]
        yTransArray = [-t[1] for t in transArray]
        zTransArray = [-t[2] for t in transArray]

        fig = plt.figure()
        ax = p3.Axes3D(fig)

        lim_offset = 50

        #ax.set_xlim3d([min(xTransArray) - lim_offset, max(xTransArray) + lim_offset])
        #ax.set_ylim3d([min(yTransArray) - lim_offset, max(yTransArray) + lim_offset])
        #ax.set_zlim3d([min(zTransArray) - lim_offset, 2000])
        ax.set_xlim3d([-500, 1500])
        ax.set_ylim3d([-1500, 500])
        ax.set_zlim3d([-500, 1500])

        ax.plot(xTransArray, yTransArray, zTransArray)
        plt.show()

    def plotTranslationAscent(self):
        transArray = [nav.getTranslation() for nav in self.navdataArray]
        xTransArray = [t[0] for t in transArray]
        yTransArray = [-t[1] for t in transArray]
        zTransArray = [-t[2] for t in transArray]

        fig = plt.figure()
        ax = p3.Axes3D(fig)

        lim_offset = 50

        #ax.set_xlim3d([min(xTransArray) - lim_offset, max(xTransArray) + lim_offset])
        #ax.set_ylim3d([min(yTransArray) - lim_offset, max(yTransArray) + lim_offset])
        #ax.set_zlim3d([min(zTransArray) - lim_offset, 2000])
        ax.set_xlim3d([-500, 1500])
        ax.set_ylim3d([-1500, 500])
        ax.set_zlim3d([-500, 1500])

        ax.plot(xTransArray, yTransArray, zTransArray)
        plt.show()

    def plotTranslationAscent2D(self):
        transArray = [nav.getTranslation() for nav in self.navdataArray[150:-250]]
        
        tTransArray = self.timeArray[150:-250]
        xTransArray = [-t[0]/1000 for t in transArray]
        yTransArray = [-t[1]/1000 for t in transArray]
        zTransArray = [t[2]/1000 for t in transArray]

        # Subplot 1 : Pos X
        plt.subplot(311)
        plt.plot(tTransArray, xTransArray)
        plt.title('Position X over time')
        plt.grid(True)

        # Subplot 2 : Pos Y
        plt.subplot(312)
        plt.plot(tTransArray, yTransArray)
        plt.title('Position Y over time')
        plt.grid(True)

        # Subplot 3 : Pos Z
        plt.subplot(313)
        plt.plot(tTransArray, zTransArray)
        plt.title('Position Z over time')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


    def plotTranslationAnim(self, n_data=1):
        transArray = [nav.getTranslation() for nav in self.navdataArray]
        xTransArray = [t[0] for t in transArray]
        yTransArray = [t[1] for t in transArray]
        zTransArray = [t[2] for t in transArray]
        
        x_data = []
        y_data = []
        z_data = []

        lim_offset = 50

        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.set_xlim3d([min(xTransArray) - lim_offset, max(xTransArray) + lim_offset])
        ax.set_ylim3d([min(yTransArray) - lim_offset, max(yTransArray) + lim_offset])
        ax.set_zlim3d([min(zTransArray) - lim_offset, max(zTransArray) + lim_offset])
        line = ax.plot(x_data, y_data, z_data)[0]
        
        def update(frame):
            x_data = xTransArray[:frame]
            y_data = yTransArray[:frame]
            z_data = zTransArray[:frame]
            #print('x:', x_data)
            #print('y:', y_data)
            #print('z:', z_data)

            line.set_data(x_data, y_data)
            line.set_3d_properties(z_data)
            return line,

        anim = animation.FuncAnimation(fig, update, frames=range(len(self.navdataArray)-250), 
                                        interval=5, blit=False)

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

    def plotAltitude(self):
        altitudeArray = [nav.getAltitude() for nav in self.navdataArray]

        plt.plot(self.timeArray, altitudeArray)
        plt.grid(True)
        plt.show()


    def plot(self, rypunit='deg'):
        # Orientation
        rollArray = [nav.getRollYawPitch(rypunit)[0] for nav in self.navdataArray[150:-250]]
        yawArray = [(nav.getRollYawPitch(rypunit)[1]) for nav in self.navdataArray[150:-250]]
        pitchArray = [nav.getRollYawPitch(rypunit)[2] for nav in self.navdataArray[150:-250]]

        # Velocity
        vxArray= [(nav.getVelocity()[0]/1000) for nav in self.navdataArray[150:-250]]
        vyArray= [(nav.getVelocity()[1]/1000) for nav in self.navdataArray[150:-250]]
        vzArray= [(nav.getVelocity()[2]/1000) for nav in self.navdataArray[150:-250]]

        # Subplot 1 : Roll
        plt.subplot(321)
        plt.plot(self.timeArray[150:-250], rollArray)
        plt.title('Roll over time')
        plt.ylabel('Roll ({})'.format(rypunit))
        plt.ylim(-30,30)
        plt.grid(True)

        # Subplot 3 : Yaw
        plt.subplot(323)
        plt.plot(self.timeArray[150:-250], yawArray)
        plt.title('Yaw over time')
        plt.ylabel('Yaw ({})'.format(rypunit))
        plt.ylim(-70,-130)
        plt.grid(True)

        # Subplot 5 : Pitch
        plt.subplot(325)
        plt.plot(self.timeArray[150:-250], pitchArray)
        plt.title('Pitch over time')
        plt.ylabel('Pitch ({})'.format(rypunit))
        plt.ylim(-30,30)
        plt.grid(True)

        # Subplot 2 : X velocity
        plt.subplot(322)
        plt.plot(self.timeArray[150:-250], vxArray)
        plt.title('X velocity over time')
        plt.ylabel('Velocity (m/s)')
        plt.grid(True)

        # Subplot 4 : Y velocity
        plt.subplot(324)
        plt.plot(self.timeArray[150:-250], vyArray)
        plt.title('Y velocity over time')
        plt.ylabel('Velocity (m/s)')
        plt.grid(True)

        # Subplot 6 : Z velocity
        plt.subplot(326)
        plt.plot(self.timeArray[150:-250], vzArray)
        plt.title('Z velocity over time')
        plt.ylabel('Velocity (m/s)')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Ascent
    FD = FlightData('C:/Users/rss75/Documents/GitHub/ta-shop/source/ardrone/nodejs/flight-data/jul_6/flight_1594006229371.json')
    
    print('Processing Flight Data with {} points'.format(len(FD.navdataArray)))
    print('Battery Percentage:',FD.navdataArray[0].getBatteryPercentage())
    FD.plot('deg')
    #FD.plotAltitude()
    #FD.plotTrajectory()
    #FD.plotTimeDelta()
    #FD.plotTranslationAscent()
    FD.plotTranslationAscent2D()
    #FD.plotTranslationAnim(n_data=len(FD.navdataArray)-20)