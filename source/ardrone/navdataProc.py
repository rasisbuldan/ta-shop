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

    def getTimestamp(self):
        return self.navdata['timestamp']

    def getAltitude(self):
        return self.navdata['demo']['altitude']

    def getTranslation(self):
        x_trans = self.navdata['demo']['drone']['camera']['translation']['x']
        y_trans = self.navdata['demo']['drone']['camera']['translation']['y']
        z_trans = self.navdata['demo']['drone']['camera']['translation']['z']

        return [x_trans, y_trans, z_trans]

    def getMotorPWM(self):
        try:
            return self.navdata['pwm']['motors']
        except:
            return [0,0,0,0]
        
    def getSatMotors(self):
        return self.navdata['pwm']['satMotors']
    
    def getMotorCurrents(self):
        return self.navdata['pwm']['motorCurrents']


class FlightData:
    def __init__(self, filename, filetype='new'):
        self.filetype = filetype

        if self.filetype == 'old':
            self.navdataArray = [NavData(n) for n in getJSONArray(filename)]
            self.timeDeltaArray = [nav.getTimeDelta() for nav in self.navdataArray]
            self.timeArray = np.cumsum(self.timeDeltaArray)

        elif self.filetype == 'new':
            self.navdataArray = [NavData(n) for n in getJSONArray(filename)]
            self.timeraw = [nav.getTimestamp() for nav in self.navdataArray]

            # Calculate timestamp
            t0 = self.timeraw[0]
            self.timeArray = []
            for t in self.timeraw:
                self.timeArray.append(t - t0)

            # Calculate timedelta
            t0 = self.timeArray[0]
            self.timeDeltaArray = []
            for t in self.timeArray:
                self.timeDeltaArray.append(t - t0)
                t0 = t
    

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

        self.plotSubplot(
            n_plot=3,
            x_data=self.timeArray,
            y_data=[rollArray,yawArray,pitchArray],
            title=['Roll','Yaw','Pitch']
        )

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

        self.plotSubplot(
            n_plot=3,
            x_data=tTransArray,
            y_data=[xTransArray,yTransArray,zTransArray],
            title=['X Position','Y Position','Z Position']
        )


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

            line.set_data(x_data, y_data)
            line.set_3d_properties(z_data)
            return line,

        anim = animation.FuncAnimation(fig, update, frames=range(len(self.navdataArray)-250), 
                                        interval=5, blit=False)

        plt.show()


    def plotAltitude(self):
        altitudeArray = [nav.getAltitude() for nav in self.navdataArray]

        plt.plot(self.timeArray, altitudeArray)
        plt.grid(True)
        plt.show()


    def plotOrientationVelocity(self, rypunit='deg'):
        # Orientation
        rollArray = [nav.getRollYawPitch(rypunit)[0] for nav in self.navdataArray]
        yawArray = [(nav.getRollYawPitch(rypunit)[1]) for nav in self.navdataArray]
        pitchArray = [nav.getRollYawPitch(rypunit)[2] for nav in self.navdataArray]

        # Velocity
        vxArray= [(nav.getVelocity()[0]/1000) for nav in self.navdataArray]
        vyArray= [(nav.getVelocity()[1]/1000) for nav in self.navdataArray]
        vzArray= [(nav.getVelocity()[2]/1000) for nav in self.navdataArray]

        self.plotSubplot(
            n_plot=6,
            x_data=self.timeArray,
            y_data=[rollArray,vxArray,yawArray,vyArray,pitchArray,vzArray],
            title=['Roll','X Velocity','Yaw','Y Velocity','Pitch','Z Velocity']
        )


    def plotPWM(self):
        pwm = [nav.getMotorPWM() for nav in self.navdataArray]
        pwm1 = [p[0] for p in pwm]
        pwm2 = [p[1] for p in pwm]
        pwm3 = [p[2] for p in pwm]
        pwm4 = [p[3] for p in pwm]
        print([(pwm1[i] - pwm2[i]) for i in range(len(pwm1))])
        
        self.plotSubplot(
            n_plot=4,
            x_data=self.timeArray,
            y_data=[pwm1,pwm2,pwm3,pwm4],
            title=['PWM1','PWM2','PWM3','PWM4']
        )


    def plotMotorCurrent(self):
        mc = [nav.getMotorCurrents() for nav in self.navdataArray]
        mc1 = [c[0] for c in mc]
        mc2 = [c[1] for c in mc]
        mc3 = [c[2] for c in mc]
        mc4 = [c[3] for c in mc]

        self.plotSubplot(
            n_plot=4,
            x_data=self.timeArray,
            y_data=[mc1,mc2,mc3,mc4],
            title=['MC1','MC2','MC3','MC4']
        )


    def plotSubplot(self, n_plot, x_data, y_data, title, grid=True, layout=None):
        if layout == None:
            subplt_num = (n_plot*100) + 11
        else:
            subplt_num = layout
            
        for p_num in range(n_plot):
            print(subplt_num + p_num)
            plt.subplot(subplt_num + p_num)
            plt.plot(x_data, y_data[0])
            plt.title(title[0])
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Ascent
    FD = FlightData('D:/Cloud/Google Drive/Tugas Akhir/data/flight-data/jul_28/flight_new_1595916173725_hover30s_4.json')
    
    print('Processing Flight Data with {} points'.format(len(FD.navdataArray)))
    print('Battery Percentage:',FD.navdataArray[0].getBatteryPercentage())
    #FD.plotOrientationVelocity('deg')
    #FD.plotAltitude()
    #FD.plotTrajectory()
    #FD.plotTimeDelta()
    #FD.plotTranslationAscent()
    #FD.plotTranslationAnim(n_data=len(FD.navdataArray)-20)
    #FD.plotTranslationAscent2D()
    FD.plotPWM()
    #FD.plotMotorCurrent()