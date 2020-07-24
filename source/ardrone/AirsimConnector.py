'''
Sign convention: 
    pitch:
        (+) front up
        (-) front down
    roll:
        (+) right down
        (-) right up
    yaw:
        (+) cw
        (-) ccw
'''


import sys
from navdataProc import FlightData
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import time

# Airsim import
import setup_path
import airsim

class AirsimConnector:
    # Attributes
    flight_plan = []
    last_id = 0

    def __init__(self, verbose=False):
        self.verbose = verbose

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        print('Client ready!')


    def loadFlightPlan(self, flight_array, control_type):
        def addNewFlightPlan():
            self.flight_plan.append({
                'type': '',
                'data': [],
            })

            last_id = len(self.flight_plan) - 1
            return last_id

        if control_type == 'velocity_z':
            flight_id = addNewFlightPlan()

            self.flight_plan[flight_id]['type'] = 'velocity_z'

            zAltPrev = flight_array[0].getAltitude()
            for nav in flight_array:
                vx, vy, = nav.getVelocity()
                zAlt = nav.getAltitude()
                td = nav.getTimeDelta()
                
                vz = ((zAltPrev - zAlt) / td) * 100
                zAltPrev = zAlt

                self.flight_plan[flight_id]['data'].append({
                    'vx': vx,
                    'vy': vy,
                    'vz': vz,
                    'td': nav.getTimeDelta()/1000
                })

            return flight_id
        
        else:
            print('Unknown control type')
    

    def executeFlightPlan(self, control_type, flight_id=None, record=True):
        def calcDuration(self, td):
            duration = td - (time.clock_gettime_ns(time.CLOCK_REALTIME) - loop_time) / 1000000000
            start_time = time.clock_gettime_ns(time.CLOCK_REALTIME)

            return duration

        print('\n-----------------')
        print('Executing Flight Plan with id:', flight_id, 'and control type:', control_type)

        # Get last flight id
        if flight_id == None:
            flight_id = self.last_id
        
        if control_type == 'velocity_z':
            # Takeoff
            self.takeoff()
            
            # Maneuver
            i = 0
            start_time = time.clock_gettime_ns(time.CLOCK_REALTIME)
            loop_time = time.clock_gettime_ns(time.CLOCK_REALTIME)
            for dataPoint in self.flight_plan[flight_id]['data']:
                i += 1
                print('Data points {}/{}'.format(i, self.last_id+1), end='\r')
                self.client.moveByVelocityAsync(
                    vx=dataPoint['vx'],
                    vy=dataPoint['vy'],
                    vz=dataPoint['vz'],
                    duration=calcDuration(dataPoint['td'])
                ).join()
            
            print('\nFlight plan executed in {:.2f}s'.format(time.clock_gettime_ns(time.CLOCK_REALTIME) - start_time)/1000000000)


    def takeoff(self, timeout_sec=6):
        self.client.takeoffAsync(timeout_sec=6).join()


    def getFlightData(self, state=True, imu=True, barometer=True, magnetometer=True,
                        gps=True, distance=True, lidar=True, lidar_seg = True):
        
        flight_data = {}

        if state:
            state_data = self.client.getMultirotorState()
            flight_data['state'] = state_data

        if imu:
            imu_data = self.client.getImuData()
            flight_data['imu'] = imu_data
        
        if barometer:
            barometer_data = self.client.getBarometerData()
            flight_data['barometer'] = barometer_data

        if magnetometer:
            magnetometer_data = self.client.getMagnetometerData()
            flight_data['magnetometer'] = magnetometer_data
            
        if distance:
            distance_data = self.client.getDistanceSensorData()
            flight_data['distance'] = distance_data

        return flight_data


    def getPosition(self):
        pos = self.getFlightData(state=True, imu=False, barometer=False, magnetometer=False, gps=False,
                                        distance=False, lidar=False, lidar_seg=False)['state'].kinematics_estimated.position

        return pos.x_val, pos.y_val, pos.z_val
    

    def moveDeltaPositionAsync(self, x_delta, y_delta, z_delta, velocity):
        initial_pos = self.getPosition()

        self.client.moveToPositionAsync(initial_pos[0] + x_delta, initial_pos[1] + y_delta, initial_pos[2] + z_delta, velocity)


    def plotPositionSavedFlight(self, x=True, y=True, z=True):
        '''
            TO DO
        '''
        pass


    def plotPositionAnim(self):
        self.x_pos_array = []
        self.y_pos_array = []
        self.z_pos_array = []

        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.set_xlim3d([-30,30])
        ax.set_ylim3d([-30,30])
        ax.set_zlim3d([-30,30])
        line = ax.plot(self.x_pos_array, self.y_pos_array, self.z_pos_array)[0]

        def update(frame):
            fd = self.getFlightData(state=True, imu=False, barometer=False, magnetometer=False, gps=False,
                                        distance=False, lidar=False, lidar_seg=False)

            self.x_pos_array.append(fd['state'].kinematics_estimated.position.x_val)
            self.y_pos_array.append(fd['state'].kinematics_estimated.position.y_val)
            self.z_pos_array.append(fd['state'].kinematics_estimated.position.z_val)

            if len(self.x_pos_array) > 10000:
                self.x_pos_array = self.x_pos_array[-10000:]
                self.y_pos_array = self.y_pos_array[-10000:]
                self.z_pos_array = self.z_pos_array[-10000:]

            line.set_data(self.x_pos_array, self.y_pos_array)
            line.set_3d_properties(self.z_pos_array)
            print('[o] frame', frame, end='\r')

            return line,

        # 10k data max
        positionAnim = animation.FuncAnimation(fig, update, frames=range(100000), interval=50, blit=True)

        plt.show(block=False)


if __name__ == '__main__':
    AirConn = AirsimConnector()
    AirConn.takeoff()
    print('Plotting')
    AirConn.plotPositionAnim()
    print('Move to pos')
    AirConn.moveDeltaPositionAsync(15, 15, -15, 1)
    

    ''' FD = FlightData('C:/Users/rss75/Documents/GitHub/ta-shop/source/ardrone/nodejs/flight-data/jul_6/flight_1594006229371.json')
    navdataArray = FD.navdataArray

    timeDeltaArray = [nav.getTimeDelta() for nav in navdataArray]
    timeDelta = np.cumsum(timeDeltaArray)
    yawArray = [nav.getRollYawPitch()[1] for nav in navdataArray]

    plt.plot(timeDelta, yawArray)
    plt.show()
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print('Takeoff...')
    #client.takeoffAsync(timeout_sec=6).join()

    imu_data = client.getImuData()
    print(imu_data.angular_velocity.x_val)

    print('Maneuver...')
    i = 0
    time_start = time.perf_counter_ns()
    zAltPrev = navdataArray[0].getAltitude()
    for nav in navdataArray[150:-250]:
        
        r, y, p = nav.getRollYawPitch(unit='rad')
        vx, vy, vz = nav.getVelocity()
        zAlt = nav.getAltitude()
        
        # Move by roll,yaw,pitch and throttle
        client.moveByRollPitchYawThrottleAsync(
            roll=r,
            yaw=y,
            pitch=p,
            throttle=0.8,
            duration=(nav.getTimeDelta()/1000)
        ).join()

        client.moveByRollPitchYawZAsync(
            roll=r,
            yaw=y,
            pitch=p,
            z=zAlt,
            duration=(nav.getTimeDelta()/1000)
        ).join()

        # Move by velocity x,y,z
        client.moveByVelocityAsync(
            vx = vx/1000,
            vy = vy/1000,
            vz = ((zAltPrev - zAlt)/nav.getTimeDelta()) * 1000,
            duration = (nav.getTimeDelta()/1000)
        ).join()
        zAltPrev = zAlt

        print('Maneuver {} / {} in {:.1f} / {:.1f}ms'.format(i, len(navdataArray)+1, (time.perf_counter_ns() - time_start)/1000000, nav.getTimeDelta()), end='\r')
        time_start = time.perf_counter_ns()
        i += 1 '''
