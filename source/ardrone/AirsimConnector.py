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
import threading
import time
from datetime import datetime
import statistics
import math

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

# Airsim import
import setup_path
import airsim

# Custom
from navdataProc import FlightData

class AirsimConnector:
    # Attributes
    flight_plan = []
    flight_saved = []
    last_id = 0
    stop_record = False

    def __init__(self, control=True, verbose=False):
        self.verbose = verbose

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        if control:
            self.client.enableApiControl(True)
            self.client.armDisarm(True)

        print('Client ready!')

    ### Flight Planning ###
    def loadFlightPlan(self, flight_array, control_type):
        def addNewFlightPlan():
            self.flight_plan.append({
                'type': '',
                'data': [],
            })

            last_id = len(self.flight_plan) - 1
            return last_id

        if control_type == 'velocity_async':
            flight_id = addNewFlightPlan()

            self.flight_plan[flight_id]['type'] = 'velocity_async'
            
            flight_data = [nav.getVelocity() for nav in flight_array]
            zAlt = [nav.getAltitude() for nav in flight_array]

            zAltPrev = flight_array[0].getAltitude()
            for nav in flight_array:
                vx, vy, vzx = nav.getVelocity()
                zAlt = nav.getAltitude()
                td = nav.getTimeDelta()
                
                vz = ((zAltPrev - zAlt) / td) * 100
                zAltPrev = zAlt

                self.flight_plan[flight_id]['data'].append({
                    'vx': vx/1000,
                    'vy': vy/1000,
                    'vz': vz/1000,
                    'td': nav.getTimeDelta()/1000
                })

            return flight_id

        elif control_type == 'velocity_z':
            flight_id = addNewFlightPlan()

            self.flight_plan[flight_id]['type'] = 'velocity_z'
            
            flight_data = [nav.getVelocity() for nav in flight_array]
            zAlt = [nav.getAltitude() for nav in flight_array]

            zAltPrev = flight_array[0].getAltitude()
            for nav in flight_array:
                vx, vy, vzx = nav.getVelocity()
                zAlt = nav.getAltitude()
                td = nav.getTimeDelta()

                self.flight_plan[flight_id]['data'].append({
                    'vx': vx/1000,
                    'vy': vy/1000,
                    'z' : zAlt,
                    'td': nav.getTimeDelta()/1000
                })

            return flight_id

        elif control_type == 'rollpitchyaw_z':
            flight_id = addNewFlightPlan()

            self.flight_plan[flight_id]['type'] = 'rollpitchyaw_z'
            
            flight_data = [nav.getVelocity() for nav in flight_array]
            zAlt = [nav.getAltitude() for nav in flight_array]

            zAltPrev = flight_array[0].getAltitude()
            initialYaw = flight_array[0].getRollYawPitch()[1]
            for nav in flight_array:
                r, y, p = nav.getRollYawPitch()
                zAlt = nav.getAltitude()
                td = nav.getTimeDelta()

                self.flight_plan[flight_id]['data'].append({
                    'roll': (r * math.pi / 180),
                    'pitch': (p * math.pi / 180),
                    'yaw' : (y * math.pi / 180) - initialYaw,
                    'z' : zAlt,
                    'td': nav.getTimeDelta()/1000
                })

            return flight_id
        
        else:
            print('Unknown control type')
    

    def executeFlightPlan(self, flight_id=None, record=True):
        def calcDuration(td):
            duration = max(0, td - ((time.time_ns() - self.loop_time) / 1000000000))
            #print(duration)
            #print(td, (time.time_ns() - loop_time) / (10 ** 9), duration)
            self.loop_time = time.time_ns()

            return duration

        control_type = self.flight_plan[flight_id]['type']
        print('\n-----------------')
        print('Executing Flight Plan with id:', flight_id, 'and control type:', control_type)

        # Get last flight id
        if flight_id == None:
            flight_id = self.last_id
        
        if control_type == 'velocity_async':
            # Takeoff
            self.takeoff()
            time.sleep(5)
            
            # Maneuver
            i = 0
            start_time = time.time_ns()
            self.loop_time = time.time_ns()
            for dataPoint in self.flight_plan[flight_id]['data']:
                i += 1
                print('Data points {}/{}'.format(i, len(self.flight_plan[flight_id]['data'])), end='\r')
                self.client.moveByVelocityAsync(
                    vx=dataPoint['vx'],
                    vy=dataPoint['vy'],
                    vz=dataPoint['vz'],
                    duration=dataPoint['td']
                ).join()
            
            print('\nFlight plan executed in {:.2f}s'.format((time.time_ns() - start_time)/1000000000))

        elif control_type == 'velocity_z':
            # Takeoff
            self.takeoff()
            time.sleep(5)
            
            # Maneuver
            initial_pos = self.getPosition()
            print(initial_pos)
            i = 0
            start_time = time.time_ns()
            self.loop_time = time.time_ns()
            for dataPoint in self.flight_plan[flight_id]['data']:
                i += 1
                print('Data points {}/{}, z: {}'.format(i, len(self.flight_plan[flight_id]['data']), dataPoint['z']), end='\r')
                self.client.moveByVelocityZAsync(
                    vx=dataPoint['vx']*2,
                    vy=dataPoint['vy']*2,
                    z=-dataPoint['z'],
                    duration=dataPoint['td']
                ).join()
            
            print('\nFlight plan executed in {:.2f}s'.format((time.time_ns() - start_time)/1000000000))

        elif control_type == 'rollpitchyaw_z':
            # Takeoff
            self.takeoff()
            time.sleep(5)
            
            # Maneuver
            initial_pos = self.getPosition()
            i = 0
            start_time = time.time_ns()
            self.loop_time = time.time_ns()
            for dataPoint in self.flight_plan[flight_id]['data']:
                i += 1
                print('Data points {}/{}, z: {}'.format(i, len(self.flight_plan[flight_id]['data']), dataPoint['z']), end='\r')
                self.client.moveByRollPitchYawZAsync(
                    roll=dataPoint['roll'],
                    pitch=dataPoint['pitch'],
                    yaw=dataPoint['yaw'],
                    z=-dataPoint['z'],
                    duration=dataPoint['td']
                ).join()
            
            print('\nFlight plan executed in {:.2f}s'.format((time.time_ns() - start_time)/1000000000))


    def takeoff(self, timeout_sec=6):
        self.client.takeoffAsync(timeout_sec=6).join()


    def moveDeltaPositionAsync(self, x_delta, y_delta, z_delta, velocity):
        initial_pos = self.getPosition()

        return self.client.moveToPositionAsync(
            initial_pos[0] + x_delta, 
            initial_pos[1] + y_delta, 
            initial_pos[2] + z_delta, 
            velocity
        )

    ### Flight Recording ###
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
    

    def loadSavedFlight(self, filename):
        flight_arr = np.genfromtxt(filename, delimiter=',', dtype='float')
        
        self.flight_saved.append({
            'filename': filename,
            'data': flight_arr
        })

        record_id = len(self.flight_saved) - 1

        return record_id


    def plotPositionSavedFlight(self, record_id, min_data=0, max_data=-1, x=True, y=True, z=True):
        flight_arr = self.flight_saved[record_id]['data'][min_data:max_data]

        t0 = flight_arr[0,0]
        time_array = [(t - t0) / 1000000000 for t in flight_arr[:,0]]
        pos_x = flight_arr[:,17]
        pos_y = flight_arr[:,18]
        pos_z = flight_arr[:,19]

        # Subplot 1 : Pos X
        plt.subplot(311)
        plt.plot(time_array, pos_x)
        plt.title('Position X over time')
        plt.grid(True)

        # Subplot 2 : Pos Y
        plt.subplot(312)
        plt.plot(time_array, pos_y)
        plt.title('Position Y over time')
        plt.grid(True)

        # Subplot 3 : Pos Z
        plt.subplot(313)
        plt.plot(time_array, pos_z)
        plt.title('Position Z over time')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


    def plotOrientationSavedFlight(self, record_id, min_data=0, max_data=-1, x=True, y=True, z=True):
        flight_arr = self.flight_saved[record_id]['data'][min_data:max_data]

        t0 = flight_arr[0,0]
        time_array = [(t - t0) / 1000000000 for t in flight_arr[:,0]]
        ori_x_rad = flight_arr[:,14]
        ori_y_rad = flight_arr[:,15]
        ori_z_rad = flight_arr[:,16]

        ori_x = [o * 180 / (2 * math.pi) for o in ori_x_rad]
        ori_y = [o * 180 / (2 * math.pi) for o in ori_y_rad]
        ori_z = [o * 180 / (2 * math.pi) for o in ori_z_rad]


        # Subplot 1 : Roll
        plt.subplot(311)
        plt.plot(time_array, ori_x)
        plt.title('Orientation X over time')
        plt.ylim(-30,30)
        plt.grid(True)

        # Subplot 2 : Pitch
        plt.subplot(312)
        plt.plot(time_array, ori_y)
        plt.title('Orientation Y over time')
        plt.ylim(-30,30)
        plt.grid(True)

        # Subplot 3 : Yaw
        plt.subplot(313)
        plt.plot(time_array, ori_z)
        plt.title('Orientation Z over time')
        plt.ylim(-30,30)
        plt.grid(True)

        plt.tight_layout()
        plt.show()


    def plotVelocitySavedFlight(self, record_id, min_data=0, max_data=-1, x=True, y=True, z=True):
        flight_arr = self.flight_saved[record_id]['data'][min_data:max_data]

        t0 = flight_arr[0,0]
        time_array = [(t - t0) / 1000000000 for t in flight_arr[:,0]]
        vel_x = flight_arr[:,10]
        vel_y = flight_arr[:,11]
        vel_z = flight_arr[:,12]

        # Subplot 1 : Roll
        plt.subplot(311)
        plt.plot(time_array, vel_x)
        plt.title('Velocity X over time')
        plt.grid(True)

        # Subplot 2 : Pitch
        plt.subplot(312)
        plt.plot(time_array, vel_y)
        plt.title('Velocity Y over time')
        plt.grid(True)

        # Subplot 3 : Yaw
        plt.subplot(313)
        plt.plot(time_array, vel_z)
        plt.title('Velocity Z over time')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


    def recordFlightData(self, state=True, imu=False, barometer=False, magnetometer=False, gps=False,
                            distance=False, lidar=False, lidar_seg=False, filename=None):

        print('[o] Start recording')
        self.fd_array = np.array([]).reshape(0,20)

        # Loop until stop_threads
        i = 0
        while not self.stop_record:
            fd = self.getFlightData(state=state, imu=False, barometer=barometer, magnetometer=magnetometer, gps=gps,
                                        distance=distance, lidar=lidar, lidar_seg=lidar_seg)['state']
            
            fd_list = [
                fd.timestamp,                                           # 0
                fd.kinematics_estimated.angular_acceleration.x_val,
                fd.kinematics_estimated.angular_acceleration.y_val,
                fd.kinematics_estimated.angular_acceleration.z_val,
                fd.kinematics_estimated.angular_velocity.x_val,
                fd.kinematics_estimated.angular_velocity.y_val,         # 5
                fd.kinematics_estimated.angular_velocity.z_val,
                fd.kinematics_estimated.linear_acceleration.x_val,
                fd.kinematics_estimated.linear_acceleration.y_val,
                fd.kinematics_estimated.linear_acceleration.z_val,
                fd.kinematics_estimated.linear_velocity.x_val,          # 10
                fd.kinematics_estimated.linear_velocity.y_val,
                fd.kinematics_estimated.linear_velocity.z_val,
                fd.kinematics_estimated.orientation.w_val,
                fd.kinematics_estimated.orientation.x_val,
                fd.kinematics_estimated.orientation.y_val,              # 15
                fd.kinematics_estimated.orientation.z_val,
                fd.kinematics_estimated.position.x_val,
                fd.kinematics_estimated.position.y_val,
                fd.kinematics_estimated.position.z_val,                 # 19
            ]

            self.fd_array = np.append(self.fd_array, np.array(fd_list).reshape(1,20), axis=0)

            i += 1
            print('[o] Recording {} data points'.format(i), end='\r')

            if i > 15000:
                self.stop_record = True
        
        if self.stop_record:
            if filename == None:
                filename = str(datetime.now().strftime('%y_%m_%d_%H_%M_%S')) + '.csv'

            print('\n[o] Flight data recording terminated, saving to', filename)
            print(self.fd_array.shape)
            np.savetxt(filename, self.fd_array, fmt='%.4f', delimiter=',')
            print('[o] Flight Data record saved to', filename)


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

        plt.show()


    def mainThreaded(self):
        '''
            Run async maneuver on main thread
            Run flight data record on branch thread
        '''
        self.stop_record = False
        """ record_thread = threading.Thread(target=self.recordFlightData)
        record_thread.start() """

        # Start flight
        print('\nTakeoff')
        self.takeoff()
        #print(self.getFlightData(True, False, False, False, False, False, False, False)['state'].kinematics_estimated)
        print('\nMoving to 15,15,-15')
        self.moveDeltaPositionAsync(20, 20, -20, 1).join()
        #self.recordFlightData()


if __name__ == '__main__':
    FD = FlightData('C:/Users/rss75/Documents/GitHub/ta-shop/source/ardrone/nodejs/flight-data/jul_6/flight_1594006229371.json')
    navdataArray = FD.navdataArray[150:-250]
    
    AirConn = AirsimConnector()
    f_id = AirConn.loadFlightPlan(flight_array=navdataArray, control_type='velocity_z')
    print('Saved flight id: ', f_id, 'with', len(AirConn.flight_plan[f_id]['data']), 'points')

    # Flight
    time.sleep(5)
    AirConn.executeFlightPlan(flight_id=f_id)

    #print(airsim.to_quaternion(30,30,30))
    
    # Plot Recorded
    #rec_id = AirConn.loadSavedFlight('20_07_10_13_59_33.csv')
    #AirConn.plotPositionSavedFlight(record_id=rec_id, min_data=7600, max_data=10500)
    #AirConn.plotOrientationSavedFlight(record_id=rec_id, min_data=7600, max_data=10500)
    #AirConn.plotVelocitySavedFlight(record_id=rec_id, min_data=7600, max_data=10500)
    
    """ FD = FlightData('C:/Users/rss75/Documents/GitHub/ta-shop/source/ardrone/nodejs/flight-data/jul_6/flight_1594006229371.json')
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

    #imu_data = client.getImuData()
    #print(imu_data.angular_velocity.x_val)

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
        i += 1 """
