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
from navDataProc import FlightData
import numpy as np
import matplotlib.pyplot as plt

# Airsim import
import setup_path
import airsim


if __name__ == '__main__':
    FD = FlightData('C:/Users/rss75/Documents/GitHub/ta-shop/source/ardrone/nodejs/flight-data/ardrone-flight-test3.json')
    navdataArray = FD.navdataArray

    timeDeltaArray = [nav.getTimeDelta() for nav in navdataArray]
    timeDelta = np.cumsum(timeDeltaArray)
    yawArray = [nav.getRollYawPitch()[1] for nav in navdataArray]

    ''' plt.plot(timeDelta, yawArray)
    plt.show() '''
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print('Takeoff...')
    client.takeoffAsync(timeout_sec=5).join()

    print('Maneuver...')
    for nav in navdataArray:
        r, y, p = nav.getRollYawPitch(unit='rad')
        vx, vy, vz = nav.getVelocity()
        
        # Move by roll,yaw,pitch and throttle
        ''' client.moveByRollPitchYawThrottleAsync(
            roll=r,
            yaw=y,
            pitch=p,
            throttle=0.5,
            duration=(nav.getTimeDelta()/1000)
        ).join() '''

        # Move by velocity x,y,z
        client.moveByVelocityAsync(
            vx=vx/1000,
            vy=vy/1000,
            vz=vz/1000,
            duration=(nav.getTimeDelta()/1000)
        ).join()