import setup_path 
import airsim

import numpy as np
import os

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

state = client.getMultirotorState()
client.takeoffAsync(timeout_sec=5).join()

while True:
    # Get IMU data
    imu_data = client.getImuData()
    #print('imu data: ', imu_data)

    client.moveByRollPitchYawThrottleAsync(
        roll=0.1,
        yaw=0,
        pitch=-0.1,
        throttle=0.6,
        duration=0.2,
    )