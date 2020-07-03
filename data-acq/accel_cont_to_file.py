'''
Continuous vibration measurement at constant speed
Sending throttle value over serial port /dev/ttyACM0
Saving file to data-acq/accel-data-cont
'''

import numpy as np
from mpu6050.mpu6050_regist import MPU6050
from datetime import datetime
from time import sleep
import serial
import sys

# Data acquisition parameter
const_speed = 1048
n_sample = 10000

# MPU6050 object declaration
mpu1 = MPU6050(i2c_addr=0x68, g_range='4g', sample_rate=1000, accel_ms=1, temp_ms=1)

# Opening serial communication to arduino
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.8)
ser.write(b'Serial initialized')

# Arming motors
sleep(10)

# Sending speed payload to serial
payload = str(const_speed) + 's'
ser.write(bytes(payload, 'ascii'))
ser_read = ser.readline()
sleep(4)

# Run infinite loop
while True:
    # Array declaration
    accel_arr = np.array([]).reshape(0,4)

    # Get measurement time (time start)
    measure_time = str(datetime.now().strftime('%y_%m_%d_%H_%M_%S'))
    print('Starting vibration measurement with speed',const_speed,'at',measure_time)

    # Get n_sample acceleration point
    try:
        for j in range(n_sample):
            accel_data = np.array(mpu1.get_accel_data()).reshape(1,4)
            accel_arr = np.append(accel_arr, accel_data, axis=0)

        # Save to file
        np.savetxt('data-acq/accel-data-cont2/accel_cont_' + measure_time + '_' + str(const_speed) + '.txt', accel_arr, delimiter=',', fmt='%.8f')

    # Quit on keyboard interrupt
    except KeyboardInterrupt:
        sys.exit()

    # Error occured while reading accel, turn off motor and print error message
    except Exception as e:
        ser.write(bytes('48s', 'ascii'))
        print('Error occured, turning off motor')
        print(repr(e))
