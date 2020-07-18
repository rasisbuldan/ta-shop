'''
Vibration measurement with sweep throttle value
from min_speed to max_speed, each throttle correspond
to one measurement file containing n_sample data point

Sending throttle value over serial port /dev/ttyACM0
Saving file to data-acq/accel-data
'''

import numpy as np
from mpu6050.mpu6050_regist import MPU6050
from datetime import datetime, timedelta
from time import sleep
import serial
import math
import time
import sys

# Data acquisition iteration
min_speed = 100
max_speed = 1800
n_sample = 10000

# MPU6050 object declaration
mpu1 = MPU6050(i2c_addr=0x68, g_range='16g', sample_rate=1000, include_ina=True)

# Opening serial communication to arduino
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.8)
ser.write(bytes('48s', 'ascii'))

# Arming motor delay and reset mpu6050 offset
sleep(10)
mpu1.reset_offset()
print(mpu1.get_accel_data())

def motorIsOn(accelData):
    xAccel = accelData[:,1]
    yAccel = accelData[:,2]
    zAccel = accelData[:,3]

    xRMS = math.sqrt(np.sum(np.square(xAccel))/xAccel.shape[0])
    yRMS = math.sqrt(np.sum(np.square(yAccel))/yAccel.shape[0])
    zRMS = math.sqrt(np.sum(np.square(zAccel))/zAccel.shape[0])
    #print(xRMS, yRMS, zRMS)

    return (xRMS >= 0.15 and yRMS >= 0.15 and zRMS >= 0.2)

def getAccelSample(n_sample):
    accel_arr = np.array([]).reshape(0,4)
    time_prev = time.clock_gettime_ns(time.CLOCK_REALTIME)
    for s in range(n_sample):
        accel = mpu1.get_accel_data()
        accel_data = np.array([time.clock_gettime_ns(time.CLOCK_REALTIME) - time_prev, accel[0], accel[1], accel[2]]).reshape(1,4)
        accel_arr = np.append(accel_arr, accel_data, axis=0)
        time_prev = time.clock_gettime_ns(time.CLOCK_REALTIME)

    return accel_arr

def armMotor():
    ser.write(bytes('48s', 'ascii'))
    ser_read = ser.readline()
    sleep(6)

def turnOnMotor(throttle):
    print('Turning on motor with throttle {}'.format(throttle))
    payload = str(throttle) + 's'
    ser.write(bytes(payload, 'ascii'))
    sleep(5)

    # Wait until motor is started successfully
    while not motorIsOn(getAccelSample(1000)):
        print('[!] Motor is not started successfully, re-arming at', datetime.now().strftime('%H:%M:%S %d-%m-%y'))
        ser.close()
        ser.open()
        # Re-arming motor
        armMotor()

        print('[!] Sending throttle to motor', end='\r')
        # Sending speed payload to serial
        ser.write(bytes(payload, 'ascii'))
        ser_read = ser.readline()
        sleep(5)

    print('[o] Motor is started successfully at', datetime.now().strftime('%H:%M:%S %d-%m-%y'))

########################
##### Main Program #####
########################

turnOnMotor(throttle=min_speed)
for speed in range(min_speed, max_speed, 10):
    measure_time = str(datetime.now().strftime('%H_%M_%S'))
    print('Measuring sample data with speed',speed,'at',measure_time)
    
    # Sending speed payload to 
    payload = str(speed) + 's'
    ser.write(bytes(payload, 'ascii'))
    print(ser.readline())
    sleep(4)

    # Array declaration
    accel_arr = getAccelSample(n_sample)

    # Save to file
    np.savetxt('data-acq/accel-data-varspeed-1/accel_speed_' + measure_time + '_' + str(speed) + '.txt', accel_arr, delimiter=',', fmt='%.5f')
