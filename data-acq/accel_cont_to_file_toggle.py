'''
Continuous vibration measurement at constant speed
Sending throttle value over serial port /dev/ttyACM0
Saving file to data-acq/accel-data-cont

Version 2
'''

import numpy as np
from mpu6050.mpu6050_regist import MPU6050
from datetime import datetime, timedelta
from time import sleep
import serial
import math
import sys
import board
import adafruit_ina260

# Data acquisition parameter
const_speed = 1400
n_sample = 10000
t_period = 3600  # seconds

# MPU6050 object declaration
mpu1 = MPU6050(i2c_addr=0x68, g_range='8g', sample_rate=1000, accel_ms=1, temp_ms=1)
#mpu1.reset_offset()
#print(mpu1.get_accel_data())

# INA260 object declaration
i2c = board.I2C()
ina260 = adafruit_ina260.INA260(i2c)
ina260.averaging_count = adafruit_ina260.AveragingCount.COUNT_16
ina260.current_conversion_time = adafruit_ina260.ConversionTime.TIME_558_us
ina260.voltage_conversion_time = adafruit_ina260.ConversionTime.TIME_558_us

# Opening serial communication to arduino
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.8)

#ser.write(b'Serial initialized')
ser.write(bytes('48s', 'ascii'))

# Arming motors
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

    return (xRMS >= 0.3 and yRMS >= 0.15 and zRMS >= 0.3)


def getAccelSample(n_sample):
    accel_arr = np.array([]).reshape(0,6)
    for s in range(n_sample):
        accel_data = np.array(mpu1.get_accel_data()).reshape(1,4)
        vi_data = np.array([ina260.voltage, ina260.current]).reshape(1,2)
        accel_data = np.append(accel_data, vi_data, axis=1)
        accel_arr = np.append(accel_arr, accel_data, axis=0)

    return accel_arr

def armMotor():
    ser.write(bytes('48s', 'ascii'))
    ser_read = ser.readline()
    sleep(6)

def turnOnMotor(throttle):
    print('Turning on motor with throttle {}'.format(throttle))

    # Wait until motor is started successfully
    while not motorIsOn(getAccelSample(1000)):
        print('[!] Motor is not started successfully, re-arming at', datetime.now().strftime('%H:%M:%S %d-%m-%y'))
        ser.close()
        ser.open()
        # Re-arming motor
        armMotor()

        print('[!] Sending throttle to motor', end='\r')
        # Sending speed payload to serial
        payload = str(throttle) + 's'
        ser.write(bytes(payload, 'ascii'))
        ser_read = ser.readline()
        sleep(4)

    print('[o] Motor is started successfully at', datetime.now().strftime('%H:%M:%S %d-%m-%y'))


def turnOffMotor():
    print('Turning off motor')

    # Wait until motor is started successfully
    while motorIsOn(getAccelSample(1000)):
        print('[!] Motor is not stopped successfully, re-stopping at', datetime.now().strftime('%H:%M:%S %d-%m-%y'))
        # Re-arming motor
        armMotor()
        ser.write(bytes('50s', 'ascii'))
        ser_read = ser.readline()

    print('[-] Motor is stopped successfully at', datetime.now().strftime('%H:%M:%S %d-%m-%y'))

# Loop variable
last_state_time = datetime.now()
current_state = 'off'
next_state = 'on'

# Run infinite loop
while True:
    # Measure time change
    time_delta = datetime.now() - last_state_time
    #print('td:',time_delta.seconds)
    if time_delta.seconds >= t_period:
        last_state_time = datetime.now()
        if current_state == 'off':
            next_state = 'on'

        else:
            next_state = 'off'

    # Continuous on
    if current_state == 'on' and next_state == 'on':
        # Get measurement time (time start)
        td_hours = time_delta.seconds // 3600
        td_minutes = (time_delta.seconds - (td_hours * 3600)) // 60
        td_seconds = (time_delta.seconds - (td_hours * 3600) - (td_minutes * 60))
        measure_time = str(datetime.now().strftime('%y_%m_%d_%H_%M_%S'))
        print('Starting vibration measurement with throttle',const_speed,'at',measure_time, 
            '| elapsed {}:{}:{}'.format(td_hours, td_minutes, td_seconds))

        # Get n_sample acceleration point
        try:
            accel_arr = getAccelSample(n_sample)

            # Save to file
            np.savetxt('accel-vi-data-cont/accel_cont_' + measure_time + '_' + str(const_speed) + '.txt', accel_arr, delimiter=',', fmt='%.5f')

        # Quit on keyboard interrupt
        except KeyboardInterrupt:
            sys.exit()

        # Error occured while reading accel, turn off motor and print error message
        except Exception as e:
            print('Error occured, turning off motor')
            turnOffMotor()
            print(e)

    # Continuous off
    elif current_state == 'off' and next_state == 'off':
        print('Motor currently off, turning on at', last_state_time + timedelta(seconds=t_period), end='\r')
        sleep(5)

    elif current_state == 'off' and next_state == 'on':
        turnOnMotor(throttle=const_speed)
        last_state_time = datetime.now()
        current_state = 'on'

    elif current_state == 'on' and next_state == 'off':
        turnOffMotor()
        last_state_time = datetime.now()
        current_state = 'off'
