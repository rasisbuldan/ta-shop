import numpy as np
from mpu6050.mpu6050_regist import MPU6050
from datetime import datetime
from time import sleep
import serial

# Serial communication to arduino
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.8)
ser.write(b'Serial initialized')

# Arming motors
sleep(10)

# MPU6050 object declaration
mpu1 = MPU6050(i2c_addr=0x68, g_range='2g', sample_rate=1000, accel_ms=1, temp_ms=1)

# Data acquisition iteration
min_speed = 200
max_speed = 800
n_sample = 10000

for speed in range(min_speed, max_speed, 10):
    measure_time = str(datetime.now().strftime('%H_%M_%S'))
    print('Measuring sample data with speed',speed,'at',measure_time)
    
    # Sending speed payload to 
    payload = str(speed) + 's'
    ser.write(bytes(payload, 'ascii'))
    print(ser.readline())
    sleep(4)

    # Array declaration
    accel_arr = np.array([]).reshape(0,4)

    print('Starting measuring vibration...')
    for j in range(n_sample):
        accel_data = np.array(mpu1.get_accel_data()).reshape(1,4)
        accel_arr = np.append(accel_arr, accel_data, axis=0)

    # Save to file
    np.savetxt('data_acq/accel_data/accel_speed_' + measure_time + '_' + speed + '.txt', accel_arr, delimiter=',', fmt='%.8f')
