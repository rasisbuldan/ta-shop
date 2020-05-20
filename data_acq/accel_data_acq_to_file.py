import numpy as np
from mpu6050_regist import MPU6050
from datetime import datetime

# MPU6050 object declaration
mpu1 = MPU6050(i2c_addr=0x68, g_range='2g', sample_rate=1000, accel_ms=1, temp_ms=1)
#mpu2 = MPU6050(i2c_addr=0x69, g_range='2g', sample_rate=1000, accel_ms=1, temp_ms=1)

# Data acquisition iteration
n_data = 10
n_sample = 10000

for i in range(n_data):
    print('Measuring sample data',i+1,'at',datetime.now().strftime('%H:%M:%S'))
    # Array declaration
    accel_arr = np.array([]).reshape(0,4)

    for j in range(n_sample):
        accel_data = np.array(mpu1.get_accel_data()).reshape(1,4)
        accel_arr = np.append(accel_arr, accel_data, axis=0)

    save_time = str(datetime.now().strftime('%H_%M_%S'))
    np.savetxt('data_acq/accel_data/accel_' + save_time + '.txt', accel_arr, delimiter=',', fmt='%.8f')
