from mpu6050.mpu6050_regist import MPU6050
from time import sleep
import serial

# Opening serial communication to arduino
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.8)
ser.write(b'Serial initialized')

# Arming motor delay
sleep(10)

# Data acquisition iteration
min_speed = 100
max_speed = 600

for speed in range(min_speed, max_speed, 25):
    print('Sending throttle value',speed,'...', end='\r')
    
    # Sending speed payload to 
    payload = str(speed) + 's'
    ser.write(bytes(payload, 'ascii'))
    print(ser.readline())
    sleep(4)

    print('Sent throttle value',speed,', start measurement!', end='\r')
    sleep(10)
    print('Done measurement with throttle', speed, end='\n')