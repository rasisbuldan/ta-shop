from time import sleep
import serial

ser = serial.Serial('COM9', 115200, timeout=0.6)
ser.write(b'hello')
#print(ser.readline())
sleep(10)

for i in range(200,600,5):
    print(i)
    payload = str(i) + 's'
    ser.write(bytes(payload, 'ascii'))
    print(ser.readline())
    sleep(.5)