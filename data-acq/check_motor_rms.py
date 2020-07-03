import numpy as np
import math
import time

def getFileRMS(filename):
    accelData = np.genfromtxt(filename, delimiter=",")
    tDelta = accelData[:,0]
    xAccel = accelData[:,1]
    yAccel = accelData[:,2]
    zAccel = accelData[:,3]

    xRMS = math.sqrt(np.sum(np.square(xAccel))/xAccel.shape[0])
    yRMS = math.sqrt(np.sum(np.square(yAccel))/yAccel.shape[0])
    zRMS = math.sqrt(np.sum(np.square(zAccel))/zAccel.shape[0])

    return xRMS, yRMS, zRMS


if __name__ == '__main__':
    xRMS, yRMS, zRMS = getFileRMS(
        filename='/home/stoorm/Downloads/accel_cont_20_06_30_10_51_22_1048.txt'
    )
    print('x: ', xRMS)
    print('y: ', yRMS)
    print('z: ', zRMS)
    