'''
ARDrone navdata JSON parser
Structure:
{
    header:
    droneState:
        flying
        videoEnabled
        visionEnabled
        controlAlgorithm
        altitudeControlAlgorithm
        startButtonState
        controlCommandAck
        cameraReady
        travellingEnabled
        usbReady
        navdataDemo
        navdataBootstrap
        motorProblem
        communicationLost
        softwareFault
        lowBattery
        userEmergencyLanding
        timerElapsed
        MagnometerNeedsCalibration
        anglesOutOfRange
        tooMuchWind
        ultrasonicSensorDeaf
        cutoutDetected
        picVersionNumberOk
        atCodecThreadOn
        navdataThreadOn
        videoThreadOn
        acquisitionThreadOn
        controlWatchdogDelay
        adcWatchdogDelay
        comWatchdogProblem
        emergencyLanding
    sequenceNumber
    visionFlag
    demo
        controlState
        flyState
        batteryPercentage
        rotation
            frontBack
            pitch
            theta
            y
            leftRight
            roll
            phi
            x
            clockwise
            yaw
            psi
            z
        frontBackDegrees
        leftRightDegrees
        clockwiseDegrees
        altitude
        altitudeMeters
        velocity
            x
            y
            z
        xVelocity
        yVelocity
        zVelocity
    .
    .
    . (not useful)
    .
    timeDelta
}
'''

import numpy as np
import json

class NavData:
    def __init__(self, navdata):
        self.header = navdata['demo']
        
        self.demo = navdata['demo']
        self.demo.batteryPercentage = self.demo['batteryPercentage']

def getDataFromFile(filename):
    with open(filename) as file:
        fileContents = file.read()
        fileArr = fileContents.splitlines()
    return fileArr

def getJSONArray(filename):
    dataArrayString = getDataFromFile(filename)
    dataArray = [json.loads(f) for f in dataArrayString]
    return dataArray

if __name__ == '__main__':
    flightDataRaw = getJSONArray('/home/stoorm/github/ta-shop/source/ardrone/nodejs/flight-data/flight_1593945457322.json')
    print('Flight data recorded {} steps'.format(len(flightDataRaw)))
    print(json.dumps(flightDataRaw[len(flightDataRaw)//2], indent=2))