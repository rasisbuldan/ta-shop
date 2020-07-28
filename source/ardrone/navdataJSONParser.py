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
    flightDataRaw = getJSONArray('D:/Cloud/Google Drive/Tugas Akhir/data/flight-data/jul_28/to_db/flight_new_1595914928926_hover_spin.json')
    print('Flight data recorded {} steps'.format(len(flightDataRaw)))
    print(json.dumps(flightDataRaw[0]['droneState'], indent=2))