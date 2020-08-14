from navdataVibProc import NavData, VibData, NavdataVib
import numpy as np
import matplotlib.pyplot as plt
import pymongo
import sys
import time

def combineArray(pwmdata, vibdata, timedelta=False):
    '''
        Combine based on smallest timestamp delta
        (Currently pwmdata to match vibdata timestamp)

        Input format:
        pwmdata = [[timestamp,...], [pwm,...]]
        vibdata = [[timestamp,...], [[vib.x,...],...]]

        Return format:
        [timestamp, pwm, vibx, viby, vibz]
    '''

    tsPWM = pwmdata[0]
    tsVib = vibdata[0]
    pwmArray = pwmdata[1]
    vibArray = vibdata[1]

    print('pwm', len(tsPWM))
    print('vib', len(tsVib))

    # Check for data duplication
    tsPWMUnique = []
    tsPWMUnique = [tsPWMUnique.append(ts) for ts in tsPWM if ts not in tsPWMUnique] 
    tsVibUnique = []
    tsVibUnique = [tsVibUnique.append(ts) for ts in tsVib if ts not in tsVibUnique]
    
    # PWMData duplicate (most likely)
    if len(tsPWMUnique) != len(tsPWM):
        print('Duplicated data (PWM)! {} -> {}'.format(len(tsPWMUnique), len(tsPWM)))
        tsPWM = tsPWM[:len(tsPWMUnique)]
        pwmArray = pwmArray[:len(tsPWMUnique)]
    
    # Vibdata duplicate (not likely, but ok)
    if len(tsVibUnique) != len(tsVib):
        print('Duplicated data (Vib)!')


    # Get average timestamp
    tsPWMAvg = (max(tsPWM) - min(tsPWM)) / len(tsPWM)
    tsVibAvg = (max(tsVib) - min(tsVib)) / len(tsVib)

    print('tsPWM: {} | tsVib: {}'.format(tsPWMAvg, tsVibAvg))
    print('tsPWM: {} - {}'.format(min(tsPWM), max(tsPWM)))
    print('tsVib: {} - {}'.format(min(tsVib), max(tsVib)))
    
    # Interpolate PWM data into Vib data
    if tsVibAvg < tsPWMAvg:
        newPWMArray = []

        for ts in tsVib:

            # Vibration data traversal
            i = len(tsPWM) - 1
            while tsPWM[i] > ts:
                i -= 1
            #print('idx:', i, end='\r')
            
            # Append inrange data into new pwm array
            newPWMArray.append(pwmArray[i])

        # Restructure data
        combinedArray = []
        for i in range(len(tsVib)):
            combinedArray.append([
                tsVib[i]-tsVib[0],
                newPWMArray[i],
                [
                    vibArray[0][i],
                    vibArray[1][i],
                    vibArray[2][i]
                ]
            ])

    # Interpolate vib data into PWM data
    elif tsPWMAvg < tsVibAvg:
        print('tspwm < tsvib')
        # Special case: trim boundary data from tsPWM and pwmArray
        startTrim = 0
        while tsPWM[startTrim] < tsVib[0]:
            startTrim += 1

        stopTrim = -1
        while tsPWM[stopTrim] > tsVib[-1]:
            stopTrim -= 1
        
        tsPWM = tsPWM[startTrim:stopTrim]
        pwmArray = pwmArray[startTrim:stopTrim]

        newVibArray = [[],[],[]]

        print('trim', startTrim, stopTrim)

        for ts in tsPWM:
            i = 0
            try:
                while tsVib[i] < ts:
                    i += 1
            except:
                print(i, ':', tsVib[i-1], '<', ts)
                sys.exit()


            # Append inrange data into new pwm array
            newVibArray[0].append(vibArray[0][i])
            newVibArray[1].append(vibArray[1][i])
            newVibArray[2].append(vibArray[2][i])
        
        # Debug
        #print(len(newVibArray[0]), newVibArray[0][:5])
        #print(len(newVibArray[1]), newVibArray[1][:5])
        #print(len(newVibArray[2]), newVibArray[2][:5])

        # Restructure data
        combinedArray = []
        for i in range(len(tsPWM)):
            combinedArray.append([
                tsPWM[i]-tsPWM[0],
                pwmArray[i],
                [
                    newVibArray[0][i],
                    newVibArray[1][i],
                    newVibArray[2][i]
                ]
            ])

    #print('Combined data result: {} | {}'.format(len(newVibArray[0]), len(tsPWM)))
    # combinedArray = [tsVib, newPWMArray, *vibArray]
    return combinedArray


desc = 'aug9_0_hover10s.json'
Nav = NavData()
Vib = VibData()

# Get timestamp range
tsStart, tsStop = Nav.getTimestampRangeByDescription(
    description=desc,
    landed=True,
)

# Store navdata to local active data
navId = Nav.storeData(
    data_array=Nav.getByDescription(
        description=desc,
        landed=True),
    description=desc
)

# Store vibdata to local active data
vibId = Vib.storeData(
    data_array=Vib.getBetweenTimestamp(
        lower=tsStart,
        upper=tsStop),
    description=desc
)

combinedArray = combineArray(
    pwmdata=[Nav.getTimestampArray(navId), Nav.getPWMArray(navId)[0]],
    vibdata=[Vib.getTimestampArray(vibId), Vib.getVibArray(vibId)[:3]],
    timedelta=True
)

plt.plot(list(range(len(combinedArray))), [data[1] for data in combinedArray])
plt.plot(list(range(len(combinedArray))), [data[2] for data in combinedArray])
plt.grid(True)
plt.show()

print(len(combinedArray))