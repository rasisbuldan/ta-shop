from navdataVibProc import NavData, VibData, NavdataVib
import pymongo
import sys

### Settings ###
startOffset = 2000
endOffset = 2000

dataTemplate = {
    'description': '',
    'timestamp': 0,
    'state': {
        'controlState': '',
        'batteryPercentage': 0,
        'batteryMillivolt': 0
    },
    'pwm': [0,0,0,0],
    'orientation': [0,0,0],
    'mpu1': [0,0,0],
    'mpu2': [0,0,0],
}

discardDesc = [
    '',
    'ascent.json',
    'ascent_trial.json',
    'calib_magneto.json',
    'front_back.json',
    'front_back_01.json',
    'front_back_02.json',
    'front_back_02_2.json',
    'hover30s.json',
    'hover30s_2.json',
    'hover30s_3.json',
    'hover30s_4.json',
    'hover_spin.json',
    'maneuv.json',
    'aug10_testfail.json'
]



onlyDesc = []


### Initialization ###
Nav = NavData()
Vib = VibData()
NV = NavdataVib()

# MongoDB Connector
clientDB = pymongo.MongoClient("localhost", 27017)
flightdataCollection = clientDB['test-db']['flightdatas']

descArray = Nav.listDescription()
#print(*descArray, sep='\n')
#sys.exit()

dArrCount = 0
for desc in descArray:
    dArrCount += 1

    if desc in discardDesc:
        continue

    if (onlyDesc != []) and (desc not in onlyDesc):
        continue

    print('[{}/{}] Processing:'.format(dArrCount, len(descArray)), desc)

    # Get navdata by description
    navArr = Nav.getByDescription(
        description=desc,
        landed=True
    )

    # Get timestamp range value
    tStart, tStop = Nav.getTimestampRangeByDescription(
        description=desc,
        landed=True
    )

    # Get vibdata by timestamp range
    print('[o] Get vibration data...', end='\r')
    vibdataArr = Vib.getBetweenTimestamp(tStart, tStop)

    # Conditional: missing data
    if len(vibdataArr) <= 0:
        print('Skipping dataset (vibdata empty)..')
        continue

    # Store navdata by description
    print('[o] Storing static data...', end='\r')
    navId = Nav.storeData(
        data_array=navArr,
        description=desc
    )

    # Store vibdata by timestamp range
    print('[o] Storing vibration data...', end='\r')
    vibId = Vib.storeData(
        data_array=vibdataArr,
        description=desc
    )

    # Combine data
    print('[o] Combining navigation and vibration data...', end='\r')
    combinedData = NV.combineDataMultiFeature(
        navdata=Nav.getMultiFeatureArray(navId),
        vibdata=Vib.getMultiFeatureArray(vibId)
    )

    
    ### Data trimming ###
    print('[o] Trimming data...', end='\r')

    # Get takeoff time
    takeoffIdx = 0
    while combinedData[takeoffIdx]['state']['controlState'] == 'CTRL_LANDED':
        takeoffIdx += 1
    
    # Get starting trim
    tTakeoff = combinedData[takeoffIdx]['timestamp']
    while (tTakeoff - combinedData[takeoffIdx]['timestamp']) < startOffset and (takeoffIdx > 0):
        takeoffIdx -= 1
    
    # Get landing time
    landingIdx = len(combinedData) - 1
    while combinedData[landingIdx]['state']['controlState'] == 'CTRL_LANDED':
        landingIdx -= 1
    
    # Get landing trim
    tLanding = combinedData[landingIdx]['timestamp']
    while (combinedData[landingIdx]['timestamp'] - tLanding) < endOffset and (landingIdx < (len(combinedData)-1)):
        landingIdx += 1

    combinedDataTrimmed = combinedData[takeoffIdx:landingIdx]

    timestart = combinedDataTrimmed[0]['timestamp']
    for data in combinedDataTrimmed:
        data['timestamp'] = data['timestamp'] - timestart
        data['description'] = desc
    
    flightdataCollection.insert_many(combinedDataTrimmed)

    print('[{}/{}] Processed:'.format(dArrCount, len(descArray)), desc)