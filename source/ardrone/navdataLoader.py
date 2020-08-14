'''
    NavData load from json file to MongoDB
    using folder file traversal 'data_path'
'''

from navdataJSONParser import getDataFromFile, getJSONArray
import pymongo
import sys
import os
import traceback

### Variable ###
data_path = 'C:/Users/rss75/Documents/GitHub/ta-shop/source/ardrone/nodejs/flight-data/aug_11'

dataTemplate = {
    "description": "",
    "timestart": 0,
    "timestamp": 0,
    "state": {
        "controlState": "",
        "batteryPercentage": 0,
        "batteryMillivolt": 0,
    },
    "navdata": {
        "altitude": 0,
        "orientation": {
            "roll": 0,
            "pitch": 0,
            "yaw": 0,
        },
        "pwm": {
            "mot1": 0,
            "mot2": 0,
            "mot3": 0,
            "mot4": 0,
        },
        "input": {
            "uroll": 0,
            "upitch": 0,
            "uyaw": 0,
        },
        "rawMeasures": {
            "accelerometers": {
                "x": 0,
                "y": 0,
                "z": 0,
            }
        }
    }
}

def getDirectoryList(data_dir):
    return [f for f in os.listdir(data_dir) if '.json' in f]

def loadDataset(filename):
    if filename == 'desktop.ini':
        return

    # Already stored in db
    if '_'.join(filename.split('_')[3:]) in descListStored:
        return

    # Filename parse
    fn = filename.split("_")
    timestart = int(fn[2].replace('.json',''))
    desc = '_'.join(fn[3:])

    # Load raw navdata from file
    rawNavData = getJSONArray(data_path + '/' + filename)

    # Load file-related values
    dataFileTemplate = dataTemplate.copy()
    dataFileTemplate['description'] = desc
    dataFileTemplate['timestart'] = timestart
    
    # Populate template with value from file
    i = 0
    rawNavDataLength = len(rawNavData)
    for nav in rawNavData:
        i += 1
        print('Loading [{}/{}] {:.1f}%'.format(i, rawNavDataLength, i/rawNavDataLength*100), end='\r')
        try:
            # Demo data
            if nav['droneState']['navdataDemo']:
                continue

            data = dataFileTemplate.copy()
            data['timestamp'] = nav['timestamp']
            

            # State
            data['state']['controlState'] = nav['demo']['controlState']
            data['state']['batteryPercentage'] = nav['demo']['batteryPercentage']
            data['state']['batteryMillivolt'] = nav['rawMeasures']['batteryMilliVolt']


            # Navdata
            data['navdata']['altitude'] = nav['demo']['altitude']
            
            data['navdata']['orientation']['roll'] = nav['demo']['leftRightDegrees']
            data['navdata']['orientation']['pitch'] = nav['demo']['frontBackDegrees']
            data['navdata']['orientation']['yaw'] = nav['demo']['clockwiseDegrees']

            data['navdata']['pwm']['mot1'] = nav['pwm']['motors'][0]
            data['navdata']['pwm']['mot2'] = nav['pwm']['motors'][1]
            data['navdata']['pwm']['mot3'] = nav['pwm']['motors'][2]
            data['navdata']['pwm']['mot4'] = nav['pwm']['motors'][3]

            data['navdata']['input']['uroll'] = nav['pwm']['uRoll']
            data['navdata']['input']['upitch'] = nav['pwm']['uPitch']
            data['navdata']['input']['uyaw'] = nav['pwm']['uYaw']

            data['navdata']['rawMeasures']['accelerometers']['x'] = nav['rawMeasures']['accelerometers']['x']
            data['navdata']['rawMeasures']['accelerometers']['y'] = nav['rawMeasures']['accelerometers']['y']
            data['navdata']['rawMeasures']['accelerometers']['z'] = nav['rawMeasures']['accelerometers']['z']
        
        except Exception as e:
            print(nav['droneState'])
            print(nav['demo'])
            print('\n')
            traceback.print_exc()
            sys.exit()

        # Insert into MongoDB
        idNum = navdataCollection.insert_one(data)
    
    print('Load {} with {} data points complete!'.format(filename, i))

### Main ###
# MongoDB Connector
clientDB = pymongo.MongoClient("localhost", 27017)
navdataCollection = clientDB['test-db']['navdatas']

dirlist = getDirectoryList(data_path)
descListLocal = ['_'.join(fname.split('_')[3:]) for fname in dirlist]
descListStored = list(navdataCollection.distinct('description'))
descDiff = [desc for desc in descListLocal if desc not in descListStored]

print('Directory list')
#print(*descListStored, sep='\n')
#print('---------------')
print(*dirlist, sep='\n')
print('---------------')
proceed = input('Proceed with dataset loading to DB? [y/n]: ')

if proceed == 'y':
    for file in dirlist:
        loadDataset(file)
else:
    sys.exit()