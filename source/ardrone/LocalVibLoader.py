'''
    NavData load from json file to MongoDB
    using folder file traversal 'data_path'
'''

import numpy as np
import pymongo
import sys
import os
import traceback

### Variable ###
stored_filename_path = 'C:/Users/rss75/Documents/GitHub/ta-shop/source/ardrone/stored_acceldatas.txt'
data_path = 'C:/Users/rss75/Documents/GitHub/ta-shop/source/ardrone/nodejs/flight-data/aug_11'

dataTemplate = {
    "timestamp": 0,
    "mpu1": {
        "x": 0,
        "y": 0,
        "z": 0
    },
    "mpu2": {
        "x": 0,
        "y": 0,
        "z": 0
    }
}

def getDataFromFile(filename):
    '''
        Retreive data from file as array
    '''
    filearr = np.genfromtxt(filename, delimiter=",")
    return filearr


def getDirectoryList(data_dir):
    '''
        Get list of file in directory
    '''
    return [f for f in os.listdir(data_dir) if '.txt' in f]


def loadDataset(filename):
    '''
        File traversal, load into database via dictionary
    '''
    # Check if filename already loaded
    with open(stored_filename_path) as stored_file:
        stored = stored_file.readlines()

        # Filename already stored
        if (filename + '\n') in stored:
            print('[!] File {} already stored!'.format(filename))
            return

    # Filename parse
    fn = filename.split("_")

    # Load raw navdata from file
    rawData = getDataFromFile(data_path + '/' + filename)
    
    # Populate template with value from file
    i = 0
    batchNum = 0
    rawDataLength = len(rawData)

    for data in rawData:
        i += 1
        print('Loading [{}/{}] {:.1f}%'.format(i, rawDataLength, i/rawDataLength*100), end='\r')
        try:
            payload = dataTemplate.copy()
            payload['timestamp'] = int(data[0]/1000000)

            payload['mpu1']['x'] = data[1]
            payload['mpu1']['y'] = data[2]
            payload['mpu1']['z'] = data[3]
            payload['mpu2']['x'] = data[4]
            payload['mpu2']['y'] = data[5]
            payload['mpu2']['z'] = data[6]


        except Exception as e:
            print('Data:', data)
            traceback.print_exc()
            sys.exit()

        # Insert into MongoDB
        idNum = staticpwmCollection.insert_one(payload)

    # Add to stored filename document
    with open(stored_filename_path, 'a+') as stored_file:
        stored_file.write(filename+'\n')
    
    print('Load {} with {} data points complete!'.format(filename, i))


### Main ###
# MongoDB Connector
clientDB = pymongo.MongoClient("localhost", 27017)
staticpwmCollection = clientDB['test-db']['acceldatas']

dirlist = getDirectoryList(data_path)
for file in dirlist:
    loadDataset(file)